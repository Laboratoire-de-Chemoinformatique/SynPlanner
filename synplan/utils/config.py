"""Module containing configuration classes."""

from pathlib import Path
from typing import Any, Literal

import yaml
from chython import smarts
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BaseConfigModel(BaseModel):
    """Base class for all SynPlanner configuration models.

    Replaces ConfigABC. Provides backward-compatible from_dict(), from_yaml(),
    to_dict(), to_yaml() methods as thin wrappers over Pydantic's native API.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        """Create instance from dictionary. Backward-compatible wrapper."""
        return cls.model_validate(config_dict)

    @classmethod
    def from_yaml(cls, file_path: str):
        """Create instance from YAML file. Backward-compatible wrapper."""
        with open(file_path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary. Backward-compatible wrapper."""
        d = self.model_dump()
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}

    def to_yaml(self, file_path: str) -> None:
        """Serialize to YAML file. Backward-compatible wrapper."""
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support unpickling from old dataclass-based configs.

        Old dataclass configs stored field values directly in __dict__.
        Pydantic models need proper __init__-based construction, so we
        re-validate the state dict through model_validate.
        """
        # Pydantic v2 pickle format uses __dict__ + __pydantic_fields_set__
        if "__dict__" in state and "__pydantic_fields_set__" in state:
            # Normal Pydantic pickle — delegate to default
            super().__setstate__(state)  # type: ignore[misc]
        else:
            # Legacy dataclass pickle: state is the raw field dict.
            # Strip fields unknown to the current model to avoid
            # extra="forbid" validation errors from removed fields.
            known = set(self.__class__.model_fields)
            cleaned = {k: v for k, v in state.items() if k in known}
            obj = self.__class__.model_validate(cleaned)
            super().__setstate__(  # type: ignore[misc]
                {
                    "__dict__": obj.__dict__,
                    "__pydantic_fields_set__": obj.__pydantic_fields_set__,
                }
            )


class RuleExtractionConfig(BaseConfigModel):
    """Configuration class for extracting reaction rules.

    :param multicenter_rules: If True, extracts a single rule
        encompassing all centers. If False, extracts separate reaction
        rules for each reaction center in a multicenter reaction.
    :param as_query_container: If True, the extracted rules are
        generated as QueryContainer objects, analogous to SMARTS objects
        for pattern matching in chemical structures.
    :param reverse_rule: If True, reverses the direction of the reaction
        for rule extraction.
    :param reactor_validation: If True, validates each generated rule in
        a chemical reactor to ensure correct generation of products from
        reactants.
    :param include_func_groups: If True, includes specific functional
        groups in the reaction rule in addition to the reaction center
        and its environment.
    :param func_groups_list: A list of functional groups to be
        considered when include_func_groups is True.
    :param include_rings: If True, includes ring structures in the
        reaction rules.
    :param keep_leaving_groups: If True, retains leaving groups in the
        extracted reaction rule.
    :param keep_incoming_groups: If True, retains incoming groups in the
        extracted reaction rule.
    :param keep_reagents: If True, includes reagents in the extracted
        reaction rule.
    :param environment_atom_count: Defines the size of the environment
        around the reaction center to be included in the rule (0 for
        only the reaction center, 1 for the first environment, etc.).
    :param min_popularity: Minimum number of times a rule must be
        applied to be considered for further analysis.
    :param keep_metadata: If True, retains metadata associated with the
        reaction in the extracted rule.
    :param single_product_only: If True, skips reactions that have more than
        one product (after reagent removal).
    :param atom_info_retention: Controls the amount of information about
        each atom to retain ('none', 'reaction_center', or 'all').
    """

    # default low-level parameters
    keep_metadata: bool = False
    reactor_validation: bool = True
    reverse_rule: bool = True
    as_query_container: bool = True
    single_product_only: bool = True

    # adjustable parameters
    environment_atom_count: int = 1
    min_popularity: int = 3
    include_rings: bool = True
    multicenter_rules: bool = True
    include_func_groups: bool = False
    keep_leaving_groups: bool = True
    keep_incoming_groups: bool = True
    keep_reagents: bool = False
    func_groups_list: list[str] = Field(default_factory=list)
    atom_info_retention: dict[str, dict[str, bool]] = Field(default_factory=dict)

    @field_validator("atom_info_retention")
    @classmethod
    def _validate_atom_info_retention(
        cls, v: dict[str, dict[str, bool]]
    ) -> dict[str, dict[str, bool]]:
        if v:
            required_keys = {"reaction_center", "environment"}
            if not required_keys.issubset(v):
                missing_keys = required_keys - set(v.keys())
                raise ValueError(
                    f"atom_info_retention missing required keys: {missing_keys}"
                )
            expected_subkeys = {"neighbors", "implicit_hydrogens", "ring_sizes"}
            for key, value in v.items():
                if key not in required_keys:
                    raise ValueError(f"Unexpected key in atom_info_retention: {key}")
                if not isinstance(value, dict) or not expected_subkeys.issubset(value):
                    missing_subkeys = expected_subkeys - set(value.keys())
                    raise ValueError(
                        f"Invalid structure for {key} in atom_info_retention. "
                        f"Missing subkeys: {missing_subkeys}"
                    )
                for subkey, subvalue in value.items():
                    if not isinstance(subvalue, bool):
                        raise ValueError(
                            f"Value for {subkey} in {key} of atom_info_retention "
                            f"must be boolean."
                        )
        return v

    @model_validator(mode="after")
    def _post_init(self) -> "RuleExtractionConfig":
        self._initialize_default_atom_info_retention()
        self._parse_functional_groups()
        return self

    def _initialize_default_atom_info_retention(self):
        default_atom_info = {
            "reaction_center": {
                "neighbors": True,
                "implicit_hydrogens": False,
                "ring_sizes": False,
            },
            "environment": {
                "neighbors": False,
                "implicit_hydrogens": False,
                "ring_sizes": False,
            },
        }

        if not self.atom_info_retention:
            self.atom_info_retention = default_atom_info
        else:
            for key in default_atom_info:
                self.atom_info_retention[key].update(
                    self.atom_info_retention.get(key, {})
                )

    def _parse_functional_groups(self):
        func_groups_list = []
        for group_smarts in self.func_groups_list:
            try:
                query = smarts(group_smarts)
                func_groups_list.append(query)
            except Exception as e:
                print(f"Functional group {group_smarts} was not parsed because of {e}")
        self.func_groups_list = func_groups_list


class PolicyNetworkConfig(BaseConfigModel):
    """Configuration class for the policy network.

    :param vector_dim: Dimension of the input vectors.
    :param batch_size: Number of samples per batch.
    :param dropout: Dropout rate for regularization.
    :param learning_rate: Learning rate for the optimizer.
    :param num_conv_layers: Number of convolutional layers in the network.
    :param num_epoch: Number of training epochs.
    :param policy_type: Mode of operation, either 'filtering' or 'ranking'.
    :param logger: Training logger configuration. ``None`` disables logging.
        A dict with ``"type"`` key (``"csv"``, ``"tensorboard"``, ``"mlflow"``,
        or ``"wandb"``) and optional logger-specific parameters passed to the
        PyTorch Lightning logger constructor.
    """

    policy_type: Literal["filtering", "ranking"] = "ranking"
    embedder_type: Literal["gcn", "gcn_concat", "gps"] = "gcn"
    vector_dim: int = Field(default=256, gt=0)
    batch_size: int = Field(default=500, gt=0)
    dropout: float = Field(default=0.4, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.008, gt=0.0)
    num_conv_layers: int = Field(default=5, gt=0)
    num_epoch: int = Field(default=100, gt=0)
    weights_path: str | Path | None = None

    # GPS embedder parameters (only used when embedder_type="gps")
    heads: int = Field(default=4, gt=0)
    attn_type: Literal["performer", "multihead"] = "performer"
    attn_dropout: float = Field(default=0.5, ge=0.0, le=1.0)

    # training logger (None disables logging, or dict with "type" + logger kwargs)
    logger: dict | None = None

    # extra Trainer kwargs (None = use defaults, or dict passed to Lightning Trainer)
    trainer: dict | None = None

    # logging gradient norms per module (embedder, y_predictor, etc.)
    log_grad_norm: bool = False

    # for filtering policy
    priority_rules_fraction: float = Field(default=0.5, ge=0.0)
    rule_prob_threshold: float = Field(default=0.0, ge=0.0)
    top_rules: int = Field(default=50, gt=0)

    @field_validator("logger")
    @classmethod
    def _validate_logger(cls, v: dict | None) -> dict | None:
        if v is not None:
            if "type" not in v:
                raise ValueError("logger dict must contain a 'type' key.")
            valid_types = ("csv", "tensorboard", "mlflow", "wandb")
            if v["type"] not in valid_types:
                raise ValueError(
                    f"logger type must be one of {valid_types}, got '{v['type']}'"
                )
        return v


class ValueNetworkConfig(BaseConfigModel):
    """Configuration class for the value network.

    :param vector_dim: Dimension of the input vectors.
    :param batch_size: Number of samples per batch.
    :param dropout: Dropout rate for regularization.
    :param learning_rate: Learning rate for the optimizer.
    :param num_conv_layers: Number of convolutional layers in the network.
    :param num_epoch: Number of training epochs.
    """

    weights_path: str | Path | None = None
    vector_dim: int = Field(default=256, gt=0)
    batch_size: int = Field(default=500, gt=0)
    dropout: float = Field(default=0.4, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.008, gt=0.0)
    num_conv_layers: int = Field(default=5, gt=0)
    num_epoch: int = Field(default=100, gt=0)


class TuningConfig(BaseConfigModel):
    """Configuration class for the network training.

    :param batch_size: The number of targets per batch in the planning simulation step.
    :param num_simulations: The number of planning simulations.
    """

    batch_size: int = Field(default=100, gt=0)
    num_simulations: int = 1


class TreeConfig(BaseConfigModel):
    """Configuration class for the tree search algorithm.

    :param max_iterations: The number of iterations to run the algorithm
        for.
    :param max_tree_size: The maximum number of nodes in the tree.
    :param max_time: The time limit (in seconds) for the algorithm to
        run.
    :param max_depth: The maximum depth of the tree.
    :param ucb_type: Type of UCB used in the search algorithm. Options
        are "puct", "uct", "value", defaults to "uct".
    :param c_ucb: The exploration-exploitation balance coefficient used
        in Upper Confidence Bound (UCB).
    :param backprop_type: Type of backpropagation algorithm. Options are
        "muzero", "cumulative", defaults to "muzero".
    :param search_strategy: The strategy used for tree search. Options
        are "expansion_first", "evaluation_first".
    :param exclude_small: Whether to exclude small molecules during the
        search.
    :param evaluation_agg: Method for aggregating evaluation scores.
        Options are "max", "average", defaults to "max".
    :param normalize_scores: Whether to normalize evaluation scores to [0, 1].
    :param init_node_value: Initial value for a new node.
    :param epsilon: A parameter in the epsilon-greedy search strategy
        representing the chance of random selection of reaction rules
        during the selection stage in Monte Carlo Tree Search,
        specifically during Upper Confidence Bound estimation. It
        balances between exploration and exploitation.
    :param min_mol_size: Defines the minimum size of a molecule that is
        have to be synthesized. Molecules with 6 or fewer heavy atoms
        are assumed to be building blocks by definition, thus setting
        the threshold for considering larger molecules in the search,
        defaults to 6.
    :param silent: Whether to suppress progress output.
    :param nmcs_level: Nesting level for NMCS and LazyNMCS algorithms.
        Higher levels provide more thorough search but are more
        computationally expensive. Defaults to 2.
    :param nmcs_playout_mode: Playout mode for NMCS base-level rollouts.
        Options are "greedy" (best value), "random", or "policy"
        (best policy probability). Defaults to "greedy".
    :param lnmcs_ratio: Pruning percentile for LazyNMCS algorithm.
        Only candidates scoring above this percentile threshold are
        explored. Value in range [0.0, 1.0]. Defaults to 0.2.
    """

    max_iterations: int = Field(default=100, gt=0)
    max_tree_size: int = Field(default=1000000, gt=0)
    max_time: float = Field(default=600, gt=0)
    max_depth: int = Field(default=6, gt=0)
    exclude_small: bool = True
    min_mol_size: int = Field(default=6, ge=0)
    silent: bool = False

    # new parameters
    algorithm: str = "uct"
    normalize_scores: bool = False
    max_rules_applied: int = 10  # needed only in pruning
    stop_at_first: bool = False
    enable_pruning: bool = False
    use_priority: bool = False
    priority_rule_multiapplication: bool = False
    policy_rule_source_name: str = "policy"
    priority_rule_source_name: str = "priority"

    # UCT configuration
    search_strategy: Literal["expansion_first", "evaluation_first"] = "expansion_first"
    ucb_type: Literal["uct", "puct", "value"] = "uct"
    c_ucb: float = Field(default=0.1, ge=0.0)
    backprop_type: Literal["muzero", "cumulative"] = "muzero"
    evaluation_agg: Literal["max", "average"] = "max"
    epsilon: float = Field(default=0.0, ge=0.0, le=1.0)
    init_node_value: float = Field(default=0.5, ge=0.0, le=1.0)
    beam_width: int = Field(default=10, gt=0)

    # NMCS configuration
    nmcs_level: int = Field(default=2, gt=0)
    nmcs_playout_mode: Literal["greedy", "random", "policy"] = "greedy"

    # LazyNMCS configuration
    lnmcs_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class RolloutEvaluationConfig(BaseConfigModel):
    """Configuration for rollout-based evaluation strategy.

    Contains all dependencies needed for rollout simulation.

    :param policy_network: Policy network function for rollout simulation.
    :param reaction_rules: List of reaction rules for applying transformations.
    :param building_blocks: Set of building block molecules.
    :param min_mol_size: Minimum molecule size to consider for expansion.
    :param max_depth: Maximum depth for rollout simulation.
    :param normalize: Whether to normalize scores to [0, 1].
    :param stochastic: If True, sample from valid rules using policy probabilities.
        If False (default), use greedy selection (first successful rule).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    policy_network: Any  # PolicyNetworkFunction - using Any to avoid circular import
    reaction_rules: Any  # List[Reactor]
    building_blocks: Any  # Set[str]
    min_mol_size: int = Field(default=0, ge=0)
    max_depth: int = Field(default=6, gt=0)
    normalize: bool = False
    stochastic: bool = False


class ValueNetworkEvaluationConfig(BaseConfigModel):
    """Configuration for value network-based evaluation strategy.

    :param weights_path: Path to the value network weights file.
    :param normalize: Whether to normalize scores to [0, 1].
    """

    weights_path: str | Path
    normalize: bool = False


class RDKitEvaluationConfig(BaseConfigModel):
    """Configuration for RDKit-based evaluation strategy.

    Uses molecular descriptors like SA score, molecular weight, etc.

    :param score_function: Name of the scoring function to use.
        Options: "sascore", "weight", "heavyAtomCount", "weightXsascore", "WxWxSAS".
    :param normalize: Whether to normalize scores to [0, 1].
    """

    score_function: Literal[
        "sascore", "weight", "heavyAtomCount", "weightXsascore", "WxWxSAS"
    ] = "sascore"
    normalize: bool = False


class PolicyEvaluationConfig(BaseConfigModel):
    """Configuration for policy-based evaluation strategy.

    Uses policy network probabilities as evaluation scores.

    :param normalize: Whether to normalize scores to [0, 1].
    """

    normalize: bool = False


class RandomEvaluationConfig(BaseConfigModel):
    """Configuration for random evaluation strategy.

    Assigns random scores - useful for testing and baseline comparisons.

    :param normalize: Whether to normalize scores to [0, 1].
    """

    normalize: bool = False


class CombinedPolicyConfig(BaseConfigModel):
    """Configuration for combined filtering + ranking policy.

    Combines filtering and ranking policies by weighted addition of logits:
        combined_logits = filtering_logits + ranking_weight * ranking_logits
        combined_probs = softmax(combined_logits / temperature)

    The filtering policy provides applicability scores (trained on multi-label applicability).
    The ranking policy provides feasibility scores (trained on actual reactions).

    :param filtering_weights_path: Path to the filtering policy network weights.
    :param ranking_weights_path: Path to the ranking policy network weights.
    :param top_rules: Number of top rules to return.
    :param rule_prob_threshold: Minimum probability threshold for returning a rule.
    :param ranking_weight: Weight for ranking logits (default 1.0).
        Values > 1.0 give more weight to ranking (feasibility).
    :param temperature: Temperature for softmax (default 1.0).
        Values > 1.0 produce softer distributions (more exploration).
    """

    filtering_weights_path: str | Path
    ranking_weights_path: str | Path
    top_rules: int = Field(default=50, gt=0)
    rule_prob_threshold: float = 0.0
    ranking_weight: float = Field(default=1.0, gt=0.0)
    temperature: float = Field(default=1.0, gt=0.0)
