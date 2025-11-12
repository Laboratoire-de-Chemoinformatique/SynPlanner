"""Module containing configuration classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from chython import smarts
import yaml


@dataclass
class ConfigABC(ABC):
    """Abstract base class for configuration classes."""

    @staticmethod
    @abstractmethod
    def from_dict(config_dict: Dict[str, Any]):
        """Create an instance of the configuration from a dictionary."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration into a dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()
        }

    @staticmethod
    @abstractmethod
    def from_yaml(file_path: str):
        """Deserialize a YAML file into a configuration object."""

    def to_yaml(self, file_path: str):
        """Serializes the configuration to a YAML file.

        :param file_path: The path to the output YAML file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(self.to_dict(), file)

    @abstractmethod
    def _validate_params(self, params: Dict[str, Any]):
        """Validate configuration parameters."""

    def __post_init__(self):
        """Validates the configuration parameters."""
        # call _validate_params method after initialization
        params = self.to_dict()
        self._validate_params(params)


@dataclass
class RuleExtractionConfig(ConfigABC):
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
    :param single_product_only: If True, includes only reaction rules
        with a single reactant molecule.
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
    func_groups_list: List[str] = field(default_factory=list)
    atom_info_retention: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self._validate_params(self.to_dict())
        self._initialize_default_atom_info_retention()
        self._parse_functional_groups()

    def _initialize_default_atom_info_retention(self):
        default_atom_info = {
            "reaction_center": {
                "neighbors": True,
                "hybridization": True,
                "implicit_hydrogens": False,
                "ring_sizes": False,
            },
            "environment": {
                "neighbors": False,
                "hybridization": False,
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

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "RuleExtractionConfig":
        return RuleExtractionConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "RuleExtractionConfig":

        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return RuleExtractionConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]) -> None:

        if not isinstance(params["multicenter_rules"], bool):
            raise ValueError("multicenter_rules must be a boolean.")

        if not isinstance(params["as_query_container"], bool):
            raise ValueError("as_query_container must be a boolean.")

        if not isinstance(params["reverse_rule"], bool):
            raise ValueError("reverse_rule must be a boolean.")

        if not isinstance(params["reactor_validation"], bool):
            raise ValueError("reactor_validation must be a boolean.")

        if not isinstance(params["include_func_groups"], bool):
            raise ValueError("include_func_groups must be a boolean.")

        if params["func_groups_list"] is not None and not all(
            isinstance(group, str) for group in params["func_groups_list"]
        ):
            raise ValueError("func_groups_list must be a list of SMARTS.")

        if not isinstance(params["include_rings"], bool):
            raise ValueError("include_rings must be a boolean.")

        if not isinstance(params["keep_leaving_groups"], bool):
            raise ValueError("keep_leaving_groups must be a boolean.")

        if not isinstance(params["keep_incoming_groups"], bool):
            raise ValueError("keep_incoming_groups must be a boolean.")

        if not isinstance(params["keep_reagents"], bool):
            raise ValueError("keep_reagents must be a boolean.")

        if not isinstance(params["environment_atom_count"], int):
            raise ValueError("environment_atom_count must be an integer.")

        if not isinstance(params["min_popularity"], int):
            raise ValueError("min_popularity must be an integer.")

        if not isinstance(params["keep_metadata"], bool):
            raise ValueError("keep_metadata must be a boolean.")

        if not isinstance(params["single_product_only"], bool):
            raise ValueError("single_product_only must be a boolean.")

        if params["atom_info_retention"] is not None:
            if not isinstance(params["atom_info_retention"], dict):
                raise ValueError("atom_info_retention must be a dictionary.")

            required_keys = {"reaction_center", "environment"}
            if not required_keys.issubset(params["atom_info_retention"]):
                missing_keys = required_keys - set(params["atom_info_retention"].keys())
                raise ValueError(
                    f"atom_info_retention missing required keys: {missing_keys}"
                )

            for key, value in params["atom_info_retention"].items():
                if key not in required_keys:
                    raise ValueError(f"Unexpected key in atom_info_retention: {key}")

                expected_subkeys = {
                    "neighbors",
                    "hybridization",
                    "implicit_hydrogens",
                    "ring_sizes",
                }
                if not isinstance(value, dict) or not expected_subkeys.issubset(value):
                    missing_subkeys = expected_subkeys - set(value.keys())
                    raise ValueError(
                        f"Invalid structure for {key} in atom_info_retention. Missing subkeys: {missing_subkeys}"
                    )

                for subkey, subvalue in value.items():
                    if not isinstance(subvalue, bool):
                        raise ValueError(
                            f"Value for {subkey} in {key} of atom_info_retention must be boolean."
                        )


@dataclass
class PolicyNetworkConfig(ConfigABC):
    """Configuration class for the policy network.

    :param vector_dim: Dimension of the input vectors.
    :param batch_size: Number of samples per batch.
    :param dropout: Dropout rate for regularization.
    :param learning_rate: Learning rate for the optimizer.
    :param num_conv_layers: Number of convolutional layers in the network.
    :param num_epoch: Number of training epochs.
    :param policy_type: Mode of operation, either 'filtering' or 'ranking'.
    """

    policy_type: str = "ranking"
    vector_dim: int = 256
    batch_size: int = 500
    dropout: float = 0.4
    learning_rate: float = 0.008
    num_conv_layers: int = 5
    num_epoch: int = 100
    weights_path: str = None

    # for filtering policy
    priority_rules_fraction: float = 0.5
    rule_prob_threshold: float = 0.0
    top_rules: int = 50

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "PolicyNetworkConfig":
        return PolicyNetworkConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "PolicyNetworkConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return PolicyNetworkConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):

        if params["policy_type"] not in ["filtering", "ranking"]:
            raise ValueError("policy_type must be either 'filtering' or 'ranking'.")

        if not isinstance(params["vector_dim"], int) or params["vector_dim"] <= 0:
            raise ValueError("vector_dim must be a positive integer.")

        if not isinstance(params["batch_size"], int) or params["batch_size"] <= 0:
            raise ValueError("batch_size must be a positive integer.")

        if (
            not isinstance(params["num_conv_layers"], int)
            or params["num_conv_layers"] <= 0
        ):
            raise ValueError("num_conv_layers must be a positive integer.")

        if not isinstance(params["num_epoch"], int) or params["num_epoch"] <= 0:
            raise ValueError("num_epoch must be a positive integer.")

        if not isinstance(params["dropout"], float) or not (
            0.0 <= params["dropout"] <= 1.0
        ):
            raise ValueError("dropout must be a float between 0.0 and 1.0.")

        if (
            not isinstance(params["learning_rate"], float)
            or params["learning_rate"] <= 0.0
        ):
            raise ValueError("learning_rate must be a positive float.")

        if (
            not isinstance(params["priority_rules_fraction"], float)
            or params["priority_rules_fraction"] < 0.0
        ):
            raise ValueError(
                "priority_rules_fraction must be a non-negative positive float."
            )

        if (
            not isinstance(params["rule_prob_threshold"], float)
            or params["rule_prob_threshold"] < 0.0
        ):
            raise ValueError("rule_prob_threshold must be a non-negative float.")

        if not isinstance(params["top_rules"], int) or params["top_rules"] <= 0:
            raise ValueError("top_rules must be a positive integer.")


@dataclass
class ValueNetworkConfig(ConfigABC):
    """Configuration class for the value network.

    :param vector_dim: Dimension of the input vectors.
    :param batch_size: Number of samples per batch.
    :param dropout: Dropout rate for regularization.
    :param learning_rate: Learning rate for the optimizer.
    :param num_conv_layers: Number of convolutional layers in the network.
    :param num_epoch: Number of training epochs.
    """

    weights_path: str = None
    vector_dim: int = 256
    batch_size: int = 500
    dropout: float = 0.4
    learning_rate: float = 0.008
    num_conv_layers: int = 5
    num_epoch: int = 100

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "ValueNetworkConfig":
        return ValueNetworkConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "ValueNetworkConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return ValueNetworkConfig.from_dict(config_dict)

    def to_yaml(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(self.to_dict(), file)

    def _validate_params(self, params: Dict[str, Any]):

        if not isinstance(params["vector_dim"], int) or params["vector_dim"] <= 0:
            raise ValueError("vector_dim must be a positive integer.")

        if not isinstance(params["batch_size"], int) or params["batch_size"] <= 0:
            raise ValueError("batch_size must be a positive integer.")

        if (
            not isinstance(params["num_conv_layers"], int)
            or params["num_conv_layers"] <= 0
        ):
            raise ValueError("num_conv_layers must be a positive integer.")

        if not isinstance(params["num_epoch"], int) or params["num_epoch"] <= 0:
            raise ValueError("num_epoch must be a positive integer.")

        if not isinstance(params["dropout"], float) or not (
            0.0 <= params["dropout"] <= 1.0
        ):
            raise ValueError("dropout must be a float between 0.0 and 1.0.")

        if (
            not isinstance(params["learning_rate"], float)
            or params["learning_rate"] <= 0.0
        ):
            raise ValueError("learning_rate must be a positive float.")


@dataclass
class TuningConfig(ConfigABC):
    """Configuration class for the network training.

    :param batch_size: The number of targets per batch in the planning simulation step.
    :param num_simulations: The number of planning simulations.
    """

    batch_size: int = 100
    num_simulations: int = 1

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "TuningConfig":
        return TuningConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "TuningConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return TuningConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):

        if not isinstance(params["batch_size"], int) or params["batch_size"] <= 0:
            raise ValueError("batch_size must be a positive integer.")


@dataclass
class TreeConfig(ConfigABC):
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
    """

    max_iterations: int = 100
    max_tree_size: int = 1000000
    max_time: float = 600
    max_depth: int = 6
    exclude_small: bool = True
    min_mol_size: int = 6
    silent: bool = False

    # new parameters
    algorithm: str = "uct"
    normalize_scores: bool = False
    max_rules_applied = 10  # needed only in pruning
    stop_at_first = False
    enable_pruning: bool = False

    # UCT configuration
    search_strategy: str = "expansion_first"
    ucb_type: str = "uct"  # one of: "uct", "puct", "value"
    c_ucb: float = 0.1  # exploration constant >= 0
    backprop_type: str = "muzero"  # one of: "muzero", "cumulative"
    evaluation_agg: str = "max"  # one of: "max", "average"
    epsilon: float = 0.0  # epsilon-greedy in [0.0, 1.0]
    init_node_value: float = 0.5  # initial node value in [0.0, 1.0]
    beam_width: int = 10

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "TreeConfig":
        return TreeConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "TreeConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return TreeConfig.from_dict(config_dict)

    def _validate_params(self, params):
        if not isinstance(params["max_depth"], int) or params["max_depth"] < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(params["max_tree_size"], int) or params["max_tree_size"] < 1:
            raise ValueError("max_tree_size must be a positive integer.")
        if (
            not isinstance(params["max_iterations"], int)
            or params["max_iterations"] < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(params["max_time"], int) or params["max_time"] < 1:
            raise ValueError("max_time must be a positive integer.")
        if not isinstance(params["exclude_small"], bool):
            raise TypeError("exclude_small must be a boolean.")
        if not isinstance(params["silent"], bool):
            raise TypeError("silent must be a boolean.")
        if params["search_strategy"] not in ["expansion_first", "evaluation_first"]:
            raise ValueError(
                f"Invalid search_strategy: {params['search_strategy']}: "
                f"Allowed values are 'expansion_first', 'evaluation_first'"
            )
        if not isinstance(params["min_mol_size"], int) or params["min_mol_size"] < 0:
            raise ValueError("min_mol_size must be a non-negative integer.")
        if not isinstance(params.get("enable_pruning", True), bool):
            raise ValueError("enable_pruning must be a boolean.")

        # Validate UCT-related parameters
        if params.get("ucb_type") not in ["uct", "puct", "value"]:
            raise ValueError("ucb_type must be 'uct', 'puct' or 'value'.")
        if not isinstance(params.get("c_ucb"), (float, int)) or params.get("c_ucb") < 0:
            raise ValueError("c_ucb must be a non-negative float.")
        if params.get("backprop_type") not in ["muzero", "cumulative"]:
            raise ValueError("backprop_type must be 'muzero' or 'cumulative'.")
        if params.get("evaluation_agg") not in ["max", "average"]:
            raise ValueError("evaluation_agg must be 'max' or 'average'.")
        if not isinstance(params.get("epsilon"), (float, int)) or not (
            0.0 <= float(params.get("epsilon")) <= 1.0
        ):
            raise ValueError("epsilon must be a float in [0.0, 1.0].")
        if not isinstance(params.get("init_node_value"), (float, int)) or not (
            0.0 <= float(params.get("init_node_value")) <= 1.0
        ):
            raise ValueError("init_node_value must be a float in [0.0, 1.0].")
        # Beam width
        if (
            not isinstance(params.get("beam_width", 10), int)
            or params.get("beam_width", 10) <= 0
        ):
            raise ValueError("beam_width must be a positive integer.")


@dataclass
class RolloutEvaluationConfig(ConfigABC):
    """Configuration for rollout-based evaluation strategy.

    Contains all dependencies needed for rollout simulation.

    :param policy_network: Policy network function for rollout simulation.
    :param reaction_rules: List of reaction rules for applying transformations.
    :param building_blocks: Set of building block molecules.
    :param min_mol_size: Minimum molecule size to consider for expansion.
    :param max_depth: Maximum depth for rollout simulation.
    :param normalize: Whether to normalize scores to [0, 1].
    """

    policy_network: Any  # PolicyNetworkFunction - using Any to avoid circular import
    reaction_rules: Any  # List[Reactor]
    building_blocks: Any  # Set[str]
    min_mol_size: int = 0
    max_depth: int = 6
    normalize: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "RolloutEvaluationConfig":
        return RolloutEvaluationConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "RolloutEvaluationConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return RolloutEvaluationConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        if (
            not isinstance(params.get("min_mol_size", 6), int)
            or params.get("min_mol_size", 6) < 0
        ):
            raise ValueError("min_mol_size must be a non-negative integer.")
        if (
            not isinstance(params.get("max_depth", 9), int)
            or params.get("max_depth", 9) < 1
        ):
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(params.get("normalize", False), bool):
            raise ValueError("normalize must be a boolean.")


@dataclass
class ValueNetworkEvaluationConfig(ConfigABC):
    """Configuration for value network-based evaluation strategy.

    :param weights_path: Path to the value network weights file.
    :param normalize: Whether to normalize scores to [0, 1].
    """

    weights_path: str
    normalize: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "ValueNetworkEvaluationConfig":
        return ValueNetworkEvaluationConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "ValueNetworkEvaluationConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return ValueNetworkEvaluationConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        if not isinstance(params.get("weights_path"), str):
            raise ValueError("weights_path must be a string.")
        if not isinstance(params.get("normalize", False), bool):
            raise ValueError("normalize must be a boolean.")


@dataclass
class RDKitEvaluationConfig(ConfigABC):
    """Configuration for RDKit-based evaluation strategy.

    Uses molecular descriptors like SA score, molecular weight, etc.

    :param score_function: Name of the scoring function to use.
        Options: "sascore", "weight", "heavyAtomCount", "weightXsascore", "WxWxSAS".
    :param normalize: Whether to normalize scores to [0, 1].
    """

    score_function: str = "sascore"
    normalize: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "RDKitEvaluationConfig":
        return RDKitEvaluationConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "RDKitEvaluationConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return RDKitEvaluationConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        valid_functions = [
            "sascore",
            "weight",
            "heavyAtomCount",
            "weightXsascore",
            "WxWxSAS",
        ]
        if params.get("score_function") not in valid_functions:
            raise ValueError(
                f"score_function must be one of {valid_functions}, got {params.get('score_function')}"
            )
        if not isinstance(params.get("normalize", False), bool):
            raise ValueError("normalize must be a boolean.")


@dataclass
class PolicyEvaluationConfig(ConfigABC):
    """Configuration for policy-based evaluation strategy.

    Uses policy network probabilities as evaluation scores.

    :param normalize: Whether to normalize scores to [0, 1].
    """

    normalize: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "PolicyEvaluationConfig":
        return PolicyEvaluationConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "PolicyEvaluationConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return PolicyEvaluationConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        if not isinstance(params.get("normalize", False), bool):
            raise ValueError("normalize must be a boolean.")


@dataclass
class RandomEvaluationConfig(ConfigABC):
    """Configuration for random evaluation strategy.

    Assigns random scores - useful for testing and baseline comparisons.

    :param normalize: Whether to normalize scores to [0, 1].
    """

    normalize: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "RandomEvaluationConfig":
        return RandomEvaluationConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "RandomEvaluationConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return RandomEvaluationConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        if not isinstance(params.get("normalize", False), bool):
            raise ValueError("normalize must be a boolean.")


def convert_config_to_dict(config_attr: ConfigABC, config_type) -> Dict | None:
    """Converts a configuration attribute to a dictionary if it's either a dictionary or
    an instance of a specified configuration type.

    :param config_attr: The configuration attribute to be converted.
    :param config_type: The type to check against for conversion.
    :return: The configuration attribute as a dictionary, or None if it's not an
        instance of the given type or dict.
    """
    if isinstance(config_attr, dict):
        return config_attr
    if isinstance(config_attr, config_type):
        return config_attr.to_dict()
    return None
