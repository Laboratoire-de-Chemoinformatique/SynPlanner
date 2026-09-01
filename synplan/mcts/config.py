"""Configuration for the search tree and its node-evaluation strategies."""

from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field

from synplan.utils.config import BaseConfigModel


class TreeConfig(BaseConfigModel):
    """Configuration class for the tree search algorithm.

    :param direction: "retro" (default) searches backwards from the target to
        the building blocks; "forward" grows forwards to them. In forward mode
        ``building_blocks`` is the GOAL rather than the stock, and
        :class:`~synplan.mcts.tree.Tree` refuses an evaluator whose own copy of
        the goal or of ``min_mol_size`` disagrees with the tree's. That
        termination condition, and its consistency with the evaluator, is the
        whole of what "forward" configures. Which rules can actually fire is a
        separate matter: unimolecular forward rules (rearrangements,
        deprotections, cyclisations) work today, bimolecular ones never fire,
        because expansion applies a rule to one structure and never supplies
        the partner — ``apply_reaction_rule`` takes ``co_reactants`` but the
        tree does not pass it, and partner selection is unimplemented.
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
    :param use_priority: When ``True``, curated priority rules passed to
        :class:`~synplan.mcts.tree.Tree` (``priority_rules=...``) are tried
        ahead of the policy on every expansion. Each priority rule that
        matches the current precursor enters with ``prob=1.0``; combined
        with the per-fragment multiplier in ``_add_child_if_new``, an
        N-fragment priority disconnect produces ``Node.prob = N`` and
        therefore dominates UCB sibling selection. Defaults to ``False``.
    :param priority_rule_multiapplication: When ``True``, priority rules are
        applied repeatedly to a single precursor until no further match is
        possible (BFS to fixpoint), instead of stopping at the first match.
        Useful for repeated motifs such as multi-site deprotections. Has no
        effect on policy rules. Defaults to ``False``.
    """

    # "retro" walks from the target down to `building_blocks`; "forward" grows a start material up
    # to it. `building_blocks` therefore means STOCK in retro and GOAL in forward — the name is
    # wrong for one of the two, but it is on Tree, RolloutSimulator, the CLI and every config file,
    # so the rename is its own change. What this flag buys today is that Tree refuses a forward
    # search whose evaluator is still scoring the retro finish line. It does NOT make bimolecular
    # forward rules fire: expansion never passes `co_reactants`, so only unimolecular ones match.
    direction: Literal["retro", "forward"] = "retro"
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

    policy_network: Any  # Policy - using Any to avoid circular import
    reaction_rules: Any  # List[Reactor]
    building_blocks: Any  # Set[str]
    building_block_candidates: Any | None = None
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
