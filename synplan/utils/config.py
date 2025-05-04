"""Module containing configuration classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union
from chython import smarts

import yaml
from CGRtools.containers import MoleculeContainer, QueryContainer


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
    :param single_reactant_only: If True, includes only reaction rules
        with a single reactant molecule.
    :param atom_info_retention: Controls the amount of information about
        each atom to retain ('none', 'reaction_center', or 'all').
    """

    # default low-level parameters
    single_reactant_only: bool = True
    keep_metadata: bool = False
    reactor_validation: bool = True
    reverse_rule: bool = True
    as_query_container: bool = True
    include_func_groups: bool = False
    func_groups_list: List[str] = field(default_factory=list)

    # adjustable parameters
    environment_atom_count: int = 1
    min_popularity: int = 3
    include_rings: bool = True
    multicenter_rules: bool = True
    keep_leaving_groups: bool = True
    keep_incoming_groups: bool = True
    keep_reagents: bool = False
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

        if not isinstance(params["single_reactant_only"], bool):
            raise ValueError("single_reactant_only must be a boolean.")

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
    :param evaluation_type: The method used for evaluating nodes.
        Options are "random", "rollout", "gcn".
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
    ucb_type: str = "uct"
    c_ucb: float = 0.1
    backprop_type: str = "muzero"
    search_strategy: str = "expansion_first"
    exclude_small: bool = True
    evaluation_agg: str = "max"
    evaluation_type: str = "gcn"
    init_node_value: float = 0.0
    epsilon: float = 0.0
    min_mol_size: int = 6
    silent: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "TreeConfig":
        return TreeConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "TreeConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return TreeConfig.from_dict(config_dict)

    def _validate_params(self, params):
        if params["ucb_type"] not in ["puct", "uct", "value"]:
            raise ValueError(
                "Invalid ucb_type. Allowed values are 'puct', 'uct', 'value'."
            )
        if params["backprop_type"] not in ["muzero", "cumulative"]:
            raise ValueError(
                "Invalid backprop_type. Allowed values are 'muzero', 'cumulative'."
            )
        if params["evaluation_type"] not in ["random", "rollout", "gcn"]:
            raise ValueError(
                "Invalid evaluation_type. Allowed values are 'random', 'rollout', 'gcn'."
            )
        if params["evaluation_agg"] not in ["max", "average"]:
            raise ValueError(
                "Invalid evaluation_agg. Allowed values are 'max', 'average'."
            )
        if not isinstance(params["c_ucb"], float):
            raise TypeError("c_ucb must be a float.")
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
        if not isinstance(params["init_node_value"], float):
            raise TypeError("init_node_value must be a float if provided.")
        if params["search_strategy"] not in ["expansion_first", "evaluation_first"]:
            raise ValueError(
                f"Invalid search_strategy: {params['search_strategy']}: "
                f"Allowed values are 'expansion_first', 'evaluation_first'"
            )
        if not isinstance(params["epsilon"], float) or 0 >= params["epsilon"] >= 1:
            raise ValueError("epsilon epsilon be a positive float between 0 and 1.")
        if not isinstance(params["min_mol_size"], int) or params["min_mol_size"] < 0:
            raise ValueError("min_mol_size must be a non-negative integer.")


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
