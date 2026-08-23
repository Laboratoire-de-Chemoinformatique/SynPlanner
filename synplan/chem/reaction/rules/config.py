"""Configuration for reaction-rule extraction."""

from chython import smarts
from pydantic import Field, field_validator, model_validator

from synplan.utils.config import BaseConfigModel


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
    :param ignore_stereo: If True, removes atom/bond stereochemistry from
        input reactions before rule extraction. This is useful for rule sets
        whose reactor/canonicalization path does not preserve stereo.
    :param worker_timeout_per_reaction: Seconds allowed per reaction in a
        parallel extraction batch. The worker timeout is this value multiplied
        by the configured batch size.
    :param atom_info_retention: Controls the amount of information about
        each atom to retain ('none', 'reaction_center', or 'all').
    """

    # default low-level parameters
    keep_metadata: bool = False
    reactor_validation: bool = True
    reverse_rule: bool = True
    as_query_container: bool = True
    single_product_only: bool = True
    ignore_stereo: bool = True
    worker_timeout_per_reaction: float = Field(default=10.0, gt=0)

    # adjustable parameters
    environment_atom_count: int = 1
    min_popularity: int = 3
    include_rings: bool = False
    multicenter_rules: bool = True
    include_func_groups: bool = False
    keep_leaving_groups: bool = True
    keep_incoming_groups: bool = False
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
