"""Module containing classes and functions for reactions standardizing.

This module contains the open-source code from
https://github.com/Laboratoire-de-Chemoinformatique/Reaction_Data_Cleaning/blob/master/scripts/standardizer.py
"""

import logging
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import ray
import yaml
from CGRtools import smiles as smiles_cgrtools
from CGRtools.containers import MoleculeContainer
from CGRtools.containers import ReactionContainer
from CGRtools.containers import ReactionContainer as ReactionContainerCGRTools
from chython import ReactionContainer as ReactionContainerChython
from chython import smiles as smiles_chython
from tqdm import tqdm

from synplan.chem.utils import unite_molecules
from synplan.utils.config import ConfigABC
from synplan.utils.files import ReactionReader, ReactionWriter
from synplan.utils.logging import HiddenPrints


class ReactionMappingConfig:
    pass


class ReactionMappingStandardizer:
    """Maps atoms of the reaction using chython (chytorch)."""

    @staticmethod
    def from_config(
        config: ReactionMappingConfig,
    ) -> "ReactionMappingStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return ReactionMappingStandardizer()

    def _map_and_remove_reagents(
        self, reaction: ReactionContainerChython
    ) -> ReactionContainerChython | None:
        try:
            reaction.reset_mapping()
            reaction.remove_reagents()
        except Exception as e:
            logging.debug(e)

        return reaction

    def _map_reaction(
        self, reaction: ReactionContainerCGRTools
    ) -> ReactionContainerCGRTools | None:
        """Reads a file of reactions and maps atoms of the reactions using chytorch.
        This function does not use the ReactionReader/ReactionWriter classes, because
        they are not compatible with chython.

        :param reaction: Input reaction.
        :return: None.
        """

        chython_reaction = smiles_chython(str(reaction))
        reaction_mapped = self._map_and_remove_reagents(chython_reaction)
        if reaction_mapped:
            reaction_mapped_cgrtools = smiles_cgrtools(format(chython_reaction, "m"))
            return reaction_mapped_cgrtools

        return None

    def __call__(
        self, reaction: ReactionContainerCGRTools
    ) -> ReactionContainerCGRTools | None:
        """Maps atoms of the reactions using chytorch.

        :param reaction: Input reaction.
        :return: Returns standardized reaction if the reaction has standardized
            successfully, else None.
        """
        try:
            self._map_reaction(reaction)
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


class FunctionalGroupsConfig:
    pass


class FunctionalGroupsStandardizer:
    """Functional groups standardization."""

    @staticmethod
    def from_config(
        config: FunctionalGroupsConfig,
    ) -> "FunctionalGroupsStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return FunctionalGroupsStandardizer()

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Functional groups standardization.

        :param reaction: Input reaction.
        :return: Returns standardized reaction if the reaction has standardized
            successfully, else None.
        """
        try:
            reaction.standardize()
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


class KekuleFormConfig:
    pass


class KekuleFormStandardizer:
    """Reactants/reagents/products kekulization."""

    @staticmethod
    def from_config(
        config: KekuleFormConfig,
    ) -> "KekuleFormStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return KekuleFormStandardizer()

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Reactants/reagents/products kekulization.

        :param reaction: Input reaction.
        :return: Returns standardized reaction if the reaction has standardized
            successfully, else None.
        """
        try:
            reaction.kekule()
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


class CheckValenceConfig:
    pass


class CheckValenceStandardizer:
    """Check valence."""

    @staticmethod
    def from_config(
        config: CheckValenceConfig,
    ) -> "CheckValenceStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return CheckValenceStandardizer()

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Check valence.

        :param reaction: Input reaction.
        :return: Returns reaction if the atom valences are correct, else None.
        """
        for molecule in reaction.reactants + reaction.products + reaction.reagents:
            valence_mistakes = molecule.check_valence()
            if valence_mistakes:
                return None
        return reaction


class ImplicifyHydrogensConfig:
    pass


class ImplicifyHydrogensStandardizer:
    """Implicify hydrogens."""

    @staticmethod
    def from_config(
        config: ImplicifyHydrogensConfig,
    ) -> "ImplicifyHydrogensStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return ImplicifyHydrogensStandardizer()

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Implicify hydrogens.

        :param reaction: Input reaction.
        :return: Returns reaction with removed hydrogens, else None.
        """
        try:
            reaction.implicify_hydrogens()
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


class CheckIsotopesConfig:
    pass


class CheckIsotopesStandardizer:
    """Check isotopes."""

    @staticmethod
    def from_config(
        config: CheckIsotopesConfig,
    ) -> "CheckIsotopesStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return CheckIsotopesStandardizer()

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Check isotopes.

        :param reaction: Input reaction.
        :return: Returns reaction with cleaned isotopes, else None.
        """
        is_isotope = False
        for molecule in reaction.reactants + reaction.products:
            for _, atom in molecule.atoms():
                if atom.isotope:
                    is_isotope = True

        if is_isotope:
            reaction.clean_isotopes()

        return reaction


class SplitIonsConfig:
    pass


class SplitIonsStandardizer:
    """Computing charge of molecule."""

    @staticmethod
    def from_config(
        config: SplitIonsConfig,
    ) -> "SplitIonsStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return SplitIonsStandardizer()

    def _calc_charge(self, molecule: MoleculeContainer) -> int:
        """Computing charge of molecule.

        :param molecule: Input reactant/reagent/product.
        :return: The total charge of the molecule.
        """
        return sum(molecule._charges.values())

    def _split_ions(self, reaction: ReactionContainer):
        """Split ions in a reaction.

        :param reaction: Input reaction.
        :return: A tuple with the corresponding reaction and
        a return code as int (0 - nothing was changed, 1 - ions were split, 2 - ions were split but the reaction
        is imbalanced).
        """
        meta = reaction.meta
        reaction_parts = []
        return_codes = []
        for molecules in (reaction.reactants, reaction.reagents, reaction.products):
            divided_molecules = [x for m in molecules for x in m.split(".")]

            total_charge = 0
            ions_present = False
            for molecule in divided_molecules:
                mol_charge = self._calc_charge(molecule)
                total_charge += mol_charge
                if mol_charge != 0:
                    ions_present = True

            if ions_present and total_charge:
                return_codes.append(2)
            elif ions_present:
                return_codes.append(1)
            else:
                return_codes.append(0)

            reaction_parts.append(tuple(divided_molecules))

        return ReactionContainer(
            reactants=reaction_parts[0],
            reagents=reaction_parts[1],
            products=reaction_parts[2],
            meta=meta,
        ), max(return_codes)

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Split ions.

        :param reaction: Input reaction.
        :return: Returns reaction with split ions, else None.
        """
        try:
            reaction, return_code = self._split_ions(reaction)
            if return_code in [0, 1]:  # ions were split
                return reaction
            if return_code == 2:  # ions were split but the reaction is imbalanced
                return None
        except Exception as e:
            logging.debug(e)
        return None


class AromaticFormConfig:
    pass


class AromaticFormStandardizer:
    """Aromatize molecules in reaction."""

    @staticmethod
    def from_config(
        config: AromaticFormConfig,
    ) -> "AromaticFormStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return AromaticFormStandardizer()

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Aromatize molecules in reaction.

        :param reaction: Input reaction.
        :return: Returns reaction with aromatized reactants/reagents/products, else
            None.
        """
        try:
            reaction.thiele()
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


class MappingFixConfig:
    pass


class MappingFixStandardizer:
    """Fix atom-to-atom mapping in reaction."""

    @staticmethod
    def from_config(
        config: MappingFixConfig,
    ) -> "MappingFixStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return MappingFixStandardizer()

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Fix atom-to-atom mapping in reaction.

        :param reaction: Input reaction.
        :return: Returns reaction with fixed atom-to-atom mapping, else None.
        """
        try:
            reaction.fix_mapping()
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


@dataclass
class UnchangedPartsConfig:
    pass


class UnchangedPartsStandardizer:
    """Ungroup molecules, remove unchanged parts from reactants and products."""

    def __init__(
        self,
        add_reagents_to_reactants: bool = False,
        keep_reagents: bool = False,
    ):
        self.add_reagents_to_reactants = add_reagents_to_reactants
        self.keep_reagents = keep_reagents

    @staticmethod
    def from_config(config: UnchangedPartsConfig) -> "UnchangedPartsStandardizer":
        """Creates an instance of CompeteProductsFilter from a configuration object."""
        return UnchangedPartsStandardizer()

    def _remove_unchanged_parts(self, reaction: ReactionContainer) -> ReactionContainer:
        """Ungroup molecules, remove unchanged parts from reactants and products.

        :param reaction: Input reaction.
        :return: Returns reaction with removed unchanged parts, else None.
        """
        meta = reaction.meta
        new_reactants = list(reaction.reactants)
        new_reagents = list(reaction.reagents)
        if self.add_reagents_to_reactants:
            new_reactants.extend(new_reagents)
            new_reagents = []
        reactants = new_reactants.copy()
        new_products = list(reaction.products)

        for reactant in reactants:
            if reactant in new_products:
                new_reagents.append(reactant)
                new_reactants.remove(reactant)
                new_products.remove(reactant)
        if not self.keep_reagents:
            new_reagents = []
        return ReactionContainer(
            reactants=tuple(new_reactants),
            reagents=tuple(new_reagents),
            products=tuple(new_products),
            meta=meta,
        )

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Ungroup molecules, remove unchanged parts from reactants and products.

        :param reaction: Input reaction.
        :return: Returns reaction with removed unchanged parts, else None.
        """
        try:
            reaction = self._remove_unchanged_parts(reaction)
            if not reaction.reactants and reaction.products:
                return None
            if not reaction.products and reaction.reactants:
                return None
            if not reaction.reactants and not reaction.products:
                return None
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


@dataclass
class SmallMoleculesConfig:
    mol_max_size: int = 6

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "SmallMoleculesConfig":
        """Create an instance of SmallMoleculesConfig from a dictionary."""
        return SmallMoleculesConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "SmallMoleculesConfig":
        """Deserialize a YAML file into a SmallMoleculesConfig object."""
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return SmallMoleculesConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        if not isinstance(params.get("mol_max_size"), int) or not (
            0 < params["mol_max_size"]
        ):
            raise ValueError("Invalid 'mol_max_size'; expected an integer more than 1")


class SmallMoleculesStandardizer:
    """Remove small molecule from reaction."""

    def __init__(self, mol_max_size: int = 6):
        self.mol_max_size = mol_max_size

    @staticmethod
    def from_config(config: SmallMoleculesConfig) -> "SmallMoleculesStandardizer":
        """Creates an instance of SmallMoleculesStandardizer from a configuration
        object."""
        return SmallMoleculesStandardizer(config.mol_max_size)

    def _split_molecules(
        self, molecules: Iterable, number_of_atoms: int
    ) -> Tuple[List[MoleculeContainer], List[MoleculeContainer]]:
        """Splits molecules according to the number of heavy atoms.

        :param molecules: Iterable of molecules.
        :param number_of_atoms: Threshold for splitting molecules.
        :return: Tuple of lists containing "big" molecules and "small" molecules.
        """
        big_molecules, small_molecules = [], []
        for molecule in molecules:
            if len(molecule) > number_of_atoms:
                big_molecules.append(molecule)
            else:
                small_molecules.append(molecule)

        return big_molecules, small_molecules

    def _remove_small_molecules(
        self,
        reaction: ReactionContainer,
        small_molecules_to_meta: bool = True,
    ) -> Union[ReactionContainer, None]:
        """Processes a reaction by removing small molecules.

        :param reaction: ReactionContainer object.
        :param small_molecules_to_meta: If True, deleted molecules are saved to meta.
        :return: Processed ReactionContainer without small molecules.
        """
        new_reactants, small_reactants = self._split_molecules(
            reaction.reactants, self.mol_max_size
        )
        new_products, small_products = self._split_molecules(
            reaction.products, self.mol_max_size
        )

        if (
            sum(len(mol) for mol in new_reactants) == 0
            or sum(len(mol) for mol in new_reactants) == 0
        ):
            return None

        new_reaction = ReactionContainer(
            new_reactants, new_products, reaction.reagents, reaction.meta
        )
        new_reaction.name = reaction.name

        if small_molecules_to_meta:
            united_small_reactants = unite_molecules(small_reactants)
            new_reaction.meta["small_reactants"] = str(united_small_reactants)

            united_small_products = unite_molecules(small_products)
            new_reaction.meta["small_products"] = str(united_small_products)

        return new_reaction

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Remove small molecule from reaction.

        :param reaction: Input reaction.
        :return: Returns reaction without small molecules, else None.
        """
        try:
            reaction = self._remove_small_molecules(reaction)
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


@dataclass
class RemoveReagentsConfig:
    reagent_max_size: int = 7

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "RemoveReagentsConfig":
        """Create an instance of RemoveReagentsConfig from a dictionary."""
        return RemoveReagentsConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "RemoveReagentsConfig":
        """Deserialize a YAML file into a RemoveReagentsConfig object."""
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return RemoveReagentsConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        if not isinstance(params.get("reagent_max_size"), int) or not (
            0 < params["reagent_max_size"]
        ):
            raise ValueError(
                "Invalid 'reagent_max_size'; expected an integer more than 1"
            )


class RemoveReagentsStandardizer:
    """Remove reagents from reaction."""

    def __init__(self, reagent_max_size: int = 7):
        self.reagent_max_size = reagent_max_size

    @staticmethod
    def from_config(config: RemoveReagentsConfig) -> "RemoveReagentsStandardizer":
        """Creates an instance of SmallMoleculesStandardizer from a configuration
        object."""
        return RemoveReagentsStandardizer(config.reagent_max_size)

    def _remove_reagents(
        self,
        reaction: ReactionContainer,
        keep_reagents: bool = True,
    ) -> Union[ReactionContainer, None]:
        """Removes reagents (not changed molecules or molecules not involved in the
        reaction) from reactants and products.

        :param reaction: Input reaction
        :param keep_reagents: If True, the reagents are written to ReactionContainer.
        :return: The cleaned reaction.
        """
        not_changed_molecules = set(reaction.reactants).intersection(reaction.products)

        cgr = ~reaction
        center_atoms = set(cgr.center_atoms)

        new_reactants = []
        new_products = []
        new_reagents = []

        for molecule in reaction.reactants:
            if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
                new_reagents.append(molecule)
            else:
                new_reactants.append(molecule)

        for molecule in reaction.products:
            if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
                new_reagents.append(molecule)
            else:
                new_products.append(molecule)

        if (
            sum(len(mol) for mol in new_reactants) == 0
            or sum(len(mol) for mol in new_reactants) == 0
        ):
            return None

        if keep_reagents:
            new_reagents = {
                molecule
                for molecule in new_reagents
                if len(molecule) <= self.reagent_max_size
            }
        else:
            new_reagents = []

        new_reaction = ReactionContainer(
            new_reactants, new_products, new_reagents, reaction.meta
        )
        new_reaction.name = reaction.name

        return new_reaction

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Remove reagents from reaction.

        :param reaction: Input reaction.
        :return: Returns reaction reagents, else None.
        """
        try:
            reaction = self._remove_reagents(
                reaction,
                keep_reagents=True,
            )
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


class RebalanceReactionConfig:
    pass


class RebalanceReactionStandardizer:
    """Rebalance reaction."""

    @staticmethod
    def from_config(
        config: RebalanceReactionConfig,
    ) -> "RebalanceReactionStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return RebalanceReactionStandardizer()

    def _rebalance_reaction(self, reaction: ReactionContainer) -> ReactionContainer:
        """Rebalances the reaction by assembling CGR and then decomposing it. Works for
        all reactions for which the correct CGR can be assembled.

        :param reaction: The reaction to be rebalanced.
        :return: The rebalanced reaction.
        """
        tmp_reaction = ReactionContainer(reaction.reactants, reaction.products)
        cgr = ~tmp_reaction
        reactants, products = ~cgr
        rebalanced_reaction = ReactionContainer(
            reactants.split(), products.split(), reaction.reagents, reaction.meta
        )
        rebalanced_reaction.name = reaction.name

        return rebalanced_reaction

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Rebalance reaction.

        :param reaction: Input reaction.
        :return: Returns rebalanced reaction, else None.
        """
        try:
            reaction = self._rebalance_reaction(reaction)
            return reaction
        except Exception as e:
            logging.debug(e)
        return None


class DuplicateReactionConfig:
    pass


class DuplicateReactionStandardizer:
    """Remove duplicate reactions."""

    def __init__(self):
        self.hash_set = set()

    @staticmethod
    def from_config(
        config: DuplicateReactionConfig,
    ) -> "DuplicateReactionStandardizer":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return DuplicateReactionStandardizer()

    def safe_reaction_smiles(self, reaction: ReactionContainer):
        reactants_smi = ".".join(str(i) for i in reaction.reactants)
        products_smi = ".".join(str(i) for i in reaction.products)
        reaction_smi = ">>".join([reactants_smi, products_smi])
        return reaction_smi

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer | None:
        """Remove duplicate reactions.

        :param reaction: Input reaction.
        :return: Returns reaction if it is unique (not duplicate), else None
        """

        h = hash(self.safe_reaction_smiles(reaction))
        if h not in self.hash_set:
            self.hash_set.add(h)
            return reaction

        return None


@dataclass
class ReactionStandardizationConfig(ConfigABC):
    """Configuration class for reaction filtering. This class manages configuration
    settings for various reaction filters, including paths, file formats, and filter-
    specific parameters.

    :param reaction_mapping_config: Configuration for reaction mapping.
    :param functional_groups_config: Configuration for functional groups
        standardization.
    :param kekule_form_config: Configuration for reactants/reagents/products
        kekulization.
    :param check_valence_config: Configuration for atom valence checking.
    :param implicify_hydrogens_config: Configuration for hydrogens removal.
    :param check_isotopes_config: Configuration for isotopes checking and cleaning.
    :param split_ions_config: Configuration for computing charge of molecule.
    :param aromatic_form_config: Configuration for molecules aromatization.
    :param unchanged_parts_config: Configuration for removal of unchanged parts in
        reaction.
    :param small_molecules_config: Configuration for removal of small molecule from
        reaction.
    :param remove_reagents_config: Configuration for removal of reagents from reaction.
    :param rebalance_reaction_config: Configuration for reaction rebalancing.
    :param duplicate_reaction_config: Configuration for removal of duplicate reactions.
    """

    # configuration for reaction standardizers
    reaction_mapping_config: Optional[ReactionMappingConfig] = None
    functional_groups_config: Optional[FunctionalGroupsConfig] = None
    kekule_form_config: Optional[KekuleFormConfig] = None
    check_valence_config: Optional[CheckValenceConfig] = None
    implicify_hydrogens_config: Optional[ImplicifyHydrogensConfig] = None
    check_isotopes_config: Optional[CheckIsotopesConfig] = None
    split_ions_config: Optional[SplitIonsConfig] = None
    aromatic_form_config: Optional[AromaticFormConfig] = None
    mapping_fix_config: Optional[MappingFixConfig] = None
    unchanged_parts_config: Optional[UnchangedPartsConfig] = None
    small_molecules_config: Optional[SmallMoleculesConfig] = None
    remove_reagents_config: Optional[RemoveReagentsConfig] = None
    rebalance_reaction_config: Optional[RebalanceReactionConfig] = None
    duplicate_reaction_config: Optional[DuplicateReactionConfig] = None

    def to_dict(self):
        """Converts the configuration into a dictionary."""
        config_dict = {
            "reaction_mapping_config": (
                {} if self.reaction_mapping_config is not None else None
            ),
            "functional_groups_config": (
                {} if self.functional_groups_config is not None else None
            ),
            "kekule_form_config": ({} if self.kekule_form_config is not None else None),
            "check_valence_config": (
                {} if self.check_valence_config is not None else None
            ),
            "implicify_hydrogens_config": (
                {} if self.implicify_hydrogens_config is not None else None
            ),
            "check_isotopes_config": (
                {} if self.check_isotopes_config is not None else None
            ),
            "aromatic_form_config": (
                {} if self.aromatic_form_config is not None else None
            ),
            "mapping_fix_config": ({} if self.mapping_fix_config is not None else None),
            "unchanged_parts_config": (
                {} if self.unchanged_parts_config is not None else None
            ),
            "small_molecules_config": (
                {} if self.small_molecules_config is not None else None
            ),
            "remove_reagents_config": (
                {} if self.remove_reagents_config is not None else None
            ),
            "rebalance_reaction_config": (
                {} if self.rebalance_reaction_config is not None else None
            ),
            "duplicate_reaction_config": (
                {} if self.duplicate_reaction_config is not None else None
            ),
        }

        filtered_config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return filtered_config_dict

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "ReactionStandardizationConfig":
        """Create an instance of ReactionCheckConfig from a dictionary."""
        # Instantiate configuration objects if their corresponding dictionary is present
        reaction_mapping_config = (
            ReactionMappingConfig()
            if "reaction_mapping_config" in config_dict
            else None
        )
        functional_groups_config = (
            FunctionalGroupsConfig()
            if "functional_groups_config" in config_dict
            else None
        )
        kekule_form_config = (
            KekuleFormConfig() if "kekule_form_config" in config_dict else None
        )
        check_valence_config = (
            CheckValenceConfig() if "check_valence_config" in config_dict else None
        )
        implicify_hydrogens_config = (
            ImplicifyHydrogensConfig()
            if "implicify_hydrogens_config" in config_dict
            else None
        )
        check_isotopes_config = (
            CheckIsotopesConfig() if "check_isotopes_config" in config_dict else None
        )
        split_ions_config = (
            SplitIonsConfig() if "split_ions_config" in config_dict else None
        )
        aromatic_form_config = (
            AromaticFormConfig() if "aromatic_form_config" in config_dict else None
        )
        mapping_fix_config = (
            MappingFixConfig() if "mapping_fix_config" in config_dict else None
        )
        unchanged_parts_config = (
            UnchangedPartsConfig() if "unchanged_parts_config" in config_dict else None
        )
        small_molecules_config = (
            SmallMoleculesConfig(**config_dict["small_molecules_config"])
            if "small_molecules_config" in config_dict
            else None
        )
        remove_reagents_config = (
            RemoveReagentsConfig() if "remove_reagents_config" in config_dict else None
        )
        rebalance_reaction_config = (
            RebalanceReactionConfig()
            if "rebalance_reaction_config" in config_dict
            else None
        )
        duplicate_reaction_config = (
            DuplicateReactionConfig()
            if "duplicate_reaction_config" in config_dict
            else None
        )

        return ReactionStandardizationConfig(
            reaction_mapping_config=reaction_mapping_config,
            functional_groups_config=functional_groups_config,
            kekule_form_config=kekule_form_config,
            check_valence_config=check_valence_config,
            implicify_hydrogens_config=implicify_hydrogens_config,
            check_isotopes_config=check_isotopes_config,
            split_ions_config=split_ions_config,
            aromatic_form_config=aromatic_form_config,
            mapping_fix_config=mapping_fix_config,
            unchanged_parts_config=unchanged_parts_config,
            small_molecules_config=small_molecules_config,
            remove_reagents_config=remove_reagents_config,
            rebalance_reaction_config=rebalance_reaction_config,
            duplicate_reaction_config=duplicate_reaction_config,
        )

    @staticmethod
    def from_yaml(file_path: str) -> "ReactionStandardizationConfig":
        """Deserializes a YAML file into a ReactionCheckConfig object."""
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return ReactionStandardizationConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        pass

    def create_standardizers(self):
        standardizer_instances = []

        if self.reaction_mapping_config is not None:
            standardizer_instances.append(
                ReactionMappingStandardizer.from_config(self.reaction_mapping_config)
            )

        if self.functional_groups_config is not None:
            standardizer_instances.append(
                FunctionalGroupsStandardizer.from_config(self.functional_groups_config)
            )

        if self.kekule_form_config is not None:
            standardizer_instances.append(
                KekuleFormStandardizer.from_config(self.kekule_form_config)
            )

        if self.check_valence_config is not None:
            standardizer_instances.append(
                CheckValenceStandardizer.from_config(self.check_valence_config)
            )

        if self.implicify_hydrogens_config is not None:
            standardizer_instances.append(
                ImplicifyHydrogensStandardizer.from_config(
                    self.implicify_hydrogens_config
                )
            )

        if self.check_isotopes_config is not None:
            standardizer_instances.append(
                CheckIsotopesStandardizer.from_config(self.check_isotopes_config)
            )

        if self.split_ions_config is not None:
            standardizer_instances.append(
                SplitIonsStandardizer.from_config(self.split_ions_config)
            )

        if self.aromatic_form_config is not None:
            standardizer_instances.append(
                AromaticFormStandardizer.from_config(self.aromatic_form_config)
            )

        if self.mapping_fix_config is not None:
            standardizer_instances.append(
                MappingFixStandardizer.from_config(self.mapping_fix_config)
            )

        if self.unchanged_parts_config is not None:
            standardizer_instances.append(
                UnchangedPartsStandardizer.from_config(self.unchanged_parts_config)
            )

        if self.small_molecules_config is not None:
            standardizer_instances.append(
                SmallMoleculesStandardizer.from_config(self.small_molecules_config)
            )

        if self.remove_reagents_config is not None:
            standardizer_instances.append(
                RemoveReagentsStandardizer.from_config(self.remove_reagents_config)
            )
        if self.rebalance_reaction_config is not None:
            standardizer_instances.append(
                RebalanceReactionStandardizer.from_config(
                    self.rebalance_reaction_config
                )
            )
        if self.duplicate_reaction_config is not None:
            standardizer_instances.append(
                DuplicateReactionStandardizer.from_config(
                    self.duplicate_reaction_config
                )
            )

        return standardizer_instances


def standardize_reaction(
    reaction: ReactionContainer, standardizers: list
) -> Tuple[bool, ReactionContainer] | None:
    """Remove duplicate reactions.

    :param reaction: Input reaction.
    :param standardizers: The list of standardizers.
    :return: Returns the standardized reaction, else None.
    """

    standardized_reaction = None
    with HiddenPrints():
        for standardizer in standardizers:
            standardized_reaction = standardizer(reaction)
            if not standardized_reaction:
                return None

    return standardized_reaction


@ray.remote
def process_batch(
    batch: List[ReactionContainer],
    standardizers: list,
) -> List[ReactionContainer]:
    """Processes a batch of reactions to standardize reactions based on the given list
    of standardizers.

    :param batch: A list of reactions to be standardized.
    :param standardizers: The list of standardizers.
    :return: The list of standardized reactions.
    """

    standardized_reaction_list = []
    for reaction in batch:
        standardized_reaction = standardize_reaction(reaction, standardizers)
        if standardized_reaction:
            standardized_reaction_list.append(standardized_reaction)
        else:
            continue
    return standardized_reaction_list


def process_completed_batch(
    futures: Dict,
    result_file: TextIOWrapper,
    n_processed: int,
) -> int:
    """Processes completed batches of standardized reactions.

    :param futures: A dictionary of futures with ongoing batch processing tasks.
    :param result_file: The path to the file where standardized reactions will be
        stored.
    :param n_processed: The number of already standardized reactions.
    :return: The number of standardized reactions after the processing of the current
        batch.
    """
    ready_id, running_id = ray.wait(list(futures.keys()), num_returns=1)
    completed_batch = ray.get(ready_id[0])

    # write results of the completed batch to file
    for reaction in completed_batch:
        result_file.write(reaction)
        n_processed += 1

    # remove completed future and update progress bar
    del futures[ready_id[0]]

    return n_processed


def standardize_reactions_from_file(
    config: ReactionStandardizationConfig,
    input_reaction_data_path: str,
    standardized_reaction_data_path: str = "reaction_data_standardized.smi",
    num_cpus: int = 1,
    batch_size: int = 100,
) -> None:
    """Reactions standardization.

    :param config: The reaction standardization configuration.
    :param input_reaction_data_path: Path to the reaction data file.
    :param standardized_reaction_data_path: Name for the file where standardized
        reactions will be stored.
    :param num_cpus: Number of CPUs to use for processing.
    :param batch_size: Size of the batch for processing reactions.
    :return: None. The function writes the processed reactions to specified smi/RDF
        files.
    """

    standardizers = config.create_standardizers()

    ray.init(num_cpus=num_cpus, ignore_reinit_error=True, logging_level=logging.ERROR)
    max_concurrent_batches = num_cpus  # limit the number of concurrent batches
    lines_counter = 0
    with ReactionReader(input_reaction_data_path) as reactions, ReactionWriter(
        standardized_reaction_data_path
    ) as result_file:

        batches_to_process, batch = {}, []
        n_processed = 0
        for index, reaction in tqdm(
            enumerate(reactions),
            desc="Number of reactions processed: ",
            bar_format="{desc}{n} [{elapsed}]",
        ):
            lines_counter += 1
            batch.append(reaction)
            if len(batch) == batch_size:
                batch_results = process_batch.remote(batch, standardizers)
                batches_to_process[batch_results] = None
                batch = []

                # check and process completed tasks if reached the concurrency limit
                while len(batches_to_process) >= max_concurrent_batches:
                    n_processed = process_completed_batch(
                        batches_to_process, result_file, n_processed
                    )

        # process the last batch if it's not empty
        if batch:
            batch_results = process_batch.remote(batch, standardizers)
            batches_to_process[batch_results] = None

        # process remaining batches
        while batches_to_process:
            n_processed = process_completed_batch(
                batches_to_process, result_file, n_processed
            )

    ray.shutdown()

    print(f"Initial number of parsed reactions: {lines_counter}")
    print(f"Standardized number of reactions: {n_processed}")
