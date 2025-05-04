"""Module containing classes and functions for reactions standardizing.

This module contains the open-source code from
https://github.com/Laboratoire-de-Chemoinformatique/Reaction_Data_Cleaning/blob/master/scripts/standardizer.py
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Sequence, TextIO
from abc import ABC, abstractmethod
from pathlib import Path
import sys


import ray
import yaml
from CGRtools import smiles as smiles_cgrtools
from CGRtools.containers import MoleculeContainer
from CGRtools.containers import ReactionContainer
from CGRtools.containers import ReactionContainer as ReactionContainerCGRTools
from chython import ReactionContainer as ReactionContainerChython
from chython import smiles as smiles_chython
from tqdm.auto import tqdm

from synplan.chem.utils import unite_molecules
from synplan.utils.config import ConfigABC
from synplan.utils.files import ReactionReader, ReactionWriter
from synplan.utils.logging import init_logger, init_ray_logging

logger = logging.getLogger("synplan.chem.data.standardizing")


class StandardizationError(RuntimeError):
    """Wraps the original exception and the reaction string that failed."""

    def __init__(self, stage: str, reaction: str, original: Exception):
        super().__init__(f"{stage} failed on {reaction}: {original}")
        self.stage = stage
        self.reaction = reaction
        self.original = original


class BaseStandardizer(ABC):
    """Template: subclasses override `_run` only."""

    @classmethod
    def from_config(cls, _cfg: object) -> "BaseStandardizer":
        return cls()

    @abstractmethod
    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Run the standardization step on the reaction.

        Args:
            rxn: The reaction to standardize

        Returns:
            The standardized reaction

        Raises:
            StandardizationError: If standardization fails
        """
        ...

    def __call__(self, rxn: ReactionContainer) -> ReactionContainer:
        """Execute the standardization step with proper error handling.

        Args:
            rxn: The reaction to standardize

        Returns:
            The standardized reaction

        Raises:
            StandardizationError: If standardization fails
        """
        try:
            return self._run(rxn)
        except Exception as exc:
            logging.debug("%s: %s", self.__class__.__name__, exc, exc_info=True)
            raise StandardizationError(self.__class__.__name__, str(rxn), exc)


# Configuration classes
@dataclass
class ReactionMappingConfig:
    pass


class ReactionMappingStandardizer(BaseStandardizer):
    """Maps atoms of the reaction using chython (chytorch)."""

    def _map_and_remove_reagents(
        self, reaction: ReactionContainerChython
    ) -> ReactionContainerChython:
        """Map and remove reagents from the reaction.

        Args:
            reaction: Input reaction

        Returns:
            The mapped reaction with reagents removed
        """
        reaction.reset_mapping()
        reaction.remove_reagents()
        return reaction

    def _run(self, rxn: ReactionContainerCGRTools) -> ReactionContainerCGRTools:
        """Map atoms of the reaction using chython.

        Args:
            rxn: Input reaction

        Returns:
            The mapped reaction

        Raises:
            StandardizationError: If mapping fails
        """
        try:
            # Convert to chython format
            if isinstance(rxn, str):
                chython_reaction = smiles_chython(rxn)
            else:
                # Convert CGRtools reaction to SMILES string, preserving reagents
                reactants = ".".join(str(m) for m in rxn.reactants)
                reagents = ".".join(str(m) for m in rxn.reagents)
                products = ".".join(str(m) for m in rxn.products)
                smiles = f"{reactants}>{reagents}>{products}"
                # Parse SMILES string with chython
                chython_reaction = smiles_chython(smiles)

            # Map and remove reagents
            reaction_mapped = self._map_and_remove_reagents(chython_reaction)
            if not reaction_mapped:
                raise StandardizationError(
                    "ReactionMapping", str(rxn), ValueError("Mapping failed")
                )

            # Convert back to CGRtools format
            mapped_smiles = format(chython_reaction, "m")
            result = smiles_cgrtools(mapped_smiles)
            result.meta.update(rxn.meta)  # Preserve metadata
            return result
        except Exception as e:
            raise StandardizationError("ReactionMapping", str(rxn), e)


@dataclass
class FunctionalGroupsConfig:
    pass


class FunctionalGroupsStandardizer(BaseStandardizer):
    """Functional groups standardization."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Standardize functional groups in the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction with standardized functional groups

        Raises:
            StandardizationError: If standardization fails
        """
        rxn.standardize()
        return rxn


@dataclass
class KekuleFormConfig:
    pass


class KekuleFormStandardizer(BaseStandardizer):
    """Reactants/reagents/products kekulization."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Kekulize the reaction.

        Args:
            rxn: The reaction to kekulize

        Returns:
            The kekulized reaction

        Raises:
            StandardizationError: If kekulization fails
        """
        rxn.kekule()
        return rxn


@dataclass
class CheckValenceConfig:
    pass


class CheckValenceStandardizer(BaseStandardizer):
    """Check valence."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Check valence of atoms in the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction if valences are correct

        Raises:
            StandardizationError: If valence check fails
        """
        for molecule in rxn.reactants + rxn.products + rxn.reagents:
            valence_mistakes = molecule.check_valence()
            if valence_mistakes:
                raise StandardizationError(
                    "CheckValence",
                    str(rxn),
                    ValueError(f"Valence errors: {valence_mistakes}"),
                )
        return rxn


@dataclass
class ImplicifyHydrogensConfig:
    pass


class ImplicifyHydrogensStandardizer(BaseStandardizer):
    """Implicify hydrogens."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Implicify hydrogens in the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction with implicified hydrogens

        Raises:
            StandardizationError: If hydrogen implicification fails
        """
        rxn.implicify_hydrogens()
        return rxn


@dataclass
class CheckIsotopesConfig:
    pass


class CheckIsotopesStandardizer(BaseStandardizer):
    """Check isotopes."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Check and clean isotopes in the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction with cleaned isotopes

        Raises:
            StandardizationError: If isotope check/cleaning fails
        """
        is_isotope = False
        for molecule in rxn.reactants + rxn.products:
            for _, atom in molecule.atoms():
                if atom.isotope:
                    is_isotope = True
                    break
            if is_isotope:
                break

        if is_isotope:
            rxn.clean_isotopes()

        return rxn


@dataclass
class SplitIonsConfig:
    pass


class SplitIonsStandardizer(BaseStandardizer):
    """Computing charge of molecule."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Split ions in the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction with split ions

        Raises:
            StandardizationError: If ion splitting fails
        """
        reaction, return_code = self._split_ions(rxn)
        if return_code == 2:  # ions were split but the reaction is imbalanced
            raise StandardizationError(
                "SplitIons",
                str(rxn),
                ValueError("Reaction is imbalanced after ion splitting"),
            )
        return reaction

    def _calc_charge(self, molecule: MoleculeContainer) -> int:
        """Compute total charge of a molecule.

        Args:
            molecule: Input molecule

        Returns:
            The total charge of the molecule
        """
        return sum(molecule._charges.values())

    def _split_ions(self, reaction: ReactionContainer) -> Tuple[ReactionContainer, int]:
        """Split ions in a reaction.

        Args:
            reaction: Input reaction

        Returns:
            A tuple containing:
            - The reaction with split ions
            - Return code (0: nothing changed, 1: ions split, 2: ions split but imbalanced)
        """
        meta = reaction.meta
        reaction_parts = []
        return_codes = []

        for molecules in (reaction.reactants, reaction.reagents, reaction.products):
            # Split molecules into individual components
            divided_molecules = []
            for molecule in molecules:
                if isinstance(molecule, str):
                    # If it's a string, try to parse it as a molecule
                    try:
                        molecule: MoleculeContainer = smiles_cgrtools(molecule)
                    except Exception as e:
                        logging.warning("Failed to parse molecule %s: %s", molecule, e)
                        continue

                # Use the split method from CGRtools
                try:
                    components = molecule.split()
                    divided_molecules.extend(components)
                except Exception as e:
                    logging.warning("Failed to split molecule %s: %s", molecule, e)
                    divided_molecules.append(molecule)

            total_charge = 0
            ions_present = False
            for molecule in divided_molecules:
                try:
                    mol_charge = self._calc_charge(molecule)
                    total_charge += mol_charge
                    if mol_charge != 0:
                        ions_present = True
                except Exception as e:
                    logging.warning(
                        "Failed to calculate charge for molecule %s: %s", molecule, e
                    )
                    continue

            if ions_present and total_charge:
                return_codes.append(2)
            elif ions_present:
                return_codes.append(1)
            else:
                return_codes.append(0)

            reaction_parts.append(tuple(divided_molecules))

        return (
            ReactionContainer(
                reactants=reaction_parts[0],
                reagents=reaction_parts[1],
                products=reaction_parts[2],
                meta=meta,
            ),
            max(return_codes),
        )


@dataclass
class AromaticFormConfig:
    pass


class AromaticFormStandardizer(BaseStandardizer):
    """Aromatize molecules in reaction."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Aromatize molecules in the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction with aromatized molecules

        Raises:
            StandardizationError: If aromatization fails
        """
        rxn.thiele()
        return rxn


@dataclass
class MappingFixConfig:
    pass


class MappingFixStandardizer(BaseStandardizer):
    """Fix atom-to-atom mapping in reaction."""

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Fix atom-to-atom mapping in the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction with fixed atom-to-atom mapping

        Raises:
            StandardizationError: If mapping fix fails
        """
        rxn.fix_mapping()
        return rxn


@dataclass
class UnchangedPartsConfig:
    pass


class UnchangedPartsStandardizer(BaseStandardizer):
    """Ungroup molecules, remove unchanged parts from reactants and products."""

    def __init__(
        self,
        add_reagents_to_reactants: bool = False,
        keep_reagents: bool = False,
    ):
        self.add_reagents_to_reactants = add_reagents_to_reactants
        self.keep_reagents = keep_reagents

    @classmethod
    def from_config(cls, config: UnchangedPartsConfig) -> "UnchangedPartsStandardizer":
        return cls()

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Remove unchanged parts from the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction with unchanged parts removed

        Raises:
            StandardizationError: If unchanged parts removal fails
        """
        meta = rxn.meta
        new_reactants = list(rxn.reactants)
        new_reagents = list(rxn.reagents)
        if self.add_reagents_to_reactants:
            new_reactants.extend(new_reagents)
            new_reagents = []
        reactants = new_reactants.copy()
        new_products = list(rxn.products)

        for reactant in reactants:
            if reactant in new_products:
                new_reagents.append(reactant)
                new_reactants.remove(reactant)
                new_products.remove(reactant)
        if not self.keep_reagents:
            new_reagents = []

        if not new_reactants and new_products:
            raise StandardizationError(
                "UnchangedParts", str(rxn), ValueError("No reactants left")
            )
        if not new_products and new_reactants:
            raise StandardizationError(
                "UnchangedParts", str(rxn), ValueError("No products left")
            )
        if not new_reactants and not new_products:
            raise StandardizationError(
                "UnchangedParts", str(rxn), ValueError("No molecules left")
            )

        new_reaction = ReactionContainer(
            reactants=tuple(new_reactants),
            reagents=tuple(new_reagents),
            products=tuple(new_products),
            meta=meta,
        )
        new_reaction.name = rxn.name
        return new_reaction


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
        mol_max_size = params.get("mol_max_size", self.mol_max_size)
        if not isinstance(mol_max_size, int) or not (0 < mol_max_size):
            raise ValueError("Invalid 'mol_max_size'; expected an integer more than 1")


class SmallMoleculesStandardizer(BaseStandardizer):
    """Remove small molecule from reaction."""

    def __init__(self, mol_max_size: int = 6):
        self.mol_max_size = mol_max_size

    @classmethod
    def from_config(cls, config: SmallMoleculesConfig) -> "SmallMoleculesStandardizer":
        return cls(config.mol_max_size)

    def _split_molecules(
        self, molecules: Iterable, number_of_atoms: int
    ) -> Tuple[List[MoleculeContainer], List[MoleculeContainer]]:
        """Split molecules according to the number of heavy atoms.

        Args:
            molecules: Iterable of molecules
            number_of_atoms: Threshold for splitting molecules

        Returns:
            Tuple of lists containing "big" molecules and "small" molecules
        """
        big_molecules, small_molecules = [], []
        for molecule in molecules:
            if len(molecule) > number_of_atoms:
                big_molecules.append(molecule)
            else:
                small_molecules.append(molecule)
        return big_molecules, small_molecules

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Remove small molecules from the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction without small molecules

        Raises:
            StandardizationError: If small molecule removal fails
        """
        new_reactants, small_reactants = self._split_molecules(
            rxn.reactants, self.mol_max_size
        )
        new_products, small_products = self._split_molecules(
            rxn.products, self.mol_max_size
        )

        if not new_reactants or not new_products:
            raise StandardizationError(
                "SmallMolecules",
                str(rxn),
                ValueError("No molecules left after removing small ones"),
            )

        new_reaction = ReactionContainer(
            new_reactants, new_products, rxn.reagents, rxn.meta
        )
        new_reaction.name = rxn.name

        # Save small molecules to meta
        united_small_reactants = unite_molecules(small_reactants)
        new_reaction.meta["small_reactants"] = str(united_small_reactants)
        united_small_products = unite_molecules(small_products)
        new_reaction.meta["small_products"] = str(united_small_products)

        return new_reaction


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
        reagent_max_size = params.get("reagent_max_size", self.reagent_max_size)
        if not isinstance(reagent_max_size, int) or not (0 < reagent_max_size):
            raise ValueError(
                "Invalid 'reagent_max_size'; expected an integer more than 1"
            )


class RemoveReagentsStandardizer(BaseStandardizer):
    """Remove reagents from reaction."""

    def __init__(self, reagent_max_size: int = 7):
        self.reagent_max_size = reagent_max_size

    @classmethod
    def from_config(cls, config: RemoveReagentsConfig) -> "RemoveReagentsStandardizer":
        return cls(config.reagent_max_size)

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Remove reagents from the reaction.

        Args:
            rxn: Input reaction

        Returns:
            The reaction without reagents

        Raises:
            StandardizationError: If reagent removal fails
        """
        not_changed_molecules = set(rxn.reactants).intersection(rxn.products)
        cgr = ~rxn
        center_atoms = set(cgr.center_atoms)

        new_reactants = []
        new_products = []
        new_reagents = []

        for molecule in rxn.reactants:
            if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
                new_reagents.append(molecule)
            else:
                new_reactants.append(molecule)

        for molecule in rxn.products:
            if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
                new_reagents.append(molecule)
            else:
                new_products.append(molecule)

        if not new_reactants or not new_products:
            raise StandardizationError(
                "RemoveReagents",
                str(rxn),
                ValueError("No molecules left after removing reagents"),
            )

        # Filter reagents by size
        new_reagents = {
            molecule
            for molecule in new_reagents
            if len(molecule) <= self.reagent_max_size
        }

        new_reaction = ReactionContainer(
            new_reactants, new_products, new_reagents, rxn.meta
        )
        new_reaction.name = rxn.name

        return new_reaction


@dataclass
class RebalanceReactionConfig:
    pass


class RebalanceReactionStandardizer(BaseStandardizer):
    """Rebalance reaction."""

    @classmethod
    def from_config(
        cls, config: RebalanceReactionConfig
    ) -> "RebalanceReactionStandardizer":
        return cls()

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        """Rebalances the reaction by assembling CGR and then decomposing it. Works for
        all reactions for which the correct CGR can be assembled.

        Args:
            rxn: Input reaction

        Returns:
            The rebalanced reaction

        Raises:
            StandardizationError: If rebalancing fails
        """
        try:
            tmp_rxn = ReactionContainer(rxn.reactants, rxn.products)
            cgr = ~tmp_rxn
            reactants, products = ~cgr
            new_rxn = ReactionContainer(
                reactants.split(), products.split(), rxn.reagents, rxn.meta
            )
            new_rxn.name = rxn.name
            return new_rxn
        except Exception as e:
            logging.debug(f"Rebalancing attempt failed: {e}")
            raise StandardizationError(
                "RebalanceReaction",
                str(rxn),
                ValueError("Failed to rebalance reaction"),
            )


@dataclass
class DuplicateReactionConfig:
    pass


class DuplicateReactionStandardizer(BaseStandardizer):
    """Cluster‑wide duplicate removal via a Ray actor."""

    def __init__(self, dedup_actor: "ray.actor.ActorHandle"):
        self._actor = dedup_actor  # global singleton handle
        # local fast‑path cache to avoid actor call on obvious repeats *in
        # the same worker*; purely an optimisation, not required.
        self._local_seen: set[int] = set()

    @classmethod
    def from_config(cls, config: DuplicateReactionConfig):
        # fallback for single‑process mode: create a dummy in‑proc actor
        if ray.is_initialized():
            dedup_actor = ray.get_actor("duplicate_rxn_actor")
        else:
            dedup_actor = None
        return cls(dedup_actor)

    # ------------------------------------------------------------------
    def safe_reaction_smiles(self, reaction: ReactionContainer) -> str:
        reactants_smi = ".".join(str(i) for i in reaction.reactants)
        products_smi = ".".join(str(i) for i in reaction.products)
        return f"{reactants_smi}>>{products_smi}"

    def _run(self, rxn: ReactionContainer) -> ReactionContainer:
        h = hash(self.safe_reaction_smiles(rxn))

        # local cache fast‑path (helps in large batches processed by same
        # worker; no correctness impact).
        if h in self._local_seen:
            raise StandardizationError(
                "DuplicateReaction", str(rxn), ValueError("Duplicate reaction found")
            )

        # ------------------- cluster‑wide check ------------------------
        if self._actor is None:  # single‑CPU fall‑back
            is_new = h not in self._local_seen
        else:
            # synchronous, returns True/False
            is_new = ray.get(self._actor.check_and_add.remote(h))

        if is_new:
            self._local_seen.add(h)
            return rxn

        raise StandardizationError(
            "DuplicateReaction", str(rxn), ValueError("Duplicate reaction found")
        )


@ray.remote
class DedupActor:
    """Cluster‑wide set of reaction hashes."""

    def __init__(self):
        self._seen: set[int] = set()

    def check_and_add(self, h: int) -> bool:
        """
        Returns True **iff** the hash was not present yet and is now stored.
        Cluster‑wide uniqueness is guaranteed because this method executes
        serially inside the actor process.
        """
        if h in self._seen:
            return False
        self._seen.add(h)
        return True


# Registry mapping config field names to standardizer classes
STANDARDIZER_REGISTRY = {
    "reaction_mapping_config": ReactionMappingStandardizer,
    "functional_groups_config": FunctionalGroupsStandardizer,
    "kekule_form_config": KekuleFormStandardizer,
    "check_valence_config": CheckValenceStandardizer,
    "implicify_hydrogens_config": ImplicifyHydrogensStandardizer,
    "check_isotopes_config": CheckIsotopesStandardizer,
    "split_ions_config": SplitIonsStandardizer,
    "aromatic_form_config": AromaticFormStandardizer,
    "mapping_fix_config": MappingFixStandardizer,
    "unchanged_parts_config": UnchangedPartsStandardizer,
    "small_molecules_config": SmallMoleculesStandardizer,
    "remove_reagents_config": RemoveReagentsStandardizer,
    "rebalance_reaction_config": RebalanceReactionStandardizer,
    "duplicate_reaction_config": DuplicateReactionStandardizer,
}


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

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        for field_name, config in self.__dict__.items():
            if config is not None and hasattr(config, "_validate_params"):
                config._validate_params(params.get(field_name, {}))

    def to_dict(self):
        """Converts the configuration into a dictionary."""
        config_dict = {}
        for field_name in STANDARDIZER_REGISTRY:
            config = getattr(self, field_name)
            if config is not None:
                config_dict[field_name] = {}
        return config_dict

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "ReactionStandardizationConfig":
        """Create an instance of ReactionCheckConfig from a dictionary."""
        config_kwargs = {}
        for field_name, std_cls in STANDARDIZER_REGISTRY.items():
            if field_name in config_dict:
                config_kwargs[field_name] = std_cls.__name__.replace(
                    "Standardizer", "Config"
                )()
        return ReactionStandardizationConfig(**config_kwargs)

    @staticmethod
    def from_yaml(file_path: str) -> "ReactionStandardizationConfig":
        """Deserializes a YAML file into a ReactionCheckConfig object."""
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return ReactionStandardizationConfig.from_dict(config_dict)

    def create_standardizers(self):
        """Create standardizer instances based on configuration."""
        standardizers = []
        for field_name, std_cls in STANDARDIZER_REGISTRY.items():
            config = getattr(self, field_name)
            if config is not None:
                standardizers.append(std_cls.from_config(config))
        return standardizers


def standardize_reaction(
    reaction: ReactionContainer,
    standardizers: Sequence,
) -> ReactionContainer | None:
    """
    Apply each standardizer in order.

    Returns
    -------
    ReactionContainer | None
        - the fully‑standardised reaction, or
        - None if *any* standardizer decides to filter it out.

    Raises
    ------
    StandardizationError
        Propagated untouched so the caller can decide what to do.
    """
    std_rxn = reaction
    for std in standardizers:
        logger.debug("  › %s(%s)", std.__class__.__name__, std_rxn)
        try:
            std_rxn = std(std_rxn)  # may return None
            if std_rxn is None:  # soft filter
                logger.info("%s filtered out reaction", std.__class__.__name__)
                return None
        except StandardizationError as exc:
            # Log *once*, then re‑raise with full traceback intact
            logger.warning(
                "%s failed on reaction %s : %s",
                std.__class__.__name__,
                std_rxn,
                exc,
            )
            raise  # re‑raise same object
    return std_rxn


def safe_standardize(
    item: str | ReactionContainer,
    standardizers: Sequence,
) -> Tuple[ReactionContainer, bool]:
    """
    Always returns a ReactionContainer. The boolean flags real success.
    """
    try:
        # Parse only if needed
        reaction = (
            item if isinstance(item, ReactionContainer) else smiles_cgrtools(item)
        )
        std = standardize_reaction(reaction, standardizers)
        if std is None:
            return reaction, False  # filtered → keep original
        return std, True
    except Exception as exc:  # noqa: BLE001
        # keep the original container (parse if it was a string)
        if isinstance(item, ReactionContainer):
            return item, False
        return smiles_cgrtools(item), False


def _process_batch(
    batch: Sequence[str | ReactionContainer],
    standardizers: Sequence,
) -> Tuple[List[ReactionContainer], int]:
    results: List[ReactionContainer] = []
    n_std = 0
    for item in batch:
        rxn, ok = safe_standardize(item, standardizers)
        results.append(rxn)
        n_std += ok
    return results, n_std


@ray.remote
def process_batch_remote(
    batch: Sequence[str | ReactionContainer],
    std_param: ray.ObjectRef,  # <-- receives a ref
    log_file_path: str | Path | None = None,
) -> Tuple[List[ReactionContainer], int]:
    # Ray keeps a local cache of fetched objects, so the list is
    # deserialised only once per worker process, not once per task.
    if isinstance(std_param, ray.ObjectRef):  # handle?   get it
        standardizers = ray.get(std_param)  # • O(once)
    else:  # plain list? use as is
        standardizers = std_param

    # --- Worker-specific logging setup ---
    worker_logger = logging.getLogger("synplan.chem.data.standardizing")
    if log_file_path:
        log_file_path = Path(log_file_path)  # Ensure it's a Path object
        # Check if a handler for this file already exists for this logger
        handler_exists = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file_path
            for h in worker_logger.handlers
        )
        if not handler_exists:
            try:
                fh = logging.FileHandler(log_file_path, encoding="utf-8")
                # Use a simple format for worker logs, or match driver's format
                formatter = logging.Formatter(
                    "%(asctime)s | %(name)s (worker) | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                fh.setFormatter(formatter)
                fh.setLevel(logging.INFO)  # Or DEBUG, or use worker_log_level if passed
                worker_logger.addHandler(fh)
                worker_logger.setLevel(
                    logging.INFO
                )  # Ensure logger passes messages to handler
                worker_logger.propagate = (
                    False  # Avoid double logging if driver also logs
                )
                # Optional: Log that the handler was added
                # worker_logger.info(f"Worker process attached file handler: {log_file_path}")
            except Exception as e:
                # Log error if handler creation fails (e.g., permissions)
                logging.error(
                    f"Worker failed to create file handler {log_file_path}: {e}"
                )

    return _process_batch(batch, standardizers)


def chunked(iterable: Iterable, size: int):
    chunk = []
    for it in iterable:
        chunk.append(it)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def standardize_reactions_from_file(
    config: "ReactionStandardizationConfig",
    input_reaction_data_path: str | Path,
    standardized_reaction_data_path: str | Path = "reaction_data_standardized.smi",
    *,
    num_cpus: int = 1,
    batch_size: int = 1_000,  # larger batches amortise overhead
    silent: bool = True,
    max_pending_factor: int = 4,  # tasks in flight = factor × CPUs
    worker_log_level: int | str = logging.WARNING,
    log_file_path: str | Path | None = None,
) -> None:
    """
    Reads reactions, standardises them in parallel with Ray, writes results.

    The function keeps at most `max_pending_factor * num_cpus` Ray tasks in
    flight to avoid flooding the scheduler and blowing up the object store.
    Standardisers are broadcast once with `ray.put`, removing per‑task
    pickling cost.  All other logic is unchanged.

    Args:
        config: Configuration object for standardizers.
        input_reaction_data_path: Path to the input reaction data file.
        standardized_reaction_data_path: Path to save the standardized reactions.
        num_cpus: Number of CPU cores to use for parallel processing.
        batch_size: Number of reactions to process in each batch.
        silent: If True, suppress the progress bar.
        max_pending_factor: Controls the number of pending Ray tasks.
        worker_log_level: Logging level for Ray workers (e.g., logging.INFO, logging.WARNING).
        log_file_path: Path to the log file for workers to write to.
    """
    output_path = Path(standardized_reaction_data_path)
    standardizers = config.create_standardizers()

    logger.info(
        "Standardizers: %s",
        ", ".join(s.__class__.__name__ for s in standardizers),
    )

    # -----------------------  Ray initialisation  -----------------------
    if num_cpus > 1:
        if not ray.is_initialized():
            ray.init(
                num_cpus=num_cpus,
                ignore_reinit_error=True,
                logging_level=worker_log_level,
                log_to_driver=False,
            )

        DEDUP_NAME = "duplicate_rxn_actor"

        try:
            dedup_actor = ray.get_actor(DEDUP_NAME)  # already running?
        except ValueError:
            dedup_actor = DedupActor.options(
                name=DEDUP_NAME, lifetime="detached"  # survives driver exit
            ).remote()

        std_ref: ray.ObjectRef | None = None
        if num_cpus > 1 and std_ref is None:  # broadcast once
            std_ref = ray.put(standardizers)

    max_pending = max_pending_factor * num_cpus
    pending: Dict[ray.ObjectRef, None] = {}

    n_processed = n_std = 0
    bar = tqdm(
        total=0,
        unit="rxn",
        desc="Standardising",
        disable=silent,
        dynamic_ncols=True,
    )

    # ------------------------  Helper function  ------------------------
    def _flush(ref: ray.ObjectRef, write_fn) -> None:
        """Fetch finished task, write its results, update counters & bar."""
        nonlocal n_processed, n_std
        res, ok = ray.get(ref)
        write_fn(res)
        bar.update(len(res))
        n_processed += len(res)
        n_std += ok

    # -----------------------------  I/O  -------------------------------
    with ReactionReader(input_reaction_data_path) as reader, ReactionWriter(
        output_path
    ) as writer:

        write_fn = lambda reactions: [writer.write(r) for r in reactions]

        # ---------------------  Main read/compute loop  -----------------
        for chunk in chunked(reader, batch_size):
            bar.total += len(chunk)
            bar.refresh()

            if num_cpus > 1:
                # ---------- back‑pressure: keep ≤ max_pending ----------
                while len(pending) >= max_pending:
                    done, _ = ray.wait(list(pending), num_returns=1)
                    _flush(done[0], write_fn)
                    pending.pop(done[0], None)

                # ----------- schedule new task -------------------------
                ref = process_batch_remote.remote(chunk, std_ref, log_file_path)
                pending[ref] = None
            else:
                # --------------- serial fall‑back ----------------------
                res, ok = _process_batch(chunk, standardizers)
                write_fn(res)
                bar.update(len(res))
                n_processed += len(res)
                n_std += ok

        # ------------------  Drain remaining Ray tasks  -----------------
        while pending:
            done, _ = ray.wait(list(pending), num_returns=1)
            _flush(done[0], write_fn)
            pending.pop(done[0], None)

    bar.close()
    ray.shutdown()

    logger.info(
        "Finished: processed %d, standardised %d, filtered %d",
        n_processed,
        n_std,
        n_processed - n_std,
    )
