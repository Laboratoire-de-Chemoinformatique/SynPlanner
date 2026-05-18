"""Module containing classes and functions for reactions standardizing.

This module contains the open-source code from
https://github.com/Laboratoire-de-Chemoinformatique/Reaction_Data_Cleaning/blob/master/scripts/standardizer.py
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from chython import smiles as smiles_chython
from chython.containers import MoleculeContainer, ReactionContainer
from pydantic import Field
from tqdm.auto import tqdm

from synplan.chem.data.pipeline import (
    build_batch_result,
    reaction_cgr_key,
    write_batch_results,
)
from synplan.chem.data.reaction_result import (
    BatchResult,
    ErrorEntry,
    PipelineSummary,
)
from synplan.chem.utils import unite_molecules
from synplan.utils.config import BaseConfigModel, NestedConfigContainer
from synplan.utils.files import (
    RawReactionReader,
    ReactionWriter,
    extract_origin_fields,
    parse_reaction,
    reaction_source_info,
    write_error_tsv_header,
)
from synplan.utils.parallel import chunked, graceful_shutdown, process_pool_map_stream

logger = logging.getLogger("synplan.chem.data.standardizing")


class StandardizationError(RuntimeError):
    """Wraps the original exception and the reaction string that failed.

    Stores the original exception type name and message as strings so the
    error is safely picklable across process boundaries (no reference to the
    live exception object which may not be importable on the worker).
    """

    def __init__(self, stage: str, reaction: str, original: Exception):
        self.stage = stage
        self.reaction = reaction
        self.original_type = type(original).__qualname__
        self.original_msg = str(original)
        super().__init__(f"{stage} failed on {reaction}: {original}")

    def __reduce__(self):
        """Make picklable (no reference to original exception object)."""
        return (
            self.__class__._from_pickle,
            (self.stage, self.reaction, self.original_type, self.original_msg),
        )

    @classmethod
    def _from_pickle(cls, stage, reaction, orig_type, orig_msg):
        err = cls.__new__(cls)
        err.stage = stage
        err.reaction = reaction
        err.original_type = orig_type
        err.original_msg = orig_msg
        RuntimeError.__init__(
            err, f"{stage} failed on {reaction}: {orig_type}: {orig_msg}"
        )
        return err


class BaseStandardizer(ABC):
    """Template: subclasses override `_run` only."""

    @classmethod
    def from_config(cls, _cfg: object) -> "BaseStandardizer":
        return cls()

    @abstractmethod
    def _run(self, rxn: ReactionContainer | str) -> ReactionContainer:
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
        except StandardizationError:
            raise
        except Exception as exc:
            logger.debug("%s: %s", self.__class__.__name__, exc, exc_info=True)
            raise StandardizationError(self.__class__.__name__, str(rxn), exc) from exc


class FunctionalGroupsConfig(BaseConfigModel):
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


class KekuleFormConfig(BaseConfigModel):
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


class CheckValenceConfig(BaseConfigModel):
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


class ImplicifyHydrogensConfig(BaseConfigModel):
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


class CheckIsotopesConfig(BaseConfigModel):
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


class SplitIonsConfig(BaseConfigModel):
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

    def _split_ions(self, reaction: ReactionContainer) -> tuple[ReactionContainer, int]:
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
                        molecule: MoleculeContainer = smiles_chython(molecule)
                    except (ValueError, TypeError) as e:
                        logger.warning("Failed to parse molecule %s: %s", molecule, e)
                        continue

                # Use the split method from chython
                try:
                    components = molecule.split()
                    divided_molecules.extend(components)
                except (ValueError, RuntimeError) as e:
                    logger.warning("Failed to split molecule %s: %s", molecule, e)
                    divided_molecules.append(molecule)

            total_charge = 0
            ions_present = False
            for molecule in divided_molecules:
                try:
                    mol_charge = self._calc_charge(molecule)
                    total_charge += mol_charge
                    if mol_charge != 0:
                        ions_present = True
                except (ValueError, RuntimeError) as e:
                    logger.warning(
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


class AromaticFormConfig(BaseConfigModel):
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


class MappingFixConfig(BaseConfigModel):
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


class UnchangedPartsConfig(BaseConfigModel):
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


class SmallMoleculesConfig(BaseConfigModel):
    mol_max_size: int = Field(default=6, ge=1)


class SmallMoleculesStandardizer(BaseStandardizer):
    """Remove small molecule from reaction."""

    def __init__(self, mol_max_size: int = 6):
        self.mol_max_size = mol_max_size

    @classmethod
    def from_config(cls, config: SmallMoleculesConfig) -> "SmallMoleculesStandardizer":
        return cls(config.mol_max_size)

    def _split_molecules(
        self, molecules: Iterable, number_of_atoms: int
    ) -> tuple[list[MoleculeContainer], list[MoleculeContainer]]:
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


class RemoveReagentsConfig(BaseConfigModel):
    reagent_max_size: int = Field(default=7, ge=1)


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


class RebalanceReactionConfig(BaseConfigModel):
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
        tmp_rxn = ReactionContainer(rxn.reactants, rxn.products)
        cgr = ~tmp_rxn
        reactants, products = ~cgr
        new_rxn = ReactionContainer(
            reactants.split(), products.split(), rxn.reagents, rxn.meta
        )
        new_rxn.name = rxn.name
        return new_rxn


# Canonical chemistry order for reaction standardization. The configuration
# only enables/disables steps and sets parameters; it does not define execution
# order.  In particular, reagents are removed before valence validation so
# reagent-only species do not create false valence failures, and aromatization
# happens after valence-sensitive steps so final serialized reactions deduplicate
# in a consistent aromatic form.
STANDARDIZER_REGISTRY = {
    "kekule_form_config": KekuleFormStandardizer,
    "functional_groups_config": FunctionalGroupsStandardizer,
    "remove_reagents_config": RemoveReagentsStandardizer,
    "check_valence_config": CheckValenceStandardizer,
    "implicify_hydrogens_config": ImplicifyHydrogensStandardizer,
    "check_isotopes_config": CheckIsotopesStandardizer,
    "split_ions_config": SplitIonsStandardizer,
    "aromatic_form_config": AromaticFormStandardizer,
    "mapping_fix_config": MappingFixStandardizer,
    "unchanged_parts_config": UnchangedPartsStandardizer,
    "small_molecules_config": SmallMoleculesStandardizer,
    "rebalance_reaction_config": RebalanceReactionStandardizer,
}


class ReactionStandardizationConfig(NestedConfigContainer):
    """Configuration class for reaction filtering. This class manages configuration
    settings for various reaction filters, including paths, file formats, and filter-
    specific parameters.

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
    """

    # configuration for reaction standardizers
    functional_groups_config: FunctionalGroupsConfig | None = None
    kekule_form_config: KekuleFormConfig | None = None
    check_valence_config: CheckValenceConfig | None = None
    implicify_hydrogens_config: ImplicifyHydrogensConfig | None = None
    check_isotopes_config: CheckIsotopesConfig | None = None
    split_ions_config: SplitIonsConfig | None = None
    aromatic_form_config: AromaticFormConfig | None = None
    mapping_fix_config: MappingFixConfig | None = None
    unchanged_parts_config: UnchangedPartsConfig | None = None
    small_molecules_config: SmallMoleculesConfig | None = None
    remove_reagents_config: RemoveReagentsConfig | None = None
    rebalance_reaction_config: RebalanceReactionConfig | None = None
    deduplicate: bool = True
    _NESTED_CONFIG_TYPES: ClassVar[dict[str, type]] = {
        "functional_groups_config": FunctionalGroupsConfig,
        "kekule_form_config": KekuleFormConfig,
        "check_valence_config": CheckValenceConfig,
        "implicify_hydrogens_config": ImplicifyHydrogensConfig,
        "check_isotopes_config": CheckIsotopesConfig,
        "split_ions_config": SplitIonsConfig,
        "aromatic_form_config": AromaticFormConfig,
        "mapping_fix_config": MappingFixConfig,
        "unchanged_parts_config": UnchangedPartsConfig,
        "small_molecules_config": SmallMoleculesConfig,
        "remove_reagents_config": RemoveReagentsConfig,
        "rebalance_reaction_config": RebalanceReactionConfig,
    }

    def create_standardizers(self):
        """Create selected standardizers in SynPlanner's canonical order."""
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
        std_rxn = std(std_rxn)  # may return None or raise
        if std_rxn is None:
            return None
    return std_rxn


def safe_standardize(
    item: str | ReactionContainer,
    standardizers: Sequence,
    *,
    ignore_errors: bool = False,
    fmt: str = "smi",
) -> tuple[ReactionContainer | None, bool, StandardizationError | None]:
    """
    Returns ``(reaction, success, error)``.

    *reaction* is a :class:`ReactionContainer` on success or when the original
    could be recovered, or ``None`` when parsing failed completely.
    The boolean flags real success.
    The third element is the :class:`StandardizationError` when the reaction
    failed, or ``None`` on success.

    If ``ignore_errors`` is False (default), any :class:`StandardizationError`
    or unexpected exception is propagated so callers can see new failure modes.
    When set to True, failures are logged and the original reaction is returned
    with the success flag set to False.
    """
    reaction: ReactionContainer | None = None
    try:
        reaction = parse_reaction(item, fmt)
        std = standardize_reaction(reaction, standardizers)
        if std is None:
            return reaction, False, None  # filtered → keep original
        return std, True, None
    except StandardizationError as exc:
        if ignore_errors:
            if reaction is not None:
                return reaction, False, exc
            return None, False, exc
        raise
    except Exception as exc:
        if ignore_errors:
            orig = item if isinstance(item, str) else str(item)
            wrapped = StandardizationError("parse", orig, exc)
            if reaction is not None:
                return reaction, False, wrapped
            return None, False, wrapped
        raise


def _process_batch(
    batch: Sequence[str | ReactionContainer],
    standardizers: Sequence,
    *,
    ignore_errors: bool = False,
    fmt: str = "smi",
) -> tuple[list[ReactionContainer], int, list[tuple[str, str, str, str, str]]]:
    """Process a batch and return (results, n_ok, errors).

    Each error entry is ``(original_smiles, source_info, stage, error_type,
    error_message)``.
    """
    results: list[ReactionContainer] = []
    errors: list[tuple[str, str, str, str, str]] = []
    n_std = 0
    for item in batch:
        rxn, ok, exc = safe_standardize(
            item, standardizers, ignore_errors=ignore_errors, fmt=fmt
        )
        if ok and rxn is not None:
            results.append(rxn)
        n_std += ok
        if exc is not None:
            if rxn is not None and hasattr(rxn, "meta"):
                orig = rxn.meta.get("init_smiles", str(rxn))
                source_info = reaction_source_info(rxn)
            else:
                orig, source_info = extract_origin_fields(item, fmt=fmt)
            errors.append(
                (orig, source_info, exc.stage, exc.original_type, exc.original_msg)
            )
    return results, n_std, errors


_worker_state: dict | None = None


def _init_standardize_worker(config_dict: dict, ignore_errors: bool, fmt: str) -> None:
    """Process initializer: build standardizers once per worker process."""
    global _worker_state
    config = ReactionStandardizationConfig(**config_dict)
    _worker_state = {
        "standardizers": config.create_standardizers(),
        "ignore_errors": ignore_errors,
        "fmt": fmt,
    }


def _standardize_batch_worker(
    items: list[str],
) -> BatchResult:
    """Top-level module function called by each worker — must be picklable.

    All serialization (CGR key + output record) is done here in the worker so
    the parent receives only plain strings and does a set lookup + file write.
    No ReactionContainer crosses the IPC boundary.
    """
    if _worker_state is None:
        raise RuntimeError(
            "_standardize_batch_worker called outside a worker process. "
            "This function requires _init_standardize_worker to run first."
        )
    reactions, _n_ok, raw_errors = _process_batch(
        items,
        _worker_state["standardizers"],
        ignore_errors=_worker_state["ignore_errors"],
        fmt=_worker_state["fmt"],
    )
    errors = [
        ErrorEntry(
            original=smi,
            source_info=source_info,
            stage=stage,
            error_type=etype,
            message=emsg,
        )
        for smi, source_info, stage, etype, emsg in raw_errors
    ]
    return build_batch_result(reactions, errors, [], _worker_state["fmt"])


# Keep public alias so any external code that imported _reaction_dedup_key still works.
_reaction_dedup_key = reaction_cgr_key


def _make_timeout_result(exc: Exception, items: list[str]) -> BatchResult:
    """Fallback for timed-out batches.

    Worker fmt is unknown here (the timeout callback signature is fixed by
    the pool helper), so default to ``"smi"`` — that matches the historical
    fmt-blind ``split_smiles_record`` call.
    """
    fmt = _worker_state["fmt"] if _worker_state is not None else "smi"
    errors = []
    for item in items:
        original, source_info = extract_origin_fields(item, fmt=fmt)
        errors.append(
            ErrorEntry(
                original=original,
                source_info=source_info,
                stage="timeout",
                error_type="TimeoutError",
                message=str(exc),
            )
        )
    return BatchResult(
        records=[],
        dedup_keys=[],
        errors=errors,
    )


# -- Error taxonomy for categorized summaries --
# Stages / chython exceptions that represent noisy *data*, not pipeline bugs.
_DATA_ERROR_STAGES = frozenset(
    {
        "CheckValence",
        "ReactionMapping",
        "SplitIons",
        "UnchangedParts",
        "SmallMolecules",
        "RemoveReagents",
        "parse",
    }
)
_DATA_ERROR_TYPES = frozenset(
    {
        "InvalidAromaticRing",
        "MappingError",
        "EmptyMolecule",
        "EmptyReaction",
        "IncorrectSmiles",
        "ValenceError",
        "ValueError",
    }
)


def _print_error_summary(
    error_counts: Counter,
    n_processed: int,
    n_ok: int,
    error_file_path: Path | None,
) -> None:
    """Print a categorized summary of errors to stdout and logger."""
    n_failed = n_processed - n_ok
    summary_lines = [
        f"Finished: processed {n_processed}, succeeded {n_ok}, failed {n_failed}"
    ]

    if error_counts:
        data_errors: list[str] = []
        pipeline_errors: list[str] = []
        for (stage, etype), count in error_counts.most_common():
            label = f"{stage}/{etype}={count}"
            if stage in _DATA_ERROR_STAGES or etype in _DATA_ERROR_TYPES:
                data_errors.append(label)
            else:
                pipeline_errors.append(label)
        if data_errors:
            summary_lines.append(f"  Data errors:     {', '.join(data_errors)}")
        if pipeline_errors:
            summary_lines.append(
                f"  Pipeline errors: {', '.join(pipeline_errors)}  <-- INVESTIGATE"
            )

    if error_file_path is not None:
        summary_lines.append(f"Errors written to: {error_file_path}")

    summary = "\n".join(summary_lines)
    print(summary)
    logger.info(summary)


def standardize_reactions_from_file(
    config: ReactionStandardizationConfig,
    input_reaction_data_path: str | Path,
    standardized_reaction_data_path: str | Path = "reaction_data_standardized.smi",
    *,
    num_cpus: int = 1,
    batch_size: int = 1_000,
    silent: bool = True,
    max_pending_factor: int = 4,
    ignore_errors: bool = False,
    error_file_path: str | Path | None = None,
) -> PipelineSummary:
    """
    Reads reactions, standardises them in parallel with ProcessPoolExecutor,
    writes results.

    Backpressure is handled by ``process_pool_map_stream``.  Standardisers
    are constructed once per worker via ``_init_standardize_worker``.
    Cross-worker deduplication is performed writer-side using worker-computed
    stable CGR keys.

    Args:
        config: Configuration object for standardizers.
        input_reaction_data_path: Path to the input reaction data file.
        standardized_reaction_data_path: Path to save the standardized reactions.
        num_cpus: Number of CPU cores to use for parallel processing.
        batch_size: Number of reactions to process in each batch.
        silent: If True, suppress the progress bar.
        max_pending_factor: Controls the number of pending tasks in flight.
        ignore_errors: If True, log standardization failures and keep processing;
            otherwise propagate exceptions to surface new issues.
        error_file_path: Path to write failed reactions (TSV).  If None and
            ``ignore_errors`` is True, defaults to ``<output>.errors.tsv``.

    Returns:
        A ``PipelineSummary`` with counts and error breakdown.
    """
    output_path = Path(standardized_reaction_data_path)

    # Resolve error file path
    _error_path: Path | None = None
    if error_file_path is not None:
        _error_path = Path(error_file_path)
    elif ignore_errors:
        _error_path = output_path.with_suffix(".errors.tsv")

    # Log which standardizers will be used (construct temporarily for logging)
    standardizers = config.create_standardizers()
    logger.info(
        "Standardizers: %s",
        ", ".join(s.__class__.__name__ for s in standardizers),
    )
    del standardizers  # workers will build their own copies

    summary = PipelineSummary()
    error_counts: Counter = Counter()
    dedup = config.deduplicate
    seen: set[str | int] = set()  # CGR key dedup (worker-computed, if enabled)
    t0 = time.monotonic()

    raw_reader = RawReactionReader(input_reaction_data_path)
    fmt = raw_reader.format

    bar = tqdm(
        total=0,
        unit="rxn",
        desc="Standardising",
        disable=silent,
        dynamic_ncols=True,
    )

    error_file = open(_error_path, "w", encoding="utf-8") if _error_path else None
    if error_file is not None:
        write_error_tsv_header(error_file)

    try:
        with (
            raw_reader,
            ReactionWriter(output_path) as writer,
        ):
            chunks = chunked(raw_reader, batch_size)

            with graceful_shutdown() as stop:
                stream = process_pool_map_stream(
                    chunks,
                    _standardize_batch_worker,
                    max_workers=max(num_cpus, 1),
                    max_pending=max_pending_factor * max(num_cpus, 1),
                    ordered=True,
                    initializer=_init_standardize_worker,
                    initargs=(config.to_dict(), ignore_errors, fmt),
                    timeout=300,
                    # Avoid ProcessPoolExecutor worker recycling here: CPython
                    # documents open hangs with max_tasks_per_child in 3.13/3.14.
                    # The shared pool helper now performs bounded cleanup of
                    # stale chemistry workers at the end of the run instead.
                    on_timeout=_make_timeout_result,
                )

                def _stoppable(s):
                    for batch in s:
                        if stop.is_set():
                            break
                        # Track error counts for the final summary print
                        for err in batch.errors:
                            error_counts[(err.stage, err.error_type)] += 1
                        yield batch

                def _on_batch(batch_total: int) -> None:
                    bar.total = (bar.total or 0) + batch_total
                    bar.update(batch_total)

                write_batch_results(
                    _stoppable(stream),
                    writer.write_string,
                    summary,
                    error_file=error_file,
                    dedup=dedup,
                    seen=seen,
                    on_batch=_on_batch,
                )
    finally:
        if error_file is not None:
            error_file.close()

    bar.close()

    summary.elapsed_seconds = time.monotonic() - t0
    summary.error_file = str(_error_path) if _error_path else None

    _print_error_summary(
        error_counts, summary.total_input, summary.succeeded, _error_path
    )

    return summary
