"""Module containing classes abd functions for reactions filtering."""

import logging
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
from chython.algorithms.fingerprints.morgan import MorganFingerprint
from chython.containers import CGRContainer, MoleculeContainer, ReactionContainer
from pydantic import Field, model_validator
from tqdm.auto import tqdm

from synplan.chem.data.pipeline import build_batch_result, write_batch_results
from synplan.chem.data.reaction_result import (
    BatchResult,
    ErrorEntry,
    FilteredEntry,
    PipelineSummary,
)
from synplan.chem.data.standardizing import (
    AromaticFormStandardizer,
    KekuleFormStandardizer,
    StandardizationError,
)
from synplan.utils.config import BaseConfigModel, NestedConfigContainer
from synplan.utils.files import (
    RawReactionReader,
    ReactionWriter,
    format_source_fields,
    parse_reaction,
    reaction_source_info,
    split_smiles_record,
)
from synplan.utils.parallel import chunked, graceful_shutdown, process_pool_map_stream

logger = logging.getLogger("synplan.chem.data.filtering")


class CompeteProductsConfig(BaseConfigModel):
    fingerprint_tanimoto_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    mcs_tanimoto_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class CompeteProductsFilter:
    """Checks if there are compete reactions."""

    def __init__(
        self,
        fingerprint_tanimoto_threshold: float = 0.3,
        mcs_tanimoto_threshold: float = 0.6,
    ):
        self.fingerprint_tanimoto_threshold = fingerprint_tanimoto_threshold
        self.mcs_tanimoto_threshold = mcs_tanimoto_threshold

    @staticmethod
    def from_config(config: CompeteProductsConfig) -> "CompeteProductsFilter":
        """Creates an instance of CompeteProductsFilter from a configuration object."""
        return CompeteProductsFilter(
            config.fingerprint_tanimoto_threshold, config.mcs_tanimoto_threshold
        )

    def __call__(self, reaction: ReactionContainer) -> bool:
        """Checks if the reaction has competing products, else False.

        :param reaction: Input reaction.
        :return: Returns True if the reaction has competing products, else False.
        """
        is_compete = False

        # check for compete products using both fingerprint similarity and maximum common substructure (MCS) similarity
        for mol in reaction.reagents:
            for other_mol in reaction.products:
                if len(mol) > 6 and len(other_mol) > 6:
                    # compute fingerprint similarity
                    molf = mol.morgan_fingerprint()
                    other_molf = other_mol.morgan_fingerprint()
                    fingerprint_tanimoto = tanimoto_kernel(molf, other_molf)[0][0]

                    # if fingerprint similarity is high enough, check for MCS similarity
                    if fingerprint_tanimoto > self.fingerprint_tanimoto_threshold:
                        try:
                            # find the maximum common substructure (MCS) and compute its size
                            clique_size = len(
                                next(mol.get_mcs_mapping(other_mol, limit=100))
                            )

                            # calculate MCS similarity based on MCS size
                            mcs_tanimoto = clique_size / (
                                len(mol) + len(other_mol) - clique_size
                            )

                            # if MCS similarity is also high enough, mark the reaction as having compete products
                            if mcs_tanimoto > self.mcs_tanimoto_threshold:
                                is_compete = True
                                break
                        except StopIteration:
                            continue

        return is_compete


class DynamicBondsConfig(BaseConfigModel):
    min_bonds_number: int = Field(default=1, ge=0)
    max_bonds_number: int = Field(default=6, ge=0)

    @model_validator(mode="after")
    def _check_min_le_max(self):
        if self.min_bonds_number > self.max_bonds_number:
            raise ValueError(
                "'min_bonds_number' cannot be greater than 'max_bonds_number'"
            )
        return self


class DynamicBondsFilter:
    """Checks if there is an unacceptable number of dynamic bonds in CGR."""

    def __init__(self, min_bonds_number: int = 1, max_bonds_number: int = 6):
        self.min_bonds_number = min_bonds_number
        self.max_bonds_number = max_bonds_number

    @staticmethod
    def from_config(config: DynamicBondsConfig):
        """Creates an instance of DynamicBondsChecker from a configuration object."""
        return DynamicBondsFilter(config.min_bonds_number, config.max_bonds_number)

    def __call__(self, reaction: ReactionContainer) -> bool:
        cgr = ~reaction
        return not (
            self.min_bonds_number <= len(cgr.center_bonds) <= self.max_bonds_number
        )


class SmallMoleculesConfig(BaseConfigModel):
    mol_max_size: int = Field(default=6, ge=1)


class SmallMoleculesFilter:
    """Checks if there are only small molecules in the reaction or if there is only one
    small reactant or product."""

    def __init__(self, mol_max_size: int = 6):
        self.limit = mol_max_size

    @staticmethod
    def from_config(config: SmallMoleculesConfig) -> "SmallMoleculesFilter":
        """Creates an instance of SmallMoleculesChecker from a configuration object."""
        return SmallMoleculesFilter(config.mol_max_size)

    def __call__(self, reaction: ReactionContainer) -> bool:
        return bool(
            (
                len(reaction.reactants) == 1
                and self.are_only_small_molecules(reaction.reactants)
            )
            or (
                len(reaction.products) == 1
                and self.are_only_small_molecules(reaction.products)
            )
            or (
                self.are_only_small_molecules(reaction.reactants)
                and self.are_only_small_molecules(reaction.products)
            )
        )

    def are_only_small_molecules(self, molecules: Iterable[MoleculeContainer]) -> bool:
        """Checks if all molecules in the given iterable are small molecules."""
        return all(len(molecule) <= self.limit for molecule in molecules)


class CGRConnectedComponentsConfig(BaseConfigModel):
    pass


class CGRConnectedComponentsFilter:
    """Checks if CGR contains unrelated components (without reagents)."""

    @staticmethod
    def from_config(
        config: CGRConnectedComponentsConfig,
    ) -> "CGRConnectedComponentsFilter":
        """Creates an instance of CGRConnectedComponentsChecker from a configuration
        object."""
        return CGRConnectedComponentsFilter()

    def __call__(self, reaction: ReactionContainer) -> bool:
        tmp_reaction = ReactionContainer(reaction.reactants, reaction.products)
        cgr = ~tmp_reaction
        return cgr.connected_components_count > 1


class RingsChangeConfig(BaseConfigModel):
    pass


class RingsChangeFilter:
    """Checks if there is changing rings number in the reaction."""

    @staticmethod
    def from_config(config: RingsChangeConfig) -> "RingsChangeFilter":
        """Creates an instance of RingsChecker from a configuration object."""
        return RingsChangeFilter()

    def __call__(self, reaction: ReactionContainer):
        """
        Returns True if there are valence mistakes in the reaction or there is a
        reaction with mismatch numbers of all rings or aromatic rings in reactants and
        products (reaction in rings)

        :param reaction: Input reaction.
        :return: Returns True if there are valence mistakes in the reaction.

        """

        r_rings, r_arom_rings = self._calc_rings(reaction.reactants)
        p_rings, p_arom_rings = self._calc_rings(reaction.products)

        return (r_arom_rings != p_arom_rings) or (r_rings != p_rings)

    @staticmethod
    def _calc_rings(molecules: Iterable) -> tuple[int, int]:
        """
        Calculates number of all rings and number of aromatic rings in molecules.

        :param molecules: Set of molecules.
        :return: Number of all rings and number of aromatic rings in molecules
        """
        rings, arom_rings = 0, 0
        for mol in molecules:
            rings += mol.rings_count
            arom_rings += len(mol.aromatic_rings)
        return rings, arom_rings


class StrangeCarbonsConfig(BaseConfigModel):
    # currently empty, but can be extended in the future if needed
    pass


class StrangeCarbonsFilter:
    """Checks if there are 'strange' carbons in the reaction."""

    @staticmethod
    def from_config(config: StrangeCarbonsConfig) -> "StrangeCarbonsFilter":
        """Creates an instance of StrangeCarbonsChecker from a configuration object."""
        return StrangeCarbonsFilter()

    def __call__(self, reaction: ReactionContainer) -> bool:
        for molecule in reaction.reactants + reaction.products:
            atoms_types = {
                a.atomic_symbol for _, a in molecule.atoms()
            }  # atoms types in molecule
            if len(atoms_types) == 1 and atoms_types.pop() == "C":
                if len(molecule) == 1:  # methane
                    return True
                bond_types = {int(b) for _, _, b in molecule.bonds()}
                if len(bond_types) == 1 and bond_types.pop() != 4:
                    return True  # C molecules with only one type of bond (not aromatic)
        return False


class NoReactionConfig(BaseConfigModel):
    # Currently empty, but can be extended in the future if needed
    pass


class NoReactionFilter:
    """Checks if there is no reaction in the provided reaction container."""

    @staticmethod
    def from_config(config: NoReactionConfig) -> "NoReactionFilter":
        """Creates an instance of NoReactionChecker from a configuration object."""
        return NoReactionFilter()

    def __call__(self, reaction: ReactionContainer) -> bool:
        cgr = ~reaction
        return not cgr.center_atoms and len(cgr.center_bonds) == 0


class MultiCenterConfig(BaseConfigModel):
    pass


class MultiCenterFilter:
    """Checks if there is a multicenter reaction."""

    @staticmethod
    def from_config(config: MultiCenterConfig) -> "MultiCenterFilter":
        return MultiCenterFilter()

    def __call__(self, reaction: ReactionContainer) -> bool:
        cgr = ~reaction
        return len(cgr.centers_list) > 1


class WrongCHBreakingConfig(BaseConfigModel):
    pass


class WrongCHBreakingFilter:
    """Checks for incorrect C-C bond formation from breaking a C-H bond."""

    @staticmethod
    def from_config(config: WrongCHBreakingConfig) -> "WrongCHBreakingFilter":
        return WrongCHBreakingFilter()

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Determines if a reaction involves incorrect C-C bond formation from breaking
        a C-H bond.

        :param reaction: The reaction to be filtered.
        :return: True if incorrect C-C bond formation is found, False otherwise.

        """

        if reaction.check_valence():
            return False

        copy_reaction = reaction.copy()
        copy_reaction.explicify_hydrogens()
        cgr = ~copy_reaction
        reduced_cgr = cgr.augmented_substructure(cgr.center_atoms, deep=1)

        return self.is_wrong_c_h_breaking(reduced_cgr)

    @staticmethod
    def is_wrong_c_h_breaking(cgr: CGRContainer) -> bool:
        """
        Checks for incorrect C-C bond formation from breaking a C-H bond in a CGR.

        :param cgr: The CGR with explicified hydrogens.
        :return: True if incorrect C-C bond formation is found, False otherwise.

        """
        for atom_id in cgr.center_atoms:
            if cgr.atom(atom_id).atomic_symbol == "C":
                is_c_h_breaking, is_c_c_formation = False, False
                c_with_h_id, another_c_id = None, None

                for neighbour_id, bond in cgr._bonds[atom_id].items():
                    neighbour = cgr.atom(neighbour_id)

                    if (
                        bond.order
                        and not bond.p_order
                        and neighbour.atomic_symbol == "H"
                    ):
                        is_c_h_breaking = True
                        c_with_h_id = atom_id

                    elif (
                        not bond.order
                        and bond.p_order
                        and neighbour.atomic_symbol == "C"
                    ):
                        is_c_c_formation = True
                        another_c_id = neighbour_id

                if is_c_h_breaking and is_c_c_formation:
                    # check for presence of heteroatoms in the first environment of 2 bonding carbons
                    return not (
                        any(
                            cgr.atom(nid).atomic_symbol not in ("C", "H")
                            for nid in cgr._bonds[c_with_h_id]
                        )
                        or any(
                            cgr.atom(nid).atomic_symbol not in ("C", "H")
                            for nid in cgr._bonds[another_c_id]
                        )
                    )

        return False


class CCsp3BreakingConfig(BaseConfigModel):
    pass


class CCsp3BreakingFilter:
    """Checks if there is C(sp3)-C bond breaking."""

    @staticmethod
    def from_config(config: CCsp3BreakingConfig) -> "CCsp3BreakingFilter":
        return CCsp3BreakingFilter()

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if there is C(sp3)-C bonds breaking, else False.

        :param reaction: Input reaction
        :return: Returns True if there is C(sp3)-C bonds breaking, else False.

        """
        cgr = ~reaction
        rc = cgr.augmented_substructure(cgr.center_atoms, deep=1)

        for n, m, bond in rc.bonds():
            atom = rc.atom(n)
            neigh = rc.atom(m)

            is_bond_broken = bond.order is not None and bond.p_order is None
            are_atoms_carbons = atom.atomic_number == 6 and neigh.atomic_number == 6
            is_atom_sp3 = (
                rc._hybridizations.get(n, 0) == 1 or rc._hybridizations.get(m, 0) == 1
            )

            if is_bond_broken and are_atoms_carbons and is_atom_sp3:
                return True
        return False


class CCRingBreakingConfig(BaseConfigModel):
    """
    Object to pass to ReactionFilterConfig if you want to enable C-C ring breaking filter

    """

    pass


class CCRingBreakingFilter:
    """Checks if a reaction involves ring C-C bond breaking."""

    @staticmethod
    def from_config(config: CCRingBreakingConfig):
        return CCRingBreakingFilter()

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if the reaction involves ring C-C bond breaking, else False.

        :param reaction: Input reaction
        :return: Returns True if the reaction involves ring C-C bond breaking, else
            False.

        """
        cgr = ~reaction

        # Extract reactants' center atoms and their rings
        reactants_center_atoms = {}
        reactants_rings = set()
        for reactant in reaction.reactants:
            reactants_rings.update(reactant.sssr)
            for n, atom in reactant.atoms():
                if n in cgr.center_atoms:
                    reactants_center_atoms[n] = atom

        # identify reaction center based on center atoms
        reaction_center = cgr.augmented_substructure(atoms=cgr.center_atoms, deep=0)

        # iterate over bonds in the reaction center and filter for ring C-C bond breaking
        for atom_id, neighbour_id, bond in reaction_center.bonds():
            try:
                # Retrieve corresponding atoms from reactants
                atom = reactants_center_atoms[atom_id]
                neighbour = reactants_center_atoms[neighbour_id]
            except KeyError:
                continue
            else:
                # Check if the bond is broken and both atoms are carbons in rings of size 5, 6, or 7
                is_bond_broken = (bond.order is not None) and (bond.p_order is None)
                are_atoms_carbons = (
                    atom.atomic_symbol == "C" and neighbour.atomic_symbol == "C"
                )
                are_atoms_in_ring = (
                    set(atom.ring_sizes).intersection({5, 6, 7})
                    and set(neighbour.ring_sizes).intersection({5, 6, 7})
                    and any(
                        atom_id in ring and neighbour_id in ring
                        for ring in reactants_rings
                    )
                )

                # If all conditions are met, indicate ring C-C bond breaking
                if is_bond_broken and are_atoms_carbons and are_atoms_in_ring:
                    return True

        return False


class ReactionFilterConfig(NestedConfigContainer):
    """
    Configuration class for reaction filtering. This class manages configuration
    settings for various reaction filters, including paths, file formats, and filter-
    specific parameters.

    :ivar dynamic_bonds_config: Configuration for dynamic bonds checking.
    :ivar small_molecules_config: Configuration for small molecules checking.
    :ivar strange_carbons_config: Configuration for strange carbons checking.
    :ivar compete_products_config: Configuration for competing products checking.
    :ivar cgr_connected_components_config: Configuration for CGR connected components checking.
    :ivar rings_change_config: Configuration for rings change checking.
    :ivar no_reaction_config: Configuration for no reaction checking.
    :ivar multi_center_config: Configuration for multi-center checking.
    :ivar wrong_ch_breaking_config: Configuration for wrong C-H breaking checking.
    :ivar cc_sp3_breaking_config: Configuration for CC sp3 breaking checking.
    :ivar cc_ring_breaking_config: Configuration for CC ring breaking checking.

    """

    # configuration for reaction filters
    dynamic_bonds_config: DynamicBondsConfig | None = None
    small_molecules_config: SmallMoleculesConfig | None = None
    strange_carbons_config: StrangeCarbonsConfig | None = None
    compete_products_config: CompeteProductsConfig | None = None
    cgr_connected_components_config: CGRConnectedComponentsConfig | None = None
    rings_change_config: RingsChangeConfig | None = None
    no_reaction_config: NoReactionConfig | None = None
    multi_center_config: MultiCenterConfig | None = None
    wrong_ch_breaking_config: WrongCHBreakingConfig | None = None
    cc_sp3_breaking_config: CCsp3BreakingConfig | None = None
    cc_ring_breaking_config: CCRingBreakingConfig | None = None

    _NESTED_CONFIG_TYPES: ClassVar[dict[str, type]] = {
        "dynamic_bonds_config": DynamicBondsConfig,
        "small_molecules_config": SmallMoleculesConfig,
        "strange_carbons_config": StrangeCarbonsConfig,
        "compete_products_config": CompeteProductsConfig,
        "cgr_connected_components_config": CGRConnectedComponentsConfig,
        "rings_change_config": RingsChangeConfig,
        "no_reaction_config": NoReactionConfig,
        "multi_center_config": MultiCenterConfig,
        "wrong_ch_breaking_config": WrongCHBreakingConfig,
        "cc_sp3_breaking_config": CCsp3BreakingConfig,
        "cc_ring_breaking_config": CCRingBreakingConfig,
    }

    def create_filters(self):
        filter_instances = []

        if self.dynamic_bonds_config is not None:
            filter_instances.append(
                DynamicBondsFilter.from_config(self.dynamic_bonds_config)
            )

        if self.small_molecules_config is not None:
            filter_instances.append(
                SmallMoleculesFilter.from_config(self.small_molecules_config)
            )

        if self.strange_carbons_config is not None:
            filter_instances.append(
                StrangeCarbonsFilter.from_config(self.strange_carbons_config)
            )

        if self.compete_products_config is not None:
            filter_instances.append(
                CompeteProductsFilter.from_config(self.compete_products_config)
            )

        if self.cgr_connected_components_config is not None:
            filter_instances.append(
                CGRConnectedComponentsFilter.from_config(
                    self.cgr_connected_components_config
                )
            )

        if self.rings_change_config is not None:
            filter_instances.append(
                RingsChangeFilter.from_config(self.rings_change_config)
            )

        if self.no_reaction_config is not None:
            filter_instances.append(
                NoReactionFilter.from_config(self.no_reaction_config)
            )

        if self.multi_center_config is not None:
            filter_instances.append(
                MultiCenterFilter.from_config(self.multi_center_config)
            )

        if self.wrong_ch_breaking_config is not None:
            filter_instances.append(
                WrongCHBreakingFilter.from_config(self.wrong_ch_breaking_config)
            )

        if self.cc_sp3_breaking_config is not None:
            filter_instances.append(
                CCsp3BreakingFilter.from_config(self.cc_sp3_breaking_config)
            )

        if self.cc_ring_breaking_config is not None:
            filter_instances.append(
                CCRingBreakingFilter.from_config(self.cc_ring_breaking_config)
            )

        return filter_instances


def tanimoto_kernel(x: MorganFingerprint, y: MorganFingerprint) -> float:
    """Calculate the Tanimoto coefficient between each element of arrays x and y."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_dot = np.dot(x, y.T)
    x2 = np.sum(x**2, axis=1)
    y2 = np.sum(y**2, axis=1)

    denominator = np.array([x2] * len(y2)).T + np.array([y2] * len(x2)) - x_dot
    result = np.divide(
        x_dot, denominator, out=np.zeros_like(x_dot), where=denominator != 0
    )

    return result


def filter_reaction(
    item: str | ReactionContainer,
    config: ReactionFilterConfig,
    filters: list,
    *,
    ignore_errors: bool = False,
    fmt: str = "smi",
) -> tuple[bool, ReactionContainer | None, str | None]:
    """Checks the input reaction. Returns True if reaction is detected as erroneous and
    returns reaction itself, which sometimes is modified and does not necessarily
    correspond to the initial reaction.

    :param item: Reaction to be filtered (raw string or already-parsed).
    :param config: Reaction filtration configuration.
    :param filters: The list of reaction filters.
    :param ignore_errors: If True, keep processing when a standardization step fails;
        otherwise re-raise to expose unexpected issues.
    :param fmt: Format hint — ``"smi"`` for SMILES strings, ``"rdf"`` for RDF blocks.
    :return: 3-tuple ``(is_filtered, reaction, reason)`` where *reason* is
        ``None`` on success, a filter class name when filtered, or
        ``"stage/ErrorType: message"`` on standardization error.
        *reaction* may be ``None`` when parsing fails completely.
    """

    # --- Parse raw string into ReactionContainer ---
    try:
        reaction = parse_reaction(item, fmt) if isinstance(item, str) else item
    except Exception as exc:
        if not ignore_errors:
            raise
        exc_type = type(exc).__qualname__
        return True, None, f"parse/{exc_type}: {exc}"

    is_filtered = False
    reason: str | None = None

    # run reaction standardization

    standardizers = [
        # RemoveReagentsStandardizer(),
        KekuleFormStandardizer(),
        AromaticFormStandardizer(),
    ]

    try:
        for reaction_standardizer in standardizers:
            reaction = reaction_standardizer(reaction)
            if not reaction:
                is_filtered = True
                reason = reaction_standardizer.__class__.__name__
                break
    except StandardizationError as error:
        if not ignore_errors:
            raise
        reaction.meta["standardization_log"] = error.stage
        is_filtered = True
        reason = f"{error.stage}/{error.original_type}: {error.original_msg}"

    # run reaction filtration
    if not is_filtered:
        for reaction_filter in filters:
            try:
                if reaction_filter(reaction):
                    reaction.meta["filtration_log"] = reaction_filter.__class__.__name__
                    is_filtered = True
                    reason = reaction_filter.__class__.__name__
                    break
            except StandardizationError as error:
                if not ignore_errors:
                    raise
                reaction.meta["standardization_log"] = error.stage
                is_filtered = True
                reason = f"{error.stage}/{error.original_type}: {error.original_msg}"
                break
            except Exception as exc:
                if not ignore_errors:
                    raise
                filter_name = reaction_filter.__class__.__name__
                exc_type = type(exc).__qualname__
                reaction.meta["filtration_log"] = filter_name
                is_filtered = True
                reason = f"{filter_name}/{exc_type}: {exc}"
                break

    return is_filtered, reaction, reason


# ---------------------------------------------------------------------------
# Worker initializer + batch worker for ProcessPoolExecutor
# ---------------------------------------------------------------------------

_filter_state: dict | None = None


def _init_filter_worker(config_dict: dict) -> None:
    """Initialise per-worker state (filter instances, config).

    Called once per worker process by ``ProcessPoolExecutor``.
    """
    global _filter_state
    cfg = ReactionFilterConfig(**config_dict)
    _filter_state = {
        "config": cfg,
        "filters": cfg.create_filters(),
    }


def _filter_batch_worker(
    items: list[tuple[str, str, bool]],
) -> BatchResult:
    """Process a batch of raw reaction strings.

    Each element of *items* is ``(raw_string, fmt, ignore_errors)``.
    All items share the same fmt; reads fmt from the first element.

    Returns a :class:`BatchResult` with pre-serialized records, filtered
    entries, and error entries.  No ReactionContainer crosses the IPC boundary.
    """
    if _filter_state is None:
        raise RuntimeError("_filter_batch_worker called before _init_filter_worker")

    config: ReactionFilterConfig = _filter_state["config"]
    filters: list = _filter_state["filters"]

    reactions: list[ReactionContainer] = []
    filtered: list[FilteredEntry] = []
    errors: list[ErrorEntry] = []
    fmt = items[0][1] if items else "smi"

    for raw, _fmt, ignore_errors in items:
        is_filtered, reaction, reason = filter_reaction(
            raw,
            config,
            filters,
            ignore_errors=ignore_errors,
            fmt=_fmt,
        )
        if not is_filtered and reaction is not None:
            reactions.append(reaction)
        elif reason is not None:
            source_info = ""
            if reaction is not None and hasattr(reaction, "meta"):
                orig_smi = reaction.meta.get("init_smiles", str(reaction))
                source_info = reaction_source_info(reaction)
            else:
                orig_smi, source_fields = split_smiles_record(raw)
                source_info = format_source_fields(source_fields)
            if "/" in reason:
                # Error: "stage/ErrorType: message"
                stage_type = reason.split(":")[0]
                parts = (
                    stage_type.split("/", 1)
                    if "/" in stage_type
                    else (stage_type, "Unknown")
                )
                msg = reason.split(":", 1)[1].strip() if ":" in reason else reason
                errors.append(
                    ErrorEntry(
                        original=orig_smi,
                        source_info=source_info,
                        stage=parts[0],
                        error_type=parts[1],
                        message=msg,
                    )
                )
            else:
                filtered.append(FilteredEntry(original=orig_smi, reason=reason))

    return build_batch_result(reactions, errors, filtered, fmt, compute_dedup=False)


def _print_filtering_summary(
    lines_counter: int,
    n_filtered: int,
    error_counts: Counter,
    error_file_path: Path | None,
) -> None:
    """Print a categorized summary of filtering results."""
    from synplan.chem.data.standardizing import _DATA_ERROR_STAGES, _DATA_ERROR_TYPES

    n_rejected = lines_counter - n_filtered
    summary_lines = [
        f"Finished: processed {lines_counter}, kept {n_filtered}, rejected {n_rejected}"
    ]

    if error_counts:
        data_reasons: list[str] = []
        pipeline_reasons: list[str] = []
        filter_reasons: list[str] = []
        for reason, count in error_counts.most_common():
            label = f"{reason}={count}"
            if "/" in reason:
                # "stage/ErrorType: msg" pattern — extract stage and type
                stage_type = reason.split(":")[0]  # "stage/ErrorType"
                stage, etype = (
                    stage_type.split("/", 1) if "/" in stage_type else (stage_type, "")
                )
                if stage in _DATA_ERROR_STAGES or etype in _DATA_ERROR_TYPES:
                    data_reasons.append(label)
                else:
                    pipeline_reasons.append(label)
            else:
                filter_reasons.append(label)
        if filter_reasons:
            summary_lines.append(f"  Filter rejections: {', '.join(filter_reasons)}")
        if data_reasons:
            summary_lines.append(f"  Data errors:       {', '.join(data_reasons)}")
        if pipeline_reasons:
            summary_lines.append(
                f"  Pipeline errors:   {', '.join(pipeline_reasons)}  <-- INVESTIGATE"
            )

    if error_file_path is not None:
        summary_lines.append(f"Errors written to: {error_file_path}")

    summary = "\n".join(summary_lines)
    print(summary)
    logger.info(summary)


def filter_reactions_from_file(
    config: ReactionFilterConfig,
    input_reaction_data_path: str,
    filtered_reaction_data_path: str = "reaction_data_filtered.smi",
    num_cpus: int = 1,
    batch_size: int = 100,
    *,
    ignore_errors: bool = False,
    error_file_path: str | Path | None = None,
) -> PipelineSummary:
    """
    Processes reaction data, applying reaction filters based on the provided
    configuration, and writes the results to specified files.

    :param config: ReactionCheckConfig object containing all filtration configuration
        settings.
    :param input_reaction_data_path: Path to the reaction data file.
    :param filtered_reaction_data_path: Name for the file that will contain filtered
        reactions.
    :param num_cpus: Number of CPUs to use for processing.
    :param batch_size: Size of the batch for processing reactions.
    :param ignore_errors: If True, suppress standardization failures and continue
        processing (logging them instead). If False, standardization errors will stop
        the pipeline so new issues are visible.
    :param error_file_path: Path to write failed/filtered reactions (TSV).  If None
        and ``ignore_errors`` is True, defaults to ``<output>.errors.tsv``.
    :return: A :class:`PipelineSummary` with counts and breakdowns.

    """

    # Resolve error file path
    _error_path: Path | None = None
    if error_file_path is not None:
        _error_path = Path(error_file_path)
    elif ignore_errors:
        _error_path = Path(filtered_reaction_data_path).with_suffix(".errors.tsv")

    config_dict = config.to_dict()

    raw_reader = RawReactionReader(input_reaction_data_path)
    fmt = raw_reader.format

    summary = PipelineSummary(
        error_file=str(_error_path) if _error_path else None,
    )
    t0 = time.monotonic()

    error_fh = None
    if _error_path is not None:
        error_fh = open(_error_path, "w", encoding="utf-8")
        error_fh.write(
            "# original_smiles\tsource_info\tstage\terror_type\terror_message\n"
        )

    try:
        # Build batches: each item is a list of (raw_string, fmt, ignore_errors)
        def _batched_items():
            with raw_reader:
                for chunk in chunked(raw_reader, batch_size):
                    yield [(raw, fmt, ignore_errors) for raw in chunk]

        max_workers = max(num_cpus, 1)

        with (
            graceful_shutdown() as stop,
            ReactionWriter(filtered_reaction_data_path) as result_file,
        ):
            bar = tqdm(
                desc="Number of reactions processed: ",
                bar_format="{desc}{n} [{elapsed}]",
            )

            stream = process_pool_map_stream(
                _batched_items(),
                _filter_batch_worker,
                max_workers=max_workers,
                ordered=True,
                initializer=_init_filter_worker,
                initargs=(config_dict,),
                # Avoid ProcessPoolExecutor worker recycling here: CPython
                # documents open hangs with max_tasks_per_child in 3.13/3.14.
                # The shared pool helper now performs bounded cleanup of
                # stale chemistry workers at the end of the run instead.
            )

            def _stoppable(s):
                for batch in s:
                    if stop.is_set():
                        logger.warning("Shutdown requested — stopping filtering")
                        break
                    yield batch

            write_batch_results(
                _stoppable(stream),
                result_file.write_string,
                summary,
                error_file=error_fh,
                dedup=False,
                on_batch=bar.update,
            )

            bar.close()
    finally:
        if error_fh is not None:
            error_fh.close()

    summary.elapsed_seconds = time.monotonic() - t0

    # Print legacy-style summary
    error_counts = Counter()
    error_counts.update(summary.filter_breakdown)
    error_counts.update(summary.error_breakdown)
    _print_filtering_summary(
        summary.total_input, summary.succeeded, error_counts, _error_path
    )

    return summary
