"""Module containing functions for protocol of reaction rules extraction."""

import logging
import tempfile
from collections import Counter, defaultdict
from io import TextIOWrapper
from itertools import islice
from os.path import splitext
from pathlib import Path

from chython.containers.bonds import Bond, QueryBond
from chython.containers.cgr import CGRContainer
from chython.containers.molecule import MoleculeContainer
from chython.containers.query import QueryContainer
from chython.containers.reaction import ReactionContainer
from chython.exceptions import InvalidAromaticRing
from chython.periodictable import QueryElement
from tqdm.auto import tqdm

from synplan.chem.data.reaction_result import (
    ErrorEntry,
    ExtractedRuleRecord,
    ExtractionBatchResult,
)
from synplan.chem.data.standardizing import RemoveReagentsStandardizer
from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.utils import (
    canonical_query_cgr_key,
    reverse_reaction,
    unite_molecules,
)
from synplan.utils.config import RuleExtractionConfig
from synplan.utils.files import (
    RawReactionReader,
    extract_origin_fields,
    load_rule_index_mapping_tsv,
    parse_reaction,
    tsv_safe,
    write_error_row,
    write_error_tsv_header,
)
from synplan.utils.parallel import graceful_shutdown, process_pool_map_stream

logger = logging.getLogger(__name__)


def molecule_substructure_as_query(mol, atoms) -> QueryContainer:
    atoms = set(atoms)
    q = QueryContainer(smarts="")
    for n in atoms:
        atom = mol.atom(n)
        xy = atom.xy if hasattr(atom, "xy") else None
        if isinstance(atom, QueryElement):
            q.add_atom(atom.copy(full=True), n, xy=xy)
        else:
            query_atom = QueryElement.from_atom(
                atom,
                neighbors=True,
                hydrogens=True,
                ring_sizes=True,
            )
            # Asymmetric hybridization copy: only stamp `(4,)` when the source
            # is aromatic, so the SMARTS writer emits lowercase `[c]`. For
            # sp3/sp2/sp atoms we deliberately leave `_hybridization=()` to
            # keep the rule lenient on hybridization for non-aromatic targets.
            if atom.hybridization == 4:
                query_atom._hybridization = (4,)
            q.add_atom(query_atom, n, xy=xy)
    for n, m, bond in mol.bonds():
        if n in atoms and m in atoms:
            if isinstance(bond, QueryBond):
                q.add_bond(n, m, bond.copy(full=True))
            elif isinstance(bond, Bond):
                q.add_bond(n, m, QueryBond.from_bond(bond))
    return q


def add_environment_atoms(
    cgr: CGRContainer, center_atoms: set[int], environment_atom_count: int
) -> set[int]:
    """
    Adds environment atoms to the set of center atoms based on the specified depth.

    :param cgr: A complete graph representation of a reaction (ReactionContainer
        object).
    :param center_atoms: A set of atom id corresponding to the center atoms of the
        reaction.
    :param environment_atom_count: An integer specifying the depth of the environment
        around the reaction center to be included. If it's 0, only the reaction center
        is included. If it's 1, the first layer of surrounding atoms is included, and so
        on.

    :return: A set of atom id including the center atoms and their environment atoms up
        to the specified depth. If environment_atom_count is 0, the original set of
        center atoms is returned unchanged.

    """
    if environment_atom_count:
        env_cgr = cgr.augmented_substructure(center_atoms, deep=environment_atom_count)
        return center_atoms | set(env_cgr)
    return center_atoms


def add_functional_groups(
    reaction: ReactionContainer,
    center_atoms: set[int],
    func_groups_list: list[QueryContainer],
) -> set[int]:
    """
    Augments the set of reaction rule atoms with functional groups if specified.

    :param reaction: The reaction object (ReactionContainer) from which molecules are
        extracted.
    :param center_atoms: A set of atom id corresponding to the center atoms of the
        reaction.
    :param func_groups_list: A list of functional group objects (MoleculeContainer or
        QueryContainer) to be considered when including functional groups. These objects
        define the structure of the functional groups to be included.

    :return: A set of atom id corresponding to the rule atoms, including atoms from the
        specified functional groups if include_func_groups is True. If
        include_func_groups is False, the original set of center atoms is returned.

    """

    rule_atoms = center_atoms.copy()
    for molecule in reaction.molecules():
        for func_group in func_groups_list:
            for mapping in func_group.get_mapping(molecule):
                func_group.remap(mapping)
                if set(func_group.atoms_numbers) & center_atoms:
                    rule_atoms |= set(func_group.atoms_numbers)
                func_group.remap({v: k for k, v in mapping.items()})
    return rule_atoms


def add_ring_structures(cgr: CGRContainer, rule_atoms: set[int]) -> set[int]:
    """
    Adds ring structures to the set of rule atoms if they intersect with the reaction
    center atoms.

    :param cgr: A condensed graph representation of a reaction (CGRContainer object).
    :param rule_atoms: A set of atom id corresponding to the center atoms of the
        reaction.

    :return: A set of atom id corresponding to the original rule atoms and the included
        ring structures.

    """
    for ring in cgr.sssr:
        # check if the current ring intersects with the set of rule atoms
        if set(ring) & rule_atoms:
            # if the intersection exists, include all atoms in the ring to the rule atoms
            rule_atoms |= set(ring)
    return rule_atoms


def add_leaving_incoming_groups(
    reaction: ReactionContainer,
    rule_atoms: set[int],
    keep_leaving_groups: bool,
    keep_incoming_groups: bool,
) -> tuple[set[int], dict[str, set]]:
    """
    Identifies and includes leaving and incoming groups to the rule atoms based on
    specified flags.

    :param reaction: The reaction object (ReactionContainer) from which leaving and
        incoming groups are extracted.
    :param rule_atoms: A set of atom id corresponding to the center atoms of the
        reaction.
    :param keep_leaving_groups: A boolean flag indicating whether to include leaving
        groups in the rule.
    :param keep_incoming_groups: A boolean flag indicating whether to include incoming
        groups in the rule.

    :return: Updated set of rule atoms including leaving and incoming groups if
        specified, and metadata about added groups.

    """

    meta_debug = {"leaving": set(), "incoming": set()}

    # extract atoms from reactants and products
    reactant_atoms = {atom for reactant in reaction.reactants for atom in reactant}
    product_atoms = {atom for product in reaction.products for atom in product}

    # identify leaving groups (reactant atoms not in products)
    if keep_leaving_groups:
        leaving_atoms = reactant_atoms - product_atoms
        new_leaving_atoms = leaving_atoms - rule_atoms
        # include leaving atoms in the rule atoms
        rule_atoms |= leaving_atoms
        # add leaving atoms to metadata
        meta_debug["leaving"] |= new_leaving_atoms

    # identify incoming groups (product atoms not in reactants)
    if keep_incoming_groups:
        incoming_atoms = product_atoms - reactant_atoms
        new_incoming_atoms = incoming_atoms - rule_atoms
        # Include incoming atoms in the rule atoms
        rule_atoms |= incoming_atoms
        # Add incoming atoms to metadata
        meta_debug["incoming"] |= new_incoming_atoms

    return rule_atoms, meta_debug


def clean_molecules(
    rule_molecules: list[MoleculeContainer],
    reaction_molecules: tuple[MoleculeContainer],
    reaction_center_atoms: set[int],
    atom_retention_details: dict[str, dict[str, bool]],
) -> list[QueryContainer]:
    """
    Cleans rule molecules by removing specified information about atoms based on
    retention details provided.

    :param rule_molecules: A list of query container objects representing the rule molecules.
    :param reaction_molecules: A list of molecule container objects involved in the reaction.
    :param reaction_center_atoms: A set of id corresponding to the atom numbers in the reaction center.
    :param atom_retention_details: A dictionary specifying what atom information to retain or remove.
                                   This dictionary should have two keys: "reaction_center" and "environment",
                                   each mapping to another dictionary. The nested dictionaries should have
                                   keys representing atom attributes (like "neighbors", "hybridization",
                                   "implicit_hydrogens", "ring_sizes") and boolean values.
                                   A value of True indicates that the corresponding attribute
                                   should be retained, while False indicates it should be removed from the atom.

    :return: A list of QueryContainer objects representing the cleaned rule molecules.

    """
    cleaned = []
    for rule_mol in rule_molecules:
        rule_atoms = set(rule_mol.atoms_numbers)
        for rxn_mol in reaction_molecules:
            rxn_atoms = set(rxn_mol.atoms_numbers)
            if rule_atoms <= rxn_atoms:
                q_rxn = molecule_substructure_as_query(rxn_mol, rxn_atoms)
                q_rule = molecule_substructure_as_query(q_rxn, rule_atoms)

                if not all(atom_retention_details["reaction_center"].values()):
                    for n in rule_atoms & reaction_center_atoms:
                        q_rule = clean_atom(
                            q_rule, atom_retention_details["reaction_center"], n
                        )

                if not all(atom_retention_details["environment"].values()):
                    for n in rule_atoms - reaction_center_atoms:
                        q_rule = clean_atom(
                            q_rule, atom_retention_details["environment"], n
                        )

                cleaned.append(q_rule)
                break
    return cleaned


def clean_atom(
    query_molecule: QueryContainer,
    attributes_to_keep: dict[str, bool],
    atom_number: int,
) -> QueryContainer:
    """
    Removes specified information from a given atom in a query molecule.

    :param query_molecule: The QueryContainer of molecule.
    :param attributes_to_keep: Dictionary indicating which attributes to keep in the atom. The keys should be strings
                               representing the attribute names, and the values should be booleans indicating whether
                               to retain (True) or remove(False) that attribute. Expected keys are:
                               - "neighbors": Indicates if neighbors of the atom should be removed.
                               - "implicit_hydrogens": Indicates if implicit hydrogen information of the atom should be removed.
                               - "ring_sizes": Indicates if ring size information of the atom should be removed.

    :param atom_number: The number of the atom to be modified in the query molecule.

    """

    target_atom = query_molecule.atom(atom_number)

    if not attributes_to_keep["neighbors"]:
        target_atom.neighbors = None
    if not attributes_to_keep["implicit_hydrogens"]:
        target_atom.implicit_hydrogens = None
    if not attributes_to_keep["ring_sizes"]:
        target_atom.ring_sizes = None

    return query_molecule


def create_substructures_and_reagents(
    reaction: ReactionContainer,
    rule_atoms: set[int],
    as_query_container: bool,
    keep_reagents: bool,
) -> tuple[list[MoleculeContainer], list[MoleculeContainer], list]:
    """
    Creates substructures for reactants and products, and optionally includes
    reagents, based on specified parameters. The function processes the reaction to
    create substructures for reactants and products based on the rule atoms. It also
    handles the inclusion of reagents based on the keep_reagents flag and converts these
    structures to query containers if required.

    :param reaction: The reaction object (ReactionContainer) from which to extract substructures.
                     This object  represents a chemical reaction with specified reactants, products, and possibly reagents.
    :param rule_atoms: A set of atom id corresponding to the rule atoms. These are used to identify relevant
                       substructures in reactants and products.
    :param as_query_container: A boolean flag indicating whether the substructures should be converted to query containers.
                               Query containers are used for pattern matching in chemical structures.
    :param keep_reagents: A boolean flag indicating whether reagents should be included in the resulting structures.
                          Reagents are additional substances that are present in the reaction but are not reactants or products.

    :return: A tuple containing three elements:
             - A list of reactant substructures, each corresponding to a part of the reactants that matches the rule atoms.
             - A list of product substructures, each corresponding to a part of the products that matches the rule atoms.
             - A list of reagents, included as is or as substructures, depending on the as_query_container flag.

    """
    reactant_substructures = []
    for reactant in reaction.reactants:
        atoms = rule_atoms & set(reactant.atoms_numbers)
        if atoms:
            reactant_substructures.append(reactant.substructure(atoms))

    product_substructures = []
    for product in reaction.products:
        atoms = rule_atoms & set(product.atoms_numbers)
        if atoms:
            product_substructures.append(product.substructure(atoms))

    reagents = list(reaction.reagents) if keep_reagents else []

    return reactant_substructures, product_substructures, reagents


def assemble_final_rule(
    reactant_substructures: list[QueryContainer],
    product_substructures: list[QueryContainer],
    reagents: list,
    meta_debug: dict[str, set],
    keep_metadata: bool,
    reaction: ReactionContainer,
) -> ReactionContainer:
    """
    Assembles the final reaction rule from the provided substructures and metadata.
    This function brings together the various components of a reaction rule, including
    reactant and product substructures, reagents, and metadata. It creates a
    comprehensive representation of the reaction rule, which can be used for further
    processing or analysis.

    :param reactant_substructures: A list of substructures derived from the reactants of
        the reaction. These substructures represent parts of reactants that are relevant
        to the rule.
    :param product_substructures: A list of substructures derived from the products of
        the reaction. These substructures represent parts of products that are relevant
        to the rule.
    :param reagents: A list of reagents involved in the reaction. These may be included
        as-is or as substructures, depending on earlier processing steps.
    :param meta_debug: A dictionary containing additional metadata about the reaction,
        such as leaving and incoming groups.
    :param keep_metadata: A boolean flag indicating whether to retain the metadata
        associated with the reaction in the rule.
    :param reaction: The original reaction object (ReactionContainer) from which the
        rule is being created.

    :return: A ReactionContainer object representing the assembled reaction rule. This
        container includes the reactant and product substructures, reagents, and any
        additional metadata if keep_metadata is True.

    """

    rule_metadata = meta_debug if keep_metadata else {}
    rule_metadata.update(reaction.meta if keep_metadata else {})

    rule = ReactionContainer(
        reactant_substructures, product_substructures, reagents, rule_metadata
    )

    if keep_metadata:
        rule.name = reaction.name

    rule.flush_cache()
    return rule


def _isomorphism_cost_estimate(query, target) -> float:
    """Cheap upper bound on subgraph-isomorphism enumeration cost.

    Buckets atoms by ``(atomic_symbol, degree, is_aromatic)`` and multiplies
    per-bucket ``perm(target_count, query_count)``. Caps at 1e15.

    Returns ``0`` if any query bucket has fewer matches in the target — but a
    zero result is NOT reliable as "will fail fast" because SMARTS D/h
    constraints are more flexible than this crude bucketing. Only use the
    high side for skip decisions (e.g. ``> _ISOMORPHISM_COST_SKIP_THRESHOLD``).
    """
    def _sig(mol, n):
        a = mol.atom(n)
        return (a.atomic_symbol, len(mol._bonds[n]),
                getattr(a, "hybridization", None) == 4)

    q_sig = Counter(_sig(query, n) for n in query.atoms_numbers)
    t_sig = Counter(_sig(target, n) for n in target.atoms_numbers)
    cost = 1.0
    for sig, k in q_sig.items():
        n = t_sig.get(sig, 0)
        if n < k:
            return 0.0
        for i in range(k):
            cost *= (n - i)
        if cost > 1e15:
            return cost
    return cost


_ISOMORPHISM_COST_SKIP_THRESHOLD = 1e9


def validate_rule(rule: ReactionContainer, reaction: ReactionContainer) -> bool:
    """
    Validates a reaction rule by ensuring it can correctly generate the products from
    the reactants. The function uses a chemical reactor to simulate the reaction based
    on the provided rule. It then compares the products generated by the simulation with
    the actual products of the reaction. If they match, the rule is considered valid. If
    not, a ValueError is raised, indicating an issue with the rule.

    :param rule: The reaction rule to be validated. This is a ReactionContainer object
        representing a chemical reaction rule, which includes the necessary information
        to perform a reaction.
    :param reaction: The original reaction object (ReactionContainer) against which the
        rule is to be validated. This object contains the actual reactants and products
        of the reaction.

    :return: The validated rule if the rule correctly generates the products from the
        reactants.

    :raises ValueError: If the rule does not correctly generate the products from the
        reactants, indicating an incorrect or incomplete rule.

    """

    # build the query patterns and products as before
    patterns = tuple(
        molecule_substructure_as_query(m, m.atoms_numbers) for m in rule.reactants
    )
    products = tuple(rule.products)
    if patterns and reaction.reactants:
        cost = _isomorphism_cost_estimate(patterns[0], reaction.reactants[0])
        if cost > _ISOMORPHISM_COST_SKIP_THRESHOLD:
            return False
    reactor = CanonicalRetroReactor(patterns=patterns, products=products, delete_atoms=False)
    try:
        for result_reaction in reactor(*reaction.reactants):  # unpack here
            try:
                result_products = []
                for result_product in result_reaction.products:
                    tmp = result_product.copy()
                    try:
                        tmp.remove_coordinate_bonds(keep_to_terminal=False)
                        tmp.kekule()
                        if tmp.check_valence():
                            continue
                    except InvalidAromaticRing:
                        continue
                    result_products.append(result_product)
                if set(reaction.products) == set(result_products) and len(
                    reaction.products
                ) == len(result_products):
                    return True
            except (KeyError, IndexError, InvalidAromaticRing):
                # KeyError - iteration over reactor is finished and products are different from the original reaction
                # IndexError - mistake in __contract_ions, possibly problems with charges in reaction rule
                # InvalidAromaticRing - aromatic ring is invalid
                continue
    except KeyError:
        # KeyError - iteration over reactor is finished and products are different from the original reaction
        return False
    return False


def create_rule(
    config: RuleExtractionConfig,
    reaction: ReactionContainer,
    _restrict_center_atoms: set[int] | None = None,
) -> ReactionContainer:
    """
    Creates a reaction rule from a given reaction based on the specified
    configuration. The function processes the reaction to create a rule that matches the
    configuration settings. It handles the inclusion of environmental atoms, functional
    groups, ring structures, and leaving and incoming groups. It also constructs
    substructures for reactants, products, and reagents, and cleans molecule
    representations if required. Optionally, it validates the rule using a reactor.

    :param config: An instance of ExtractRuleConfig, containing various settings that
                   determine how the rule is created, such as environmental atom count, inclusion
                   of functional groups, rings, leaving and incoming groups, and other parameters.
    :param reaction: The reaction object (ReactionContainer) from which to create the
                     rule. This object represents a chemical reaction with specified reactants,
                     products, and possibly reagents.
    :param _restrict_center_atoms: Optional override of the reaction-center atom set.
                     When given, only these atoms (plus their environment/leaving groups
                     per config) are included in the rule, instead of all CGR center atoms.
                     Used by extract_rules when multicenter_rules=False to build a separate
                     rule per disconnected reaction-center component.
    :return: A ReactionContainer object representing the extracted reaction rule.

    """

    # 1. create reaction CGR
    cgr = ~reaction
    if _restrict_center_atoms is not None:
        center_atoms = set(_restrict_center_atoms)
    else:
        center_atoms = set(cgr.center_atoms)

    # 2. add atoms of reaction environment based on config settings
    center_atoms = add_environment_atoms(
        cgr, center_atoms, config.environment_atom_count
    )

    # 3. include functional groups in the rule if specified in config
    if config.include_func_groups and config.func_groups_list:
        rule_atoms = add_functional_groups(
            reaction, center_atoms, config.func_groups_list
        )
    else:
        rule_atoms = center_atoms.copy()

    # 4. include ring structures in the rule if specified in config
    if config.include_rings:
        rule_atoms = add_ring_structures(cgr, rule_atoms)

    # 5. add leaving and incoming groups to the rule based on config settings
    rule_atoms, meta_debug = add_leaving_incoming_groups(
        reaction, rule_atoms, config.keep_leaving_groups, config.keep_incoming_groups
    )

    # 6. create substructures for reactants, products, and reagents
    reactant_substructures, product_substructures, reagents = (
        create_substructures_and_reagents(
            reaction, rule_atoms, config.as_query_container, config.keep_reagents
        )
    )
    # 7. clean atom marks in the molecules if they are being converted to query containers
    if config.as_query_container:
        reactant_substructures = clean_molecules(
            reactant_substructures,
            reaction.reactants,
            center_atoms,
            config.atom_info_retention,
        )
        product_substructures = clean_molecules(
            product_substructures,
            reaction.products,
            center_atoms,
            config.atom_info_retention,
        )

    # 8. assemble the final rule including metadata if specified
    rule = assemble_final_rule(
        reactant_substructures,
        product_substructures,
        reagents,
        meta_debug,
        config.keep_metadata,
        reaction,
    )

    # 9. reverse extracted reaction rule and reaction
    if config.reverse_rule:
        rule = reverse_reaction(rule)
        reaction = reverse_reaction(reaction)

    # 10. validate the rule using a reactor if validation is enabled in config
    if config.reactor_validation:
        if validate_rule(rule, reaction):
            rule.meta["reactor_validation"] = "passed"
        else:
            rule.meta["reactor_validation"] = "failed"

    return rule


def extract_rules(
    config: RuleExtractionConfig, reaction: ReactionContainer
) -> tuple[list[ReactionContainer], bool]:
    """
    Extracts reaction rules from a given reaction based on the specified
    configuration.

    :param config: An instance of ExtractRuleConfig, which contains various
        configuration settings for rule extraction, such as whether to include
        multicenter rules, functional groups, ring structures, leaving and incoming
        groups, etc.
    :param reaction: The reaction object (ReactionContainer) from which to extract
        rules. The reaction object represents a chemical reaction with specified
        reactants, products, and possibly reagents.
    :return: A tuple of (rules, skipped_multi_product) where *rules* is a list of
        ReactionContainer objects and *skipped_multi_product* is True if the reaction
        was skipped because it has multiple products.

    """

    standardizer = RemoveReagentsStandardizer()
    reaction = standardizer(reaction)

    # skip reactions with multiple products (checked after reagent removal)
    if config.single_product_only and len(reaction.products) != 1:
        return [], True

    if config.multicenter_rules:
        return [create_rule(config, reaction)], False

    # extract one rule per disconnected reaction-center component, dedup by CGR
    cgr = ~reaction
    seen_cgrs = {}
    for component in islice(cgr.centers_list, 15):
        rule = create_rule(config, reaction, _restrict_center_atoms=set(component))
        rule_cgr = ~rule
        if rule_cgr not in seen_cgrs:
            seen_cgrs[rule_cgr] = rule

    return list(seen_cgrs.values()), False


def _rule_to_reactor_smarts(rule: ReactionContainer) -> str:
    """Serialize an extracted rule to the final TSV SMARTS representation."""
    patterns = tuple(
        molecule_substructure_as_query(m, m.atoms_numbers) for m in rule.reactants
    )
    products = tuple(rule.products)
    reactor = CanonicalRetroReactor(patterns=patterns, products=products, delete_atoms=False)
    return str(reactor)


def _make_extracted_rule_record(rule: ReactionContainer) -> ExtractedRuleRecord:
    """Build the lightweight rule payload sent from worker to parent.

    The parent must not parse rule SMARTS back into ReactionContainer objects.
    Chython can emit query SMARTS that are valid enough for final Reactor TSV
    loading but fail a parser round-trip during aggregation. Computing the CGR
    key and final TSV SMARTS in the worker avoids that serial bottleneck and
    avoids parser crashes in the parent.
    """
    query_cgr = ~rule
    return ExtractedRuleRecord(
        cgr_key=canonical_query_cgr_key(query_cgr),
        rule_smarts=_rule_to_reactor_smarts(rule),
        reactor_validation=rule.meta.get("reactor_validation"),
    )


_worker_state: dict | None = None


def _init_extraction_worker(config_dict: dict, ignore_errors: bool, fmt: str) -> None:
    """Process initializer: build config once per worker process."""
    global _worker_state
    _worker_state = {
        "config": RuleExtractionConfig(**config_dict),
        "ignore_errors": ignore_errors,
        "fmt": fmt,
    }


def _make_audit_entry(
    index: int,
    raw_item: object,
    fmt: str,
    stage: str,
    error_type: str,
    message: str,
) -> ErrorEntry:
    """Build an audit ErrorEntry tagged with the input ``line_number``.

    Audit entries describe non-error reasons a reaction did not contribute
    to the final rule set (multi-product skip, no-rules-extracted, all
    rules filtered out). They share the ErrorEntry shape so the parent can
    fold them into a single per-reaction audit file alongside true errors.
    """
    original, source_info = extract_origin_fields(raw_item, fmt)
    return ErrorEntry(
        original=original,
        source_info=source_info,
        stage=stage,
        error_type=error_type,
        message=tsv_safe(message),
        line_number=index,
    )


def _extract_rules_batch_worker(
    batch: list[tuple[int, str]],
) -> ExtractionBatchResult:
    """Top-level picklable worker function for rule extraction.

    Called by each worker process via ``process_pool_map_stream``.
    Requires ``_init_extraction_worker`` to have been called first.

    Rules are serialized to SMARTS strings (~200 B each) rather than
    sending ReactionContainer objects (~19 KB each) over IPC.

    :param batch: List of ``(index, raw_string)`` pairs.
    :return: ExtractionBatchResult with rule SMARTS strings plus lightweight
        validation metadata, errors, and the count of skipped multi-product
        reactions.
    """
    if _worker_state is None:
        raise RuntimeError(
            "_extract_rules_batch_worker called outside a worker process. "
            "This function requires _init_extraction_worker to run first."
        )
    config = _worker_state["config"]
    ignore_errors = _worker_state["ignore_errors"]
    fmt = _worker_state["fmt"]

    rule_records: list[tuple[int, list[ExtractedRuleRecord], str]] = []
    errors: list[ErrorEntry] = []
    audit_entries: list[ErrorEntry] = []
    n_multi_product = 0
    for index, raw_item in batch:
        try:
            reaction = parse_reaction(raw_item, fmt=fmt)
            product_smi = str(unite_molecules(reaction.products))
            extracted_rules, skipped = extract_rules(config, reaction)
            if skipped:
                n_multi_product += 1
                audit_entries.append(
                    _make_audit_entry(
                        index,
                        raw_item,
                        fmt,
                        "extract_rules",
                        "SkippedMultiProduct",
                        "single_product_only=True and reaction has multiple "
                        "products after reagent removal",
                    )
                )
                continue
            rules_payload = [
                _make_extracted_rule_record(rule) for rule in extracted_rules
            ]
            if not rules_payload:
                audit_entries.append(
                    _make_audit_entry(
                        index,
                        raw_item,
                        fmt,
                        "extract_rules",
                        "NoRulesExtracted",
                        "reaction parsed successfully but no rules were extracted",
                    )
                )
                continue
            rule_records.append((index, rules_payload, product_smi))
        except Exception as e:
            if not ignore_errors:
                raise
            orig, source_info = extract_origin_fields(raw_item, fmt)
            etype = type(e).__qualname__
            stage = e.stage if hasattr(e, "stage") else "extract_rules"
            errors.append(
                ErrorEntry(
                    original=orig,
                    source_info=source_info,
                    stage=stage,
                    error_type=etype,
                    message=tsv_safe(str(e)),
                    line_number=index,
                )
            )
    return ExtractionBatchResult(
        rule_records=rule_records,
        errors=errors,
        n_multi_product=n_multi_product,
        audit_entries=audit_entries,
    )


def _update_rules_statistics(
    rules_statistics: dict,
    cgr_to_rule: dict,
    index: int,
    rule_records: list[ExtractedRuleRecord],
) -> None:
    """Update rules statistics with the indices of reactions they came from.

    Deduplication is performed by ``cgr_key``, which is canonical with
    respect to atom numbering (see
    :func:`synplan.chem.utils.canonical_query_cgr_key`) — so chemically
    identical rules from different workers collapse here at ingest time.
    The parent never parses rule SMARTS back into ReactionContainers; that
    keeps aggregation cheap and avoids parser round-trip failures for query
    SMARTS.
    """
    for rule_record in rule_records:
        prev_stats_len = len(rules_statistics)
        rules_statistics[rule_record.cgr_key].append(index)
        if len(rules_statistics) != prev_stats_len:
            cgr_to_rule[rule_record.cgr_key] = rule_record


def _process_extraction_result(
    result: ExtractionBatchResult,
    rules_statistics: dict,
    cgr_to_rule: dict,
    error_file: TextIOWrapper | None = None,
    error_counts: Counter | None = None,
    multi_product_count: list[int] | None = None,
    products_file: TextIOWrapper | None = None,
    audit_entries_by_index: dict[int, ErrorEntry] | None = None,
    reaction_rule_keys_by_index: dict[int, list[str]] | None = None,
    audit_counts: Counter | None = None,
) -> int:
    """Process a single ExtractionBatchResult, updating rules statistics.

    :param result: ExtractionBatchResult returned by a worker.
    :param rules_statistics: Dict mapping CGR key strings to lists of reaction indices.
    :param cgr_to_rule: Dict mapping CGR key strings to the first rule seen.
    :param error_file: Optional file handle to write failed reactions.
    :param error_counts: Optional counter to accumulate error categories.
    :param multi_product_count: Single-element list used as mutable accumulator.
    :param products_file: Optional file handle to write ``reaction_id\\tproduct_smiles``.
    :param audit_entries_by_index: Optional mapping populated with informational
        per-reaction audit entries (multi-product skips, no-rules-extracted,
        parse errors). Used to build the final audit file in input-line order.
    :param reaction_rule_keys_by_index: Optional mapping from reaction index to
        the canonical CGR keys of the rules it produced. Used after
        :func:`sort_rules` to attribute final filtering reasons back to the
        source reaction.
    :param audit_counts: Optional counter to accumulate audit-entry categories.
    :return: Number of reactions processed in this batch (for progress bar).
    """
    for index, rule_records, product_smi in result.rule_records:
        _update_rules_statistics(rules_statistics, cgr_to_rule, index, rule_records)
        if reaction_rule_keys_by_index is not None:
            reaction_rule_keys_by_index[index] = [
                rule_record.cgr_key for rule_record in rule_records
            ]
        if products_file is not None:
            products_file.write(f"{index}\t{product_smi}\n")

    for err in result.errors:
        if audit_entries_by_index is not None and err.line_number is not None:
            audit_entries_by_index[err.line_number] = err
        if error_file is not None:
            write_error_row(
                error_file,
                err.original,
                err.source_info,
                err.stage,
                err.error_type,
                err.message,
            )
        if error_counts is not None:
            error_counts[(err.stage, err.error_type)] += 1

    for entry in result.audit_entries:
        if audit_entries_by_index is not None and entry.line_number is not None:
            audit_entries_by_index[entry.line_number] = entry
        if audit_counts is not None:
            audit_counts[(entry.stage, entry.error_type)] += 1

    if multi_product_count is not None:
        multi_product_count[0] += result.n_multi_product

    return len(result.rule_records) + len(result.errors) + len(result.audit_entries)


def sort_rules(
    rules_stats: dict,
    cgr_to_rule: dict,
    min_popularity: int,
) -> tuple[list[tuple[ExtractedRuleRecord, list[int]]], dict[str, int]]:
    """
    Sorts reaction rules based on their popularity and validation status. This
    function sorts the given rules according to their popularity (i.e., the number of
    times they have been applied) and filters out rules that haven't passed reactor
    validation or are less popular than the specified minimum popularity threshold.

    :param rules_stats: A dictionary where each key is a rule CGR and the value is
        a list of integers. Each integer represents an index where the rule was applied.
    :param cgr_to_rule: A dictionary mapping rule CGRs to the first serialized
        rule record seen for that CGR.
    :param min_popularity: The minimum number of times a rule must be applied to be
        considered. Default is 3.

    :return: A tuple of (sorted_rules, filter_stats) where *sorted_rules* is a list of
        tuples with a serialized rule record and its application indices, sorted by
        descending popularity, and *filter_stats* is a dict with counts of rules
        rejected at each filtering stage.

    """
    passed = []
    filter_stats = {
        "total_unique_rules": 0,
        "rejected_reactor_validation": 0,
        "rejected_popularity": 0,
        "passed": 0,
    }

    for cgr, indices in rules_stats.items():
        rule = cgr_to_rule[cgr]
        filter_stats["total_unique_rules"] += 1
        # Reject only rules that explicitly failed validation. ``None`` means
        # validation was disabled in the extraction config; those rules pass
        # through unfiltered (the user opted out of the check).
        if rule.reactor_validation == "failed":
            filter_stats["rejected_reactor_validation"] += 1
            continue
        if len(indices) < min_popularity:
            filter_stats["rejected_popularity"] += 1
            continue
        filter_stats["passed"] += 1
        passed.append((rule, indices))

    passed.sort(key=lambda x: -len(x[1]))
    return passed, filter_stats


def _filtered_rule_reason(
    cgr_key: str,
    rules_statistics: dict,
    cgr_to_rule: dict,
    min_popularity: int,
) -> str:
    """Mirror :func:`sort_rules`' filtering decision for one rule key.

    Returns one of ``"reactor_validation_failed"``,
    ``"reactor_validation_<state>"`` (for non-passed sentinels),
    ``"below_min_popularity"`` or ``"retained"``.
    """
    rule = cgr_to_rule[cgr_key]
    validation = rule.reactor_validation
    if validation == "failed":
        return "reactor_validation_failed"
    if validation is not None and validation != "passed":
        return f"reactor_validation_{validation}"
    if len(rules_statistics[cgr_key]) < min_popularity:
        return "below_min_popularity"
    return "retained"


def _make_rule_filter_audit_entry(
    index: int,
    rule_keys: list[str],
    rules_statistics: dict,
    cgr_to_rule: dict,
    min_popularity: int,
) -> ErrorEntry:
    """Build an audit entry for a reaction whose extracted rules all got filtered."""
    reason_counts = Counter(
        _filtered_rule_reason(cgr_key, rules_statistics, cgr_to_rule, min_popularity)
        for cgr_key in rule_keys
    )
    reason_counts.pop("retained", None)

    if not reason_counts:
        error_type = "UnknownRuleFilter"
        message = "reaction did not map to a retained rule for an unknown reason"
    elif set(reason_counts) == {"below_min_popularity"}:
        error_type = "BelowMinPopularity"
        message = f"all extracted rules were below min_popularity={min_popularity}"
    elif all(reason.startswith("reactor_validation_") for reason in reason_counts):
        error_type = "ReactorValidationFailed"
        details = ", ".join(
            f"{reason.removeprefix('reactor_validation_')}={count}"
            for reason, count in sorted(reason_counts.items())
        )
        message = f"all extracted rules failed reactor validation ({details})"
    else:
        error_type = "AllRulesFiltered"
        details = ", ".join(
            f"{reason}={count}" for reason, count in sorted(reason_counts.items())
        )
        message = f"all extracted rules were filtered out ({details})"

    return ErrorEntry(
        original="",
        stage="rule_filtering",
        error_type=error_type,
        message=tsv_safe(message),
        line_number=index,
    )


def _add_rule_filter_audit_entries(
    audit_entries_by_index: dict[int, ErrorEntry],
    reaction_rule_keys_by_index: dict[int, list[str]],
    retained_reaction_indices: set[int],
    rules_statistics: dict,
    cgr_to_rule: dict,
    min_popularity: int,
    audit_counts: Counter | None = None,
) -> None:
    """Post-sort sweep: tag reactions whose rules were all filtered out.

    A reaction that produced rules but had every rule rejected by
    :func:`sort_rules` (reactor validation, min popularity) is otherwise
    silently dropped from the output. This sweep adds an audit entry
    explaining the rejection so users can debug their config.
    """
    for index, rule_keys in reaction_rule_keys_by_index.items():
        if index in retained_reaction_indices or index in audit_entries_by_index:
            continue
        entry = _make_rule_filter_audit_entry(
            index,
            rule_keys,
            rules_statistics,
            cgr_to_rule,
            min_popularity,
        )
        audit_entries_by_index[index] = entry
        if audit_counts is not None:
            audit_counts[(entry.stage, entry.error_type)] += 1


def _write_reaction_audit_file(
    audit_file_path: Path,
    reaction_data_path: str,
    audit_entries_by_index: dict[int, ErrorEntry],
) -> None:
    """Write a per-reaction audit TSV, sorted by input line index.

    The audit row carries the original SMILES/source from the *input* file
    so it is reproducible against the user's data, plus the stage, error
    type and message captured during extraction or post-sort filtering.
    """
    with open(audit_file_path, "w", encoding="utf-8") as audit_file:
        audit_file.write(
            "# reaction_index\toriginal_smiles\tsource_info\tstage\t"
            "error_type\terror_message\n"
        )
        if not audit_entries_by_index:
            return
        raw_reader = RawReactionReader(reaction_data_path)
        for index, raw_item in enumerate(raw_reader):
            entry = audit_entries_by_index.get(index)
            if entry is None:
                continue
            if entry.original:
                original, source_info = entry.original, entry.source_info
            else:
                original, source_info = extract_origin_fields(
                    raw_item, raw_reader.format
                )
            audit_file.write(
                f"{index}\t{tsv_safe(original)}\t{tsv_safe(source_info)}\t"
                f"{tsv_safe(entry.stage)}\t{tsv_safe(entry.error_type)}\t"
                f"{tsv_safe(entry.message)}\n"
            )


def _extract_rules_serial(
    config: RuleExtractionConfig,
    reaction_data_path: str,
    rules_statistics: dict,
    cgr_to_rule: dict,
    *,
    ignore_errors: bool = False,
    error_file: TextIOWrapper | None = None,
    error_counts: Counter | None = None,
    products_file: TextIOWrapper | None = None,
    fmt: str = "smi",
    audit_entries_by_index: dict[int, ErrorEntry] | None = None,
    reaction_rule_keys_by_index: dict[int, list[str]] | None = None,
    audit_counts: Counter | None = None,
) -> tuple[int, int]:
    """Serial rules extraction path used when a single CPU is requested.

    Returns ``(n_processed, n_multi_product)``.
    """
    n_processed = 0
    n_multi_product = 0
    raw_reader = RawReactionReader(reaction_data_path)
    for index, raw_item in tqdm(
        enumerate(raw_reader),
        desc="Number of reactions processed: ",
        bar_format="{desc}{n} [{elapsed}]",
    ):
        n_processed += 1
        try:
            reaction = parse_reaction(raw_item, fmt=fmt)
            product_smi = str(unite_molecules(reaction.products))
            extracted_rules, skipped = extract_rules(config, reaction)
            if skipped:
                n_multi_product += 1
                entry = _make_audit_entry(
                    index,
                    raw_item,
                    fmt,
                    "extract_rules",
                    "SkippedMultiProduct",
                    "single_product_only=True and reaction has multiple "
                    "products after reagent removal",
                )
                if audit_entries_by_index is not None:
                    audit_entries_by_index[index] = entry
                if audit_counts is not None:
                    audit_counts[(entry.stage, entry.error_type)] += 1
                continue
            rule_smarts = [
                _make_extracted_rule_record(rule) for rule in extracted_rules
            ]
            if not rule_smarts:
                entry = _make_audit_entry(
                    index,
                    raw_item,
                    fmt,
                    "extract_rules",
                    "NoRulesExtracted",
                    "reaction parsed successfully but no rules were extracted",
                )
                if audit_entries_by_index is not None:
                    audit_entries_by_index[index] = entry
                if audit_counts is not None:
                    audit_counts[(entry.stage, entry.error_type)] += 1
                continue
            _update_rules_statistics(rules_statistics, cgr_to_rule, index, rule_smarts)
            if reaction_rule_keys_by_index is not None:
                reaction_rule_keys_by_index[index] = [
                    rule_record.cgr_key for rule_record in rule_smarts
                ]
            if products_file is not None:
                products_file.write(f"{index}\t{product_smi}\n")
        except Exception as e:
            if not ignore_errors:
                raise
            orig, source_info = extract_origin_fields(raw_item, fmt)
            etype = type(e).__qualname__
            stage = e.stage if hasattr(e, "stage") else "extract_rules"
            err_entry = ErrorEntry(
                original=orig,
                source_info=source_info,
                stage=stage,
                error_type=etype,
                message=tsv_safe(str(e)),
                line_number=index,
            )
            if audit_entries_by_index is not None:
                audit_entries_by_index[index] = err_entry
            if error_file is not None:
                write_error_row(error_file, orig, source_info, stage, etype, str(e))
            if error_counts is not None:
                error_counts[(stage, etype)] += 1
    return n_processed, n_multi_product


def _print_extraction_summary(
    n_processed: int,
    sorted_rules: list[tuple[ExtractedRuleRecord, list[int]]],
    filter_stats: dict[str, int],
    error_counts: Counter,
    error_file_path: Path | None,
    audit_counts: Counter | None = None,
    audit_file_path: Path | None = None,
) -> None:
    """Print a categorized summary of rule extraction results."""
    n_rules = len(sorted_rules)
    n_errors = sum(error_counts.values())
    covered_reactions = set()
    for _rule, indices in sorted_rules:
        covered_reactions.update(indices)
    n_covered = len(covered_reactions)
    if n_processed:
        summary_lines = [
            f"Finished: processed {n_processed}, extracted {n_rules} rules "
            f"from {n_covered} reactions ({n_covered * 100 / n_processed:.1f}%), "
            f"failed {n_errors}"
        ]
    else:
        summary_lines = [f"Finished: processed 0 reactions, failed {n_errors}"]
    # Reaction-level filtering
    n_multi_product = filter_stats.get("skipped_multi_product", 0)
    if n_multi_product and n_processed:
        summary_lines.append(
            f"Reactions skipped (multi-product): {n_multi_product} "
            f"({n_multi_product * 100 / n_processed:.1f}%)"
        )
    # Rule filtering breakdown
    total_unique = filter_stats.get("total_unique_rules", 0)
    if total_unique:
        summary_lines.append(f"Rule filtering ({total_unique} unique rules extracted):")
        for key, label in [
            ("rejected_reactor_validation", "reactor validation failed"),
            ("rejected_popularity", "below min popularity"),
        ]:
            count = filter_stats.get(key, 0)
            if count:
                summary_lines.append(
                    f"  {label}: {count} ({count * 100 / total_unique:.1f}%)"
                )
        n_passed = filter_stats.get("passed", 0)
        summary_lines.append(
            f"  passed all filters: {n_passed} ({n_passed * 100 / total_unique:.1f}%)"
        )
    if error_counts:
        summary_lines.append("Extraction errors:")
        for (stage, etype), count in error_counts.most_common():
            summary_lines.append(f"  {stage}/{etype}={count}")
    if audit_counts:
        n_audited = sum(audit_counts.values())
        summary_lines.append(f"Non-retained reactions: {n_audited}")
        for (stage, etype), count in audit_counts.most_common():
            summary_lines.append(f"  {stage}/{etype}={count}")
    if error_file_path is not None:
        summary_lines.append(f"Errors written to: {error_file_path}")
    if audit_file_path is not None:
        summary_lines.append(f"Reaction audit written to: {audit_file_path}")
    summary = "\n".join(summary_lines)
    print(summary)
    logger.info(summary)


def extract_rules_from_reactions(
    config: RuleExtractionConfig,
    reaction_data_path: str,
    reaction_rules_path: str,
    num_cpus: int,
    batch_size: int,
    *,
    ignore_errors: bool = False,
    error_file_path: str | Path | None = None,
    audit_file_path: str | Path | None = None,
) -> None:
    """
    Extracts reaction rules from a set of reactions based on the given configuration.
    This function uses ProcessPoolExecutor for parallel processing of reactions in
    batches to extract reaction rules. Extracted rules and their statistics are
    collected, then saved as a TSV file.

    :param config: Configuration settings for rule extraction, including file paths,
        batch size, and other parameters.
    :param reaction_data_path: Path to the file containing reaction database.
    :param reaction_rules_path: Name of the file to store the extracted rules.
    :param num_cpus: Number of CPU cores to use for processing. Defaults to 1.
    :param batch_size: Number of reactions to process in each batch. Defaults to 10.
    :param ignore_errors: If True, log extraction failures and keep processing.
    :param error_file_path: Path to write failed reactions (TSV). If None and
        ``ignore_errors`` is True, defaults to ``<output>.errors.tsv``.
    :param audit_file_path: Path to write the per-reaction audit TSV (one row
        per non-retained reaction, sorted by input line). If None and
        ``ignore_errors`` is True, defaults to ``<output>.audit.tsv``. The
        audit complements the streaming error file: errors record what
        crashed during workers, the audit records every reaction whose rules
        did not survive sort (multi-product skip, no-rules-extracted,
        below-min-popularity, reactor-validation-failed).
    :return: None

    """

    reaction_rules_path_base, _ = splitext(reaction_rules_path)
    extracted_rules_and_statistics = defaultdict(list)  # CGR -> list[int]
    cgr_to_rule: dict = {}  # CGR -> first ReactionContainer seen

    # Resolve error file path
    _error_path: Path | None = None
    if error_file_path is not None:
        _error_path = Path(error_file_path)
    elif ignore_errors:
        _error_path = Path(f"{reaction_rules_path_base}.errors.tsv")

    # Resolve audit file path
    _audit_path: Path | None = None
    if audit_file_path is not None:
        _audit_path = Path(audit_file_path)
    elif ignore_errors:
        _audit_path = Path(f"{reaction_rules_path_base}.audit.tsv")

    error_counts: Counter = Counter()
    audit_counts: Counter = Counter()
    audit_entries_by_index: dict[int, ErrorEntry] = {}
    reaction_rule_keys_by_index: dict[int, list[str]] = {}
    error_file = None
    if _error_path is not None:
        error_file = open(_error_path, "w", encoding="utf-8")
        write_error_tsv_header(error_file)

    n_processed = 0
    n_multi_product = 0

    raw_reader = RawReactionReader(reaction_data_path)
    fmt = raw_reader.format

    # Temp file to store reaction_id → product_smiles during extraction,
    # so we don't need to re-parse the entire input file for policy data.
    products_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", delete=False, encoding="utf-8"
    )

    try:
        # Simple serial path for a single CPU.
        if num_cpus <= 1:
            n_processed, n_multi_product = _extract_rules_serial(
                config,
                reaction_data_path,
                extracted_rules_and_statistics,
                cgr_to_rule,
                ignore_errors=ignore_errors,
                error_file=error_file,
                error_counts=error_counts,
                products_file=products_tmp,
                fmt=fmt,
                audit_entries_by_index=audit_entries_by_index,
                reaction_rule_keys_by_index=reaction_rule_keys_by_index,
                audit_counts=audit_counts,
            )
        else:
            multi_product_count = [0]  # mutable accumulator

            def _make_batches():
                batch = []
                for index, raw_item in enumerate(raw_reader):
                    batch.append((index, raw_item))
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

            bar = tqdm(
                desc="Number of reactions processed: ",
                bar_format="{desc}{n} [{elapsed}]",
            )

            with graceful_shutdown() as stop:
                for result in process_pool_map_stream(
                    _make_batches(),
                    _extract_rules_batch_worker,
                    max_workers=num_cpus,
                    max_pending=4 * num_cpus,
                    ordered=True,
                    initializer=_init_extraction_worker,
                    initargs=(config.to_dict(), ignore_errors, fmt),
                    timeout=300,
                    # Avoid ProcessPoolExecutor worker recycling here: CPython
                    # documents open hangs with max_tasks_per_child in 3.13/3.14.
                    # The shared pool helper now performs bounded cleanup of
                    # stale chemistry workers at the end of the run instead.
                ):
                    if stop.is_set():
                        break

                    batch_count = _process_extraction_result(
                        result,
                        extracted_rules_and_statistics,
                        cgr_to_rule,
                        error_file=error_file,
                        error_counts=error_counts,
                        multi_product_count=multi_product_count,
                        products_file=products_tmp,
                        audit_entries_by_index=audit_entries_by_index,
                        reaction_rule_keys_by_index=reaction_rule_keys_by_index,
                        audit_counts=audit_counts,
                    )
                    n_processed += batch_count
                    bar.update(batch_count)

            bar.close()
            n_multi_product = multi_product_count[0]
    finally:
        if error_file is not None:
            error_file.close()
        products_tmp.close()

    sorted_rules, filter_stats = sort_rules(
        extracted_rules_and_statistics,
        cgr_to_rule,
        min_popularity=config.min_popularity,
    )
    filter_stats["skipped_multi_product"] = n_multi_product

    # Reactions whose rules all got filtered: add a per-reaction audit entry
    # explaining why none of their rules survived sort_rules.
    retained_reaction_indices = {
        index for _rule, indices in sorted_rules for index in indices
    }
    _add_rule_filter_audit_entries(
        audit_entries_by_index,
        reaction_rule_keys_by_index,
        retained_reaction_indices,
        extracted_rules_and_statistics,
        cgr_to_rule,
        config.min_popularity,
        audit_counts=audit_counts,
    )

    if _audit_path is not None:
        _write_reaction_audit_file(
            _audit_path,
            reaction_data_path,
            audit_entries_by_index,
        )

    # Always save rules as TSV (primary format: human-readable, safe, reproducible).
    rules_tsv_path = f"{reaction_rules_path_base}.tsv"
    with open(rules_tsv_path, "w", encoding="utf-8") as tsv_file:
        tsv_file.write("rule_smarts\tpopularity\treaction_indices\n")
        for rule, indices in sorted_rules:
            tsv_file.write(
                f"{rule.rule_smarts}\t{len(indices)}\t{','.join(map(str, indices))}\n"
            )

    _print_extraction_summary(
        n_processed,
        sorted_rules,
        filter_stats,
        error_counts,
        _error_path,
        audit_counts=audit_counts,
        audit_file_path=_audit_path,
    )

    # Generate policy training mapping file from the temp products file.
    # No re-parsing needed — product SMILES were captured during extraction.
    policy_data_path = f"{reaction_rules_path_base}_policy_data.tsv"
    reaction_rule_pairs = load_rule_index_mapping_tsv(rules_tsv_path)
    n_mapped = 0
    _n_small_products = 0
    products_tmp_path = Path(products_tmp.name)
    try:
        with (
            open(products_tmp_path, encoding="utf-8") as products_in,
            open(policy_data_path, "w", encoding="utf-8") as out,
        ):
            out.write("product_smiles\trule_id\n")
            for line in products_in:
                reaction_id_str, product_smi = line.rstrip("\n").split("\t", 1)
                rule_id = reaction_rule_pairs.get(int(reaction_id_str))
                if rule_id is None:
                    continue
                out.write(f"{product_smi}\t{rule_id}\n")
                n_mapped += 1
    finally:
        products_tmp_path.unlink(missing_ok=True)

    n_expected = len(reaction_rule_pairs)
    print(
        f"Policy training data: {n_mapped}/{n_expected} examples written to "
        f"{policy_data_path}"
    )
