"""Module containing functions for protocol of reaction rules extraction."""

import logging
import pickle
import warnings
from collections import Counter, defaultdict
from io import TextIOWrapper
from itertools import islice
from os.path import splitext
from pathlib import Path
from typing import Dict, List, Set, Tuple

import ray
from tqdm.auto import tqdm

from chython.containers.cgr import CGRContainer
from chython.containers.molecule import MoleculeContainer
from chython.containers.query import QueryContainer
from chython.containers.reaction import ReactionContainer
from chython.containers.bonds import QueryBond, Bond
from chython.periodictable import QueryElement
from chython.exceptions import InvalidAromaticRing
from chython.reactor import Reactor

from synplan.chem.data.standardizing import RemoveReagentsStandardizer
from synplan.chem.utils import reverse_reaction
from synplan.utils.config import RuleExtractionConfig
from synplan.utils.files import RawReactionReader, ReactionReader, ReactionWriter, parse_reaction


logger = logging.getLogger(__name__)


def molecule_substructure_as_query(mol, atoms) -> QueryContainer:
    atoms = set(atoms)
    q = QueryContainer(smarts="")
    for n in atoms:
        atom = mol.atom(n)
        xy = atom.xy if hasattr(atom, 'xy') else None
        if isinstance(atom, QueryElement):
            q.add_atom(atom.copy(full=True), n, xy=xy)
        else:
            q.add_atom(
                QueryElement.from_atom(
                    atom,
                    neighbors=True,
                    hydrogens=True,
                    ring_sizes=True,
                ),
                n,
                xy=xy,
            )
    for n, m, bond in mol.bonds():
        if n in atoms and m in atoms:
            if isinstance(bond, QueryBond):
                q.add_bond(n, m, bond.copy(full=True))
            elif isinstance(bond, Bond):
                q.add_bond(n, m, QueryBond.from_bond(bond))
    return q


def add_environment_atoms(
    cgr: CGRContainer, center_atoms: Set[int], environment_atom_count: int
) -> Set[int]:
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
    center_atoms: Set[int],
    func_groups_list: List[QueryContainer],
) -> Set[int]:
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


def add_ring_structures(cgr: CGRContainer, rule_atoms: Set[int]) -> Set[int]:
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
    rule_atoms: Set[int],
    keep_leaving_groups: bool,
    keep_incoming_groups: bool,
) -> Tuple[Set[int], Dict[str, Set]]:
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
    rule_molecules: List[MoleculeContainer],
    reaction_molecules: Tuple[MoleculeContainer],
    reaction_center_atoms: Set[int],
    atom_retention_details: Dict[str, Dict[str, bool]],
) -> List[QueryContainer]:
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
    attributes_to_keep: Dict[str, bool],
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
    rule_atoms: Set[int],
    as_query_container: bool,
    keep_reagents: bool,
) -> Tuple[List[MoleculeContainer], List[MoleculeContainer], List]:
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
    reactant_substructures: List[QueryContainer],
    product_substructures: List[QueryContainer],
    reagents: List,
    meta_debug: Dict[str, Set],
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
    reactor = Reactor(patterns=patterns, products=products)
    try:
        for result_reaction in reactor(*reaction.reactants):  # unpack here
            result_products = []
            for result_product in result_reaction.products:
                tmp = result_product.copy()
                try:
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
        return False
    return False


def create_rule(
    config: RuleExtractionConfig, reaction: ReactionContainer
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
    :return: A ReactionContainer object representing the extracted reaction rule.

    """

    # 1. create reaction CGR
    cgr = ~reaction
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
) -> Tuple[List[ReactionContainer], bool]:
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

    # extract separate rules for each distinct reaction center
    distinct_rules = set()
    for center_reaction in islice(reaction.enumerate_centers(), 15):
        distinct_rules.add(create_rule(config, center_reaction))

    return list(distinct_rules), False


@ray.remote
def process_reaction_batch(
    batch: List[Tuple[int, str]],
    config: RuleExtractionConfig,
    ignore_errors: bool = False,
    fmt: str = "smi",
) -> Tuple[List[Tuple[int, List[ReactionContainer]]], List[Tuple[str, str, str, str]], int]:
    """Process a batch of reactions for rule extraction.

    Raw strings are parsed inside the worker so that SMILES parsing is
    distributed across Ray workers instead of being a main-thread bottleneck.

    :param batch: List of ``(index, raw_string)`` pairs.
    :param config: Rule extraction configuration.
    :param ignore_errors: If True, log failures and continue.
    :param fmt: Format hint — ``"smi"`` or ``"rdf"``.
    :return: ``(results, errors, n_multi_product)`` where *errors* is a list of
        ``(original_smiles, stage, error_type, error_message)`` tuples and
        *n_multi_product* is the number of reactions skipped due to multiple products.
    """
    extracted_rules_list = []
    errors: List[Tuple[str, str, str, str]] = []
    n_multi_product = 0
    for index, raw_item in batch:
        try:
            reaction = parse_reaction(raw_item, fmt=fmt)
            extracted_rules, skipped = extract_rules(config, reaction)
            if skipped:
                n_multi_product += 1
            extracted_rules_list.append((index, extracted_rules))
        except Exception as e:
            if not ignore_errors:
                raise
            orig = raw_item if isinstance(raw_item, str) else str(raw_item)
            etype = type(e).__qualname__
            stage = e.stage if hasattr(e, "stage") else "extract_rules"
            errors.append((orig, stage, etype, str(e)))
    return extracted_rules_list, errors, n_multi_product


def _update_rules_statistics(
    rules_statistics: Dict[ReactionContainer, List[int]],
    index: int,
    extracted_rules: List[ReactionContainer],
) -> None:
    """Update rules statistics with the indices of reactions they came from."""
    for rule in extracted_rules:
        prev_stats_len = len(rules_statistics)
        rules_statistics[rule].append(index)
        if len(rules_statistics) != prev_stats_len:
            rule.meta["first_reaction_index"] = index


def process_completed_batch(
    futures: Dict,
    rules_statistics: Dict,
    error_file: TextIOWrapper | None = None,
    error_counts: Counter | None = None,
    multi_product_count: List[int] | None = None,
) -> None:
    """
    Processes completed batches of reactions, updating the rules statistics and
    writing rules to a file. This function waits for the completion of a batch of
    reactions processed in parallel (using Ray), updates the statistics for each
    extracted rule, and writes the rules to a result file if they are new. It also
    updates the progress bar with the size of the processed batch.

    :param futures: A dictionary of futures representing ongoing batch processing tasks.
    :param rules_statistics: A dictionary to keep track of statistics for each rule.
    :param error_file: Optional file handle to write failed reactions.
    :param error_counts: Optional counter to accumulate error categories.
    :param multi_product_count: Single-element list used as mutable accumulator for
        multi-product reaction count.
    :return: None

    """

    ready_id, running_id = ray.wait(list(futures.keys()), num_returns=1)
    try:
        results, errors, n_multi_product = ray.get(ready_id[0])
    except Exception as exc:
        logger.error("Batch failed entirely: %s", exc)
        del futures[ready_id[0]]
        return

    for index, extracted_rules in results:
        _update_rules_statistics(rules_statistics, index, extracted_rules)

    for orig, stage, etype, emsg in errors:
        if error_file is not None:
            error_file.write(f"{orig}\t{stage}\t{etype}\t{emsg}\n")
        if error_counts is not None:
            error_counts[(stage, etype)] += 1

    if multi_product_count is not None:
        multi_product_count[0] += n_multi_product

    del futures[ready_id[0]]


def sort_rules(
    rules_stats: Dict, min_popularity: int
) -> Tuple[List[Tuple[ReactionContainer, List[int]]], Dict[str, int]]:
    """
    Sorts reaction rules based on their popularity and validation status. This
    function sorts the given rules according to their popularity (i.e., the number of
    times they have been applied) and filters out rules that haven't passed reactor
    validation or are less popular than the specified minimum popularity threshold.

    :param rules_stats: A dictionary where each key is a reaction rule and the value is
        a list of integers. Each integer represents an index where the rule was applied.
    :type rules_stats: The number of occurrence of the reaction rules.
    :param min_popularity: The minimum number of times a rule must be applied to be
        considered. Default is 3.
    :type min_popularity: The minimum number of occurrence of the reaction rule to be
        selected.

    :return: A tuple of (sorted_rules, filter_stats) where *sorted_rules* is a list of
        tuples with a reaction rule and its application indices, sorted by descending
        popularity, and *filter_stats* is a dict with counts of rules rejected at each
        filtering stage.

    """
    passed = []
    filter_stats = {
        "total_unique_rules": 0,
        "rejected_reactor_validation": 0,
        "rejected_popularity": 0,
        "passed": 0,
    }

    for rule, indices in rules_stats.items():
        filter_stats["total_unique_rules"] += 1
        validation = rule.meta.get("reactor_validation", "not_set")
        if validation != "passed":
            filter_stats["rejected_reactor_validation"] += 1
            continue
        if len(indices) < min_popularity:
            filter_stats["rejected_popularity"] += 1
            continue
        filter_stats["passed"] += 1
        passed.append((rule, indices))

    passed.sort(key=lambda x: -len(x[1]))
    return passed, filter_stats


def _extract_rules_serial(
    config: RuleExtractionConfig,
    reaction_data_path: str,
    rules_statistics: Dict[ReactionContainer, List[int]],
    *,
    ignore_errors: bool = False,
    error_file: TextIOWrapper | None = None,
    error_counts: Counter | None = None,
    fmt: str = "smi",
) -> Tuple[int, int]:
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
            extracted_rules, skipped = extract_rules(config, reaction)
            if skipped:
                n_multi_product += 1
            _update_rules_statistics(rules_statistics, index, extracted_rules)
        except Exception as e:
            if not ignore_errors:
                raise
            orig = raw_item if isinstance(raw_item, str) else str(raw_item)
            etype = type(e).__qualname__
            stage = e.stage if hasattr(e, "stage") else "extract_rules"
            if error_file is not None:
                error_file.write(f"{orig}\t{stage}\t{etype}\t{e}\n")
            if error_counts is not None:
                error_counts[(stage, etype)] += 1
    return n_processed, n_multi_product


def _print_extraction_summary(
    n_processed: int,
    sorted_rules: List[Tuple[ReactionContainer, List[int]]],
    filter_stats: Dict[str, int],
    error_counts: Counter,
    error_file_path: Path | None,
) -> None:
    """Print a categorized summary of rule extraction results."""
    n_rules = len(sorted_rules)
    n_errors = sum(error_counts.values())
    covered_reactions = set()
    for _rule, indices in sorted_rules:
        covered_reactions.update(indices)
    n_covered = len(covered_reactions)
    summary_lines = [
        f"Finished: processed {n_processed}, extracted {n_rules} rules "
        f"from {n_covered} reactions ({n_covered*100/n_processed:.1f}%), "
        f"failed {n_errors}"
    ]
    # Reaction-level filtering
    n_multi_product = filter_stats.get("skipped_multi_product", 0)
    if n_multi_product:
        summary_lines.append(
            f"Reactions skipped (multi-product): {n_multi_product} "
            f"({n_multi_product*100/n_processed:.1f}%)"
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
                    f"  {label}: {count} ({count*100/total_unique:.1f}%)"
                )
        n_passed = filter_stats.get("passed", 0)
        summary_lines.append(
            f"  passed all filters: {n_passed} ({n_passed*100/total_unique:.1f}%)"
        )
    if error_counts:
        summary_lines.append("Extraction errors:")
        for (stage, etype), count in error_counts.most_common():
            summary_lines.append(f"  {stage}/{etype}={count}")
    if error_file_path is not None:
        summary_lines.append(f"Errors written to: {error_file_path}")
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
) -> None:
    """
    Extracts reaction rules from a set of reactions based on the given configuration.
    This function initializes a Ray environment for distributed computing and processes
    each reaction in the provided reaction database to extract reaction rules. It
    handles the reactions in batches, parallelize the rule extraction process. Extracted
    rules and their statistics are collected, then saved both as a pickle with
    statistics and as mapped reaction SMILES (.smi) records for interoperability.

    :param config: Configuration settings for rule extraction, including file paths,
        batch size, and other parameters.
    :param reaction_data_path: Path to the file containing reaction database.
    :param reaction_rules_path: Name of the file to store the extracted rules.
    :param num_cpus: Number of CPU cores to use for processing. Defaults to 1.
    :param batch_size: Number of reactions to process in each batch. Defaults to 10.
    :param ignore_errors: If True, log extraction failures and keep processing.
    :param error_file_path: Path to write failed reactions (TSV). If None and
        ``ignore_errors`` is True, defaults to ``<output>.errors.tsv``.
    :return: None

    """

    reaction_rules_path_base, _ = splitext(reaction_rules_path)
    extracted_rules_and_statistics = defaultdict(list)

    # Resolve error file path
    _error_path: Path | None = None
    if error_file_path is not None:
        _error_path = Path(error_file_path)
    elif ignore_errors:
        _error_path = Path(f"{reaction_rules_path_base}.errors.tsv")

    error_counts: Counter = Counter()
    error_file = None
    if _error_path is not None:
        error_file = open(_error_path, "w", encoding="utf-8")
        error_file.write("# original_smiles\tstage\terror_type\terror_message\n")

    n_processed = 0
    n_multi_product = 0

    raw_reader = RawReactionReader(reaction_data_path)
    fmt = raw_reader.format

    try:
        # Simple serial path for a single CPU.
        if num_cpus <= 1:
            n_processed, n_multi_product = _extract_rules_serial(
                config,
                reaction_data_path,
                extracted_rules_and_statistics,
                ignore_errors=ignore_errors,
                error_file=error_file,
                error_counts=error_counts,
                fmt=fmt,
            )
        else:
            ray.init(
                num_cpus=num_cpus, ignore_reinit_error=True, logging_level=logging.ERROR
            )

            futures = {}
            batch = []
            max_concurrent_batches = num_cpus
            multi_product_count = [0]  # mutable accumulator

            for index, raw_item in tqdm(
                enumerate(raw_reader),
                desc="Number of reactions processed: ",
                bar_format="{desc}{n} [{elapsed}]",
            ):
                n_processed += 1
                batch.append((index, raw_item))
                if len(batch) == batch_size:
                    future = process_reaction_batch.remote(
                        batch, config, ignore_errors, fmt
                    )
                    futures[future] = None
                    batch = []

                    while len(futures) >= max_concurrent_batches:
                        process_completed_batch(
                            futures,
                            extracted_rules_and_statistics,
                            error_file=error_file,
                            error_counts=error_counts,
                            multi_product_count=multi_product_count,
                        )

            if batch:
                future = process_reaction_batch.remote(
                    batch, config, ignore_errors, fmt
                )
                futures[future] = None

            while futures:
                process_completed_batch(
                    futures,
                    extracted_rules_and_statistics,
                    error_file=error_file,
                    error_counts=error_counts,
                    multi_product_count=multi_product_count,
                )

            n_multi_product = multi_product_count[0]
            ray.shutdown()
    finally:
        if error_file is not None:
            error_file.close()

    sorted_rules, filter_stats = sort_rules(
        extracted_rules_and_statistics,
        min_popularity=config.min_popularity,
    )
    filter_stats["skipped_multi_product"] = n_multi_product

    # Save rules as TSV (primary format: human-readable, safe, reproducible).
    rules_tsv_path = f"{reaction_rules_path_base}.tsv"
    with open(rules_tsv_path, "w", encoding="utf-8") as tsv_file:
        tsv_file.write("rule_smarts\tpopularity\treaction_indices\n")
        for rule, indices in sorted_rules:
            patterns = tuple(
                molecule_substructure_as_query(m, m.atoms_numbers)
                for m in rule.reactants
            )
            products = tuple(rule.products)
            reactor = Reactor(patterns=patterns, products=products)
            smarts_str = str(reactor)
            tsv_file.write(
                f"{smarts_str}\t{len(indices)}\t{','.join(map(str, indices))}\n"
            )

    # Also save pickle for backward compatibility (deprecated).
    pickle_path = f"{reaction_rules_path_base}.pickle"
    with open(pickle_path, "wb") as statistics_file:
        pickle.dump(sorted_rules, statistics_file)
    warnings.warn(
        f"Pickle file '{pickle_path}' written for backward compatibility. "
        "Prefer the TSV file for loading rules. "
        "Pickle support will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

    _print_extraction_summary(n_processed, sorted_rules, filter_stats, error_counts, _error_path)
