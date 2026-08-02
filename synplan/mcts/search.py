"""Module containing functions for running tree search for the set of target
molecules."""

import csv
import gzip
import json
import logging
import os.path
from collections.abc import Iterator
from pathlib import Path

from chython.containers import MoleculeContainer
from rdkit import Chem
from tqdm.auto import tqdm

from synplan import __version__
from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.routes.io import (
    make_json,
    write_routes_csv,
    write_routes_json,
)
from synplan.chem.reaction.routes.quality.scorer import RouteScorer
from synplan.chem.reaction.routes.representation import extract_reactions
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.tree import Tree, TreeConfig
from synplan.utils.config import PolicyNetworkConfig
from synplan.utils.files import iter_csv_smiles, iter_smiles_records
from synplan.utils.loading import (
    load_building_blocks,
    load_evaluation_function,
    load_policy_function,
    load_reaction_rules,
)
from synplan.utils.visualisation import extract_routes, generate_results_html

#: Versioned identifier for the public route-export contract emitted by
#: :func:`export_routes_artifact`. Bump when the envelope/manifest shape changes.
ROUTE_EXPORT_SCHEMA_VERSION = "synplan-routes/1"


def _canonical_target_key(smiles: str) -> str:
    """Canonical SMILES key for the route-export artifact.

    Mirrors retrocast's ``canonicalize_smiles`` default flags so keys match
    ``retrocast.curation...Target.smiles`` byte-for-byte: RDKit
    ``MolFromSmiles`` (sanitize=True) then ``MolToSmiles(canonical=True,
    isomericSmiles=True)``, with atom mapping left intact. Falls back to the raw
    string (with a warning) when RDKit cannot parse the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(
            "Could not RDKit-canonicalize target SMILES %r for route export; "
            "keying by the raw string.",
            smiles,
        )
        return smiles
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _iter_target_smiles(targets_path: str) -> Iterator[str]:
    """Yield target SMILES from a targets file.

    Supports plain SMILES-per-line files and tabular ``.tsv``/``.csv`` files
    with a SMILES column (read via :func:`iter_csv_smiles`).
    """
    suffix = Path(targets_path).suffix.lower()
    if suffix in (".tsv", ".csv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        yield from iter_csv_smiles(targets_path, delimiter=delimiter)
        return
    yield from iter_smiles_records(targets_path)


def extract_tree_stats(
    tree: Tree, target: str | MoleculeContainer, init_smiles: str | None = None
):
    """Collects various statistics from a tree and returns them in a dictionary format.

    :param tree: The built search tree.
    :param target: The target molecule associated with the tree.
    :param init_smiles: initial SMILES of the molecule, optional.
    :return: A dictionary with the calculated statistics.
    """

    newick_tree, newick_meta = tree.newickify(visits_threshold=0)
    newick_meta_line = ";".join(
        [f"{nid},{v[0]},{v[1]},{v[2]}" for nid, v in newick_meta.items()]
    )

    stats = tree.to_stats_dict()
    stats["target_smiles"] = init_smiles if init_smiles is not None else str(target)
    stats["newick_tree"] = newick_tree
    stats["newick_meta"] = newick_meta_line
    return stats


def build_target_routes(tree, reactions: dict | None = None) -> list[dict]:
    """Public per-target route shape for the route-export contract.

    Returns the list of route-tree dicts for one solved target. Each route tree
    is a recursive bipartite structure of ``{"type": "mol", ...}`` and
    ``{"type": "reaction", ...}`` nodes whose format is exactly the output of
    :func:`synplan.chem.reaction.routes.io.make_json` (node shape is not
    reshaped). Returns ``[]`` when the tree has no winning nodes.

    :param tree: A completed :class:`~synplan.mcts.tree.Tree`.
    :param reactions: Optional precomputed ``extract_reactions(tree)`` result to
        avoid a second tree traversal. When ``None`` it is computed here.
    """
    if not bool(tree.winning_nodes):
        return []
    if reactions is None:
        reactions = extract_reactions(tree)
    return list(make_json(reactions, keep_ids=True).values())


def export_routes_artifact(
    results: dict,
    results_root,
    *,
    filename: str = "results.json.gz",
) -> Path:
    """Write the target-keyed route-export artifact (public contract).

    Gzip-writes ``results`` as JSON to ``results_root/filename``. ``results`` is
    the public envelope: a top-level dict keyed by the RDKit-canonical target
    SMILES (matching retrocast's ``Target.smiles``; see
    :func:`_canonical_target_key`) mapping to ``[route_tree, ...]`` with ``[]``
    for unsolved targets, where each ``route_tree`` is a
    :func:`build_target_routes` / ``make_json`` node tree.

    Also writes ``results_root/manifest.json`` with a top-level ``directives``
    dict (``{"adapter": "synplanner", "raw_results_filename": filename}``) plus
    top-level ``schema_version`` and ``synplan_version`` keys.

    :return: Path to the written gzipped results file.
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    results_path = results_root.joinpath(filename)
    with gzip.open(results_path, "wt", encoding="utf-8") as fh:
        json.dump(results, fh)

    manifest = {
        "schema_version": ROUTE_EXPORT_SCHEMA_VERSION,
        "synplan_version": __version__,
        "directives": {
            "adapter": "synplanner",
            "raw_results_filename": filename,
        },
    }
    with open(results_root.joinpath("manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return results_path


def run_search(
    targets_path: str,
    search_config: dict,
    policy_config: PolicyNetworkConfig,
    evaluation_config,
    reaction_rules_path: str,
    building_blocks_path: str,
    results_root: str = "search_results",
    route_scorer: RouteScorer | None = None,
    priority_rules: dict[str, list[CanonicalRetroReactor]] | None = None,
    reconcile_atom_mapping: bool = False,
    export_routes: bool = False,
    routes_filename: str = "results.json.gz",
) -> None:
    """Performs a tree search on a set of target molecules using specified configuration
    and reaction rules, logging the results and statistics.

    :param targets_path: The path to the file containing the target molecules (in SDF or
        SMILES format).
    :param search_config: The config object containing the configuration for the tree
        search.
    :param policy_config: The config object containing the configuration for the policy.
    :param evaluation_config: The evaluation configuration object (e.g., RolloutEvaluationConfig).
    :param reaction_rules_path: The path to the file containing reaction rules.
    :param building_blocks_path: The path to the file containing building blocks.
    :param results_root: The name of the folder where the results of the tree search
        will be saved.
    :param route_scorer: Optional post-search route scorer for re-ranking
        winning routes (e.g. ProtectionRouteScorer).
    :param priority_rules: Optional mapping of curated rule sets
        (``{set_name: [Reactor, ...]}``) forwarded to every per-target
        :class:`Tree`. See :meth:`Tree.__init__` for the full semantics.
        When supplied, set ``search_config["use_priority"] = True`` (or pass
        a :class:`TreeConfig` with ``use_priority=True``); otherwise the rules
        are accepted but never tried.
    :param reconcile_atom_mapping: If False (default), build the exported
        routes_dict directly from the tree (fast path, per-step-local atom
        numbering). If True, use the slower ``compose_route_cgr`` path to give
        cross-step-reconciled atom-map numbering in the exported reactions.
    :param export_routes: When True, additionally emit the public route-export
        artifact (``routes_filename`` + ``manifest.json``) keyed by the
        RDKit-canonical target SMILES (matching retrocast's ``Target.smiles``;
        ``[]`` for unsolved targets), for downstream consumers. Defaults to
        False, leaving the existing outputs byte-identical.
    :param routes_filename: Filename (under ``results_root``) for the gzipped
        target-keyed routes artifact. Only used when ``export_routes`` is True.
    :return: None.
    """

    # results folder
    results_root = Path(results_root)
    if not results_root.exists():
        results_root.mkdir()

    # output files
    stats_file = results_root.joinpath("tree_search_stats.csv")
    routes_file = results_root.joinpath("extracted_routes.json")
    routes_folder = results_root.joinpath("extracted_routes_html")
    routes_folder.mkdir(exist_ok=True)

    # stats header
    stats_header = [
        "target_smiles",
        "num_routes",
        "num_nodes",
        "num_iter",
        "tree_depth",
        "search_time",
        "solved",
        # Policy performance
        "expansion_calls",
        "expansion_successes",
        "total_rules_tried",
        "total_rules_succeeded",
        "rule_applicability_rate",
        "dead_end_nodes",
        # Search dynamics
        "first_solution_iteration",
        "first_solution_time",
        # Tree shape
        "max_branching_factor",
        "mean_branching_factor",
        # Route quality
        "best_route_score",
        "mean_winning_rule_rank",
        # Tree structure
        "newick_tree",
        "newick_meta",
        "error",
        # Priority rules
        "fraction_routes_with_priority",
        "n_routes_with_priority",
        "per_priority_source",
        "priority_rules_tried",
        "policy_rules_tried",
        "priority_rules_succeeded",
        "policy_rules_succeeded",
    ]

    # Load resources
    policy_function = load_policy_function(policy_config=policy_config)
    reaction_rules = load_reaction_rules(reaction_rules_path)
    building_blocks = load_building_blocks(building_blocks_path, standardize=False)

    # Create evaluation strategy from config
    evaluation_function = load_evaluation_function(evaluation_config)

    # run search
    n_solved = 0
    extracted_routes = []
    # Public route-export accumulator keyed by RDKit-canonical target SMILES:
    # {canonical_target_smiles: [route_tree, ...]}.
    exported_routes: dict[str, list[dict]] = {}

    tree_config = TreeConfig.from_dict(search_config)
    tree_config.silent = True
    with open(stats_file, "w", encoding="utf-8", newline="\n") as csvfile:
        statswriter = csv.DictWriter(csvfile, delimiter=",", fieldnames=stats_header)
        statswriter.writeheader()

        for ti, target_smi in tqdm(
            enumerate(_iter_target_smiles(targets_path)),
            leave=True,
            desc="Number of target molecules processed: ",
            bar_format="{desc}{n} [{elapsed}]",
        ):
            target_smi = target_smi.strip()
            # Key the export dict by the RDKit-canonical target SMILES so keys
            # match retrocast's Target.smiles byte-for-byte. Every target starts
            # empty; only a solved one overwrites it.
            export_key = None
            if export_routes:
                export_key = _canonical_target_key(target_smi)
                exported_routes[export_key] = []
            try:
                target_mol = mol_from_smiles(target_smi)
                # run search
                tree = Tree(
                    target=target_mol,
                    config=tree_config,
                    reaction_rules=reaction_rules,
                    building_blocks=building_blocks,
                    expansion_function=policy_function,
                    evaluation_function=evaluation_function,
                    route_scorer=route_scorer,
                    priority_rules=priority_rules,
                )

                _ = list(tree)

            except Exception as e:
                extracted_routes.append(
                    [
                        {
                            "type": "mol",
                            "smiles": target_smi,
                            "in_stock": False,
                            "children": [],
                        }
                    ]
                )
                logging.warning(
                    f"Retrosynthetic_planning {target_smi} failed with the following error: {e}"
                )

                continue

            # is solved
            n_solved += bool(tree.winning_nodes)
            if bool(tree.winning_nodes):
                # extract routes
                extracted_routes.append(extract_routes(tree))

                # save routes
                generate_results_html(
                    tree,
                    os.path.join(routes_folder, f"retroroutes_target_{ti}.html"),
                    extended=True,
                )

                # save json routes
                with open(routes_file, "w", encoding="utf-8") as f:
                    json.dump(extracted_routes, f)

                # Save mapped reactions (CSV)
                routes_dict = extract_reactions(
                    tree, reconcile_atom_mapping=reconcile_atom_mapping
                )
                write_routes_csv(
                    routes_dict, os.path.join(routes_folder, f"mapped_routes_{ti}.csv")
                )

                # save mapped reactions (JSON)
                write_routes_json(
                    routes_dict, os.path.join(routes_folder, f"mapped_routes_{ti}.json")
                )

                # public route export (reuse extract_reactions result)
                if export_routes:
                    exported_routes[export_key] = build_target_routes(
                        tree, reactions=routes_dict
                    )

            # save stats
            statswriter.writerow(extract_tree_stats(tree, target_smi))
            csvfile.flush()

    if export_routes:
        export_routes_artifact(exported_routes, results_root, filename=routes_filename)

    print(f"Number of solved target molecules: {n_solved}")
