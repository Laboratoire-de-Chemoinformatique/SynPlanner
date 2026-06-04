import csv
import json
from typing import TYPE_CHECKING, Any

from chython import smiles as read_smiles
from chython.exceptions import InvalidAromaticRing

if TYPE_CHECKING:
    from synplan.mcts.tree import Tree


def _route_molecule_smiles(mol) -> str:
    """Return the route-IO molecule string using the existing preparation flow."""
    try:
        mol.kekule()
        mol.implicify_hydrogens()
        mol.thiele()
    except InvalidAromaticRing:
        # Keep serializing the original molecule string when aromatic
        # preparation fails; route export should remain best-effort.
        pass
    return str(mol)


def _route_step_metadata_from_tree(
    tree: "Tree", route_id: int
) -> dict[int, dict[str, Any]]:
    """Map route step ids from ``extract_reactions`` to tree rule metadata."""
    details = tree.route_details(route_id)
    steps = details.get("steps", [])
    total_steps = len(steps)
    metadata_by_step_id = {}

    for step_index, step in enumerate(steps):
        step_id = total_steps - 1 - step_index
        metadata_by_step_id[step_id] = {
            "step_id": step_id,
            "tree_node_id": step.get("node_id"),
            "rule_id": step.get("rule_id"),
            "rule_source": step.get("rule_source"),
            "rule_key": step.get("rule_key"),
        }

    return metadata_by_step_id


def _collect_reactions(tree):
    """
    Traverse a reaction tree in post-order and collect all ReactionContainers.
    Returns a dict mapping each reaction's new step ID (0, 1, …) to its container.
    """
    rxn_list = []

    def recurse(node):
        if not isinstance(node, dict):
            return
        for child in node.get("children", []) or []:
            recurse(child)
        if node.get("type") == "reaction":
            rxn_list.append(read_smiles(node["smiles"]))

    recurse(tree)
    return {i: rxn for i, rxn in enumerate(rxn_list)}


def make_dict(routes_json):
    """
    routes_json : dict or list of tree-dicts as produced by make_json()

    Returns a dict mapping each route index (0, 1, …) to a sub-dict
    of {new_step_id: ReactionContainer}, where the step IDs run
    from the earliest reaction (0) up to the final (max).
    """
    routes_dict = {}

    # Normalize to iterable of (route_idx, tree)
    if isinstance(routes_json, dict):
        items = ((int(k), v) for k, v in routes_json.items())
    else:
        items = enumerate(routes_json)

    for route_idx, tree in items:
        try:
            routes_dict[int(route_idx)] = _collect_reactions(tree)
        except Exception as e:
            print(f"Error processing route {route_idx}: {e}")

    return routes_dict


def read_routes_json(file_path="routes.json", to_dict=False):
    with open(file_path) as file:
        routes_json = json.load(file)
    if to_dict:
        return make_dict(routes_json)
    return routes_json


def read_routes_csv(file_path="routes.csv"):
    """Read route reactions from a CSV file.

    The input CSV is expected to contain ``route_id``, ``step_id``, ``smiles``,
    and ``meta`` columns. The ``meta`` value is currently ignored.

    Returns a nested dictionary: ``route_id -> step_id -> ReactionContainer``.
    """
    routes_dict = {}
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            route_id = int(row["route_id"])
            step_id = int(row["step_id"])
            smiles = row["smiles"]
            # adjust this constructor to your actual API
            reaction = read_smiles(smiles)
            routes_dict.setdefault(route_id, {})[step_id] = reaction
    return routes_dict


def make_json(
    routes_dict,
    keep_ids=True,
    tree: "Tree | None" = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
):
    """
    Convert routes into a nested JSON tree of reaction and molecule nodes.

    Args:
        routes_dict (dict[int, dict[int, Reaction]]): Mapping route IDs to steps (step_id -> Reaction).
        keep_ids (bool): If True, returns a list of route trees; otherwise returns a dict mapping route IDs to trees.
        tree (Tree | None): Optional source tree used to attach rule metadata to
            reaction nodes.
        route_metadata (dict | None): Optional per-route metadata mapping
            ``route_id -> step_id -> metadata``. This overrides metadata derived
            from ``tree`` when provided.

    Returns:
        list or dict: JSON-like tree(s) of routes.
    """
    # Prepare output
    all_routes = {} if keep_ids else []

    for route_id, steps in routes_dict.items():
        if not steps:
            continue
        route_step_metadata = (
            route_metadata.get(route_id) if route_metadata is not None else None
        )
        if route_step_metadata is None and tree is not None:
            route_step_metadata = _route_step_metadata_from_tree(tree, route_id)
        try:
            # Determine target molecule atoms from the final step of this route
            final_step = max(steps)
            target = steps[final_step].products[0]
            atom_nums = set(target._atoms.keys())

            # Precompute canonical SMILES and producer mapping for all products
            prod_map = {}  # smiles -> list of step_ids
            for sid, rxn in steps.items():
                for prod in rxn.products:
                    s = _route_molecule_smiles(prod)
                    prod_map.setdefault(s, []).append(sid)
        except Exception as e:
            print(f"Error processing route {route_id}: {e}")
            continue

        def build_mol_node(sid, _steps=steps, _atom_nums=atom_nums):
            """Find the product with any overlap to target atoms and recurse into its reaction."""
            rxn = _steps[sid]
            for p in rxn.products:
                if _atom_nums & set(p._atoms.keys()):
                    smiles = _route_molecule_smiles(p)
                    return {
                        "type": "mol",
                        "smiles": smiles,
                        "children": [build_reaction_node(sid)],
                        "in_stock": False,
                    }
            # Shouldn't reach here if tree is consistent
            return None

        def build_reaction_node(
            sid,
            _steps=steps,
            _route_step_metadata=route_step_metadata,
            _prod_map=prod_map,
        ):
            """Build reaction node and recurse into reactant molecule nodes."""
            rxn = _steps[sid]
            node = {"type": "reaction", "smiles": format(rxn, "m"), "children": []}
            if _route_step_metadata and sid in _route_step_metadata:
                node.update(_route_step_metadata[sid])

            for react in rxn.reactants:
                r_smi = _route_molecule_smiles(react)
                # Look up any prior step producing this reactant
                prior = [ps for ps in _prod_map.get(r_smi, []) if ps < sid]
                if prior:
                    node["children"].append(build_mol_node(max(prior)))
                else:
                    node["children"].append(
                        {"type": "mol", "smiles": r_smi, "in_stock": True}
                    )

            return node

        # Build route tree and store
        route_tree = build_mol_node(final_step)
        if keep_ids:
            all_routes[int(route_id)] = route_tree
        else:
            all_routes.append(route_tree)

    return all_routes


def write_routes_json(
    routes_dict,
    file_path,
    tree: "Tree | None" = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
):
    """Serialize reaction routes to a JSON file."""
    routes_json = make_json(
        routes_dict,
        tree=tree,
        route_metadata=route_metadata,
    )
    with open(file_path, "w") as f:
        json.dump(routes_json, f, indent=2)


def write_routes_csv(routes_dict, file_path="routes.csv"):
    """Write route reactions to a CSV file.

    ``routes_dict`` is a nested ``route_id -> step_id -> reaction`` mapping. The
    output file contains ``route_id``, ``step_id``, ``smiles``, and ``meta``
    columns; ``smiles`` is written with ``format(reaction, "m")`` and ``meta``
    is left blank.
    """
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # header row
        writer.writerow(["route_id", "step_id", "smiles", "meta"])
        # sort routes and steps for deterministic output
        for route_id in sorted(routes_dict):
            steps = routes_dict[route_id]
            for step_id in sorted(steps):
                reaction = steps[step_id]
                smiles = format(reaction, "m")
                meta = ""  # or reaction.meta if you add that later
                writer.writerow([route_id, step_id, smiles, meta])


def export_tree_to_json(tree: "Tree", file_path: str, route_id=None):
    """
    Export a retrosynthetic search tree directly to a JSON file.

    Args:
        tree: synplan.mcts.tree.Tree instance.
        file_path: Output JSON path.
        route_id: If provided, export only this specific route (node id).
    """
    from synplan.routes.route_cgr import extract_reactions

    routes_dict = extract_reactions(tree, route_id)
    if routes_dict is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")
    write_routes_json(routes_dict, file_path, tree=tree)


def export_tree_to_csv(tree: "Tree", file_path: str = "routes.csv", route_id=None):
    """
    Export a retrosynthetic search tree directly to a CSV file.

    Args:
        tree: synplan.mcts.tree.Tree instance.
        file_path: Output CSV path.
        route_id: If provided, export only this specific route (node id).
    """
    from synplan.routes.route_cgr import extract_reactions

    routes_dict = extract_reactions(tree, route_id)
    if routes_dict is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")
    write_routes_csv(routes_dict, file_path)
