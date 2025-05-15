import csv
import json

from CGRtools import smiles as read_smiles

def make_dict(routes_json):
    """
    routes_json : list of tree-dicts as produced by make_json()
    
    Returns a dict mapping each route index (0, 1, …) to a sub-dict
    of {new_step_id: ReactionContainer}, where the step IDs run
    from the earliest reaction (0) up to the final (max).
    """
    routes_dict = {}
    if isinstance(routes_json, dict):
        for route_idx, tree in routes_json.items():
            rxn_list = []
            
            def _postorder(node):
                # first dive into any children, then record this reaction
                for child in node.get('children', []):
                    _postorder(child)
                if node['type'] == 'reaction':
                    rxn_list.append(read_smiles(node['smiles']))
                # mol-nodes simply recurse (no record)
            
            # collect all reactions in leaf→root order
            _postorder(tree)
            
            # now assign 0,1,2,… in that order
            reactions = {i: rxn for i, rxn in enumerate(rxn_list)}
            routes_dict[route_idx] = reactions

        return routes_dict
    else:
        for route_idx, tree in enumerate(routes_json):
            rxn_list = []
            
            def _postorder(node):
                # first dive into any children, then record this reaction
                for child in node.get('children', []):
                    _postorder(child)
                if node['type'] == 'reaction':
                    rxn_list.append(read_smiles(node['smiles']))
                # mol-nodes simply recurse (no record)
            
            # collect all reactions in leaf→root order
            _postorder(tree)
            
            # now assign 0,1,2,… in that order
            reactions = {i: rxn for i, rxn in enumerate(rxn_list)}
            routes_dict[route_idx] = reactions

        return routes_dict

def read_routes_json(file_path='routes.csv'):
    with open(file_path, 'r') as file:
        routes_json = json.load(file)
    return routes_json

def read_routes_csv(file_path='routes.csv'):
    """
    Read a CSV with columns: route_id, step_id, smiles, meta
    and return a nested dict mapping
        route_id (int) -> step_id (int) -> ReactionContainer
    (ignoring meta for now, but you could extract it if needed).
    """
    reaction_dict = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            route_id = int(row['route_id'])
            step_id = int(row['step_id'])
            smiles = row['smiles']
            # adjust this constructor to your actual API
            reaction = read_smiles(smiles)  
            reaction_dict.setdefault(route_id, {})[step_id] = reaction
    return reaction_dict


def make_json(routes_dict, keep_ids=True):
    """
    Convert routes into a nested JSON tree of reaction and molecule nodes.

    Args:
        routes_dict (dict[int, dict[int, Reaction]]): Mapping route IDs to steps (step_id -> Reaction).
        keep_ids (bool): If True, returns a list of route trees; otherwise returns a dict mapping route IDs to trees.

    Returns:
        list or dict: JSON-like tree(s) of routes.
    """
    # Prepare output
    all_routes = {} if keep_ids else []

    for route_id, steps in routes_dict.items():
        if not steps:
            continue

        # Determine target molecule atoms from the final step of this route
        final_step = max(steps)
        target = steps[final_step].products[0]
        atom_nums = set(target._atoms.keys())

        # Precompute canonical SMILES and producer mapping for all products
        prod_map = {}  # smiles -> list of step_ids
        for sid, rxn in steps.items():
            for prod in rxn.products:
                prod.kekule()
                prod.implicify_hydrogens()
                prod.thiele()
                s = str(prod)
                prod_map.setdefault(s, []).append(sid)

        def transform(mol):
            mol.kekule(); mol.implicify_hydrogens(); mol.thiele()
            return str(mol)

        def build_mol_node(sid):
            """Find the product with any overlap to target atoms and recurse into its reaction."""
            rxn = steps[sid]
            for p in rxn.products:
                if atom_nums & set(p._atoms.keys()):
                    smiles = str(p)
                    return {
                        "type": "mol",
                        "smiles": smiles,
                        "children": [build_reaction_node(sid)],
                        "in_stock": False
                    }
            # Shouldn't reach here if tree is consistent
            return None

        def build_reaction_node(sid):
            """Build reaction node and recurse into reactant molecule nodes."""
            rxn = steps[sid]
            node = {
                "type": "reaction",
                "smiles": format(rxn, 'm'),
                "children": []
            }

            for react in rxn.reactants:
                r_smi = transform(react)
                # Look up any prior step producing this reactant
                prior = [ps for ps in prod_map.get(r_smi, []) if ps < sid]
                if prior:
                    node["children"].append(build_mol_node(max(prior)))
                else:
                    node["children"].append({"type": "mol", "smiles": r_smi, "in_stock": True})

            return node

        # Build route tree and store
        tree = build_mol_node(final_step)
        if keep_ids:
            all_routes[route_id] = tree
        else:
            all_routes.append(tree)

    return all_routes

def write_routes_json(routes_dict, file_path):
    """Serialize reaction routes to a JSON file."""
    routes_json = make_json(routes_dict)
    with open(file_path, 'w') as f:
        json.dump(routes_json, f, indent=2)

def write_routes_csv(routes_dict, file_path='routes.csv'):
    """
    Write out a nested routes_dict of the form
        { route_id: { step_id: reaction_obj, ... }, ... }
    to a CSV with columns: route_id, step_id, smiles, meta
    where smiles is format(reaction, 'm') and meta is left blank.
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # header row
        writer.writerow(['route_id', 'step_id', 'smiles', 'meta'])
        # sort routes and steps for deterministic output
        for route_id in sorted(routes_dict):
            steps = routes_dict[route_id]
            for step_id in sorted(steps):
                reaction = steps[step_id]
                smiles = format(reaction, 'm')
                meta = ''   # or reaction.meta if you add that later
                writer.writerow([route_id, step_id, smiles, meta])