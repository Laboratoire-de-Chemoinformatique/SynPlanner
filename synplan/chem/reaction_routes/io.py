import csv
import json
import pickle
import os

from CGRtools import smiles as read_smiles
from synplan.mcts.tree import Tree
from synplan.chem.reaction_routes.route_cgr import extract_reactions


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


def read_routes_json(file_path="routes.csv", to_dict=False):
    with open(file_path, "r") as file:
        routes_json = json.load(file)
    if to_dict:
        return make_dict(routes_json)
    return routes_json


def read_routes_csv(file_path="routes.csv"):
    """
    Read a CSV with columns: route_id, step_id, smiles, meta
    and return a nested dict mapping
        route_id (int) -> step_id (int) -> ReactionContainer
    (ignoring meta for now, but you could extract it if needed).
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
        try:
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
        except Exception as e:
            print(f"Error processing route {route_id}: {e}")
            continue

        def transform(mol):
            mol.kekule()
            mol.implicify_hydrogens()
            mol.thiele()
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
                        "in_stock": False,
                    }
            # Shouldn't reach here if tree is consistent
            return None

        def build_reaction_node(sid):
            """Build reaction node and recurse into reactant molecule nodes."""
            rxn = steps[sid]
            node = {"type": "reaction", "smiles": format(rxn, "m"), "children": []}

            for react in rxn.reactants:
                r_smi = transform(react)
                # Look up any prior step producing this reactant
                prior = [ps for ps in prod_map.get(r_smi, []) if ps < sid]
                if prior:
                    node["children"].append(build_mol_node(max(prior)))
                else:
                    node["children"].append(
                        {"type": "mol", "smiles": r_smi, "in_stock": True}
                    )

            return node

        # Build route tree and store
        tree = build_mol_node(final_step)
        if keep_ids:
            all_routes[int(route_id)] = tree
        else:
            all_routes.append(tree)

    return all_routes


def write_routes_json(routes_dict, file_path):
    """Serialize reaction routes to a JSON file."""
    routes_json = make_json(routes_dict)
    with open(file_path, "w") as f:
        json.dump(routes_json, f, indent=2)


def write_routes_csv(routes_dict, file_path="routes.csv"):
    """
    Write out a nested routes_dict of the form
        { route_id: { step_id: reaction_obj, ... }, ... }
    to a CSV with columns: route_id, step_id, smiles, meta
    where smiles is format(reaction, 'm') and meta is left blank.
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


def export_tree_to_json(tree: Tree, file_path: str, route_id=None):
    """
    Export a retrosynthetic search tree directly to a JSON file.

    Args:
        tree: synplan.mcts.tree.Tree instance.
        file_path: Output JSON path.
        route_id: If provided, export only this specific route (node id).
    """
    routes_dict = extract_reactions(tree, route_id)
    if routes_dict is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")
    write_routes_json(routes_dict, file_path)


def export_tree_to_csv(tree: Tree, file_path: str = "routes.csv", route_id=None):
    """
    Export a retrosynthetic search tree directly to a CSV file.

    Args:
        tree: synplan.mcts.tree.Tree instance.
        file_path: Output CSV path.
        route_id: If provided, export only this specific route (node id).
    """
    routes_dict = extract_reactions(tree, route_id)
    if routes_dict is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")
    write_routes_csv(routes_dict, file_path)


class TreeWrapper:

    def __init__(self, tree, mol_id=1, config=1, path="planning_results/forest"):
        """Initializes the TreeWrapper."""
        self.tree = tree
        self.mol_id = mol_id
        self.config = config
        self.path = path
        # Ensure the directory exists before creating the filename
        os.makedirs(self.path, exist_ok=True)
        self.filename = os.path.join(self.path, f"tree_{mol_id}_{config}.pkl")

    def __getstate__(self):
        state = self.__dict__.copy()
        tree_state = self.tree.__dict__.copy()
        # Reset or remove non-pickleable attributes (e.g., _tqdm, policy_network, value_network)
        if "_tqdm" in tree_state:
            tree_state["_tqdm"] = True  # Reset to a simple flag
        for attr in ["policy_network", "value_network"]:
            if attr in tree_state:
                tree_state[attr] = None
        state["tree_state"] = tree_state
        del state["tree"]
        return state

    def __setstate__(self, state):
        tree_state = state.pop("tree_state")
        self.__dict__.update(state)
        new_tree = Tree.__new__(Tree)
        new_tree.__dict__.update(tree_state)
        self.tree = new_tree

    def save_tree(self):
        """Saves the TreeWrapper instance (including the tree state) to a file."""
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(self, f)
            print(
                f"Tree wrapper for mol_id '{self.mol_id}', config '{self.config}' saved to '{self.filename}'."
            )
        except Exception as e:
            print(f"Error saving tree to {self.filename}: {e}")

    @classmethod
    def load_tree_from_id(cls, mol_id, config=1, path="planning_results/forest"):
        """
        Loads a Tree object from a saved file using mol_id and config.

        Args:
            mol_id: The molecule ID used for saving.
            config: The configuration used for saving.
            path: The directory where the file is located

        Returns:
            The loaded Tree object, or None if loading fails.
        """
        filename = os.path.join(path, f"tree_{mol_id}_{config}.pkl")
        print(f"Attempting to load tree from: {filename}")
        try:
            # Ensure the 'Tree' class is defined in the current scope
            if "Tree" not in globals() and "Tree" not in locals():
                raise NameError(
                    "The 'Tree' class definition is required to load the object."
                )

            with open(filename, "rb") as f:
                loaded_wrapper = pickle.load(f)  # This implicitly calls __setstate__

            print(
                f"Tree object for mol_id '{mol_id}', config '{config}' successfully loaded from '{filename}'."
            )
            # The __setstate__ method already reconstructed the tree inside the wrapper
            return loaded_wrapper.tree

        except FileNotFoundError:
            print(f"Error: File not found at {filename}")
            return None
        except (pickle.UnpicklingError, EOFError) as e:
            print(
                f"Error: Could not unpickle file {filename}. It might be corrupted or empty. Details: {e}"
            )
            return None
        except NameError as e:
            print(f"Error during loading: {e}. Ensure 'Tree' class is defined.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred loading tree from {filename}: {e}")
            return None
