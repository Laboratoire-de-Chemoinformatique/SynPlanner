"""Molecule extraction and SVG depiction helpers for tree visualizations."""

from __future__ import annotations

import html
import math
from typing import Optional

from synplan.chem.utils import mol_from_smiles
from synplan.mcts.tree import Tree


def node_primary_molecule(node) -> Optional[object]:
    if getattr(node, "curr_precursor", None) and hasattr(
        node.curr_precursor, "molecule"
    ):
        return node.curr_precursor.molecule
    if getattr(node, "new_precursors", None):
        precursor = node.new_precursors[0]
        if hasattr(precursor, "molecule"):
            return precursor.molecule
    if getattr(node, "precursors_to_expand", None):
        precursor = node.precursors_to_expand[0]
        if hasattr(precursor, "molecule"):
            return precursor.molecule
    return None


def depict_molecule_svg(molecule) -> Optional[str]:
    if molecule is None:
        return None
    try:
        molecule.clean2d()
        return molecule.depict()
    except Exception:
        return None


def svg_from_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        molecule = mol_from_smiles(smiles)
    except Exception:
        return None
    return depict_molecule_svg(molecule)


def molecule_key(molecule: Optional[object]) -> Optional[str]:
    if molecule is None:
        return None
    try:
        return str(molecule)
    except Exception:
        return None


def molecule_smiles_and_svg(
    molecule: Optional[object], *, with_svg: bool
) -> tuple[Optional[str], Optional[str]]:
    if molecule is None:
        return None, None
    try:
        smiles = str(molecule)
    except Exception:
        smiles = None
    if not with_svg:
        return smiles, None
    return smiles, depict_molecule_svg(molecule)


def node_product_molecules(node) -> list[object]:
    """Return product molecules for a node, falling back to one primary molecule."""
    molecules: list[object] = []
    for precursor in getattr(node, "new_precursors", None) or ():
        mol = getattr(precursor, "molecule", None)
        if mol is not None:
            molecules.append(mol)
    if molecules:
        return molecules

    primary = node_primary_molecule(node)
    return [primary] if primary is not None else []


def curr_precursor_key(node) -> Optional[str]:
    curr = getattr(node, "curr_precursor", None)
    if curr is None:
        return None
    molecule = getattr(curr, "molecule", curr)
    key = molecule_key(molecule)
    if key:
        return key
    try:
        return str(curr)
    except Exception:
        return None


def target_molecule(tree: Tree) -> Optional[object]:
    target_node = tree.nodes.get(1)
    if target_node is None:
        return None

    candidates = []
    curr_precursor = getattr(target_node, "curr_precursor", None)
    if curr_precursor:
        candidates.append(curr_precursor)
    for attr_name in ("precursors_to_expand", "new_precursors"):
        for precursor in getattr(target_node, attr_name, None) or ():
            candidates.append(precursor)

    for candidate in candidates:
        molecule = getattr(candidate, "molecule", candidate)
        if molecule is not None:
            return molecule
    return None


def molecule_atom_coordinates(molecule: object) -> dict[int, tuple[float, float]]:
    try:
        molecule.clean2d()
    except Exception:
        pass

    plane = getattr(molecule, "_plane", None)
    if plane:
        coordinates: dict[int, tuple[float, float]] = {}
        for atom_id, coord in plane.items():
            try:
                x, y = coord
                coordinates[int(atom_id)] = (float(x), float(y))
            except Exception:
                continue
        if coordinates:
            return coordinates

    coordinates = {}
    try:
        atom_items = molecule.atoms()
    except Exception:
        return coordinates

    for atom_id, atom in atom_items:
        try:
            x = getattr(atom, "x")
            y = getattr(atom, "y")
            if callable(x):
                x = x()
            if callable(y):
                y = y()
            x_float = float(x)
            y_float = float(y)
            if math.isfinite(x_float) and math.isfinite(y_float):
                coordinates[int(atom_id)] = (x_float, y_float)
        except Exception:
            continue
    return coordinates


def ordered_bond_id(atom_a: object, atom_b: object) -> str:
    try:
        a_int = int(atom_a)
        b_int = int(atom_b)
        return f"{a_int}-{b_int}" if a_int < b_int else f"{b_int}-{a_int}"
    except Exception:
        a_str = str(atom_a)
        b_str = str(atom_b)
        return f"{a_str}-{b_str}" if a_str < b_str else f"{b_str}-{a_str}"


def target_svg_from_coordinates(molecule: object) -> str:
    coordinates = molecule_atom_coordinates(molecule)
    if not coordinates:
        return ""

    xs = [coord[0] for coord in coordinates.values()]
    ys = [coord[1] for coord in coordinates.values()]
    if not xs or not ys:
        return ""

    pad = 0.7
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad

    min_y_svg = -max_y
    max_y_svg = -min_y
    width = max(max_x - min_x, 1e-6)
    height = max(max_y_svg - min_y_svg, 1e-6)

    bond_lines: list[str] = []
    bond_offset = 0.18
    try:
        bonds = molecule.bonds()
    except Exception:
        bonds = ()

    for atom_a, atom_b, bond in bonds:
        try:
            a_id = int(atom_a)
            b_id = int(atom_b)
        except Exception:
            continue
        if a_id not in coordinates or b_id not in coordinates:
            continue

        x1, y1 = coordinates[a_id]
        x2, y2 = coordinates[b_id]
        y1 = -y1
        y2 = -y2
        dx = x2 - x1
        dy = y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        if norm == 0:
            continue

        perp_x = -dy / norm
        perp_y = dx / norm
        bond_id = ordered_bond_id(a_id, b_id)
        order = getattr(bond, "order", 1)
        try:
            order = int(order)
        except Exception:
            order = 1

        if order == 2:
            offsets = (-bond_offset, bond_offset)
            bond_class = "target-bond bond-double"
        elif order == 3:
            offsets = (-bond_offset, 0.0, bond_offset)
            bond_class = "target-bond bond-triple"
        elif order == 4:
            offsets = (0.0,)
            bond_class = "target-bond bond-aromatic"
        else:
            offsets = (0.0,)
            bond_class = "target-bond bond-single"

        for offset in offsets:
            ox = perp_x * offset
            oy = perp_y * offset
            bond_lines.append(
                f'<line class="{bond_class}" data-atom1="{a_id}" '
                f'data-atom2="{b_id}" data-bond="{bond_id}" '
                f'x1="{x1 + ox:.2f}" y1="{y1 + oy:.2f}" '
                f'x2="{x2 + ox:.2f}" y2="{y2 + oy:.2f}" />'
            )

    atom_marks: list[str] = []
    atom_radius = 0.14
    label_size = 0.5
    atom_colors = {
        "N": "#2f6fd0",
        "O": "#e14b4b",
        "S": "#d3b338",
        "F": "#2aa84a",
        "Cl": "#2aa84a",
        "Br": "#2aa84a",
        "I": "#2aa84a",
    }
    try:
        atom_items = molecule.atoms()
    except Exception:
        atom_items = ()

    for atom_id, atom in atom_items:
        try:
            atom_id_int = int(atom_id)
        except Exception:
            continue
        if atom_id_int not in coordinates:
            continue
        x, y = coordinates[atom_id_int]
        y = -y
        symbol = html.escape(str(getattr(atom, "atomic_symbol", "C")))
        fill = atom_colors.get(symbol, "#1f242a55")
        if symbol == "C":
            atom_marks.append(
                f'<circle class="target-atom" cx="{x:.2f}" cy="{y:.2f}" '
                f'r="{atom_radius:.2f}" fill="{fill}" />'
            )
        else:
            mask_radius = atom_radius * 1.9
            atom_marks.append(
                f'<circle class="target-atom-mask" cx="{x:.2f}" cy="{y:.2f}" '
                f'r="{mask_radius:.2f}" />'
            )
            atom_marks.append(
                f'<text class="target-atom-label" x="{x:.2f}" y="{y:.2f}" '
                f'font-size="{label_size:.2f}" fill="{fill}" '
                f'text-anchor="middle" dominant-baseline="central">{symbol}</text>'
            )

    return f"""
    <svg id="target-svg" viewBox="{min_x:.2f} {min_y_svg:.2f} {width:.2f} {height:.2f}" preserveAspectRatio="xMidYMid meet">
      <g class="target-bonds">
        {"".join(bond_lines)}
      </g>
      <g class="target-atoms">
        {"".join(atom_marks)}
      </g>
    </svg>
    """


def build_target_svg(tree: Tree) -> str:
    molecule = target_molecule(tree)
    if molecule is None:
        return ""

    target_svg = target_svg_from_coordinates(molecule)
    if target_svg:
        return target_svg

    return depict_molecule_svg(molecule) or ""
