"""Molecule extraction and SVG depiction helpers for tree visualizations."""

from __future__ import annotations

import html
import math
from contextlib import suppress

from synplan.mcts.tree import Tree


def node_primary_molecule(node) -> object:
    if node.curr_precursor:
        return node.curr_precursor.molecule
    if node.new_precursors:
        return node.new_precursors[0].molecule
    return node.precursors_to_expand[0].molecule


def molecule_smiles_and_svg(
    molecule: object, *, with_svg: bool
) -> tuple[str, str | None]:
    smiles = str(molecule)
    if not with_svg:
        return smiles, None

    with suppress(Exception):
        molecule.clean2d()
    return smiles, molecule.depict()


def node_product_molecules(node) -> list[object]:
    """Return product molecules for a node, falling back to one primary molecule."""
    if node.new_precursors:
        return [precursor.molecule for precursor in node.new_precursors]

    return [node_primary_molecule(node)]


def curr_precursor_key(node) -> str | None:
    return str(node.curr_precursor.molecule) if node.curr_precursor else None


def target_molecule(tree: Tree) -> object:
    return tree.nodes[1].curr_precursor.molecule


def molecule_atom_coordinates(molecule: object) -> dict[int, tuple[float, float]]:
    with suppress(Exception):
        molecule.clean2d()

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
            x = atom.x
            y = atom.y
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


def bond_order(bond: object) -> int:
    try:
        return int(getattr(bond, "order", 1))
    except Exception:
        return 1


def aromatic_component_centers(
    bonds: tuple[tuple[object, object, object], ...],
    coordinates: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    aromatic_neighbors: dict[int, set[int]] = {}
    for atom_a, atom_b, bond in bonds:
        try:
            a_id = int(atom_a)
            b_id = int(atom_b)
        except Exception:
            continue
        if bond_order(bond) != 4 or a_id not in coordinates or b_id not in coordinates:
            continue
        aromatic_neighbors.setdefault(a_id, set()).add(b_id)
        aromatic_neighbors.setdefault(b_id, set()).add(a_id)

    centers: dict[int, tuple[float, float]] = {}
    visited: set[int] = set()
    for start_atom in aromatic_neighbors:
        if start_atom in visited:
            continue

        stack = [start_atom]
        component: list[int] = []
        while stack:
            atom_id = stack.pop()
            if atom_id in visited:
                continue
            visited.add(atom_id)
            component.append(atom_id)
            stack.extend(aromatic_neighbors.get(atom_id, set()) - visited)

        if not component:
            continue
        x_center = sum(coordinates[atom_id][0] for atom_id in component) / len(
            component
        )
        y_center = -sum(coordinates[atom_id][1] for atom_id in component) / len(
            component
        )
        for atom_id in component:
            centers[atom_id] = (x_center, y_center)

    return centers


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
        bonds = tuple(molecule.bonds())
    except Exception:
        bonds = ()

    aromatic_centers = aromatic_component_centers(bonds, coordinates)

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
        unit_x = dx / norm
        unit_y = dy / norm
        bond_id = ordered_bond_id(a_id, b_id)
        order = bond_order(bond)

        def append_line(
            bond_class: str,
            start_x: float,
            start_y: float,
            end_x: float,
            end_y: float,
            *,
            atom_a_id: int = a_id,
            atom_b_id: int = b_id,
            current_bond_id: str = bond_id,
        ) -> None:
            bond_lines.append(
                f'<line class="{bond_class}" data-atom1="{atom_a_id}" '
                f'data-atom2="{atom_b_id}" data-bond="{current_bond_id}" '
                f'x1="{start_x:.2f}" y1="{start_y:.2f}" '
                f'x2="{end_x:.2f}" y2="{end_y:.2f}" />'
            )

        if order == 2:
            offsets = (-bond_offset, bond_offset)
            bond_class = "target-bond bond-double"
        elif order == 3:
            offsets = (-bond_offset, 0.0, bond_offset)
            bond_class = "target-bond bond-triple"
        elif order == 4:
            append_line("target-bond bond-aromatic bond-aromatic-outer", x1, y1, x2, y2)
            center = aromatic_centers.get(a_id)
            if center is not None:
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0
                if (center[0] - mid_x) * perp_x + (center[1] - mid_y) * perp_y < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y
            trim = min(0.22, norm * 0.2)
            inner_offset = bond_offset * 1.15
            append_line(
                "target-bond bond-aromatic bond-aromatic-inner",
                x1 + unit_x * trim + perp_x * inner_offset,
                y1 + unit_y * trim + perp_y * inner_offset,
                x2 - unit_x * trim + perp_x * inner_offset,
                y2 - unit_y * trim + perp_y * inner_offset,
            )
            continue
        else:
            offsets = (0.0,)
            bond_class = "target-bond bond-single"

        for offset in offsets:
            ox = perp_x * offset
            oy = perp_y * offset
            append_line(bond_class, x1 + ox, y1 + oy, x2 + ox, y2 + oy)

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
    return target_svg_from_coordinates(molecule) or molecule.depict()
