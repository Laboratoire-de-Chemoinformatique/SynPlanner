"""Rigid 2D alignment of route precursors onto the product they came from.

Kabsch/Procrustes on the shared atom-map numbers, closed form for 2x2. Reflection is
allowed: chython derives wedges from coordinates at render time, so mirroring is
stereo-safe provided the ``_wedge_map`` cache is dropped.
"""

from __future__ import annotations

from math import atan2, cos, degrees, hypot, sin
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chython.containers.molecule import MoleculeContainer

__all__ = ["align_molecule", "align_route", "apply_transform"]


def _centroid(points):
    n = len(points)
    return sum(x for x, _ in points) / n, sum(y for _, y in points) / n


def _rmsd(a, b):
    return (
        sum(hypot(ax - bx, ay - by) ** 2 for (ax, ay), (bx, by) in zip(a, b)) / len(a)
    ) ** 0.5


def _mean_dist(a, b):
    return sum(hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(a, b)) / len(a)


def _fit(src, dst, reflect: bool):
    """Best rigid map src -> dst. Returns (matrix, src centre, dst centre, angle, rmsd)."""

    cx, cy = _centroid(src)
    dx, dy = _centroid(dst)
    a = [(x - cx, -(y - cy) if reflect else y - cy) for x, y in src]
    b = [(x - dx, y - dy) for x, y in dst]

    angle = atan2(
        sum(ax * by - ay * bx for (ax, ay), (bx, by) in zip(a, b)),
        sum(ax * bx + ay * by for (ax, ay), (bx, by) in zip(a, b)),
    )
    c, s = cos(angle), sin(angle)
    flip = -1 if reflect else 1
    matrix = (c, -s * flip, s, c * flip)  # rotation with the mirror folded in
    rmsd = _rmsd([(c * ax - s * ay, s * ax + c * ay) for ax, ay in a], b)
    return matrix, (cx, cy), (dx, dy), angle, rmsd


def apply_transform(
    mol: MoleculeContainer,
    matrix: tuple[float, float, float, float],
    src_centre: tuple[float, float],
    dst_centre: tuple[float, float],
) -> None:
    """Map every atom of `mol` through `matrix` taken about `src_centre`."""

    a, b, c, d = matrix
    cx, cy = src_centre
    dx, dy = dst_centre
    for _, atom in mol.atoms():
        x, y = atom.x - cx, atom.y - cy
        atom.xy = (a * x + b * y + dx, c * x + d * y + dy)
    # wedges are derived from coordinates and cached; a mirror silently inverts every
    # stereocentre unless the cache is dropped
    mol.__dict__.pop("_wedge_map", None)
    mol.__dict__.pop("__cached_method__repr_svg_", None)


def align_molecule(
    mol: MoleculeContainer, ref: MoleculeContainer, allow_reflection: bool = True
) -> dict[str, Any]:
    """Rotate/reflect `mol` in place so its shared atoms sit as they do in `ref`.

    Shared atoms are those carrying the same atom-map number in both. `mode` is
    ``rigid``, ``underdetermined`` (fewer than two usable shared atoms, layout left
    alone) or ``skipped`` (nothing shared).
    """

    shared = sorted(set(mol) & set(ref))
    stats = {
        "n_shared": len(shared),
        "mode": "skipped",
        "reflected": False,
        "angle_deg": 0.0,
        "before": None,
        "after": None,
    }
    if not shared:
        return stats

    src = [(mol.atom(n).x, mol.atom(n).y) for n in shared]
    dst = [(ref.atom(n).x, ref.atom(n).y) for n in shared]

    # translation is free (the renderer re-origins every molecule anyway), so measure
    # the mismatch that only a rotation or a mirror can remove
    cx, cy = _centroid(src)
    dx, dy = _centroid(dst)
    stats["before"] = stats["after"] = _mean_dist(
        [(x - cx, y - cy) for x, y in src], [(x - dx, y - dy) for x, y in dst]
    )

    if len(shared) < 2 or max(hypot(x - cx, y - cy) for x, y in src) < 1e-6:
        stats["mode"] = "underdetermined"
        return stats

    best = _fit(src, dst, False)
    if allow_reflection:
        mirrored = _fit(src, dst, True)
        if mirrored[4] < best[4] - 1e-9:
            best, stats["reflected"] = mirrored, True
    matrix, src_centre, dst_centre, angle, _ = best

    apply_transform(mol, matrix, src_centre, dst_centre)
    stats["mode"] = "rigid"
    stats["angle_deg"] = degrees(angle)
    stats["after"] = _mean_dist([(mol.atom(n).x, mol.atom(n).y) for n in shared], dst)
    return stats


def align_route(
    steps: Sequence[Any], allow_reflection: bool = True
) -> list[dict[str, Any]]:
    """Align every precursor of a `Tree.synthesis_route` onto its product.

    `steps` comes back leaf-first, so walk it in reverse: the target keeps its layout
    and each disconnection inherits from the molecule above it.
    """

    report = []
    for reaction in reversed(steps):
        product = reaction.products[0]
        for precursor in reaction.reactants:
            stats = align_molecule(precursor, product, allow_reflection)
            stats["product"] = str(product)
            stats["precursor"] = str(precursor)
            report.append(stats)
    return report
