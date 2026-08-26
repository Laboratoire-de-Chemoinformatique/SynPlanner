from math import cos, hypot, sin

from chython import smiles as read_smiles

from synplan.chem.reaction.reactor import Reaction
from synplan.utils.align2d import align_molecule, align_route, apply_transform


def _scramble(mol, angle, reflect, shift):
    """Move `mol` somewhere else on the plane, the way clean2d would."""
    c, s = cos(angle), sin(angle)
    flip = -1 if reflect else 1
    apply_transform(mol, (c, -s * flip, s, c * flip), (0.0, 0.0), shift)


def _mean_dist(mol, ref):
    shared = sorted(set(mol) & set(ref))
    a = [(mol.atom(n).x, mol.atom(n).y) for n in shared]
    b = [(ref.atom(n).x, ref.atom(n).y) for n in shared]
    ca = (sum(x for x, _ in a) / len(a), sum(y for _, y in a) / len(a))
    cb = (sum(x for x, _ in b) / len(b), sum(y for _, y in b) / len(b))
    return sum(
        hypot(x - ca[0] - u + cb[0], y - ca[1] - v + cb[1])
        for (x, y), (u, v) in zip(a, b)
    ) / len(a)


def _route():
    """A two-step route whose precursors carry the target's coordinates.

    Substructures keep the parent's atom numbers and coordinates, which is exactly
    what chython's reactor hands back for a real disconnection.
    """
    target = read_smiles("CC(=O)Nc1ccccc1CCO")
    target.clean2d()

    acid = target.substructure([1, 2, 3])
    amine = target.substructure([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    aniline = amine.substructure([4, 5, 6, 7, 8, 9, 10])
    alcohol = amine.substructure([11, 12, 13])

    _scramble(acid, 0.7, False, (12.0, -4.0))
    _scramble(amine, 2.4, True, (-6.0, 9.0))
    _scramble(aniline, 1.1, True, (3.0, 3.0))
    _scramble(alcohol, -2.0, False, (-1.0, 7.0))

    # synthesis_route order: leaf first
    return target, [
        Reaction([aniline, alcohol], [amine]),
        Reaction([acid, amine], [target]),
    ]


def test_align_route_puts_every_precursor_back_in_the_target_frame():
    target, steps = _route()
    precursors = [m for step in steps for m in step.reactants]

    before = [_mean_dist(m, target) for m in precursors]
    assert min(before) > 0.3, before

    report = align_route(steps)

    assert [s["mode"] for s in report] == ["rigid"] * 4
    assert all(s["after"] < 1e-6 < s["before"] for s in report), report
    # the cascade must run target first, or the deeper precursors inherit a frame
    # that is itself still scrambled
    assert all(_mean_dist(m, target) < 1e-6 for m in precursors)


def test_align_molecule_leaves_a_single_shared_atom_alone():
    ref = read_smiles("CCO")
    ref.clean2d()
    mol = read_smiles("CCO")
    mol.clean2d()
    mol._atoms = {1: mol._atoms[1]}
    mol._bonds = {1: {}}
    before = (mol.atom(1).x, mol.atom(1).y)

    stats = align_molecule(mol, ref)

    assert stats["mode"] == "underdetermined"
    assert (mol.atom(1).x, mol.atom(1).y) == before


def test_mirroring_redraws_the_wedges_instead_of_inverting_the_centres():
    """Chython caches wedges derived from coordinates; a mirror must drop that cache."""
    mol = read_smiles("CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O")
    mol.clean2d()
    before = {(n, m): s for n, m, s in mol._wedge_map}  # warms the cache
    assert before

    apply_transform(mol, (1.0, 0.0, 0.0, -1.0), (0.0, 0.0), (0.0, 0.0))

    after = {(n, m): s for n, m, s in mol._wedge_map}
    assert set(after) == set(before)
    assert all(after[k] == -s for k, s in before.items()), (before, after)
