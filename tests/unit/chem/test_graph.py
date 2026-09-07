"""Native storage must survive SynPlanner's atom ordering and canonicalization."""

import pytest
from chython import smiles, synthon_smiles
from chython._engine import native_graph

from synplan.chem.graph import bond_items, neighbors, ordered_copy
from synplan.chem.stereo import assert_stereo_preserved
from synplan.chem.utils import in_atom_order, safe_canonicalization


@pytest.mark.parametrize(
    "text",
    [
        "[NH2:40][C@@H:7]([CH3:23])[C:11](=[O:9])[OH:31]",
        "[CH3:40]/[CH:7]=[CH:23]/[CH2:11][OH:9]",
        "[CH3:40][CH:7]=[C@:23]=[CH:11][CH3:9]",
        "[CH3:40][O:7][c:23]1[cH:11][cH:9][n:31][cH:8][cH:2]1",
    ],
)
def test_ordered_molecules_keep_native_storage_and_stereo(text):
    molecule = smiles(text, remap=False)
    molecule.meta["source"] = "ordering regression"
    molecule.name = "mapped molecule"
    coordinates = {}
    for n, atom in molecule.atoms():
        atom.xy = (n / 10, -n / 10)
        coordinates[n] = tuple(atom.xy)
    order = tuple(molecule)
    environments = {n: tuple(neighbors(molecule, n)) for n in molecule}
    result = ordered_copy(molecule)
    assert native_graph(result) is not None
    assert tuple(result) == tuple(sorted(molecule))
    assert {n: tuple(neighbors(result, n)) for n in result} == environments
    assert {n: tuple(a.xy) for n, a in result.atoms()} == coordinates
    assert result.name == molecule.name
    assert result.meta == molecule.meta and result.meta is not molecule.meta
    assert_stereo_preserved(molecule, result)
    ordered = in_atom_order(molecule)
    assert native_graph(ordered) is not None
    assert all(
        tuple(neighbors(ordered, n)) == tuple(sorted(environments[n])) for n in ordered
    )
    assert_stereo_preserved(molecule, ordered)
    canonical = safe_canonicalization(molecule)
    assert native_graph(canonical) is not None
    assert_stereo_preserved(molecule, canonical)
    assert tuple(molecule) == order
    assert {n: tuple(neighbors(molecule, n)) for n in molecule} == environments


def test_ordered_synthon_keeps_attachment_labels():
    molecule = synthon_smiles("[NH_nuc:9][CH3:2]", remap=False)
    result = ordered_copy(molecule)
    assert native_graph(result) is not None
    assert result.atom(9).label == molecule.atom(9).label == "nuc"
    assert result.atom(9).implicit_hydrogens == molecule.atom(9).implicit_hydrogens
    assert str(result) == str(molecule)


def test_reordered_uspto_polycycle_refreshes_ring_membership():
    # Ligand from the frozen large USPTO reaction cohort (reaction 58).
    molecule = smiles(
        "[CH:128]1=[CH:127][C:126](=[CH:131][CH:130]=[CH:129]1)[P:125]"
        "([C:132]=2[CH:133]=[CH:134][CH:135]=[CH:136][CH:137]=2)"
        "[C:124]3=[C:123]4[CH2:139][CH2:140][C:141]5=[CH:159][C:145]"
        "(=[C:144]([CH2:160][CH2:119][C:120](=[CH:138]3)[CH:121]=[CH:122]4)"
        "[CH:143]=[CH:142]5)[P:146]([C:153]=6[CH:154]=[CH:155][CH:156]"
        "=[CH:157][CH:158]=6)[C:147]=7[CH:148]=[CH:149][CH:150]"
        "=[CH:151][CH:152]=7",
        remap=False,
    )
    result = in_atom_order(molecule)
    assert native_graph(result) is not None
    for n, atom in result.atoms():
        assert atom.ring_sizes == result.atoms_rings_sizes.get(n, set())


def test_cgr_neighbors_include_both_states_and_keep_shared_bonds():
    cgr = ~smiles("[CH3:9][OH:2].[Cl-:5]>>[CH3:9][Cl:5].[OH-:2]", remap=False)
    backend = native_graph(cgr)
    assert backend is not None
    assert set(neighbors(cgr, 9)) == {2, 5}
    assert {(n, b.order, b.p_order) for n, b in bond_items(cgr, 9)} == {
        (2, 1, None),
        (5, None, 1),
    }
    assert cgr.bond(9, 5) is cgr.bond(5, 9)
    assert native_graph(cgr) is backend
