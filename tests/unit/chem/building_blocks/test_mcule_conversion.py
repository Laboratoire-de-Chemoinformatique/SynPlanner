"""Mcule material labels become linked groups without inventing configurations."""

import json

import pytest
from chython import inchi_key, smiles

from synplan.chem.building_blocks import mcule_to_cxsmiles
from synplan.chem.building_blocks.io import _prepare_catalogue_batch


@pytest.mark.parametrize(
    "identifier,source,kind,group",
    [
        ("MCULE-9522123708", "[C@@H]1(C(OC)=O)C[C@@H](CC1)C(=O)O", "REL", -1),
        ("MCULE-8361652176", "CCC[C@H]1[C@@H](C1)C(=O)O", "RAC", 1),
    ],
)
def test_real_mcule_declarations_survive_preparation(identifier, source, kind, group):
    text, normalized = mcule_to_cxsmiles(source, kind)
    assert mcule_to_cxsmiles(text, normalized) == (text, normalized)
    original = {"vendor": "MC", "id": identifier, "smiles": source, "stereo_type": kind}
    records, errors = _prepare_catalogue_batch(
        [
            (
                2,
                {
                    "SMILES": text,
                    "sources": json.dumps([original]),
                    "stereo_type": normalized,
                },
            )
        ],
        smiles_column="SMILES",
        price_columns=[],
    )
    assert not errors
    key, record = records[0]
    restored = smiles(record["smiles"], strict_stereo=True)
    assert inchi_key(restored) == key == inchi_key(smiles(source))
    assert {a.extended_stereo for _, a in restored.atoms() if a.stereo is not None} == {
        group
    }
    assert record["sources"] == [{k: v for k, v in original.items() if k != "smiles"}]
    assert record["stereo_type"] == normalized


@pytest.mark.parametrize("kind", ["REL", "RAC"])
def test_conversion_preserves_ez_and_does_not_assign_unmarked_centres(kind):
    source = "C[C@H](O)[C@H](F)/C=C/C(Cl)Br"
    text, _ = mcule_to_cxsmiles(source, kind)
    restored = smiles(text, strict_stereo=True)
    assert inchi_key(restored) == inchi_key(smiles(source))
    assert sum(a.stereo is not None for _, a in restored.atoms()) == 2
    assert sum(b.stereo is not None for *_, b in restored.bonds()) == 1


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("ABS", "absolute"),
        ("UNK", "unknown"),
        ("unknown (not confirmed)", "unknown"),
        ("", "unknown"),
    ],
)
def test_unknown_material_type_is_not_rewritten_as_or_or_and(kind, expected):
    source = "C[C@H](O)C(=O)O"
    assert mcule_to_cxsmiles(source, kind) == (str(smiles(source)), expected)
    assert mcule_to_cxsmiles("CCO", "") == (str(smiles("CCO")), "")


@pytest.mark.parametrize(
    "source,kind",
    [
        ("C[C@H](O)C(=O)O", "typo"),
        ("C[C@H](O)C(=O)O |r|", "REL"),
        ("C[C@H](O)C(=O)O |o1:1|", "RAC"),
        ("C[C@H](O)C(=O)O |&1:1|", "ABS"),
        ("C[C@H](O)[C@H](F)Cl |o1:1,o2:3|", "REL"),
        ("CCO>>CC=O", "ABS"),
    ],
)
def test_conflicting_declarations_are_not_silently_overwritten(source, kind):
    with pytest.raises(ValueError):
        mcule_to_cxsmiles(source, kind)


def test_conversion_preserves_existing_non_stereo_cx_annotations():
    source = "[CH2]C[C@H](F)Cl"
    text, _ = mcule_to_cxsmiles(source, "REL")
    restored = smiles(text, strict_stereo=True)
    assert sum(a.is_radical for _, a in restored.atoms()) == 1
    assert inchi_key(restored) == inchi_key(smiles(source))
