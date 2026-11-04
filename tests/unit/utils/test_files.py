import pytest
from chython import smiles
from chython.containers import ReactionContainer

from synplan.utils.files import (
    MoleculeReader,
    MoleculeWriter,
    ReactionReader,
    ReactionWriter,
    load_rule_index_mapping_tsv,
    parse_reaction,
    read_json,
    write_json_atomic,
)


@pytest.mark.parametrize("suffix", [".json", ".json.gz", ".JSON.GZ"])
def test_json_file_roundtrip(tmp_path, suffix):
    path = tmp_path / "results" / f"routes{suffix}"
    data = {"target": [{"label": "réaction", "score": 0.5, "rank": None}]}

    write_json_atomic(path, data)

    assert read_json(path) == data
    assert list(path.parent.iterdir()) == [path]
    if suffix.lower().endswith(".gz"):
        assert path.read_bytes().startswith(b"\x1f\x8b")


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_failed_json_write_preserves_destination(tmp_path, suffix):
    path = tmp_path / f"cache{suffix}"
    write_json_atomic(path, {"completed": True})
    original = path.read_bytes()

    with pytest.raises(ValueError, match="Out of range float"):
        write_json_atomic(path, {"score": float("nan")})

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_load_rule_index_mapping_tsv_preserves_multiple_rules_per_reaction(tmp_path):
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        "rule_smarts\tpopularity\treaction_indices\n"
        "rule-a\t3\t10,11\n"
        "rule-b\t2\t10\n"
        "rule-c\t1\t11,12\n",
        encoding="utf-8",
    )

    assert load_rule_index_mapping_tsv(rules_path) == {
        10: [0, 1],
        11: [0, 2],
        12: [2],
    }


@pytest.mark.parametrize(
    "extension,reader,writer",
    [("sdf", MoleculeReader, MoleculeWriter), ("rdf", ReactionReader, ReactionWriter)],
)
def test_mdl_reader_stereo_defaults_and_overrides(tmp_path, extension, reader, writer):
    path = tmp_path / f"stereo.{extension}"
    for text in ("C/C=C/C", "CCO"):
        molecule = smiles(text)
        structure = (
            ReactionContainer((molecule,), (molecule.copy(),))
            if extension == "rdf"
            else molecule
        )
        with writer(path) as output:
            output.write(structure)
        with reader(path) as source:
            assert str(source.read_structure()) == str(structure)
        if extension == "rdf":
            assert str(parse_reaction(path.read_text(), fmt="rdf")) == str(structure)

    # A wedge on achiral ethanol must fail instead of silently losing stereo.
    original = path.read_text()
    malformed = original.replace("M  V30 1 1 1 2\n", "M  V30 1 1 1 2 CFG=1\n", 1)
    assert malformed != original
    path.write_text(malformed)
    with reader(path) as source, pytest.raises(ValueError, match="stereo"):
        source.read_structure()
    for options in ({"strict_stereo": False}, {"ignore_stereo": True}):
        with reader(path, **options) as source:
            assert str(source.read_structure()) == str(structure)
    if extension == "rdf":
        with pytest.raises((ValueError, StopIteration)):
            parse_reaction(malformed, fmt="rdf")
        assert str(parse_reaction(malformed, fmt="rdf", ignore_stereo=True)) == str(
            structure
        )
