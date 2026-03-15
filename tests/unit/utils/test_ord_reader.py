"""Tests for ORD .pb file reading (synplan/utils/ord/reader.py)."""

import pytest

from chython.containers import ReactionContainer

from synplan.utils.ord import dataset_pb2, reaction_pb2
from synplan.utils.ord.reader import (
    _get_name,
    _get_smiles,
    _get_yield,
    _reaction_to_smiles,
    convert_ord_to_smiles,
    count_ord_reactions,
    iter_ord_reactions,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal protobuf messages
# ---------------------------------------------------------------------------


def _make_compound(smiles=None, name=None, role=1):
    """Build a Compound protobuf message."""
    rxn_pb = reaction_pb2
    comp = rxn_pb.Compound()
    comp.reaction_role = role
    if smiles:
        ident = comp.identifiers.add()
        ident.type = 2  # SMILES
        ident.value = smiles
    if name:
        ident = comp.identifiers.add()
        ident.type = 6  # NAME
        ident.value = name
    return comp


def _make_dataset(reactions_data):
    """Build a Dataset with reactions from a list of dicts.

    Each dict: {"reactants": [smi, ...], "products": [smi, ...],
                "reagents": [(smi, role), ...], "reaction_id": str,
                "yields": [float, ...]}
    """
    ds = dataset_pb2.Dataset()
    ds.name = "test_dataset"

    for rdata in reactions_data:
        rxn = ds.reactions.add()
        rxn.reaction_id = rdata.get("reaction_id", "test-rxn")

        # Add reactants
        ri = rxn.inputs["reactants"]
        for smi in rdata.get("reactants", []):
            ri.components.append(_make_compound(smiles=smi, role=1))

        # Add reagents/solvents/catalysts
        for smi, role in rdata.get("reagents", []):
            ri2 = rxn.inputs["reagents"]
            ri2.components.append(_make_compound(smiles=smi, role=role))

        # Add products
        outcome = rxn.outcomes.add()
        yields = rdata.get("yields", [])
        for i, smi in enumerate(rdata.get("products", [])):
            prod = outcome.products.add()
            ident = prod.identifiers.add()
            ident.type = 2
            ident.value = smi
            if i < len(yields):
                m = prod.measurements.add()
                m.type = 3  # YIELD
                m.percentage.value = yields[i]

        return ds


def _write_dataset(tmp_path, reactions_data, filename="test.pb"):
    ds = _make_dataset(reactions_data)
    pb_file = tmp_path / filename
    pb_file.write_bytes(ds.SerializeToString())
    return pb_file


# ---------------------------------------------------------------------------
# Unit tests for extraction helpers
# ---------------------------------------------------------------------------


class TestGetSmiles:
    def test_returns_smiles_value(self):
        comp = _make_compound(smiles="CCO", name="ethanol")
        assert _get_smiles(comp.identifiers) == "CCO"

    def test_returns_none_when_no_smiles(self):
        comp = _make_compound(name="water")
        assert _get_smiles(comp.identifiers) is None

    def test_skips_empty_smiles(self):
        rxn_pb = reaction_pb2
        comp = rxn_pb.Compound()
        ident = comp.identifiers.add()
        ident.type = 2  # SMILES
        ident.value = ""
        assert _get_smiles(comp.identifiers) is None


class TestGetName:
    def test_returns_name(self):
        comp = _make_compound(smiles="CCO", name="ethanol")
        assert _get_name(comp.identifiers) == "ethanol"

    def test_returns_none_when_no_name(self):
        comp = _make_compound(smiles="CCO")
        assert _get_name(comp.identifiers) is None


class TestGetYield:
    def test_returns_yield_percentage(self):
        rxn_pb = reaction_pb2
        prod = rxn_pb.ProductCompound()
        m = prod.measurements.add()
        m.type = 3  # YIELD
        m.percentage.value = 85.2
        assert _get_yield(prod.measurements) == pytest.approx(85.2)

    def test_returns_none_when_no_yield(self):
        rxn_pb = reaction_pb2
        prod = rxn_pb.ProductCompound()
        m = prod.measurements.add()
        m.type = 6  # AREA, not YIELD
        assert _get_yield(prod.measurements) is None

    def test_returns_none_when_empty(self):
        assert _get_yield([]) is None


# ---------------------------------------------------------------------------
# Unit tests for _reaction_to_smiles
# ---------------------------------------------------------------------------


class TestReactionToSmiles:
    def test_basic_reaction(self):
        ds = _make_dataset([{
            "reactants": ["CC(=O)O", "CCO"],
            "products": ["CC(=O)OCC"],
            "reaction_id": "rxn-001",
        }])
        rxn = ds.reactions[0]
        smi, meta = _reaction_to_smiles(rxn)
        assert smi is not None
        assert "CC(=O)O" in smi
        assert "CCO" in smi
        assert "CC(=O)OCC" in smi
        assert ">" in smi  # has reaction arrow
        assert meta["ord_reaction_id"] == "rxn-001"

    def test_with_reagents(self):
        ds = _make_dataset([{
            "reactants": ["CC(=O)O"],
            "reagents": [("CCN(CC)CC", 2)],  # role=2 REAGENT
            "products": ["CC(=O)OCC"],
        }])
        smi, meta = _reaction_to_smiles(ds.reactions[0])
        parts = smi.split(">")
        assert len(parts) == 3
        assert "CCN(CC)CC" in parts[1]  # middle = reagents

    def test_returns_none_when_no_products(self):
        ds = _make_dataset([{
            "reactants": ["CCO"],
            "products": [],
        }])
        smi, meta = _reaction_to_smiles(ds.reactions[0])
        assert smi is None

    def test_returns_none_when_no_reactants(self):
        ds = _make_dataset([{
            "reactants": [],
            "products": ["CCO"],
        }])
        smi, meta = _reaction_to_smiles(ds.reactions[0])
        assert smi is None

    def test_yields_in_meta(self):
        ds = _make_dataset([{
            "reactants": ["CCO"],
            "products": ["CC"],
            "yields": [92.5],
        }])
        smi, meta = _reaction_to_smiles(ds.reactions[0])
        assert "ord_yields" in meta
        assert "92.5" in meta["ord_yields"]


# ---------------------------------------------------------------------------
# Integration: iter_ord_reactions
# ---------------------------------------------------------------------------


class TestIterOrdReactions:
    def test_yields_reaction_containers(self, tmp_path):
        pb_file = _write_dataset(tmp_path, [{
            "reactants": ["CC(=O)O", "CCO"],
            "products": ["CC(=O)OCC"],
            "reaction_id": "rxn-001",
        }])
        results = list(iter_ord_reactions(pb_file))
        assert len(results) == 1
        assert isinstance(results[0], ReactionContainer)
        assert results[0].meta["ord_reaction_id"] == "rxn-001"
        assert "init_smiles" in results[0].meta

    def test_skips_unparseable_smiles(self, tmp_path, caplog):
        import logging

        ds = dataset_pb2.Dataset()
        ds.name = "bad"
        rxn = ds.reactions.add()
        rxn.reaction_id = "bad-rxn"
        ri = rxn.inputs["r"]
        comp = ri.components.add()
        comp.reaction_role = 1
        ident = comp.identifiers.add()
        ident.type = 2
        ident.value = "INVALID!!!"
        outcome = rxn.outcomes.add()
        prod = outcome.products.add()
        pid = prod.identifiers.add()
        pid.type = 2
        pid.value = "ALSO_INVALID"
        pb_file = tmp_path / "bad.pb"
        pb_file.write_bytes(ds.SerializeToString())

        with caplog.at_level(logging.WARNING):
            results = list(iter_ord_reactions(pb_file))
        assert results == []

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_ord_reactions(tmp_path / "nonexistent.pb"))


# ---------------------------------------------------------------------------
# count_ord_reactions
# ---------------------------------------------------------------------------


class TestCountOrdReactions:
    def test_counts_reactions(self, tmp_path):
        pb_file = _write_dataset(tmp_path, [{
            "reactants": ["CCO"],
            "products": ["CC"],
        }])
        assert count_ord_reactions(pb_file) == 1


# ---------------------------------------------------------------------------
# convert_ord_to_smiles
# ---------------------------------------------------------------------------


class TestConvertOrdToSmiles:
    def test_writes_smi_file(self, tmp_path):
        pb_file = _write_dataset(tmp_path, [{
            "reactants": ["CC(=O)O", "CCO"],
            "products": ["CC(=O)OCC"],
        }])
        out_file = tmp_path / "out.smi"
        n = convert_ord_to_smiles(pb_file, out_file)
        assert n == 1
        lines = [l.strip() for l in out_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert ">" in lines[0]  # reaction SMILES format


# ---------------------------------------------------------------------------
# RawReactionReader integration
# ---------------------------------------------------------------------------


class TestRawReactionReaderPb:
    def test_accepts_pb_extension(self, tmp_path):
        from synplan.utils.files import RawReactionReader

        pb_file = _write_dataset(tmp_path, [{
            "reactants": ["CCO"],
            "products": ["CC"],
        }], filename="reactions.pb")
        reader = RawReactionReader(pb_file)
        assert reader.format == "pb"
        items = list(reader)
        assert len(items) == 1
        assert isinstance(items[0], ReactionContainer)

    def test_rejects_unknown_extension(self, tmp_path):
        from synplan.utils.files import RawReactionReader

        with pytest.raises(ValueError, match="Unsupported"):
            RawReactionReader(tmp_path / "data.xyz")


# ---------------------------------------------------------------------------
# ReactionReader integration
# ---------------------------------------------------------------------------


class TestReactionReaderPb:
    def test_reads_pb_file(self, tmp_path):
        from synplan.utils.files import ReactionReader

        pb_file = _write_dataset(tmp_path, [{
            "reactants": ["CC(=O)O", "CCO"],
            "products": ["CC(=O)OCC"],
        }], filename="reactions.pb")
        with ReactionReader(pb_file) as r:
            rxns = list(r)
        assert len(rxns) == 1
        assert isinstance(rxns[0], ReactionContainer)
