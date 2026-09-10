"""Cache lifecycle, independent readers and detached route presentation."""

import gzip
import json
import multiprocessing
import os
import re
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from html import unescape

import pytest
from chython import smiles

from synplan.chem.building_blocks import SQLiteBuildingBlockCatalogue, io
from synplan.chem.building_blocks.database import build_catalogue
from synplan.chem.precursor import Precursor
from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.route import Route, Step
from synplan.chem.utils import standardize_building_blocks
from synplan.utils.loading import load_building_blocks
from synplan.utils.visualisation import routes_report_html

RAW = "SMILES\tchosen_ppg\tsecond_ppg\nC[C@H](O)C(=O)O\t5\t7\nC[C@@H](O)C(=O)O\t1\t0\nCCO\t4\t0\nOCC\t2\t3\n"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def read_in_worker(path, prefix):
    stock = load_building_blocks(path)
    result = stock[prefix]
    connection = stock._reader().connection
    assert connection.execute("PRAGMA cache_size").fetchone()[0] == -8192
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        connection.execute("DELETE FROM blocks")
    stock.close()
    return result


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_raw_sqlite_matches_prepared_json_and_errors(tmp_path, compressed, workers):
    source = tmp_path / ("stock.tsv.gz" if compressed else "stock.tsv")
    raw = RAW + "C1CC\t1\t0\nCCN\tbad\t0\n"
    source.write_bytes(gzip.compress(raw.encode()) if compressed else raw.encode())
    prepared = tmp_path / "stock.json"
    standardize_building_blocks(source, prepared)
    destination = tmp_path / "stock.sqlite"
    standardize_building_blocks(source, destination, num_workers=workers)
    stock = load_building_blocks(destination)
    expected = load_building_blocks(prepared)
    assert stock == expected
    assert stock.record_count == 3
    assert stock.metadata["rejected_rows"] == 2
    assert (
        destination.with_name("stock.sqlite.errors.tsv").read_bytes()
        == prepared.with_name("stock.json.errors.tsv").read_bytes()
    )
    assert load_building_blocks(source, num_workers=workers) == stock


def test_cache_reuse_invalidation_and_release_identity(tmp_path, monkeypatch):
    source = tmp_path / "stock.tsv"
    source.write_text(RAW)
    stock = load_building_blocks(source)

    def no_chemistry(*args, **kwargs):
        pytest.fail("Chemistry repeated for cached input")

    with monkeypatch.context() as patch:
        patch.setattr(io, "_prepare_catalogue_batches", no_chemistry)
        assert load_building_blocks(source) is stock
        assert load_building_blocks(stock.path) is stock
    meta = tmp_path / "meta.yaml"
    meta.write_text("release: test-v2\n")
    changed = load_building_blocks(source)
    assert changed is not stock
    assert changed.metadata["source_metadata"] == meta.read_text()
    assert changed.metadata["cache_id"] != stock.metadata["cache_id"]
    source.write_text(RAW.replace("5\t7", "9\t7"))
    assert (
        load_building_blocks(source).metadata["cache_id"]
        != changed.metadata["cache_id"]
    )
    assert stock == changed  # old snapshot remains usable


def test_empty_or_invalid_build_preserves_existing_database(tmp_path, monkeypatch):
    source = tmp_path / "stock.tsv"
    destination = tmp_path / "stock.sqlite"
    source.write_text(RAW)
    build_catalogue(source, destination)
    before = destination.read_bytes()
    source.write_text("SMILES\tv_ppg\ninvalid!\t1\n")
    with pytest.raises(ValueError, match="no valid rows"):
        build_catalogue(source, destination)
    assert destination.read_bytes() == before
    assert destination.with_name("stock.sqlite.errors.tsv").exists()
    prepared = tmp_path / "bad.json"
    prepared.write_text('{"bad": {}}')
    with pytest.raises(ValueError):
        build_catalogue(prepared, destination)
    assert destination.read_bytes() == before
    source.write_text(RAW)
    replace = io.os.replace

    def fail_publication(source_path, destination_path):
        if destination_path == destination:
            raise OSError("publication failed")
        replace(source_path, destination_path)

    with monkeypatch.context() as patch:
        patch.setattr(io.os, "replace", fail_publication)
        with pytest.raises(OSError, match="publication failed"):
            build_catalogue(source, destination)
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob(".stock.sqlite.*"))
    build_catalogue(source, destination)
    assert not destination.with_name("stock.sqlite.errors.tsv").exists()
    if os.name == "posix":
        assert destination.stat().st_mode & 0o777 == 0o600


def test_thread_and_spawn_readers_open_database_paths(tmp_path):
    source = tmp_path / "stock.tsv"
    source.write_text(RAW)
    stock = load_building_blocks(source)
    prefix = next(iter(stock))
    expected = stock[prefix]
    release_id = stock.metadata["cache_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert (
            list(pool.map(read_in_worker, [stock.path] * 2, [prefix] * 2))
            == [expected] * 2
        )
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        assert (
            list(pool.map(read_in_worker, [stock.path] * 2, [prefix] * 2))
            == [expected] * 2
        )
    stock.close()
    source.write_text(RAW.replace("5\t7", "9\t7"))
    build_catalogue(source, stock.path)
    with pytest.raises(ValueError, match="release changed"):
        SQLiteBuildingBlockCatalogue(stock.path, expected_cache_id=release_id)


@pytest.mark.parametrize("priced", [True, False])
def test_cost_export_and_html_use_selected_isomer_without_scanning(
    tmp_path, monkeypatch, priced
):
    source = tmp_path / "stock.tsv"
    raw = RAW.replace("chosen_ppg", "<chosen>_ppg")
    source.write_text(raw if priced else raw.replace("\t5\t7\n", "\t0\t0\n"))
    stock = load_building_blocks(source)

    def no_scan(self):
        pytest.fail("Catalogue iteration is not needed for route costing/rendering")

    monkeypatch.setattr(SQLiteBuildingBlockCatalogue, "__iter__", no_scan)
    leaf = smiles("C[C@H](O)C(=O)O", ignore_stereo=False)
    precursor = Precursor(leaf)
    assert precursor.is_building_block(stock)
    leaf = precursor.molecule
    selected = leaf.meta["selected_stock"]
    target = smiles("CCOC(=O)C(O)C")
    route = Route((Step(Reaction([leaf], [target]), target),))
    cost = route.calculate_cost(stock)["leaves"][0]
    assert cost["vendor"] == ("<chosen>" if priced else None)
    assert cost["status"] == ("priced" if priced else "unpriced")
    detached = Route.from_json(json.loads(json.dumps(route.to_json())))
    assert detached.leaves()[0].meta["selected_stock"] == selected
    stock.close()
    stock.path.unlink()
    html = routes_report_html([detached], None)
    assert "<svg" in html
    pills = re.findall(r'<g class="sp-price" data-offers="([^"]*)">', html)
    if priced:
        # The pill prices the isomer the search chose, not the cheaper opposite one.
        assert "&lt;chosen&gt;" in html and "<chosen>" not in html
        (payload,) = pills
        assert json.loads(unescape(payload))["rows"] == [
            ["<chosen>", "5"],
            ["second", "7"],
        ]
        assert selected["sources"] == [
            {"vendor": "<chosen>", "ppg": "5.0"},
            {"vendor": "second", "ppg": "7.0"},
        ]
    else:
        # No offer behind the record, but the page still names what was selected.
        (payload,) = pills
        offer = json.loads(unescape(payload))
        assert offer["rows"] == [["Price", "unavailable"]]
        assert offer["title"] == selected["inchikey"]
        assert (
            '<div class="eyebrow">Price per g of target</div><div class="v">—</div>'
            in html
        )
    # 1 is the opposite isomer's price; no pill ever reaches for it.
    assert all("1" not in str(json.loads(unescape(p))) for p in pills)


class EmptyPolicy:
    def predict_reaction_rules(self, precursor, reaction_rules):
        return iter(())


def test_tree_init_does_not_materialize_stock(tmp_path, monkeypatch):
    from synplan.mcts.config import TreeConfig
    from synplan.mcts.evaluation import RolloutEvaluationStrategy
    from synplan.mcts.tree import Tree
    from synplan.ml.training.reinforcement import run_tree_search

    source = tmp_path / "stock.tsv"
    source.write_text(RAW)
    stock = load_building_blocks(source)

    def no_scan(self):
        pytest.fail("Tree construction must use stored record counts")

    monkeypatch.setattr(SQLiteBuildingBlockCatalogue, "__iter__", no_scan)
    policy = EmptyPolicy()
    evaluator = RolloutEvaluationStrategy(
        policy_network=policy,
        reaction_rules=(),
        building_blocks=stock,
        min_mol_size=0,
        max_depth=2,
    )
    tree = Tree(
        target=smiles("CCCCCCC"),
        config=TreeConfig(max_iterations=1, silent=True),
        reaction_rules=(),
        building_blocks=stock,
        expansion_function=policy,
        evaluation_function=evaluator,
    )
    assert tree.building_blocks is evaluator.rollout.building_blocks is stock
    assert tree.building_blocks.record_count == 3
    with pytest.raises(ValueError, match="RL tree search requires SMILES stock"):
        run_tree_search(
            smiles("CCCCCCC"), tree.config, None, None, "unused", str(stock.path)
        )
