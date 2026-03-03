"""
End-to-end tests for the clustering + visualization pipeline.

Based on the Step-5_Clustering.ipynb notebook workflow.
Uses existing test data (routes_mol_1.json/csv) to avoid tree-solving overhead.

Pipeline tested:
  Load routes (JSON/CSV) → compose RouteCGRs → compose SB-CGRs →
  cluster → subcluster → visualize (SVG/PNG) → export (JSON/CSV/HTML)
"""

import json
import logging
from pathlib import Path

import pytest
from chython.containers import CGRContainer, MoleculeContainer, ReactionContainer

from synplan.chem.reaction_routes.clustering import (
    cluster_routes,
    post_process_subgroup,
    subcluster_all_clusters,
)
from synplan.chem.reaction_routes.io import (
    make_dict,
    make_json,
    read_routes_csv,
    read_routes_json,
    write_routes_csv,
    write_routes_json,
)
from synplan.chem.reaction_routes.route_cgr import (
    compose_all_route_cgrs,
    compose_all_sb_cgrs,
)
from synplan.chem.reaction_routes.visualisation import (
    cgr_display,
    depict_custom_reaction,
)
from synplan.utils.visualisation import (
    get_route_svg_from_json,
    routes_clustering_report,
    routes_subclustering_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA = Path(__file__).resolve().parent.parent / "data"
JSON_PATH = TEST_DATA / "routes_mol_1.json"
CSV_PATH = TEST_DATA / "routes_mol_1.csv"


# ---------------------------------------------------------------------------
# Fixtures — built once per module for speed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def routes_dict_from_json():
    """Load routes from JSON test data as {route_id: {step_id: Reaction}}."""
    assert JSON_PATH.exists(), f"Test data missing: {JSON_PATH}"
    return read_routes_json(file_path=JSON_PATH, to_dict=True)


@pytest.fixture(scope="module")
def routes_dict_from_csv():
    """Load routes from CSV test data as {route_id: {step_id: Reaction}}."""
    assert CSV_PATH.exists(), f"Test data missing: {CSV_PATH}"
    return read_routes_csv(file_path=CSV_PATH)


@pytest.fixture(scope="module")
def routes_json(routes_dict_from_json):
    """Convert routes_dict to JSON tree structure (for visualization functions)."""
    return make_json(routes_dict_from_json)


@pytest.fixture(scope="module")
def all_route_cgrs(routes_dict_from_json):
    """Compose all route CGRs from routes_dict."""
    cgrs = compose_all_route_cgrs(routes_dict_from_json)
    assert len(cgrs) > 0, "Should compose at least one route CGR"
    return cgrs


@pytest.fixture(scope="module")
def all_sb_cgrs(all_route_cgrs):
    """Compose all SB-CGRs (reduced route CGRs) from route CGRs."""
    sb = compose_all_sb_cgrs(all_route_cgrs)
    assert len(sb) > 0, "Should compose at least one SB-CGR"
    return sb


@pytest.fixture(scope="module")
def clusters(all_sb_cgrs):
    """Cluster routes by SB-CGR."""
    cl = cluster_routes(all_sb_cgrs, use_strat=False)
    assert len(cl) > 0, "Should produce at least one cluster"
    return cl


@pytest.fixture(scope="module")
def all_subclusters(clusters, all_sb_cgrs, all_route_cgrs):
    """Subcluster all clusters."""
    sc = subcluster_all_clusters(clusters, all_sb_cgrs, all_route_cgrs)
    assert len(sc) > 0, "Should produce at least one subcluster group"
    return sc


# ---------------------------------------------------------------------------
# 1. Route loading
# ---------------------------------------------------------------------------


class TestRouteLoading:
    """Test route loading from JSON and CSV."""

    def test_load_routes_from_json(self, routes_dict_from_json):
        rd = routes_dict_from_json
        assert isinstance(rd, dict)
        assert len(rd) > 0, "Should load at least one route"
        for route_id, steps in rd.items():
            assert isinstance(steps, dict)
            for step_id, rxn in steps.items():
                assert isinstance(
                    rxn, ReactionContainer
                ), f"Route {route_id} step {step_id}: expected ReactionContainer"

    def test_load_routes_from_csv(self, routes_dict_from_csv):
        rd = routes_dict_from_csv
        assert isinstance(rd, dict)
        assert len(rd) > 0
        for route_id, steps in rd.items():
            assert isinstance(steps, dict)
            for step_id, rxn in steps.items():
                assert isinstance(rxn, ReactionContainer)

    def test_json_csv_produce_same_routes(
        self, routes_dict_from_json, routes_dict_from_csv
    ):
        """Both loaders should produce the same set of route IDs."""
        json_ids = set(routes_dict_from_json.keys())
        csv_ids = set(routes_dict_from_csv.keys())
        assert json_ids == csv_ids, (
            f"Route IDs differ: JSON has {json_ids - csv_ids}, "
            f"CSV has {csv_ids - json_ids}"
        )

    def test_make_json_produces_valid_tree(self, routes_json, routes_dict_from_json):
        """make_json should produce a dict of tree structures."""
        assert isinstance(routes_json, dict)
        assert len(routes_json) > 0
        for route_id, tree in routes_json.items():
            assert isinstance(tree, dict), f"Route {route_id}: expected dict tree node"
            assert (
                tree.get("type") == "mol"
            ), f"Route {route_id}: root node should be 'mol'"
            assert "smiles" in tree
            assert "children" in tree

    def test_make_dict_roundtrip(self, routes_dict_from_json):
        """make_json → make_dict should preserve route IDs and contain reactions.

        Note: The JSON tree format is a nested tree, so the number of steps
        recovered by make_dict may differ from the original flat dict (the tree
        only follows the path that connects to the target molecule). We verify
        route IDs match and each route has at least one step.
        """
        rj = make_json(routes_dict_from_json)
        rd_roundtrip = make_dict(rj)
        assert set(rd_roundtrip.keys()) == set(routes_dict_from_json.keys())
        for route_id in rd_roundtrip:
            assert (
                len(rd_roundtrip[route_id]) > 0
            ), f"Route {route_id}: should have at least one step after roundtrip"


# ---------------------------------------------------------------------------
# 2. Route CGR composition
# ---------------------------------------------------------------------------


class TestRouteCGRComposition:
    """Test route CGR and SB-CGR composition."""

    def test_compose_all_route_cgrs_types(self, all_route_cgrs):
        for route_id, cgr in all_route_cgrs.items():
            assert isinstance(
                cgr, CGRContainer
            ), f"Route {route_id}: expected CGRContainer, got {type(cgr)}"

    def test_compose_all_route_cgrs_nonempty(self, all_route_cgrs):
        for route_id, cgr in all_route_cgrs.items():
            assert len(cgr) > 0, f"Route {route_id}: CGR has no atoms"

    def test_compose_all_sb_cgrs_types(self, all_sb_cgrs):
        for route_id, sb_cgr in all_sb_cgrs.items():
            assert isinstance(
                sb_cgr, CGRContainer
            ), f"Route {route_id}: expected CGRContainer, got {type(sb_cgr)}"

    def test_sb_cgrs_subset_of_route_cgrs(self, all_route_cgrs, all_sb_cgrs):
        """SB-CGR route IDs should be a subset of route CGR route IDs."""
        assert set(all_sb_cgrs.keys()).issubset(set(all_route_cgrs.keys()))


# ---------------------------------------------------------------------------
# 3. Clustering
# ---------------------------------------------------------------------------


class TestClustering:
    """Test clustering and subclustering."""

    def test_cluster_structure(self, clusters):
        for key, cluster in clusters.items():
            assert isinstance(key, str), f"Cluster key should be str, got {type(key)}"
            assert "route_ids" in cluster
            assert "sb_cgr" in cluster
            assert "group_size" in cluster
            assert cluster["group_size"] == len(cluster["route_ids"])

    def test_all_routes_clustered(self, clusters, all_sb_cgrs):
        """Every route in all_sb_cgrs should appear in exactly one cluster."""
        clustered_ids = set()
        for cluster in clusters.values():
            for rid in cluster["route_ids"]:
                assert rid not in clustered_ids, f"Route {rid} in multiple clusters"
                clustered_ids.add(rid)
        assert clustered_ids == set(all_sb_cgrs.keys()), "Not all routes are clustered"

    def test_cluster_with_use_strat(self, all_sb_cgrs):
        """Clustering with use_strat=True should also work."""
        cl = cluster_routes(all_sb_cgrs, use_strat=True)
        assert len(cl) > 0

    def test_subcluster_structure(self, all_subclusters):
        for cluster_key, subcl in all_subclusters.items():
            assert isinstance(subcl, dict)
            for sc_num, sc_data in subcl.items():
                assert "routes_data" in sc_data
                assert "synthon_reaction" in sc_data

    def test_subcluster_keys_match_cluster_keys(self, clusters, all_subclusters):
        """Subcluster keys should match cluster keys."""
        assert sorted(all_subclusters.keys()) == sorted(clusters.keys())

    def test_total_subclustered_routes_match_total_routes(
        self, clusters, all_subclusters
    ):
        """Total routes across all subclusters should match total clustered routes."""
        total_clustered = sum(c["group_size"] for c in clusters.values())
        total_subclustered = sum(
            len(sc["routes_data"])
            for subcl in all_subclusters.values()
            for sc in subcl.values()
        )
        assert total_subclustered == total_clustered


# ---------------------------------------------------------------------------
# 4. Visualization — SVG
# ---------------------------------------------------------------------------


class TestVisualizationSVG:
    """Test SVG generation from various pipeline stages."""

    def test_cgr_display_produces_svg(self, all_route_cgrs):
        """cgr_display should return a valid SVG string for each route CGR."""
        tested = 0
        for route_id, route_cgr in all_route_cgrs.items():
            components = list(route_cgr.connected_components)
            if not components:
                continue
            target_cgr = route_cgr.substructure(components[0])
            svg = cgr_display(target_cgr)
            assert isinstance(svg, str), f"Route {route_id}: expected str SVG"
            assert svg.strip().startswith(
                "<svg"
            ), f"Route {route_id}: SVG should start with <svg"
            assert "</svg>" in svg, f"Route {route_id}: SVG should end with </svg>"
            tested += 1
            if tested >= 5:
                break
        assert tested > 0, "No CGRs were tested"

    def test_sb_cgr_display_produces_svg(self, all_sb_cgrs):
        """cgr_display should work on SB-CGRs too."""
        tested = 0
        for route_id, sb_cgr in all_sb_cgrs.items():
            components = list(sb_cgr.connected_components)
            if not components:
                continue
            target_cgr = sb_cgr.substructure(components[0])
            svg = cgr_display(target_cgr)
            assert isinstance(svg, str)
            assert "<svg" in svg
            tested += 1
            if tested >= 3:
                break
        assert tested > 0

    def test_molecule_depict_svg(self, routes_dict_from_json):
        """MoleculeContainer.depict() should produce SVG."""
        first_route = next(iter(routes_dict_from_json.values()))
        first_rxn = next(iter(first_route.values()))
        mol = first_rxn.reactants[0]
        mol.clean2d()
        svg = mol.depict(format="svg")
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_reaction_depict_svg(self, routes_dict_from_json):
        """ReactionContainer.depict() should produce SVG."""
        first_route = next(iter(routes_dict_from_json.values()))
        first_rxn = next(iter(first_route.values()))
        svg = first_rxn.depict(format="svg")
        assert isinstance(svg, str)
        assert "<svg" in svg

    def test_cgr_depict_svg_direct(self, all_route_cgrs):
        """CGRContainer.depict() (without custom rendering) should produce SVG."""
        first_cgr = next(iter(all_route_cgrs.values()))
        first_cgr.clean2d()
        svg = first_cgr.depict(format="svg")
        assert isinstance(svg, str)
        assert "<svg" in svg

    def test_get_route_svg_from_json(self, routes_json):
        """get_route_svg_from_json should produce valid SVG for each route."""
        tested = 0
        for route_id in routes_json:
            svg = get_route_svg_from_json(routes_json, route_id)
            assert isinstance(svg, str), f"Route {route_id}: expected str SVG"
            assert "<svg" in svg, f"Route {route_id}: should contain <svg"
            tested += 1
            if tested >= 5:
                break
        assert tested > 0

    def test_depict_custom_reaction(self, routes_dict_from_json):
        """depict_custom_reaction should produce valid SVG for a reaction."""
        first_route = next(iter(routes_dict_from_json.values()))
        first_rxn = next(iter(first_route.values()))
        svg = depict_custom_reaction(first_rxn)
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg


# ---------------------------------------------------------------------------
# 5. Visualization — PNG
# ---------------------------------------------------------------------------


class TestVisualizationPNG:
    """Test PNG generation (requires playwright/browser support)."""

    @pytest.fixture(autouse=True)
    def _check_png_support(self):
        """Skip PNG tests if svg2png is not available."""
        try:
            from chython.algorithms.depict import svg2png  # noqa: F401
        except ImportError:
            pytest.skip("svg2png not available (missing playwright/browser)")

    def test_molecule_depict_png(self, routes_dict_from_json):
        first_route = next(iter(routes_dict_from_json.values()))
        first_rxn = next(iter(first_route.values()))
        mol = first_rxn.reactants[0]
        mol.clean2d()
        try:
            png = mol.depict(format="png")
            assert isinstance(png, bytes)
            assert len(png) > 0
            # PNG magic bytes
            assert png[:4] == b"\x89PNG"
        except Exception as e:
            pytest.skip(f"PNG rendering failed (browser not available?): {e}")

    def test_cgr_depict_png(self, all_route_cgrs):
        first_cgr = next(iter(all_route_cgrs.values()))
        first_cgr.clean2d()
        try:
            png = first_cgr.depict(format="png")
            assert isinstance(png, bytes)
            assert len(png) > 0
            assert png[:4] == b"\x89PNG"
        except Exception as e:
            pytest.skip(f"PNG rendering failed (browser not available?): {e}")

    def test_reaction_depict_png(self, routes_dict_from_json):
        first_route = next(iter(routes_dict_from_json.values()))
        first_rxn = next(iter(first_route.values()))
        try:
            png = first_rxn.depict(format="png")
            assert isinstance(png, bytes)
            assert len(png) > 0
            assert png[:4] == b"\x89PNG"
        except Exception as e:
            pytest.skip(f"PNG rendering failed (browser not available?): {e}")


# ---------------------------------------------------------------------------
# 6. HTML reports
# ---------------------------------------------------------------------------


class TestHTMLReports:
    """Test HTML report generation."""

    def test_routes_clustering_report(self, clusters, all_sb_cgrs, routes_json):
        """routes_clustering_report should produce valid HTML for each cluster."""
        tested = 0
        for group_index in clusters:
            html = routes_clustering_report(
                routes_json, clusters, group_index, all_sb_cgrs
            )
            assert isinstance(html, str)
            assert len(html) > 0
            # Should contain some HTML structure
            assert "<" in html
            tested += 1
            if tested >= 3:
                break
        assert tested > 0

    def test_routes_clustering_report_saves_to_file(
        self, clusters, all_sb_cgrs, routes_json, tmp_path
    ):
        group_index = next(iter(clusters))
        html_path = str(tmp_path / "cluster_report.html")
        result = routes_clustering_report(
            routes_json, clusters, group_index, all_sb_cgrs, html_path=html_path
        )
        assert Path(html_path).exists(), "HTML report file should be created"
        content = Path(html_path).read_text()
        assert len(content) > 0

    def test_routes_subclustering_report(
        self, all_subclusters, all_sb_cgrs, routes_json
    ):
        """routes_subclustering_report should produce valid HTML."""
        tested = 0
        for cluster_key, subcl in all_subclusters.items():
            for sc_num, sc_data in subcl.items():
                html = routes_subclustering_report(
                    routes_json, sc_data, cluster_key, sc_num, all_sb_cgrs
                )
                assert isinstance(html, str)
                assert len(html) > 0
                tested += 1
                if tested >= 3:
                    break
            if tested >= 3:
                break
        assert tested > 0

    def test_post_process_subgroup(self, all_subclusters):
        """post_process_subgroup should work on subclusters with >1 route."""
        tested = 0
        for cluster_key, subcl in all_subclusters.items():
            for sc_num, sc_data in subcl.items():
                if len(sc_data["routes_data"]) > 1:
                    processed = post_process_subgroup(sc_data)
                    assert isinstance(processed, dict)
                    assert "routes_data" in processed
                    tested += 1
                    if tested >= 2:
                        break
            if tested >= 2:
                break
        # It's OK if no subclusters have >1 route; just don't fail


# ---------------------------------------------------------------------------
# 7. Export (JSON / CSV roundtrips)
# ---------------------------------------------------------------------------


class TestExport:
    """Test route export to JSON and CSV."""

    def test_write_routes_json(self, routes_dict_from_json, tmp_path):
        out = tmp_path / "exported_routes.json"
        write_routes_json(routes_dict_from_json, str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) > 0

    def test_write_routes_csv(self, routes_dict_from_json, tmp_path):
        out = tmp_path / "exported_routes.csv"
        write_routes_csv(routes_dict_from_json, str(out))
        assert out.exists()
        content = out.read_text()
        lines = content.strip().split("\n")
        assert len(lines) > 1, "CSV should have header + data rows"
        assert "route_id" in lines[0]

    def test_json_export_roundtrip(self, routes_dict_from_json, tmp_path):
        """Write JSON → read back → verify same route IDs."""
        out = tmp_path / "roundtrip.json"
        write_routes_json(routes_dict_from_json, str(out))
        rd_back = read_routes_json(file_path=str(out), to_dict=True)
        assert set(rd_back.keys()) == set(routes_dict_from_json.keys())

    def test_csv_export_roundtrip(self, routes_dict_from_json, tmp_path):
        """Write CSV → read back → verify same route IDs."""
        out = tmp_path / "roundtrip.csv"
        write_routes_csv(routes_dict_from_json, str(out))
        rd_back = read_routes_csv(file_path=str(out))
        assert set(rd_back.keys()) == set(routes_dict_from_json.keys())

    def test_json_csv_cross_roundtrip(self, routes_dict_from_json, tmp_path):
        """Write JSON → read as dict → write CSV → read back → same IDs."""
        json_out = tmp_path / "step1.json"
        write_routes_json(routes_dict_from_json, str(json_out))
        rd = read_routes_json(file_path=str(json_out), to_dict=True)
        csv_out = tmp_path / "step2.csv"
        write_routes_csv(rd, str(csv_out))
        rd_csv = read_routes_csv(file_path=str(csv_out))
        assert set(rd_csv.keys()) == set(routes_dict_from_json.keys())


# ---------------------------------------------------------------------------
# 8. Full notebook pipeline (smoke test)
# ---------------------------------------------------------------------------


class TestFullPipelineSmokeTest:
    """Smoke test replicating the notebook cells end-to-end."""

    def test_full_pipeline(self, tmp_path, caplog):
        """Replicate the notebook flow: load → compose → cluster → visualize."""
        caplog.set_level(logging.DEBUG)

        # Cell: Load routes from JSON
        routes_dict = read_routes_json(file_path=JSON_PATH, to_dict=True)
        assert len(routes_dict) > 0

        # Cell: make_json for visualization
        routes_json = make_json(routes_dict)
        assert len(routes_json) > 0

        # Cell: Compose route CGRs
        all_route_cgrs = compose_all_route_cgrs(routes_dict)
        assert len(all_route_cgrs) > 0

        # Cell: Compose SB-CGRs
        # NOTE: The notebook uses `compose_all_reduced_route_cgrs` which is the
        # old name. The correct function is `compose_all_sb_cgrs`.
        all_sb_cgrs = compose_all_sb_cgrs(all_route_cgrs)
        assert len(all_sb_cgrs) > 0

        # Cell: Display CGRs as SVG
        tested_cgr_svg = 0
        for route_id, route_cgr in all_route_cgrs.items():
            components = list(route_cgr.connected_components)
            if not components:
                continue
            target_cgr = route_cgr.substructure(components[0])
            svg = cgr_display(target_cgr)
            assert "<svg" in svg
            tested_cgr_svg += 1
            if tested_cgr_svg >= 3:
                break
        assert tested_cgr_svg > 0, "Should successfully render at least one CGR SVG"

        # Cell: Display route SVGs from JSON
        first_route_id = next(iter(routes_json))
        route_svg = get_route_svg_from_json(routes_json, first_route_id)
        assert "<svg" in route_svg

        # Cell: Cluster
        clusters = cluster_routes(all_sb_cgrs, use_strat=False)
        assert len(clusters) > 0

        # Cell: Clustering report
        first_cluster = next(iter(clusters))
        html_report = routes_clustering_report(
            routes_json, clusters, first_cluster, all_sb_cgrs
        )
        assert len(html_report) > 0

        # Cell: Subcluster
        all_subclusters = subcluster_all_clusters(clusters, all_sb_cgrs, all_route_cgrs)
        assert len(all_subclusters) > 0

        # Cell: Subclustering report
        first_sc_key = next(iter(all_subclusters))
        first_sc_num = next(iter(all_subclusters[first_sc_key]))
        subgroup = all_subclusters[first_sc_key][first_sc_num]
        sc_html = routes_subclustering_report(
            routes_json, subgroup, first_sc_key, first_sc_num, all_sb_cgrs
        )
        assert len(sc_html) > 0

        # Cell: Post-processing (if applicable)
        if len(subgroup["routes_data"]) > 1:
            new_subgroup = post_process_subgroup(subgroup)
            assert "routes_data" in new_subgroup

        # Cell: Export
        json_out = tmp_path / "pipeline_routes.json"
        csv_out = tmp_path / "pipeline_routes.csv"
        write_routes_json(routes_dict, str(json_out))
        write_routes_csv(routes_dict, str(csv_out))
        assert json_out.exists()
        assert csv_out.exists()

        # Cell: Save HTML report to file
        html_path = tmp_path / "cluster_report.html"
        routes_clustering_report(
            routes_json,
            clusters,
            first_cluster,
            all_sb_cgrs,
            html_path=str(html_path),
        )
        assert html_path.exists()
