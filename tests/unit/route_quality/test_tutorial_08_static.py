"""Static checks for tutorial 08 protection APIs."""

import ast
import json
from pathlib import Path


def test_tutorial_08_uses_canonical_protection_revision_apis():
    notebook = json.loads(Path("tutorials/08_Protection_Scoring.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "# Protection Strategy Scoring and Revision" in source
    assert "from synplan.routes.quality.protection import" in source
    assert "from synplan.routes.route_cgr import extract_reactions" in source
    assert "from synplan.routes.route_cgr import compose_route_cgr" in source
    assert "from synplan.routes.io import read_routes_json, write_routes_json" in source
    assert "from synplan.routes.io import make_json" in source
    assert "from synplan.utils.visualisation import get_route_svg_json" in source
    assert "ProtectionRouteReviser" in source
    assert "scanner.scan_route(first_route, detailed=True)" in source
    assert "route_metadata={problem_rid: revision.route_metadata}" in source
    assert "display(SVG(revised_svg))" in source
    assert "protection_group_templates.csv" in source
    assert "Chemformer" in source


def test_tutorial_08_code_cells_are_valid_python():
    notebook = json.loads(Path("tutorials/08_Protection_Scoring.ipynb").read_text())

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"tutorial 08 cell {index}: {cell.get('id')}")


def test_tutorial_08_has_no_stored_error_outputs():
    notebook = json.loads(Path("tutorials/08_Protection_Scoring.ipynb").read_text())

    for index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"tutorial 08 cell {index}: {cell.get('id')} stores "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


def test_tutorial_08_revision_output_uses_current_action_schema():
    raw = Path("tutorials/08_Protection_Scoring.ipynb").read_text()

    assert "deprotection_class" not in raw
    assert "example_reagent" not in raw
    assert "label=None" not in raw
    assert "template=''" not in raw
