"""Execute the tutorial's setup cell from the supported notebook directories."""

import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[5]
NOTEBOOKS = (
    "tutorials/21_AIMNet_Thermochemistry.ipynb",
    "docs/user_guide/21_AIMNet_Thermochemistry.ipynb",
)


def setup_cell(notebook):
    content = json.loads((REPOSITORY / notebook).read_text())
    return "".join(
        next(c for c in content["cells"] if c["cell_type"] == "code")["source"]
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS)
@pytest.mark.parametrize("directory", [".", "tutorials", "docs/user_guide"])
def test_setup_finds_the_complete_collection(notebook, directory, monkeypatch):
    import torch

    threads = torch.get_num_threads()
    monkeypatch.chdir(REPOSITORY / directory)
    monkeypatch.delenv("SYNPLAN_AIMNET_ROUTES", raising=False)
    namespace = {}
    try:
        exec(compile(setup_cell(notebook), notebook, "exec"), namespace)
    finally:
        torch.set_num_threads(threads)
    assert namespace["repository"] == REPOSITORY
    assert len(namespace["routes"]) == 435
    assert sum(len(route) for route in namespace["routes"]) == 1192


def test_setup_outside_checkout_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        RuntimeError, match="Open this notebook from the SynPlanner checkout"
    ):
        exec(compile(setup_cell(NOTEBOOKS[0]), NOTEBOOKS[0], "exec"), {})
