"""Execute the tutorial's setup cell from the supported notebook directories."""

import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[5]
NOTEBOOKS = (
    "tutorials/22_AIMNet_Thermochemistry.ipynb",
    "docs/user_guide/22_AIMNet_Thermochemistry.ipynb",
)


def setup_cell(notebook):
    content = json.loads((REPOSITORY / notebook).read_text(encoding="utf-8"))
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


@pytest.mark.parametrize("directory", [".", "tutorials", "docs/user_guide"])
def test_config_cell_loads_shipped_yaml_and_checkpoint_overrides(
    directory, tmp_path, monkeypatch
):
    import torch

    monkeypatch.chdir(REPOSITORY / directory)
    monkeypatch.delenv("SYNPLAN_AIMNET_ROUTES", raising=False)
    checkpoint = tmp_path / "local-aimnet.pt"
    monkeypatch.setenv("SYNPLAN_AIMNET_MODEL", str(checkpoint))
    monkeypatch.delenv("SYNPLAN_LOQI_MODEL", raising=False)
    namespace = {}
    threads = torch.get_num_threads()
    try:
        exec(compile(setup_cell(NOTEBOOKS[0]), NOTEBOOKS[0], "exec"), namespace)
    finally:
        torch.set_num_threads(threads)
    received = []

    def run(source, output, config, cache, **kwargs):
        received.append(config)
        return {"statistics": {}}

    namespace["run_thermochemistry_assessment"] = run
    namespace["read_routes_json"] = lambda *a, **kw: namespace["routes"]
    notebook = json.loads((REPOSITORY / NOTEBOOKS[0]).read_text(encoding="utf-8"))
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code"
        and "config = AIMNetConfig.from_yaml" in "".join(c["source"])
    )
    exec(compile(source, NOTEBOOKS[0], "exec"), namespace)
    assert len(received) == 1
    assert received[0].device == "cpu" and received[0].temperature == 298.15
    assert received[0].aimnet_checkpoint == checkpoint
    assert received[0].loqi_checkpoint is None
