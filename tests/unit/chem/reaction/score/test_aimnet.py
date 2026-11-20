"""Assessment and cache contracts; no model downloads or model imports."""

import json
import subprocess
import sys

import pytest
from chython import smiles

from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.aimnet.config import MolecularCalculationError
from synplan.chem.reaction.score.aimnet.molecular import EV_TO_KCAL_MOL
from synplan.chem.reaction.score.aimnet.results import reaction_thermochemistry


class Calculator:
    def __init__(self, identity="weights-1", fail=None):
        self.calls = []
        self.key = identity
        self.fail = fail

    def identity(self):
        return {"weights_sha256": self.key, "software_version": "test"}

    def calculate(self, text):
        self.calls.append(text)
        if self.fail is not None:
            raise self.fail
        # Deliberately structure-sensitive numbers, independent of orchestration.
        energy = -float(sum(text.encode()))
        return {"energy_eV": energy, "enthalpy_eV": energy + 1, "gibbs_eV": energy - 1}


def route(text="CC(=O)O.CCO>>CC(=O)OCC"):
    reaction = smiles(text)
    return Route((Step(reaction, reaction.products[0]),))


@pytest.mark.parametrize(
    ("reaction", "reason"),
    [
        ("CCO>>CC=O", "unsupported_redox"),
        ("C[13CH3]>>CC", "not_conserved"),
        ("[Na+].CCO>>[Na+].CCO", "unsupported_element"),
        ("[CH3]>>[CH3]", "radical"),
        ("CS(C)=O>>CS(C)=O", None),
        ("CS(CC)=O>>CS(CC)=O", "unsupported_stereo"),
        ("FC=C=CCl>>FC=C=CCl", "unsupported_stereo"),
    ],
)
def test_filtering_and_balancing_failures(reaction, reason):
    calculator = Calculator()
    result = AIMNetReactionAssessor(
        cache_path=None, calculator=calculator
    ).assess_reactions([smiles(reaction)])[0]
    if reason is None:
        assert result["status"] == "ok"
    else:
        assert result["status"] == (
            "failed" if reason == "not_conserved" else "skipped"
        )
        assert result["failure"]["reason"] == reason
        assert calculator.calls == []


def test_reaction_sign_units_duplicates_and_all_products():
    molecules = {
        "A": {"energy_eV": 1.0, "enthalpy_eV": 2.0, "gibbs_eV": 3.0},
        "B": {"energy_eV": 5.0, "enthalpy_eV": 6.0, "gibbs_eV": 7.0},
        "C": {"energy_eV": 3.0, "enthalpy_eV": 4.0, "gibbs_eV": 5.0},
    }
    actual = reaction_thermochemistry(["A", "A"], ["B", "C"], molecules)
    for key in ("delta_e", "delta_h", "delta_g"):
        assert actual[key] == pytest.approx(6 * EV_TO_KCAL_MOL)
    reverse = reaction_thermochemistry(["B", "C"], ["A", "A"], molecules)
    assert reverse["delta_g"] == -actual["delta_g"]


@pytest.mark.parametrize("filename", ["cache.json", "cache.sqlite"])
def test_persistent_cache_and_no_input_mutation(tmp_path, filename):
    source = route()
    before = source.to_json()
    calculator = Calculator()
    cache = tmp_path / "nested" / filename
    service = AIMNetReactionAssessor(cache_path=cache, calculator=calculator)
    result = service.assess([source, source])
    assert len(calculator.calls) == 4
    assert source.to_json() == before
    assert result[0] is not source
    assert result[0].steps[0].reaction is source.steps[0].reaction
    assert result[0].steps[0].product is source.steps[0].product
    assert service.last_stats["cache_hits"] == 4
    repeated = Calculator()
    rerun = AIMNetReactionAssessor(cache_path=cache, calculator=repeated)
    assert [r.to_json() for r in rerun.assess([source, source])] == [
        r.to_json() for r in result
    ]
    assert repeated.calls == []
    result[0].steps[0].thermochemistry["molecules"]["O"]["energy_eV"] = 999
    assert result[1].steps[0].thermochemistry["molecules"]["O"]["energy_eV"] != 999


@pytest.mark.parametrize(
    "change",
    ["model", "temperature", "pressure", "seed", "symmetry", "conformers", "fmax"],
)
@pytest.mark.parametrize("filename", ["cache.json", "cache.sqlite"])
def test_cache_invalidated_by_calculation_settings(tmp_path, change, filename):
    cache = tmp_path / filename
    AIMNetReactionAssessor(cache_path=cache, calculator=Calculator()).assess(
        [route("CCO>>CCO")]
    )
    changes = {
        "temperature": {"temperature": 310},
        "pressure": {"pressure": 200000},
        "seed": {"seed": 1},
        "symmetry": {"symmetry_numbers": {"OCC": 2}},
        "conformers": {"n_conformers": 1},
        "fmax": {"fmax": 0.02},
    }
    calculator = Calculator(identity="weights-2" if change == "model" else "weights-1")
    if change == "symmetry":
        calculate = calculator.calculate
        calculator.calculate = lambda text: {**calculate(text), "symmetry_number": 2}
    AIMNetReactionAssessor(
        AIMNetConfig(**changes.get(change, {})), cache, calculator=calculator
    ).assess([route("CCO>>CCO")])
    assert calculator.calls == ["CCO"]


def test_stereo_isotope_and_charge_cache_identities(tmp_path):
    calculator = Calculator()
    source = [
        route(f"{s}>>{s}")
        for s in ["C[C@H](O)F", "C[C@@H](O)F", "[13CH3]CO", "CCO", "CC[O-]"]
    ]
    service = AIMNetReactionAssessor(
        cache_path=tmp_path / "cache.json", calculator=calculator
    )
    service.assess(source)
    assert len(set(calculator.calls)) == 5


def test_expected_failure_not_cached_as_energy_and_retried(tmp_path):
    cache = tmp_path / "cache.json"
    fail = MolecularCalculationError("optimization", "not_converged", "test")
    calculator = Calculator(fail=fail)
    service = AIMNetReactionAssessor(cache_path=cache, calculator=calculator)
    result = service.assess([route("CCO>>CCO"), route("CCO>>CCO")])
    assert len(calculator.calls) == 1
    variant = result[0].steps[0].thermochemistry["variants"][0]
    assert variant["status"] == "failed"
    assert all(variant[k] is None for k in ("delta_e", "delta_h", "delta_g"))
    assert not cache.exists()
    calculator.fail = None
    assert (
        service.assess([route("CCO>>CCO")])[0].steps[0].thermochemistry["status"]
        == "ok"
    )


def test_unexpected_error_propagates_and_checkpoints_survive(tmp_path):
    class Broken(Calculator):
        def calculate(self, text):
            if text == "O":
                raise RuntimeError("model bug")
            return super().calculate(text)

    cache = tmp_path / "cache.json"
    with pytest.raises(RuntimeError, match="model bug"):
        AIMNetReactionAssessor(cache_path=cache, calculator=Broken()).assess([route()])
    assert json.loads(cache.read_text(encoding="utf-8"))["calculations"]


def test_partial_stereo_coverage_is_explicit():
    class Partial(Calculator):
        def calculate(self, text):
            if "@@" in text:
                raise MolecularCalculationError("generation", "stereo_mismatch", "test")
            return super().calculate(text)

    result = AIMNetReactionAssessor(cache_path=None, calculator=Partial()).assess(
        [route("CC(O)F>>CC(O)F")]
    )[0]
    annotation = result.steps[0].thermochemistry
    assert annotation["status"] == "partial"
    assert annotation["total_variants"] == 4
    assert annotation["successful_variants"] == 1
    assert result.assessment["aimnet"]["partial_steps"] == 1


def test_optional_dependencies_are_lazy():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'loqi', 'aimnet', 'ase'}:
            raise ImportError('blocked optional dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from chython import smiles
from synplan.chem.reaction.score.aimnet import AIMNetConfig, AIMNetReactionAssessor
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'config.yaml'
    path.write_text('device: cpu\\ntemperature: 310\\n', encoding='utf-8')
    config = AIMNetConfig.from_yaml(str(path))
service = AIMNetReactionAssessor(config, cache_path=None)
assert service.config.temperature == 310
assert service.assess_reactions([smiles('[Na+]>>[Na+]')])[0]['status'] == 'skipped'
try:
    service.assess_reactions([smiles('CCO>>CCO')])
except ImportError as exc:
    assert 'Python 3.11' in str(exc) or 'pip install' in str(exc)
else:
    raise AssertionError('Missing runtime must not silently produce results')
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_old_proton_cache_invalidated_without_discarding_other_species(tmp_path):
    from synplan.chem.reaction.score.aimnet.molecular import PROTON_REFERENCE

    class Exact(Calculator):
        def calculate(self, text):
            result = super().calculate(text)
            if text == "[H+]":
                result.update(energy_eV=0, electronic_energy_reference=PROTON_REFERENCE)
            return result

    path = tmp_path / "cache.json"
    old = Calculator()
    rows = [route("CCO>>CCO"), route("[H+]>>[H+]")]
    AIMNetReactionAssessor(cache_path=path, calculator=old).assess(rows)
    exact = Exact()
    fixed = AIMNetReactionAssessor(cache_path=path, calculator=exact).assess(rows)
    assert exact.calls == ["[H+]"]
    assert fixed[1].steps[0].thermochemistry["molecules"]["[H+]"]["energy_eV"] == 0
    exact.calls.clear()
    AIMNetReactionAssessor(cache_path=path, calculator=exact).assess(rows)
    assert exact.calls == []


def test_completed_results_survive_unexpected_failure(tmp_path):
    class Broken(Calculator):
        def calculate(self, text):
            if text == "O":
                raise RuntimeError("test failure")
            return super().calculate(text)

    service = AIMNetReactionAssessor(cache_path=None, calculator=Broken())
    checkpoints = []
    with pytest.raises(RuntimeError):
        service.assess(
            [route("CCO>>CCO"), route("O>>O")],
            on_reaction=lambda i, r: checkpoints.append((i, r)),
        )
    assert len(service.completed_reactions) == len(checkpoints) == 1
    assert checkpoints[0][0] == 0
    assert checkpoints[0][1]["status"] == "ok"
