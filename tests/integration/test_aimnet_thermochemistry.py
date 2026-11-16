"""Opt-in real CPU pipeline. Model downloads are never performed by tests.

Set SYNPLAN_AIMNET_MODEL and SYNPLAN_LOQI_MODEL to existing released checkpoint
files. The LoQI package must be installed separately at the documented revision.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from chython import smiles

from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.molecular import AIMNetMolecularCalculator


@pytest.fixture(scope="module")
def calculation(tmp_path_factory):
    paths = [
        os.environ.get(name) for name in ("SYNPLAN_AIMNET_MODEL", "SYNPLAN_LOQI_MODEL")
    ]
    if not all(paths):
        pytest.skip(
            "Set SYNPLAN_AIMNET_MODEL and SYNPLAN_LOQI_MODEL to local checkpoints"
        )
    if not all(Path(path).is_file() for path in paths):
        pytest.fail("Configured AIMNet/LoQI checkpoints must exist locally")
    pytest.importorskip("loqi")
    pytest.importorskip("aimnet")
    config = AIMNetConfig(
        aimnet_checkpoint=Path(paths[0]), loqi_checkpoint=Path(paths[1]), device="cpu"
    )
    reactions = [
        "CC(=O)O.CCO>>CC(=O)OCC.O",  # Already balanced.
        "CC(=O)O.CCO>>CC(=O)OCC",  # Missing water; identical molecular cache.
        "CCC(C)=O.[H][H]>>CCC(C)O",  # Two product stereoisomers.
    ]
    routes = []
    for text in reactions:
        reaction = smiles(text)
        routes.append(Route((Step(reaction, reaction.products[0]),)))
    cache = tmp_path_factory.mktemp("aimnet") / "cache.json"
    assessor = AIMNetReactionAssessor(config, cache)
    assessed = assessor.assess(routes)
    return config, cache, routes, assessed, assessor


def test_real_cpu_reactions_and_stereo(calculation):
    _, _, _, assessed, assessor = calculation
    annotations = [route.steps[0].thermochemistry for route in assessed]
    assert [r["status"] for r in annotations] == ["ok", "ok", "ok"]
    assert annotations[1]["inferred_species"]["products"] == [
        {"smiles": "O", "count": 1}
    ]
    assert annotations[0]["variants"] == annotations[1]["variants"]
    assert annotations[2]["total_variants"] == 2
    assert assessor.last_stats["molecular_calculations"] == 8
    for row in annotations:
        for variant in row["variants"]:
            assert np.isfinite(
                [variant[k] for k in ("delta_e", "delta_h", "delta_g")]
            ).all()
        assert all(m["requested_conformers"] == 10 for m in row["molecules"].values())


def test_selected_minimum_matches_direct_aimnet_ase(calculation):
    from aimnet.calculators import AIMNet2Calculator
    from aimnet.calculators.vibrations import analyze_hessian
    from ase import Atoms
    from ase.thermochemistry import IdealGasThermo

    config, _, _, assessed, _ = calculation
    molecule = assessed[0].steps[0].thermochemistry["molecules"]["CCO"]
    atoms = Atoms(
        numbers=molecule["atomic_numbers"],
        positions=molecule["coordinates_angstrom"],
        masses=molecule["masses_amu"],
    )
    calc = AIMNet2Calculator(
        str(config.aimnet_checkpoint),
        device="cpu",
        compile_model=False,
        cuda_graph=False,
    )
    raw = calc(
        {
            "coord": atoms.positions,
            "numbers": atoms.numbers,
            "charge": molecule["charge"],
        },
        forces=True,
        hessian=True,
    )
    energy = float(raw["energy"].item())
    assert np.linalg.norm(raw["forces"].detach().numpy(), axis=-1).max() <= config.fmax
    modes = analyze_hessian(raw["hessian"], atoms.positions, atoms.get_masses())
    assert modes.n_imaginary == 0
    direct = IdealGasThermo(
        vib_energies=modes.energies_ev,
        potentialenergy=energy,
        atoms=atoms,
        geometry="linear" if modes.is_linear else "nonlinear",
        symmetrynumber=molecule["symmetry_number"],
        spin=0,
    )
    assert molecule["energy_eV"] == pytest.approx(energy, abs=1e-5)
    assert molecule["enthalpy_eV"] == pytest.approx(
        direct.get_enthalpy(config.temperature, verbose=False), abs=1e-5, rel=0
    )
    assert molecule["gibbs_eV"] == pytest.approx(
        direct.get_gibbs_energy(config.temperature, config.pressure, verbose=False),
        abs=1e-5,
        rel=0,
    )


def test_reopened_cache_performs_no_calculations(calculation, monkeypatch):
    config, cache, routes, original, _ = calculation
    calculator = AIMNetMolecularCalculator(config)

    def unexpected(smiles):
        pytest.fail(f"Cached species unexpectedly recalculated: {smiles}")

    monkeypatch.setattr(calculator, "calculate", unexpected)
    reopened = AIMNetReactionAssessor(config, cache, calculator=calculator)
    restored = reopened.assess(routes)
    assert reopened.last_stats["molecular_calculations"] == 0
    assert [r.to_json() for r in restored] == [r.to_json() for r in original]
