"""Opt-in real CPU/CUDA pipeline. Model downloads are never performed by tests.

Set SYNPLAN_AIMNET_MODEL and SYNPLAN_LOQI_MODEL to existing released checkpoint
files. The LoQI package must be installed separately at the documented revision.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from chython import smiles
from chython.containers import ReactionContainer

from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.aimnet.molecular import AIMNetMolecularCalculator


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
        aimnet_checkpoint=Path(paths[0]),
        loqi_checkpoint=Path(paths[1]),
        device=os.environ.get("SYNPLAN_AIMNET_DEVICE", "cpu"),
    )
    reactions = [
        "CC(=O)O.CCO>>CC(=O)OCC.O",  # Already balanced.
        "CC(=O)O.CCO>>CC(=O)OCC",  # Missing water; identical molecular cache.
        "CCC(C)=O.[H][H]>>CCC(C)O",  # Two product stereoisomers.
    ]
    routes = []
    for text in reactions:
        reaction = smiles(text)
        assert isinstance(reaction, ReactionContainer)
        routes.append(Route((Step(reaction, reaction.products[0]),)))
    cache = tmp_path_factory.mktemp("aimnet") / "cache.json"
    assessor = AIMNetReactionAssessor(config, cache)
    assessed = assessor.assess(routes)
    return config, cache, routes, assessed, assessor


def test_real_reactions_and_stereo(calculation):
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
        device=config.device,
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
    assert (
        np.linalg.norm(raw["forces"].detach().cpu().numpy(), axis=-1).max()
        <= config.fmax
    )
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


def test_real_serial_parallel_symmetry_override(calculation, tmp_path):
    from scripts.aimnet_workers import CPUMolecularWorkers

    config, *_ = calculation
    if config.device != "cpu":
        pytest.skip("The worker pool supports CPU only")
    import torch

    torch.set_num_threads(1)
    config = config.model_copy(update={"symmetry_numbers": {"[OH2]": 2}})
    reactions = [smiles("O.[H][H]>>O.[H][H]")]
    serial = AIMNetReactionAssessor(config, tmp_path / "serial.json")
    expected = serial.assess_reactions(reactions)
    parallel = CPUMolecularWorkers(config, tmp_path / "workers", workers=2, threads=1)
    try:
        assessor = AIMNetReactionAssessor(
            config, tmp_path / "parallel.json", calculator=parallel
        )
        assert parallel.prime(reactions, assessor.cache) == (2, 2)
        actual = assessor.assess_reactions(reactions)
    finally:
        parallel.close()
    assert actual[0]["status"] == expected[0]["status"] == "ok"
    for row in (actual[0], expected[0]):
        assert row["molecules"]["O"]["symmetry_number"] == 2
        assert row["calculation"]["config"]["symmetry_numbers"] == {"O": 2}
    for species in ("O", "[H][H]"):
        for key in ("energy_eV", "enthalpy_eV", "gibbs_eV"):
            assert actual[0]["molecules"][species][key] == pytest.approx(
                expected[0]["molecules"][species][key], abs=1e-6, rel=0
            )
    restored = AIMNetReactionAssessor(config, tmp_path / "parallel.json")
    restored.assess_reactions(reactions)
    assert restored.last_stats["molecular_calculations"] == 0


def test_cpcm_hessian_matches_finite_difference_forces():
    """Check the TorchScript-to-ASE derivative sign, units and atom/axis order."""
    if not os.environ.get("SYNPLAN_LOQI_MODEL"):
        pytest.skip("Set SYNPLAN_LOQI_MODEL for the opt-in CPCM check")
    from importlib.resources import files

    pytest.importorskip("loqi")
    from synplan.chem.reaction.score.aimnet.config import LOQI_CPCM_SHA256
    from synplan.chem.reaction.score.aimnet.cpcm import WaterPotential
    from synplan.chem.reaction.score.aimnet.molecular import file_digest

    path = Path(
        str(
            files("megalodon").joinpath(
                "metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt"
            )
        )
    )
    assert file_digest(path) == LOQI_CPCM_SHA256
    potential = WaterPotential(path, "cpu")
    numbers = [8, 1, 1]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    hessian = potential.evaluate(coordinates, numbers, 0, hessian=True)
    displacements = np.eye(9).reshape(9, 3, 3) * 0.002
    for index, delta in enumerate(displacements):
        _, high = potential.evaluate(coordinates + delta, numbers, 0)
        _, low = potential.evaluate(coordinates - delta, numbers, 0)
        assert hessian[:, index] == pytest.approx(
            ((low - high) / 0.004).ravel(), abs=0.08, rel=0.01
        )
