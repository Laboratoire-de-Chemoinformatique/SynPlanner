"""Geometry and thermodynamic checks independent of learned model predictions."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from synplan.chem.reaction.score.config import AIMNetConfig, MolecularCalculationError
from synplan.chem.reaction.score.molecular import (
    AIMNetMolecularCalculator,
    species_seed,
    thermochemistry,
    validate_geometry,
)


def embedded(text):
    mol = Chem.AddHs(Chem.MolFromSmiles(text))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol, mol.GetConformer().GetPositions()


def test_coordinate_stereo_checks_do_not_trust_graph_labels():
    mol, xyz = embedded("C[C@H](O)F")
    validate_geometry(mol, xyz, optimized=False)
    mirrored = xyz.copy()
    mirrored[:, 0] *= -1
    with pytest.raises(MolecularCalculationError) as exc:
        validate_geometry(mol, mirrored, optimized=True)
    assert exc.value.reason == "stereo_mismatch"


def test_coordinate_ez_stereo_is_verified():
    mol, xyz = embedded("F/C=C/Cl")
    validate_geometry(mol, xyz, optimized=False)
    other, other_xyz = embedded("F/C=C\\Cl")
    # Atom order agrees; bond stereo on mol must not hide the Z coordinates.
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
        a.GetAtomicNum() for a in other.GetAtoms()
    ]
    with pytest.raises(MolecularCalculationError) as exc:
        validate_geometry(mol, other_xyz, optimized=True)
    assert exc.value.reason == "stereo_mismatch"


def test_nonfinite_collapsed_and_changed_connectivity():
    mol, xyz = embedded("CCO")
    for positions, reason in [
        (np.full_like(xyz, np.nan), "nonfinite_geometry"),
        (np.zeros_like(xyz), "collapsed_geometry"),
        (xyz * 10, "connectivity_changed"),
    ]:
        with pytest.raises(MolecularCalculationError) as exc:
            validate_geometry(mol, positions, optimized=True)
        assert exc.value.reason == reason


def test_hydrogen_collapse_check_exempts_h2():
    hydrogen = Chem.MolFromSmiles("[H][H]")
    validate_geometry(hydrogen, [[0, 0, 0], [0.74, 0, 0]], optimized=False)
    methane, xyz = embedded("C")
    # Two geminal Hs 0.7 Å apart must fail before optimization.
    xyz[1], xyz[2] = [1.0, 0.0, 0.0], [1.0, 0.7, 0.0]
    with pytest.raises(MolecularCalculationError) as exc:
        validate_geometry(methane, xyz, optimized=False)
    assert exc.value.reason == "collapsed_hydrogens"


@pytest.fixture
def ase():
    pytest.importorskip("aimnet.calculators.vibrations")
    return pytest.importorskip("ase")


def test_linear_oscillator_matches_independent_rrho_and_isotope_mass(ase):
    from ase import units
    from ase.thermochemistry import IdealGasThermo

    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    hessian = np.zeros((6, 6))
    hessian[0, 0] = hessian[3, 3] = 1
    hessian[0, 3] = hessian[3, 0] = -1
    config = AIMNetConfig()
    result = thermochemistry(atoms, -1.0, hessian, config, 2)
    assert result["geometry"] == "linear"
    assert len(result["frequencies_cm1"]) == 1
    # Independent diatomic oscillator: omega² = k * (1/m1 + 1/m2).
    omega = np.sqrt(2 / atoms.get_masses()[0] * units._e / (units._amu * 1e-20))
    quantum = units._hbar * omega / units._e
    reference = IdealGasThermo(
        vib_energies=[quantum],
        potentialenergy=-1.0,
        atoms=atoms,
        geometry="linear",
        symmetrynumber=2,
        spin=0,
    )
    assert result["enthalpy_eV"] == pytest.approx(
        reference.get_enthalpy(config.temperature, verbose=False), rel=1e-8
    )
    assert result["gibbs_eV"] == pytest.approx(
        reference.get_gibbs_energy(config.temperature, config.pressure, verbose=False),
        rel=1e-8,
    )
    deuterium = atoms.copy()
    deuterium.set_masses([2.014101778, 2.014101778])
    heavy = thermochemistry(deuterium, -1.0, hessian, config, 2)
    assert heavy["frequencies_cm1"][0] / result["frequencies_cm1"][0] == pytest.approx(
        np.sqrt(atoms.get_masses()[0] / 2.014101778)
    )


def test_monatomic_nonlinear_modes_and_symmetry_correction(ase):
    from ase.units import kB

    config = AIMNetConfig()
    mono = ase.Atoms("F", positions=[[0, 0, 0]])
    mono_result = thermochemistry(mono, 0.0, np.zeros((3, 3)), config, 1)
    assert mono_result["frequencies_cm1"] == []
    assert mono_result["geometry"] == "monatomic"
    water = ase.Atoms("OH2", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
    result = thermochemistry(water, 0.0, np.eye(9), config, 1)
    symmetric = thermochemistry(water, 0.0, np.eye(9), config, 2)
    assert len(result["frequencies_cm1"]) == 3
    assert symmetric["gibbs_eV"] - result["gibbs_eV"] == pytest.approx(
        kB * config.temperature * np.log(2)
    )


@pytest.mark.parametrize("scale", [-1.0, 0.0, float("nan")])
def test_invalid_modes_are_not_abs_or_discarded(ase, scale):
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    with pytest.raises(MolecularCalculationError):
        thermochemistry(atoms, 0.0, np.eye(6) * scale, AIMNetConfig(), 1)


def test_species_seed_stable_and_stereo_sensitive():
    assert species_seed("C[C@H](O)F", 42) == species_seed("C[C@H](O)F", 42)
    assert (
        len(
            {species_seed(s, n) for s in ("C[C@H](O)F", "C[C@@H](O)F") for n in (0, 42)}
        )
        == 4
    )


def test_optimizer_nonconvergence_is_reported(monkeypatch, ase):
    from ase.optimize import BFGS

    mol, _ = embedded("CCO")
    backend = AIMNetMolecularCalculator(AIMNetConfig())
    monkeypatch.setattr(backend, "identity", lambda: {})
    monkeypatch.setattr(backend, "_generate", lambda smiles: mol)
    backend._aimnet = type(
        "Fake", (), {"is_nse": False, "device": "cpu", "model": None}
    )()
    monkeypatch.setattr(BFGS, "run", lambda *a, **kw: False)
    with pytest.raises(MolecularCalculationError) as exc:
        backend.calculate("CCO")
    assert exc.value.reason == "no_valid_minimum"
    assert exc.value.conformer_failures[0]["reason"] == "not_converged"


@pytest.mark.parametrize("text", ["[H+]", "[2H+]", "[3H+]"])
def test_proton_uses_exact_electronic_reference_and_monatomic_thermo(
    monkeypatch, ase, text
):
    from ase.thermochemistry import IdealGasThermo
    from ase.units import kB

    from synplan.chem.reaction.score.molecular import PROTON_REFERENCE

    backend = AIMNetMolecularCalculator(AIMNetConfig())
    monkeypatch.setattr(backend, "identity", lambda: {})

    def unexpected(*args, **kwargs):
        pytest.fail("A proton must not use LoQI or model evaluation")

    monkeypatch.setattr(backend, "_generate", unexpected)
    backend._aimnet = unexpected
    result = backend.calculate(text)
    assert result["energy_eV"] == 0
    assert result["electronic_energy_reference"] == PROTON_REFERENCE
    assert result["frequencies_cm1"] == []
    mass = Chem.MolFromSmiles(text).GetAtomWithIdx(0).GetMass()
    assert result["masses_amu"] == [mass]
    reference = IdealGasThermo(
        vib_energies=[],
        potentialenergy=0,
        geometry="monatomic",
        atoms=ase.Atoms(numbers=[1], positions=[[0, 0, 0]], masses=[mass]),
        symmetrynumber=1,
        spin=0,
    )
    assert result["enthalpy_eV"] == pytest.approx(2.5 * kB * 298.15)
    assert result["gibbs_eV"] == pytest.approx(
        reference.get_gibbs_energy(298.15, 101325, verbose=False)
    )
