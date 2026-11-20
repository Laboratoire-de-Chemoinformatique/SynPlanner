"""Projected harmonic thermochemistry."""

import numpy as np

from .config import AIMNetConfig, MolecularCalculationError


def thermochemistry(
    atoms, energy: float, hessian, config: AIMNetConfig, symmetry: int
) -> dict:
    """Projected harmonic modes and one-conformer ideal-gas E/H/G, in eV."""
    from aimnet.calculators.vibrations import analyze_hessian
    from ase.thermochemistry import IdealGasThermo

    hessian = np.asarray(hessian, dtype=float)
    if not np.isfinite(hessian).all() or not np.isfinite(energy):
        raise MolecularCalculationError(
            "frequencies", "nonfinite_hessian", "Energy and Hessian must be finite."
        )
    try:
        vibrations = analyze_hessian(
            hessian, atoms.get_positions(), atoms.get_masses(), project_tr=True
        )
    except np.linalg.LinAlgError as exc:
        raise MolecularCalculationError(
            "frequencies", "diagonalization_failed", str(exc)
        ) from exc
    frequencies = np.asarray(vibrations.frequencies_cm1)
    expected = (
        0 if len(atoms) == 1 else 3 * len(atoms) - (5 if vibrations.is_linear else 6)
    )
    if len(frequencies) != expected:
        raise MolecularCalculationError(
            "frequencies",
            "mode_count",
            "Projected vibrational mode count is inconsistent with molecular geometry.",
        )
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise MolecularCalculationError(
            "frequencies",
            "not_a_minimum",
            "Imaginary/nonpositive vibrational modes; no modes were discarded or made positive.",
        )
    geometry = (
        "monatomic"
        if len(atoms) == 1
        else ("linear" if vibrations.is_linear else "nonlinear")
    )
    thermo = IdealGasThermo(
        vib_energies=vibrations.energies_ev,
        potentialenergy=energy,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry,
        spin=0,
        ignore_imag_modes=False,
    )
    enthalpy = float(thermo.get_enthalpy(config.temperature, verbose=False))
    gibbs = float(
        thermo.get_gibbs_energy(config.temperature, config.pressure, verbose=False)
    )
    if not np.isfinite([energy, enthalpy, gibbs]).all():
        raise MolecularCalculationError(
            "thermochemistry", "nonfinite_thermochemistry", "Nonfinite E/H/G."
        )
    return {
        "energy_eV": float(energy),
        "enthalpy_eV": enthalpy,
        "gibbs_eV": gibbs,
        "frequencies_cm1": frequencies.tolist(),
        "geometry": geometry,
        "symmetry_number": symmetry,
        "spin": 0,
        "temperature_K": config.temperature,
        "pressure_Pa": config.pressure,
        "atomic_numbers": atoms.numbers.tolist(),
        "masses_amu": atoms.get_masses().tolist(),
        "coordinates_angstrom": atoms.get_positions().tolist(),
    }
