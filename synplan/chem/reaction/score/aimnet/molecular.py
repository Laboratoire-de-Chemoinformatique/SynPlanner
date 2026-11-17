"""Lazy LoQI/AIMNet2/ASE adapter for one stereochemically specified species."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from importlib import import_module
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .config import AIMNetConfig, MolecularCalculationError, normalize_config
from .species import (
    canonical_species,
    is_proton,
    parse_species,
    validate_loqi_charges,
)

LOQI_MODEL = "loqi_flow_v1.1.0"
LOQI_COMMIT = "04f3caef4521b3fbcb236f8b01f4bc80c83cb918"
AIMNET_MODEL = "aimnet2"
PROTON_REFERENCE = "gas_phase_proton_zero_v1"
# CODATA conversion, also ASE's units.mol / units.kcal to rounding.
EV_TO_KCAL_MOL = 23.06054783061903


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def species_seed(smiles: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{smiles}".encode()).digest()[:4], "big"
    )


def _with_coordinates(mol, positions):
    result = Chem.Mol(mol)
    result.RemoveAllConformers()
    conformer = Chem.Conformer(result.GetNumAtoms())
    conformer.Set3D(True)
    for index, xyz in enumerate(positions):
        conformer.SetAtomPosition(index, tuple(map(float, xyz)))
    result.AddConformer(conformer, assignId=True)
    return result


def validate_geometry(mol, positions, *, optimized: bool) -> None:
    """Check coordinates independently of the labels carried by the input graph."""
    positions = np.asarray(positions, dtype=float)
    stage = "optimization" if optimized else "generation"
    if positions.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(positions).all():
        raise MolecularCalculationError(
            stage, "nonfinite_geometry", "Coordinates must be finite with shape (N, 3)."
        )
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    if np.any(distances < 0.4):
        raise MolecularCalculationError(
            stage, "collapsed_geometry", "An interatomic distance is below 0.4 Å."
        )
    # The AIMNet2 compressed-H defect concerns geminal H atoms. A real H-H
    # bond in H2 (~0.74 Å) must pass the pre-optimization 0.8 Å check.
    for atom in mol.GetAtoms():
        hydrogens = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        for i, left in enumerate(hydrogens):
            for right in hydrogens[i + 1 :]:
                if distances[left, right] < (0.6 if optimized else 0.8):
                    raise MolecularCalculationError(
                        stage,
                        "collapsed_hydrogens",
                        "Geminal hydrogens are too close for AIMNet2.",
                    )
    geometry = _with_coordinates(mol, positions)
    inferred = Chem.Mol(geometry)
    rdDetermineBonds.DetermineConnectivity(inferred, useVdw=True, covFactor=1.3)

    def edges(m):
        return {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in m.GetBonds()
        }

    if edges(inferred) != edges(mol):
        raise MolecularCalculationError(
            stage,
            "connectivity_changed",
            "Coordinates do not match the requested molecular connectivity.",
        )
    Chem.RemoveStereochemistry(geometry)
    Chem.AssignStereochemistryFrom3D(geometry, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(geometry, cleanIt=True, force=True)
    if canonical_species(geometry) != canonical_species(mol):
        raise MolecularCalculationError(
            stage,
            "stereo_mismatch",
            "3D stereochemistry does not match the requested stereoisomer.",
        )


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


class AIMNetMolecularCalculator:
    """Reuse loaded models, optimizing LoQI conformers with the gas-phase model.

    Expected chemical failures raise MolecularCalculationError. Missing packages,
    incompatible checkpoints, device failures and unexpected model errors propagate.
    """

    def __init__(self, config: AIMNetConfig):
        self.config = normalize_config(config)
        self._identity = None

    def identity(self) -> dict:
        if self._identity is not None:
            return self._identity
        if sys.version_info < (3, 11):
            raise ImportError(
                "AIMNet thermochemistry requires Python 3.11+; base SynPlanner remains usable on Python 3.10."
            )
        try:
            import loqi

            import_module("ase")
            from aimnet.calculators import AIMNet2Calculator
            from aimnet.calculators.model_registry import (
                get_model_path,
                load_model_registry,
            )
            from loqi.registry import MODELS, checkpoint_path
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError(
                "AIMNet thermochemistry needs SynPlanner[aimnet] (also in [cpu]/[all]) "
                f"and LoQI: pip install 'loqi @ git+https://github.com/isayevlab/LoQI@{LOQI_COMMIT}'"
            ) from exc
        aimnet_path = Path(
            get_model_path(
                str(self.config.aimnet_checkpoint.resolve())
                if self.config.aimnet_checkpoint
                else AIMNET_MODEL
            )
        )
        aimnet_digest = file_digest(aimnet_path)
        released = {
            item["sha256"]: name
            for name, item in load_model_registry()["models"].items()
            if item.get("family") == "wb97m-d3"
        }
        if aimnet_digest not in released:
            raise ValueError(
                "Use a released gas-phase AIMNet2 wB97M-D3 checkpoint for this workflow."
            )
        loqi_path = checkpoint_path(
            str(self.config.loqi_checkpoint or LOQI_MODEL), progress=False
        )
        loqi_digest = file_digest(loqi_path)
        if loqi_digest != MODELS[LOQI_MODEL].sha256:
            raise ValueError(
                "Use the released LoQI Flow v1.1.0 checkpoint for this workflow."
            )
        self._aimnet = AIMNet2Calculator(
            str(aimnet_path.resolve()),
            device=self.config.device,
            compile_model=False,
            cuda_graph=False,
        )
        self._loqi = loqi.load_model(
            loqi_path,
            config="loqi_flow.yaml",
            device=self.config.device,
            progress=False,
        )
        source = distribution("loqi").read_text("direct_url.json")
        self._identity = {
            "aimnet_model": released[aimnet_digest],
            "aimnet_sha256": aimnet_digest,
            "loqi_model": LOQI_MODEL,
            "loqi_sha256": loqi_digest,
            "loqi_source": json.loads(source) if source else None,
            "loqi_config": OmegaConf.to_container(self._loqi.config, resolve=True),
            "versions": {
                name: version(name)
                for name in (
                    "aimnet",
                    "loqi",
                    "torch",
                    "torch-geometric",
                    "lightning",
                    "rdkit",
                    "numpy",
                    "scipy",
                    "ase",
                    "warp-lang",
                    "nvalchemi-toolkit-ops",
                )
            },
        }
        return self._identity

    def _generate(self, smiles: str):
        import loqi
        import torch

        # LoQI seeds Python, NumPy, and Torch internally. Restore caller RNGs.
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(
                devices=[] if self.config.device == "cpu" else None
            ):
                return loqi.generate_conformers(
                    smiles,
                    self.config.n_conformers,
                    model=self._loqi,
                    seed=species_seed(smiles, self.config.seed),
                    steps=self.config.loqi_steps,
                    batch_atoms=self.config.batch_atoms,
                    add_hs=True,
                )[0]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def calculate(self, smiles: str) -> dict:
        expected = Chem.AddHs(parse_species(smiles))
        validate_loqi_charges(expected)
        self.identity()
        from aimnet.calculators import AIMNet2ASE
        from ase import Atoms
        from ase.optimize import BFGS

        if is_proton(smiles):
            # A nucleus with no electrons has exactly zero electronic energy.
            # Model inference on H+ introduces a spurious learned offset.
            atoms = Atoms(
                numbers=[1],
                positions=[[0, 0, 0]],
                masses=[expected.GetAtomWithIdx(0).GetMass()],
            )
            result = thermochemistry(
                atoms,
                0.0,
                np.zeros((3, 3)),
                self.config,
                self.config.symmetry_numbers.get(smiles, 1),
            )
            result.update(
                smiles=smiles,
                charge=1,
                conformer_id=0,
                optimization_steps=0,
                generated_conformers=1,
                requested_conformers=self.config.n_conformers,
                generation_failures=0,
                conformer_failures=[],
                symmetry_assumed=smiles not in self.config.symmetry_numbers,
                electronic_energy_reference=PROTON_REFERENCE,
            )
            return result
        if expected.GetNumAtoms() == 1:
            # A single atom has no internal geometry or conformational freedom.
            generated = _with_coordinates(expected, np.zeros((1, 3)))
        else:
            generated = self._generate(smiles)
        if canonical_species(generated) != canonical_species(expected):
            raise MolecularCalculationError(
                "generation",
                "identity_changed",
                "LoQI changed the requested molecular identity.",
            )
        charge = sum(a.GetFormalCharge() for a in generated.GetAtoms())
        numbers = [a.GetAtomicNum() for a in generated.GetAtoms()]
        masses = [a.GetMass() for a in generated.GetAtoms()]
        candidates, failures = [], []
        for conformer in generated.GetConformers():
            try:
                positions = conformer.GetPositions()
                validate_geometry(generated, positions, optimized=False)
                atoms = Atoms(numbers=numbers, positions=positions, masses=masses)
                atoms.calc = AIMNet2ASE(self._aimnet, charge=charge, mult=1)
                optimizer = BFGS(atoms, logfile=None)
                if not optimizer.run(
                    fmax=self.config.fmax, steps=self.config.max_steps
                ):
                    raise MolecularCalculationError(
                        "optimization",
                        "not_converged",
                        f"Exceeded {self.config.max_steps} optimization steps.",
                    )
                energy = float(atoms.get_potential_energy())
                if not np.isfinite(energy) or not np.isfinite(atoms.get_forces()).all():
                    raise MolecularCalculationError(
                        "optimization",
                        "nonfinite_energy",
                        "Nonfinite optimized energy or forces.",
                    )
                validate_geometry(generated, atoms.positions, optimized=True)
                candidates.append((energy, conformer.GetId(), atoms, optimizer.nsteps))
            except MolecularCalculationError as exc:
                failures.append({"conformer_id": conformer.GetId(), **exc.to_dict()})
            except np.linalg.LinAlgError as exc:
                failures.append(
                    {
                        "conformer_id": conformer.GetId(),
                        "stage": "optimization",
                        "reason": "numerical_failure",
                        "detail": str(exc),
                    }
                )
        for energy, conformer_id, atoms, nsteps in sorted(
            candidates, key=lambda row: (row[0], row[1])
        ):
            try:
                raw = self._aimnet(
                    {
                        "coord": atoms.positions,
                        "numbers": atoms.numbers,
                        "charge": charge,
                    },
                    hessian=True,
                )
                hessian = raw["hessian"].detach().cpu().numpy()
                result = thermochemistry(
                    atoms,
                    energy,
                    hessian,
                    self.config,
                    self.config.symmetry_numbers.get(smiles, 1),
                )
            except MolecularCalculationError as exc:
                failures.append({"conformer_id": conformer_id, **exc.to_dict()})
                continue
            result.update(
                {
                    "smiles": smiles,
                    "charge": charge,
                    "conformer_id": conformer_id,
                    "optimization_steps": nsteps,
                    "generated_conformers": generated.GetNumConformers(),
                    "requested_conformers": self.config.n_conformers,
                    "generation_failures": generated.GetIntProp("loqi_failed")
                    if generated.HasProp("loqi_failed")
                    else 0,
                    "conformer_failures": failures,
                    "symmetry_assumed": smiles not in self.config.symmetry_numbers,
                }
            )
            return result
        failure = MolecularCalculationError(
            "conformers",
            "no_valid_minimum",
            "No generated conformer passed optimization, geometry, stereo, and frequency checks.",
        )
        failure.conformer_failures = failures
        raise failure
