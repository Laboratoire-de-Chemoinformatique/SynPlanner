"""Lazy LoQI/AIMNet2/ASE adapter for one stereochemically specified species."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import import_module
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
from rdkit import Chem

from .config import (
    LOQI_CPCM_SHA256,
    AIMNetConfig,
    MolecularCalculationError,
    normalize_config,
)
from .conformers import LoQIGenerator
from .geometry import _with_coordinates, matches_stereo, validate_geometry
from .species import (
    is_proton,
    parse_species,
    validate_loqi_charges,
)
from .vibrations import thermochemistry

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


class AIMNetMolecularCalculator:
    """Reuse loaded models for conformer generation, optimization and Hessians.

    Expected chemical failures raise MolecularCalculationError. Missing packages,
    incompatible checkpoints, device failures and unexpected model errors propagate.
    """

    def __init__(self, config: AIMNetConfig):
        self.config = normalize_config(config)
        self.generator = LoQIGenerator(self.config)
        self._identity = None
        self._water = None

    def identity(self) -> dict:
        if self._identity is not None:
            return self._identity
        if sys.version_info < (3, 11):
            raise ImportError(
                "AIMNet thermochemistry requires Python 3.11+; base SynPlanner remains usable on Python 3.10."
            )
        try:
            import_module("loqi")
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
        if self.config.phase == "gas":
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
            model_name = released[aimnet_digest]
        else:
            from importlib.resources import files

            checkpoint = self.config.cpcm_checkpoint
            if checkpoint is None:
                checkpoint = Path(
                    str(
                        files("megalodon").joinpath(
                            "metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt"
                        )
                    )
                )
            aimnet_path = checkpoint.resolve(strict=True)
            aimnet_digest = file_digest(aimnet_path)
            if aimnet_digest != self.config.cpcm_sha256:
                raise ValueError(
                    "CPCM-water checkpoint does not match the configured SHA-256"
                )
            if any(
                entry.get("sha256") == aimnet_digest
                for entry in load_model_registry()["models"].values()
            ):
                raise ValueError(
                    "A gas-phase or other registry model cannot serve as the CPCM-water checkpoint"
                )
            model_name = "cpcm_water"
        loqi_path = checkpoint_path(
            str(self.config.loqi_checkpoint or LOQI_MODEL), progress=False
        )
        loqi_digest = file_digest(loqi_path)
        if loqi_digest != MODELS[LOQI_MODEL].sha256:
            raise ValueError(
                "Use the released LoQI Flow v1.1.0 checkpoint for this workflow."
            )
        if self.config.phase == "cpcm_water" and aimnet_path.suffix == ".jpt":
            from .cpcm import WaterPotential

            if aimnet_digest != LOQI_CPCM_SHA256:
                raise ValueError(
                    "Unknown CPCM-water TorchScript format; use a compatible eager AIMNet checkpoint"
                )
            self._water = WaterPotential(aimnet_path, self.config.device)
        else:
            self._aimnet = AIMNet2Calculator(
                str(aimnet_path.resolve()),
                device=self.config.device,
                compile_model=False,
                cuda_graph=False,
            )
        loqi_model = self.generator.load(loqi_path)
        source = distribution("loqi").read_text("direct_url.json")
        self._identity = {
            "aimnet_model": model_name,
            "aimnet_sha256": aimnet_digest,
            "loqi_model": LOQI_MODEL,
            "loqi_sha256": loqi_digest,
            "loqi_source": json.loads(source) if source else None,
            "loqi_config": OmegaConf.to_container(loqi_model.config, resolve=True),
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
        return self.generator.generate(smiles, species_seed(smiles, self.config.seed))

    def _embed_rdkit(self, smiles):
        from rdkit.Chem import AllChem

        molecule = Chem.AddHs(parse_species(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = species_seed(smiles, self.config.seed) % (2**31 - 1)
        AllChem.EmbedMultipleConfs(
            molecule, numConfs=self.config.n_conformers, params=params
        )
        molecule.SetProp("conformer_source", "ETKDGv3")
        return molecule

    def calculate(self, smiles: str) -> dict:
        expected = Chem.AddHs(parse_species(smiles))
        validate_loqi_charges(expected)
        self.identity()
        from ase import Atoms

        if is_proton(smiles) and self.config.phase != "gas":
            raise MolecularCalculationError(
                "reference",
                "aqueous_proton_reference_required",
                "The aqueous proton requires a single-ion solvation convention.",
            )
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
        elif (
            self.config.etkdg_max_heavy_atoms
            and expected.GetNumHeavyAtoms() <= self.config.etkdg_max_heavy_atoms
        ):
            generated = self._embed_rdkit(smiles)
        else:
            generated = self._generate(smiles)
        if not matches_stereo(
            generated,
            expected,
            self.config.stereo_policy == "conformer",
            self.config.allow_mirror_image,
        ):
            raise MolecularCalculationError(
                "generation",
                "identity_changed",
                "LoQI changed the requested molecular identity.",
            )
        return self._optimize_pool(
            smiles,
            generated,
            allow_reseed=self.config.etkdg_fallback
            and not generated.HasProp("conformer_source"),
        )

    def _optimize_pool(self, smiles, generated, *, allow_reseed):
        from aimnet.calculators import AIMNet2ASE
        from ase import Atoms
        from ase.optimize import BFGS, LBFGS

        charge = sum(a.GetFormalCharge() for a in generated.GetAtoms())
        numbers = [a.GetAtomicNum() for a in generated.GetAtoms()]
        masses = [a.GetMass() for a in generated.GetAtoms()]
        candidates, failures = [], []
        seeds = []
        for conformer in generated.GetConformers():
            try:
                positions = conformer.GetPositions()
                validate_geometry(
                    generated,
                    positions,
                    optimized=False,
                    conformer_stereo=self.config.stereo_policy == "conformer",
                    allow_mirror=self.config.allow_mirror_image,
                )
                atoms = Atoms(numbers=numbers, positions=positions, masses=masses)
                if self._water is None:
                    atoms.calc = AIMNet2ASE(self._aimnet, charge=charge, mult=1)
                else:
                    from .cpcm import WaterASE

                    atoms.calc = WaterASE(self._water, charge)
                score = (
                    atoms.get_potential_energy()
                    if self.config.conformer_selection == "lowest_seed_energy"
                    else 0.0
                )
                if not np.isfinite(score):
                    raise MolecularCalculationError(
                        "generation", "nonfinite_energy", "Nonfinite seed energy."
                    )
                seeds.append((score, conformer.GetId(), atoms))
            except MolecularCalculationError as exc:
                failures.append({"conformer_id": conformer.GetId(), **exc.to_dict()})
        valid_seed_count = len(seeds)
        if self.config.conformer_selection == "lowest_seed_energy" and seeds:
            seeds = [min(seeds, key=lambda x: (x[0], x[1]))]
        for _, conformer_id, atoms in seeds:
            try:
                optimizer_class = LBFGS if self.config.optimizer == "LBFGS" else BFGS
                optimizer = optimizer_class(atoms, logfile=None)
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
                selected_stereo = validate_geometry(
                    generated,
                    atoms.positions,
                    optimized=True,
                    conformer_stereo=self.config.stereo_policy == "conformer",
                    allow_mirror=self.config.allow_mirror_image,
                )
                candidates.append(
                    (energy, conformer_id, atoms, optimizer.nsteps, selected_stereo)
                )
            except MolecularCalculationError as exc:
                failures.append({"conformer_id": conformer_id, **exc.to_dict()})
            except np.linalg.LinAlgError as exc:
                failures.append(
                    {
                        "conformer_id": conformer_id,
                        "stage": "optimization",
                        "reason": "numerical_failure",
                        "detail": str(exc),
                    }
                )
        for energy, conformer_id, atoms, nsteps, selected_stereo in sorted(
            candidates, key=lambda row: (row[0], row[1])
        ):
            try:
                if self._water is not None:
                    hessian = self._water.evaluate(
                        atoms.positions, atoms.numbers, charge, hessian=True
                    )
                else:
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
                    "selected_stereoisomer": selected_stereo,
                    "valid_seed_count": valid_seed_count,
                    "conformer_source": generated.GetProp("conformer_source")
                    if generated.HasProp("conformer_source")
                    else ("atomic" if len(atoms) == 1 else LOQI_MODEL),
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
        if allow_reseed and generated.GetNumAtoms() > 1:
            reseeded = self._embed_rdkit(smiles)
            if reseeded.GetNumConformers():
                try:
                    result = self._optimize_pool(smiles, reseeded, allow_reseed=False)
                except MolecularCalculationError as exc:
                    exc.conformer_failures = failures + exc.conformer_failures
                    raise
                result["conformer_source"] = "ETKDGv3"
                result["loqi_conformer_failures"] = failures
                return result
        failure = MolecularCalculationError(
            "conformers",
            "no_valid_minimum",
            "No generated conformer passed optimization, geometry, stereo, and frequency checks.",
        )
        failure.conformer_failures = failures
        raise failure
