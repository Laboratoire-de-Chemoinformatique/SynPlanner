"""ASE forces and analytical Hessians for the LoQI-distributed CPCM-water TorchScript model."""

import torch
from ase.calculators.calculator import Calculator, all_changes


class WaterPotential:
    def __init__(self, path, device):
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.device = device

    def evaluate(self, positions, numbers, charge, *, hessian=False):
        xyz = torch.as_tensor(
            positions, device=self.device, dtype=torch.float32
        ).requires_grad_(True)
        atomic_numbers = torch.as_tensor(numbers, device=self.device, dtype=torch.long)[
            None
        ]
        total_charge = torch.tensor([charge], device=self.device, dtype=torch.float32)

        def energy(coords):
            return self.model(
                {
                    "coord": coords[None],
                    "numbers": atomic_numbers,
                    "charge": total_charge,
                }
            )["energy"].sum()

        with torch.enable_grad(), torch.jit.optimized_execution(False):
            if hessian:
                return (
                    torch.autograd.functional.hessian(energy, xyz)
                    .reshape(xyz.numel(), xyz.numel())
                    .detach()
                    .cpu()
                    .numpy()
                )
            value = energy(xyz)
            forces = -torch.autograd.grad(value, xyz)[0]
            return float(value.detach()), forces.detach().cpu().numpy()


class WaterASE(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, potential, charge):
        super().__init__()
        self.potential = potential
        self.charge = charge

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        energy, forces = self.potential.evaluate(
            self.atoms.positions, self.atoms.numbers, self.charge
        )
        self.results = {"energy": energy, "forces": forces}
