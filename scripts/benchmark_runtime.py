"""Lightweight, model-free execution identity for reproducible Linux benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import distribution, version
from pathlib import Path

FORMAT_VERSION = 2
PACKAGES = (
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


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def execution_identity(profile, runtime):
    """Exclude changing load, clocks and free memory from compatibility checks."""
    import torch

    cpuinfo = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
    stable_fields = {
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "microcode",
        "flags",
    }
    cpu = {}
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in stable_fields:
            cpu[key.strip()] = value.strip()
    topology = []
    for directory in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        folder = directory / "topology"
        topology.append(
            {
                "cpu": int(directory.name[3:]),
                "core": (folder / "core_id").read_text().strip(),
                "socket": (folder / "physical_package_id").read_text().strip(),
                "nodes": sorted(p.name for p in directory.glob("node[0-9]*")),
            }
        )
    identity = {
        "profile": profile,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "build": list(platform.python_build()),
            "executable": sys.executable,
        },
        "host": {
            "name": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu": cpu,
            "topology": topology,
        },
        "versions": {name: version(name) for name in PACKAGES},
        "package_origins": {
            name: distribution(name).read_text("direct_url.json") for name in PACKAGES
        },
        "runtime": {
            key: runtime[key]
            for key in (
                "affinity",
                "torch_threads",
                "torch_interop_threads",
                "thread_environment",
                "torch",
                "cuda_runtime",
            )
        },
        "cudnn": torch.backends.cudnn.version(),
        "precision": {
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul": torch.get_float32_matmul_precision(),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }
    if profile["device"].startswith("cuda"):
        device = torch.cuda.get_device_properties(profile["device"])
        identity["gpu"] = {
            "uuid": str(device.uuid),
            "name": device.name,
            "memory_bytes": device.total_memory,
            "compute_capability": [device.major, device.minor],
            # Works without NVML, including hosts with mismatched NVML libraries.
            "driver": Path("/proc/driver/nvidia/version").read_text().splitlines()[0],
        }
    return identity


def probe_runtime(profile, repo, destination):
    """Use the selected interpreter; never import calculators or load weights."""
    from scripts.benchmark_thermochemistry import THREAD_VARIABLES

    destination = Path(destination)
    environment = {**os.environ, "PYTHONPATH": str(repo)}
    for name in THREAD_VARIABLES:
        environment[name] = str(profile["threads"])
    result = subprocess.run(
        [
            profile["python"],
            "-m",
            "scripts.benchmark_runtime",
            json.dumps(profile),
            str(destination),
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Runtime preflight failed for {profile['id']}: {result.stderr[-4000:]}"
        )
    return json.loads(destination.read_text())


def main():
    from scripts.benchmark_thermochemistry import configure_runtime

    profile = json.loads(sys.argv[1])
    synchronize, runtime = configure_runtime(profile)
    synchronize()
    Path(sys.argv[2]).write_text(
        json.dumps(execution_identity(profile, runtime), indent=2)
    )


if __name__ == "__main__":
    main()
