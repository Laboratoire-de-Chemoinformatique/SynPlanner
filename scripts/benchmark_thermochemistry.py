#!/usr/bin/env python3
"""Measure cold-result-cache reaction latency on pinned CPUs and CUDA.

Model loading and warm-up are separate. Each reaction/repetition gets a new
assessor and new molecular cache. Only completed trial records may be resumed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch
from uuid import uuid4

from scripts.benchmark_runtime import (
    FORMAT_VERSION,
    PACKAGES,
    execution_identity,
    probe_runtime,
)

PROCESS_STARTED = perf_counter_ns()
CASES = (
    ("3", 1, "Apatinib substitution; proton and chloride"),
    ("376", 1, "Already-balanced nitrile hydration"),
    ("4", 1, "Amidation; water completion"),
    ("453", 2, "Larger protected intermediate; water completion"),
    ("654", 1, "Two stereochemical variants"),
    ("654", 2, "Four stereochemical variants"),
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def interpreter_path(path):
    """Keep the venv entry point: resolving its symlink bypasses site-packages."""
    return str(Path(path).expanduser().absolute())


def read_json(path):
    return json.loads(Path(path).read_text())


def physical_cpus(count, allowed=None, topology_root=Path("/sys/devices/system/cpu")):
    """Choose one thread per physical core, all on one socket/NUMA node."""
    allowed = sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    groups = {}
    for cpu in allowed:
        folder = topology_root / f"cpu{cpu}"
        topology = folder / "topology"
        socket = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        nodes = sorted(folder.glob("node[0-9]*"))
        node = nodes[0].name if nodes else str(socket)
        groups.setdefault((socket, node), {}).setdefault(core, cpu)
    for _, cores in sorted(groups.items()):
        if len(cores) >= count > 0:
            return sorted(cores.values())[:count]
    raise ValueError(
        f"Cannot allocate {count} distinct physical cores on one NUMA node"
    )


def configure_runtime(profile):
    os.sched_setaffinity(0, profile["cpus"])
    for name in THREAD_VARIABLES:
        os.environ[name] = str(profile["threads"])
    import torch

    torch.set_num_threads(profile["threads"])
    torch.set_num_interop_threads(1)
    if profile["device"].startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected GPU environment")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def synchronize():
            torch.cuda.synchronize(profile["device"])
    else:

        def synchronize():
            return None

    metadata = {
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "load_average": list(os.getloadavg()),
    }
    if profile["device"].startswith("cuda"):
        device = torch.cuda.get_device_properties(profile["device"])
        metadata["gpu"] = {
            "name": device.name,
            "memory_bytes": device.total_memory,
            "compute_capability": [device.major, device.minor],
            "free_total_bytes": list(torch.cuda.mem_get_info()),
        }
    return synchronize, metadata


class TimedCalculator:
    """Synchronize GPU work and measure only the actual molecular call."""

    def __init__(self, backend, synchronize=lambda: None, clock=perf_counter_ns):
        self.backend, self.synchronize, self.clock = backend, synchronize, clock
        self.nanoseconds = 0
        self.calls = []

    def identity(self):
        return self.backend.identity()

    def calculate(self, smiles):
        self.synchronize()
        started = self.clock()
        try:
            result = self.backend.calculate(smiles)
        except BaseException as exc:
            try:
                self.synchronize()
            except Exception as sync_error:
                exc.add_note(f"CUDA synchronization also failed: {sync_error}")
            duration = self.clock() - started
            self.nanoseconds += duration
            self.calls.append(
                {"smiles": smiles, "status": "failed", "seconds": duration / 1e9}
            )
            raise
        self.synchronize()
        duration = self.clock() - started
        self.nanoseconds += duration
        self.calls.append({"smiles": smiles, "status": "ok", "seconds": duration / 1e9})
        return result


class TimedWriter:
    """Time the production atomic writer, including JSON serialization."""

    def __init__(self, writer, clock=perf_counter_ns):
        self.writer, self.clock = writer, clock
        self.nanoseconds = 0
        self.calls = 0

    def __call__(self, path, data):
        started = self.clock()
        try:
            return self.writer(path, data)
        finally:
            self.nanoseconds += self.clock() - started
            self.calls += 1


def run_trial(
    case, config, backend, directory, synchronize=lambda: None, *, execution=None
):
    from chython import smiles

    from synplan.chem.reaction.score import AIMNetReactionAssessor
    from synplan.utils.files import write_json_atomic

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (directory / f"attempt-{attempt:03}").exists():
        attempt += 1
    attempt_path = directory / f"attempt-{attempt:03}"
    attempt_path.mkdir()
    cache = attempt_path / "cache.json"
    result_path = attempt_path / "thermochemistry.json"
    reaction = smiles(case["original_reaction"])
    calculator = TimedCalculator(backend, synchronize)
    cache_writer, result_writer = (
        TimedWriter(write_json_atomic),
        TimedWriter(write_json_atomic),
    )
    record = {
        **(execution or {}),
        "case_id": case["id"],
        "route_id": case["route_id"],
        "step": case["step"],
        "attempt": attempt,
        "status": "errored",
        "result_path": str(result_path),
    }
    started = perf_counter_ns()
    try:
        with patch(
            "synplan.chem.reaction.score.aimnet.write_json_atomic", cache_writer
        ):
            assessor = AIMNetReactionAssessor(config, cache, calculator=calculator)
            assert not assessor.cache["calculations"]
            annotation = assessor.assess_reactions([reaction])[0]
        if annotation["status"] == "ok":
            assert set(annotation["molecules"]) == set(case["expected_species"])
            assert len(annotation["variants"]) == case["expected_variants"]
            assert assessor.last_stats["molecular_calculations"] == len(
                case["expected_species"]
            )
            assert assessor.last_stats["cache_hits"] == 0
        result_writer(result_path, annotation)
    except BaseException as exc:
        finished = perf_counter_ns()
        record["error"] = {"type": type(exc).__name__, "detail": str(exc)}
        (attempt_path / "error.log").write_text(traceback.format_exc())
        raise
    else:
        finished = perf_counter_ns()
        record.update(
            status=annotation["status"],
            successful_variants=annotation.get("successful_variants", 0),
            total_variants=len(annotation["variants"]),
            molecular_calculations=assessor.last_stats["molecular_calculations"],
            cache_hits=assessor.last_stats["cache_hits"],
            variant_values=[
                {
                    k: v.get(k)
                    for k in (
                        "id",
                        "reaction",
                        "status",
                        "delta_e",
                        "delta_h",
                        "delta_g",
                        "failures",
                    )
                }
                for v in annotation["variants"]
            ],
            conformer_outcomes={
                s: {
                    k: m.get(k)
                    for k in (
                        "generated_conformers",
                        "optimization_steps",
                        "conformer_id",
                        "conformer_failures",
                    )
                }
                for s, m in annotation["molecules"].items()
            },
            balanced_reaction=annotation.get("balanced_reaction"),
        )
    finally:
        total = finished - started
        measured = (
            calculator.nanoseconds
            + cache_writer.nanoseconds
            + result_writer.nanoseconds
        )
        record.update(
            calculation_s=calculator.nanoseconds / 1e9,
            cache_write_s=cache_writer.nanoseconds / 1e9,
            thermo_write_s=result_writer.nanoseconds / 1e9,
            other_s=(total - measured) / 1e9,
            total_s=total / 1e9,
            cache_writes=cache_writer.calls,
            thermo_writes=result_writer.calls,
            thermo_bytes=result_path.stat().st_size if result_path.exists() else 0,
            molecular_calls=calculator.calls,
        )
        write_json_atomic(attempt_path / "timings.json", record)
    write_json_atomic(directory / "trial.json", record)
    return record


def load_cases(input_path):
    from scripts.assess_thermochemistry import load_routes
    from synplan.chem.reaction.score.chemistry import (
        prepare_reaction,
        reaction_variants,
    )
    from synplan.chem.utils import mapped_smiles

    labels, routes = load_routes(input_path)
    routes = dict(zip(labels, routes, strict=True))
    cases = []
    for route_id, step, description in CASES:
        reaction = routes[route_id].steps[step - 1].reaction
        info, left, right = prepare_reaction(reaction, 0.8)
        variants = list(reaction_variants(left, right))
        species = sorted({s for sides in variants for side in sides for s in side})
        cases.append(
            {
                "id": f"route-{route_id}-step-{step}",
                "route_id": route_id,
                "step": step,
                "description": description,
                "original_reaction": mapped_smiles(reaction),
                "balanced_reaction": info["balanced_reaction"],
                "balancing_confidence": info["balancing_confidence"],
                "expected_variants": len(variants),
                "expected_species": species,
            }
        )
    return cases


def worker(job_path):
    job = read_json(job_path)
    profile, output = job["profile"], Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    execution_id = job["execution_id"]
    startup_path = output / "startups" / f"{execution_id}.json"
    startup_path.parent.mkdir(exist_ok=True)
    if startup_path.exists():
        raise ValueError("An execution ID must never overwrite a startup record")
    startup = {"execution_id": execution_id, "profile": profile, "status": "failed"}
    try:
        synchronize, runtime = configure_runtime(profile)
        actual_execution = execution_identity(profile, runtime)
        if actual_execution != job["execution_identity"]:
            raise ValueError("Worker execution identity changed after preflight")
        execution = {
            "execution_id": execution_id,
            "execution_fingerprint": fingerprint(actual_execution),
        }
        startup.update(execution, execution_identity=actual_execution, runtime=runtime)
        from synplan.chem.reaction.score import AIMNetConfig
        from synplan.chem.reaction.score.molecular import AIMNetMolecularCalculator
        from synplan.utils.files import write_json_atomic

        config = AIMNetConfig(**{**job["config"], "device": profile["device"]})
        load_started = perf_counter_ns()
        backend = AIMNetMolecularCalculator(config)
        identity = backend.identity()
        if (
            identity["aimnet_sha256"] != job["checkpoint_hashes"]["aimnet"]
            or identity["loqi_sha256"] != job["checkpoint_hashes"]["loqi"]
        ):
            raise ValueError("Worker model checkpoints changed after preflight")
        synchronize()
        loaded = perf_counter_ns()
        warm = backend.calculate("O")
        synchronize()
        ready = perf_counter_ns()
        assert warm["generated_conformers"] == 10
        if profile["device"].startswith("cuda"):
            import torch

            runtime["gpu"]["allocated_bytes_after_warmup"] = (
                torch.cuda.memory_allocated()
            )
            runtime["gpu"]["peak_allocated_bytes_startup"] = (
                torch.cuda.max_memory_allocated()
            )
            assert runtime["gpu"]["allocated_bytes_after_warmup"] > 0
        startup = {
            **startup,
            "status": "ready",
            "profile": profile,
            "runtime": runtime,
            "calculator": identity,
            "model_loading_s": (loaded - load_started) / 1e9,
            "warmup_s": (ready - loaded) / 1e9,
            "startup_s": (ready - PROCESS_STARTED) / 1e9,
            "warmup_species": "O",
            "warmup_cached": False,
        }
        write_json_atomic(startup_path, startup)
        print(
            f"BENCH ready {profile['id']}; startup {startup['startup_s']:.2f}s",
            flush=True,
        )
        for case in job["cases"]:
            directory = output / case["id"]
            if (directory / "trial.json").exists():
                print(f"BENCH resumed {profile['id']} {case['id']}", flush=True)
                continue
            result = run_trial(
                case, config, backend, directory, synchronize, execution=execution
            )
            print(
                f"BENCH {profile['id']} repeat {job['repeat']} {case['id']} {result['status']}: total={result['total_s']:.3f}s calculation={result['calculation_s']:.3f}s cache_write={result['cache_write_s']:.6f}s thermo_write={result['thermo_write_s']:.6f}s",
                flush=True,
            )
        write_json_atomic(
            output / "complete.json",
            {"complete": True, "repeat": job["repeat"], **execution},
        )
    except BaseException as exc:
        from synplan.utils.files import write_json_atomic

        try:
            (output / f"error-{execution_id}.log").write_text(traceback.format_exc())
            if not startup_path.exists():
                startup["error"] = {"type": type(exc).__name__, "detail": str(exc)}
                write_json_atomic(startup_path, startup)
        except Exception as diagnostic_error:
            exc.add_note(f"Writing worker diagnostics also failed: {diagnostic_error}")
        raise


def launch(job, python, repo):
    from synplan.utils.files import write_json_atomic

    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "job.json"
    write_json_atomic(path, job)
    environment = {**os.environ, "PYTHONPATH": str(repo), "PYTHONUNBUFFERED": "1"}
    for name in THREAD_VARIABLES:
        environment[name] = str(job["profile"]["threads"])
    with (output / "process.log").open("a") as log:
        process = subprocess.Popen(
            [
                str(python),
                "-u",
                "-m",
                "scripts.benchmark_thermochemistry",
                "--worker",
                str(path),
            ],
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return process.wait()


def run_benchmark(args):
    from scripts.benchmark_thermochemistry_report import export_benchmark
    from synplan.chem.reaction.score import AIMNetConfig
    from synplan.utils.files import write_json_atomic

    if sys.platform != "linux":
        raise RuntimeError(
            "Thermochemistry benchmarks require Linux CPU affinity and topology interfaces"
        )
    repo, output = Path(__file__).resolve().parents[1], args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = [
        {
            "id": f"cpu-{n}",
            "device": "cpu",
            "threads": n,
            "cpus": physical_cpus(n),
            "python": sys.executable,
        }
        for n in args.cpu_cores
    ]
    profiles.append(
        {
            "id": "gpu",
            "device": "cuda:0",
            "threads": args.gpu_host_cores,
            "cpus": physical_cpus(args.gpu_host_cores),
            "python": interpreter_path(args.gpu_python),
        }
    )
    config = AIMNetConfig(
        aimnet_checkpoint=args.aimnet_model.resolve(),
        loqi_checkpoint=args.loqi_model.resolve(),
    ).model_dump(mode="json")
    cases = load_cases(args.input)
    sources = [
        Path(__file__),
        repo / "scripts/benchmark_runtime.py",
        repo / "scripts/benchmark_thermochemistry_report.py",
        *sorted((repo / "synplan/chem/reaction/score").glob("*.py")),
        repo / "synplan/chem/reaction/curation/rebalancing.py",
    ]
    definition = {
        "format_version": FORMAT_VERSION,
        "input_sha256": digest(args.input),
        "input_path": str(args.input.resolve()),
        "config": config,
        "profiles": profiles,
        "repeats": args.repeats,
        "cases": cases,
        "checkpoint_hashes": {
            "aimnet": digest(args.aimnet_model),
            "loqi": digest(args.loqi_model),
        },
        "source_hashes": {str(p.relative_to(repo)): digest(p) for p in sources},
    }
    definition["expected_versions"] = {name: version(name) for name in PACKAGES}
    key = fingerprint(definition)
    manifest_path = output / "manifest.json"
    manifest: dict
    existing = manifest_path.exists()
    if existing:
        manifest = read_json(manifest_path)
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "Legacy benchmark: use a new output directory; --report-only remains supported"
            )
        if manifest["fingerprint"] != key:
            raise ValueError(
                "Benchmark definition changed; choose a new output directory"
            )
    else:
        manifest = {
            **definition,
            "fingerprint": key,
            "status": "running",
            "unavailable": {},
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "hardware": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
            "filesystem": subprocess.check_output(["df", "-T", str(output)], text=True),
            "timing_scope": {
                "calculation_s": "Synchronized molecular calculation calls: LoQI, optimization, Hessians and RRHO.",
                "cache_write_s": "Production atomic molecular-cache writes, including JSON serialization.",
                "thermo_write_s": "Atomic complete reaction annotation write, including serialization.",
                "other_s": "Preparation, balancing, enumeration, aggregation, synchronization outside model calls, and bookkeeping.",
                "total_s": "Reaction assessment through thermochemistry-file close and atomic replacement. Excludes startup, metric-file writes and report generation.",
                "durability": "Buffered file close and atomic replacement; no forced fsync.",
                "fresh_cache": "Empty molecular-results cache for every reaction/repetition; loaded models and OS file caches are warm.",
            },
        }
    attempt = output / "attempts" / uuid4().hex
    attempt.mkdir(parents=True)
    accepted = False
    if not existing:
        write_json_atomic(manifest_path, manifest)
    try:
        # GPU comes first. Any probe or model preflight error stops CPU work.
        identities = {}
        for profile in [profiles[-1], *profiles[:-1]]:
            observed = probe_runtime(profile, repo, attempt / f"{profile['id']}.json")
            identities[profile["id"]] = observed
            for name, expected in definition["expected_versions"].items():
                actual = observed["versions"][name]
                # CPU/CUDA wheels may differ BETWEEN profiles. Within each
                # profile the full build string is part of the exact identity.
                if name == "torch":
                    actual, expected = actual.split("+")[0], expected.split("+")[0]
                if actual != expected:
                    raise ValueError(f"Dependency mismatch in {profile['id']}: {name}")
        recorded = manifest.get("execution_identities")
        if recorded is not None and recorded != identities:
            raise ValueError(
                "Benchmark execution identity changed; choose a new output directory"
            )
        if recorded is None and any(output.glob("*/repeat-*/*/trial.json")):
            raise ValueError(
                "Trials have no execution identity; choose a new output directory"
            )
        # Validate saved provenance before accepting a resume or replacing exports.
        export_benchmark(output, publish=False)
        manifest["execution_identities"] = identities
        manifest["status"] = "running"
        write_json_atomic(manifest_path, manifest)
        accepted = True

        def job_for(profile, block, repeat, selected):
            return {
                "profile": profile,
                "config": config,
                "cases": selected,
                "repeat": repeat,
                "output": str(block),
                "execution_id": uuid4().hex,
                "execution_identity": identities[profile["id"]],
                "checkpoint_hashes": definition["checkpoint_hashes"],
            }

        gpu = profiles[-1]
        preflight = output / "gpu-preflight"
        if not (preflight / "complete.json").exists():
            if launch(job_for(gpu, preflight, 0, []), gpu["python"], repo):
                raise RuntimeError(
                    f"GPU model preflight failed: {preflight}; see process.log"
                )
        for repeat in range(args.repeats):
            order = (
                profiles[repeat % len(profiles) :] + profiles[: repeat % len(profiles)]
            )
            reaction_order = cases[repeat % len(cases) :] + cases[: repeat % len(cases)]
            for profile in order:
                block = output / profile["id"] / f"repeat-{repeat + 1}"
                if (block / "complete.json").exists():
                    continue
                job = job_for(profile, block, repeat + 1, reaction_order)
                print(f"BENCH block {profile['id']} repeat {repeat + 1}", flush=True)
                if launch(job, profile["python"], repo):
                    raise RuntimeError(
                        f"Benchmark block failed: {block}; see process.log"
                    )
                export_benchmark(output)
        manifest["status"] = "complete"
    except BaseException as exc:
        try:
            (attempt / "error.log").write_text(traceback.format_exc())
            write_json_atomic(
                attempt / "failure.json",
                {
                    "status": "aborted",
                    "stage": "execution" if accepted else "preflight",
                    "type": type(exc).__name__,
                    "detail": str(exc),
                },
            )
            if accepted or not existing:
                manifest["status"] = "aborted"
                manifest["failure"] = {"type": type(exc).__name__, "detail": str(exc)}
                write_json_atomic(manifest_path, manifest)
                export_benchmark(output)
        except Exception as diagnostic_error:
            exc.add_note(
                f"Writing benchmark diagnostics also failed: {diagnostic_error}"
            )
        raise
    else:
        manifest.pop("failure", None)
        write_json_atomic(manifest_path, manifest)
        export_benchmark(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker", type=Path, help="Internal: execute one configuration block"
    )
    parser.add_argument(
        "--report-only",
        type=Path,
        help="Regenerate a completed or partial benchmark report",
    )
    parser.add_argument(
        "--input", type=Path, default=Path("tutorials/data/apatinib_routes.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aimnet-model", type=Path)
    parser.add_argument("--loqi-model", type=Path)
    parser.add_argument("--gpu-python", type=Path)
    parser.add_argument("--cpu-cores", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--gpu-host-cores", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("AIMNet thermochemistry benchmarks require Python 3.11+")
    if args.worker:
        worker(args.worker)
    elif args.report_only:
        from scripts.benchmark_thermochemistry_report import export_benchmark

        export_benchmark(args.report_only)
    else:
        if not all((args.output, args.aimnet_model, args.loqi_model, args.gpu_python)):
            parser.error(
                "--output, --aimnet-model, --loqi-model and --gpu-python are required"
            )
        if args.repeats < 1 or len(set(args.cpu_cores)) != len(args.cpu_cores):
            parser.error("Repetitions must be positive and CPU core counts unique")
        run_benchmark(args)


if __name__ == "__main__":
    main()
