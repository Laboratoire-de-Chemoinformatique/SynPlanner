"""Timing boundaries and isolated trial state without model downloads."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_thermochemistry as bench
from scripts.benchmark_thermochemistry_report import aggregate, export_benchmark
from synplan.chem.reaction.score import AIMNetConfig
from synplan.chem.reaction.score.aimnet import cache as cache_module
from synplan.chem.reaction.score.aimnet.config import MolecularCalculationError

from .test_aimnet import Calculator

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Benchmark execution requires Python 3.11+"
)


def case(text="CCO>>CCO", species=("CCO",)):
    return {
        "id": "test",
        "route_id": "<route>",
        "step": 1,
        "description": "<script>bad</script>",
        "original_reaction": text,
        "balanced_reaction": text,
        "expected_species": list(species),
        "expected_variants": 1,
    }


def test_gpu_sync_brackets_model_time_and_original_errors_survive():
    events = []
    clock = iter([100, 150, 200, 290])

    def sync():
        events.append("sync")

    def calculate(text):
        events.append(text)
        if text == "bad":
            raise RuntimeError("model error")
        return {"energy_eV": 1}

    timed = bench.TimedCalculator(
        SimpleNamespace(calculate=calculate), sync, lambda: next(clock)
    )
    assert timed.calculate("good") == {"energy_eV": 1}
    with pytest.raises(RuntimeError, match="model error"):
        timed.calculate("bad")
    assert events == ["sync", "good", "sync", "sync", "bad", "sync"]
    assert timed.nanoseconds == 140


def test_writer_includes_serialization_and_close_even_on_failure():
    clock = iter([5, 25])

    def writer(path, data):
        raise OSError("disk full")

    timed = bench.TimedWriter(writer, lambda: next(clock))
    with pytest.raises(OSError, match="disk full"):
        timed("file", {})
    assert timed.nanoseconds == 20 and timed.calls == 1


def test_each_trial_recalculates_and_writes_cache_and_annotation(tmp_path):
    calculator = Calculator()
    first = bench.run_trial(case(), AIMNetConfig(), calculator, tmp_path)
    second = bench.run_trial(case(), AIMNetConfig(), calculator, tmp_path)
    assert calculator.calls == ["CCO", "CCO"]
    assert (first["attempt"], second["attempt"]) == (1, 2)
    for record in (first, second):
        assert record["molecular_calculations"] == 1 and record["cache_hits"] == 0
        assert record["cache_writes"] == record["thermo_writes"] == 1
        assert record["thermo_bytes"] > 0
        assert record["other_s"] >= 0
        assert record["total_s"] == pytest.approx(
            sum(
                record[k]
                for k in ("calculation_s", "cache_write_s", "thermo_write_s", "other_s")
            )
        )
        assert json.loads(open(record["result_path"]).read())["status"] == "ok"


def test_abort_restores_writer_and_retry_does_not_reuse_partial_cache(tmp_path):
    original = cache_module.write_json_atomic

    class Broken(Calculator):
        broken = True

        def calculate(self, text):
            if text == "O" and self.broken:
                raise RuntimeError("unexpected model bug")
            return super().calculate(text)

    calculator = Broken()
    source = case("CCO.O>>CCO.O", ("CCO", "O"))
    with pytest.raises(RuntimeError, match="model bug"):
        bench.run_trial(source, AIMNetConfig(), calculator, tmp_path)
    assert cache_module.write_json_atomic is original
    assert not (tmp_path / "trial.json").exists()
    assert (
        json.loads((tmp_path / "attempt-001/timings.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "errored"
    )
    assert (tmp_path / "attempt-001/cache.json").exists()
    calculator.broken = False
    result = bench.run_trial(source, AIMNetConfig(), calculator, tmp_path)
    assert result["attempt"] == 2 and result["molecular_calculations"] == 2
    assert calculator.calls.count("CCO") == 2


def test_expected_molecular_failure_is_retained(tmp_path):
    calculator = Calculator(
        fail=MolecularCalculationError("frequencies", "not_a_minimum", "test")
    )
    result = bench.run_trial(case(), AIMNetConfig(), calculator, tmp_path)
    assert result["status"] == "failed"
    assert result["successful_variants"] == 0 and result["total_variants"] == 1
    assert result["thermo_writes"] == 1
    assert (
        json.loads((tmp_path / "trial.json").read_text(encoding="utf-8"))["status"]
        == "failed"
    )


def test_core_selection_excludes_smt_and_cross_numa(tmp_path):
    for cpu, core in [(0, 0), (1, 1), (2, 0), (3, 1)]:
        folder = tmp_path / f"cpu{cpu}"
        (folder / "topology").mkdir(parents=True)
        (folder / "node0").mkdir()
        (folder / "topology/physical_package_id").write_text("0", encoding="utf-8")
        (folder / "topology/core_id").write_text(str(core), encoding="utf-8")
    assert bench.physical_cpus(2, [0, 1, 2, 3], tmp_path) == [0, 1]
    with pytest.raises(ValueError, match="physical cores"):
        bench.physical_cpus(3, [0, 1, 2, 3], tmp_path)
    for cpu in (1, 3):
        (tmp_path / f"cpu{cpu}/node0").rename(tmp_path / f"cpu{cpu}/node1")
    with pytest.raises(ValueError):
        bench.physical_cpus(2, [0, 1, 2, 3], tmp_path)


def fixture_trials():
    manifest = {
        "status": "complete",
        "repeats": 3,
        "cases": [case()],
        "profiles": [{"id": "cpu-1"}, {"id": "gpu"}],
        "unavailable": {},
    }
    trials = []
    for profile, totals in [("cpu-1", [10, 12, 14]), ("gpu", [2, 3, 4])]:
        for repeat, total in enumerate(totals, 1):
            trials.append(
                {
                    "case_id": "test",
                    "profile": profile,
                    "repeat": repeat,
                    "route_id": "<route>",
                    "step": 1,
                    "status": "ok",
                    "successful_variants": 1,
                    "total_variants": 1,
                    "calculation_s": total - 0.3,
                    "cache_write_s": 0.1,
                    "thermo_write_s": 0.1,
                    "other_s": 0.1,
                    "total_s": total,
                }
            )
    return manifest, trials


def test_summaries_include_failures_and_speedups_require_all_repeats():
    manifest, trials = fixture_trials()
    rows = aggregate(manifest, trials)
    assert rows[1]["speedup_vs_cpu1"] == 4
    trials[-1]["status"] = "failed"
    trials[-1]["successful_variants"] = 0
    rows = aggregate(manifest, trials)
    assert rows[1]["speedup_vs_cpu1"] is None
    assert rows[1]["timings"]["total_s"]["max"] == 4
    assert aggregate(manifest, trials[:-1])[1]["speedup_vs_cpu1"] is None


def test_export_round_trip_escaping_and_hidden_manifest(tmp_path, monkeypatch):
    # Emulate a Windows legacy locale: report/CSV I/O must choose UTF-8 itself.
    original_open = Path.open

    def legacy_locale(self, mode="r", buffering=-1, encoding=None, **kwargs):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return original_open(self, mode, buffering, encoding=encoding, **kwargs)

    monkeypatch.setattr(Path, "open", legacy_locale)
    with pytest.raises(UnicodeEncodeError):
        (tmp_path / "locale.txt").write_text("Δ")
    manifest, trials = fixture_trials()
    before = deepcopy(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for trial in trials:
        directory = (
            tmp_path / trial["profile"] / f"repeat-{trial['repeat']}" / trial["case_id"]
        )
        directory.mkdir(parents=True)
        (directory / "trial.json").write_text(json.dumps(trial), encoding="utf-8")
    annotation = tmp_path / "cpu-1/repeat-1/test/attempt-001/thermochemistry.json"
    annotation.parent.mkdir()
    annotation.write_text(
        json.dumps(
            {
                "calculation": {
                    "method": "gas_phase_single_conformer_RRHO",
                    "reference_conventions": {"proton_electronic_energy": "0 eV"},
                },
                "molecules": {"coordinates_angstrom": [[12345, 23456, 34567]]},
            }
        ),
        encoding="utf-8",
    )
    result = export_benchmark(tmp_path)
    assert result["manifest"]["thermochemistry"]["reference_conventions"] == {
        "proton_electronic_energy": "0 eV"
    }
    assert len(result["trials"]) == 6
    assert manifest == before
    assert len((tmp_path / "trials.csv").read_text(encoding="utf-8").splitlines()) == 7
    page = (tmp_path / "benchmark.html").read_text(encoding="utf-8")
    assert (
        "<script>bad</script>" not in page
        and "&lt;script&gt;bad&lt;/script&gt;" in page
    )
    assert "<dialog id='manifest'" in page and "<dialog open" not in page
    assert "Policy log likelihood" not in page and "forward_prediction" not in page
    assert "showModal()" in page
    assert "coordinates_angstrom" not in page
    assert page.count("proton_electronic_energy") == 1


def test_energy_comparison_matches_equation_and_repeat_not_variant_order():
    manifest, trials = fixture_trials()
    for trial in trials:
        delta = 2 if trial["profile"] == "gpu" else 0
        value = 10 * trial["repeat"] + delta
        trial["variant_values"] = [
            {
                "id": 2,
                "reaction": "CCO>>CCO",
                "status": "ok",
                "delta_e": value,
                "delta_h": value + 1,
                "delta_g": value - 3,
            },
            {
                "id": 1,
                "reaction": "CCN>>CCN",
                "status": "failed",
                "delta_e": None,
                "delta_h": None,
                "delta_g": None,
            },
        ]
    rows = aggregate(manifest, trials)
    for record in rows[1]["energy_differences_vs_cpu1"]:
        assert record["variant"] == 2
        assert record["units"] == "kcal/mol"
        assert record["delta_e"] == record["delta_h"] == record["delta_g"] == 2
    assert len(rows[1]["energy_differences_vs_cpu1"]) == 3


def test_gpu_interpreter_keeps_virtual_environment_packages(tmp_path):
    import subprocess
    import venv

    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin/python"
    if not python.is_symlink():
        pytest.skip("This host cannot create a symlinked Python environment")
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    (site_packages / "benchmark_env_marker.py").write_text(
        'value = "selected-venv"\n', encoding="utf-8"
    )
    result = subprocess.check_output(
        [
            bench.interpreter_path(python),
            "-c",
            "import benchmark_env_marker; print(benchmark_env_marker.value)",
        ],
        text=True,
    )
    assert result.strip() == "selected-venv"


@pytest.fixture
def benchmark_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.sys, "platform", "linux")
    source = tmp_path / "routes.json"
    source.write_text("{}", encoding="utf-8")
    aimnet, loqi = tmp_path / "aimnet.pt", tmp_path / "loqi.ckpt"
    aimnet.write_bytes(b"fake aimnet")
    loqi.write_bytes(b"fake loqi")
    args = SimpleNamespace(
        output=tmp_path / "run",
        input=source,
        aimnet_model=aimnet,
        loqi_model=loqi,
        gpu_python=tmp_path / "gpu-env/bin/python",
        cpu_cores=[1],
        gpu_host_cores=1,
        repeats=1,
    )
    monkeypatch.setattr(bench, "physical_cpus", lambda count: list(range(count)))
    monkeypatch.setattr(
        bench, "load_cases", lambda path: [case(), {**case(), "id": "second"}]
    )
    monkeypatch.setattr(bench, "version", lambda name: "1")
    monkeypatch.setattr(
        bench.subprocess,
        "check_output",
        lambda command, **kwargs: '{"lscpu": []}' if command[0] == "lscpu" else "test",
    )
    state = SimpleNamespace(calls=[], probes=[], fail=None, identity_change={})

    def probe(profile, repo, destination):
        state.probes.append(profile["id"])
        return {
            "versions": {name: "1" for name in bench.PACKAGES},
            "profile": profile,
            **state.identity_change,
        }

    def launch(job, python, repo):
        assert bench.read_json(args.output / "manifest.json")["status"] == "running"
        state.calls.append((job["profile"]["id"], job["repeat"]))
        block = Path(job["output"])
        block.mkdir(parents=True, exist_ok=True)
        startup = block / "startups" / f"{job['execution_id']}.json"
        startup.parent.mkdir(exist_ok=True)
        execution = {
            "execution_id": job["execution_id"],
            "execution_fingerprint": bench.fingerprint(job["execution_identity"]),
        }
        startup.write_text(
            json.dumps(
                {
                    **execution,
                    "execution_identity": job["execution_identity"],
                    "status": "ready",
                    "calculator": {},
                    "profile": job["profile"],
                }
            ),
            encoding="utf-8",
        )
        for selected in job["cases"]:
            directory = block / selected["id"]
            if (directory / "trial.json").exists():
                continue
            if state.fail == (job["profile"]["id"], selected["id"]):
                state.fail = None
                return 1
            bench.run_trial(
                selected, AIMNetConfig(), Calculator(), directory, execution=execution
            )
        (block / "complete.json").write_text(
            json.dumps({**execution, "complete": True}), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(bench, "probe_runtime", probe)
    monkeypatch.setattr(bench, "launch", launch)
    return args, state


def test_partial_block_resume_preserves_execution_provenance(benchmark_run):
    args, state = benchmark_run
    state.fail = ("cpu-1", "second")
    with pytest.raises(RuntimeError, match="Benchmark block failed"):
        bench.run_benchmark(args)
    assert bench.read_json(args.output / "manifest.json")["status"] == "aborted"
    first = args.output / "cpu-1/repeat-1/test/trial.json"
    before = first.read_bytes()
    startups = {
        p: p.read_bytes() for p in args.output.glob("*/repeat-*/startups/*.json")
    }
    bench.run_benchmark(args)
    assert first.read_bytes() == before
    assert all(p.read_bytes() == data for p, data in startups.items())
    result = bench.read_json(args.output / "benchmark.json")
    records = [t for t in result["trials"] if t["profile"] == "cpu-1"]
    assert len({t["execution_id"] for t in records}) == 2
    assert len({t["execution_fingerprint"] for t in records}) == 1
    assert result["manifest"]["status"] == "complete"
    assert state.calls == [("gpu", 0), ("cpu-1", 1), ("cpu-1", 1), ("gpu", 1)]
    calls_before, probes_before = list(state.calls), len(state.probes)
    bench.run_benchmark(args)
    assert state.calls == calls_before  # no model worker, warm-up or weight load
    assert len(state.probes) == probes_before + 2


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("missing GPU interpreter"), RuntimeError("CUDA unavailable")],
)
def test_gpu_probe_failure_aborts_before_cpu_work(benchmark_run, monkeypatch, failure):
    args, state = benchmark_run

    def fail(profile, *args):
        assert profile["id"] == "gpu"
        raise failure

    monkeypatch.setattr(bench, "probe_runtime", fail)
    with pytest.raises(type(failure), match=str(failure)):
        bench.run_benchmark(args)
    assert not state.calls
    assert bench.read_json(args.output / "manifest.json")["status"] == "aborted"
    assert bench.read_json(args.output / "benchmark.json")["trials"] == []
    assert "aborted" in (args.output / "benchmark.html").read_text(encoding="utf-8")
    assert next((args.output / "attempts").glob("*/failure.json"))


def test_gpu_model_preflight_failure_stops_cpu_work(benchmark_run, monkeypatch):
    args, state = benchmark_run

    def fail(job, *unused):
        state.calls.append((job["profile"]["id"], job["repeat"]))
        return 1

    monkeypatch.setattr(bench, "launch", fail)
    with pytest.raises(RuntimeError, match="GPU model preflight failed"):
        bench.run_benchmark(args)
    assert state.calls == [("gpu", 0)]
    assert bench.read_json(args.output / "manifest.json")["status"] == "aborted"


@pytest.mark.parametrize(
    "change",
    [
        {"torch_build": "2.11+cu130"},
        {"gpu_uuid": "changed"},
        {"driver": "changed"},
        {"python": "changed"},
        {"cpu_topology": "changed"},
        {"affinity": [9]},
        {"precision": "medium"},
        {"cudnn": 999},
    ],
)
def test_changed_execution_refused_without_overwriting_results(benchmark_run, change):
    args, state = benchmark_run
    bench.run_benchmark(args)
    previous = {p: p.read_bytes() for p in args.output.rglob("*") if p.is_file()}
    state.identity_change = change
    before = list(state.calls)
    with pytest.raises(ValueError, match="execution identity changed"):
        bench.run_benchmark(args)
    assert state.calls == before
    assert all(p.read_bytes() == data for p, data in previous.items())


def test_legacy_resume_refused_but_report_reading_supported(benchmark_run):
    args, state = benchmark_run
    args.output.mkdir()
    manifest, _ = fixture_trials()
    (args.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    export_benchmark(args.output)
    before = (args.output / "benchmark.html").read_bytes()
    with pytest.raises(ValueError, match="Legacy benchmark"):
        bench.run_benchmark(args)
    assert not state.calls and not state.probes
    assert (args.output / "benchmark.html").read_bytes() == before


def test_tampered_trial_identity_is_not_aggregated(benchmark_run):
    args, _ = benchmark_run
    bench.run_benchmark(args)
    path = args.output / "cpu-1/repeat-1/test/trial.json"
    record = bench.read_json(path)
    record["execution_id"] = "missing-startup"
    path.write_text(json.dumps(record), encoding="utf-8")
    page = (args.output / "benchmark.html").read_bytes()
    with pytest.raises(ValueError, match="matching execution provenance"):
        export_benchmark(args.output)
    assert (args.output / "benchmark.html").read_bytes() == page


def test_runtime_identity_keeps_builds_and_precision_but_not_live_load(monkeypatch):
    import sys

    from scripts import benchmark_runtime as runtime

    if sys.platform != "linux":
        pytest.skip("The benchmark runtime probe requires Linux")
    properties = SimpleNamespace(
        uuid="gpu-uuid", name="fake GPU", total_memory=100, major=9, minor=0
    )
    torch = SimpleNamespace(
        get_default_dtype=lambda: "float32",
        get_float32_matmul_precision=lambda: "highest",
        are_deterministic_algorithms_enabled=lambda: False,
        cuda=SimpleNamespace(get_device_properties=lambda _: properties),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(
                version=lambda: 91900,
                allow_tf32=False,
                deterministic=False,
                benchmark=False,
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        runtime, "version", lambda name: "2.11+cu128" if name == "torch" else "1"
    )
    monkeypatch.setattr(
        runtime, "distribution", lambda _: SimpleNamespace(read_text=lambda _: None)
    )
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: "NVRM driver 1\n"
        if str(path) == "/proc/driver/nvidia/version"
        else original_read(path, *a, **kw),
    )
    metadata = {
        "affinity": [1],
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "thread_environment": {},
        "torch": "2.11+cu128",
        "cuda_runtime": "12.8",
        "load_average": [1, 2, 3],
        "gpu": {"free_total_bytes": [20, 100]},
    }
    first = runtime.execution_identity({"device": "cuda:0"}, metadata)
    assert first["versions"]["torch"] == "2.11+cu128"
    assert first["gpu"]["uuid"] == "gpu-uuid"
    metadata["load_average"] = [40, 60, 80]
    metadata["gpu"]["free_total_bytes"] = [1, 100]
    assert runtime.execution_identity({"device": "cuda:0"}, metadata) == first
    torch.backends.cuda.matmul.allow_tf32 = True
    assert runtime.execution_identity({"device": "cuda:0"}, metadata) != first
