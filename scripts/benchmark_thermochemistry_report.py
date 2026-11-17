"""Standalone performance reports; no model loading or forward-assessment fields."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

METRICS = ("calculation_s", "cache_write_s", "thermo_write_s", "other_s", "total_s")


def summarize(values):
    if not values:
        return None
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(manifest, trials):
    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        for profile in manifest["profiles"]:
            selected = [
                r
                for r in trials
                if r["case_id"] == case["id"] and r["profile"] == profile["id"]
            ]
            complete = len(selected) == manifest["repeats"] and all(
                r["status"] == "ok"
                and r.get("successful_variants")
                == r.get("total_variants")
                == case["expected_variants"]
                for r in selected
            )
            rows.append(
                {
                    "case_id": case["id"],
                    "profile": profile["id"],
                    "trials": len(selected),
                    "successful_trials": sum(r["status"] == "ok" for r in selected),
                    "comparable": complete,
                    "timings": {
                        k: summarize([r[k] for r in selected]) for k in METRICS
                    },
                    "speedup_vs_cpu1": None,
                    "energy_differences_vs_cpu1": [],
                }
            )
    baseline = {r["case_id"]: r for r in rows if r["profile"] == "cpu-1"}
    for row in rows:
        reference = baseline.get(row["case_id"])
        if reference and reference["comparable"] and row["comparable"]:
            row["speedup_vs_cpu1"] = (
                reference["timings"]["total_s"]["median"]
                / row["timings"]["total_s"]["median"]
            )
        for trial in trials:
            if trial["case_id"] != row["case_id"] or trial["profile"] != row["profile"]:
                continue
            other = next(
                (
                    t
                    for t in trials
                    if t["case_id"] == row["case_id"]
                    and t["profile"] == "cpu-1"
                    and t["repeat"] == trial["repeat"]
                ),
                None,
            )
            if not other:
                continue
            variants = {
                v["reaction"]: v
                for v in other.get("variant_values", [])
                if v["status"] == "ok"
            }
            for variant in trial.get("variant_values", []):
                ref = variants.get(variant["reaction"])
                if variant["status"] == "ok" and ref:
                    row["energy_differences_vs_cpu1"].append(
                        {
                            "repeat": trial["repeat"],
                            "variant": variant["id"],
                            "units": "kcal/mol",
                            **{
                                k: variant[k] - ref[k]
                                for k in ("delta_e", "delta_h", "delta_g")
                            },
                        }
                    )
    return rows


def format_timing(summary, scale=1):
    if summary is None:
        return "—"
    return (
        f"{summary['median'] * scale:.3f}<small>range {summary['min'] * scale:.3f} to {summary['max'] * scale:.3f}"
        f"<br>mean {summary['mean'] * scale:.3f}</small>"
    )


def report_html(manifest, summary, trials):
    tables = []
    for case in manifest["cases"]:
        rows = []
        for group in [r for r in summary if r["case_id"] == case["id"]]:
            profile = group["profile"]
            state = f"{group['successful_trials']} / {manifest['repeats']} successful"
            if profile in manifest.get("unavailable", {}):
                state = "Unavailable"
            speed = group["speedup_vs_cpu1"]
            rows.append(
                f"<tr><th>{escape(profile)}</th><td>{state}</td>"
                + "".join(
                    f"<td>{format_timing(group['timings'][k], 1000 if 'write' in k else 1)}</td>"
                    for k in METRICS
                )
                + f"<td>{'—' if speed is None else f'{speed:.2f}x'}</td></tr>"
            )
        details = []
        for trial in [t for t in trials if t["case_id"] == case["id"]]:
            values = "".join(
                f"<td>{trial[k] * (1000 if 'write' in k else 1):.4f}</td>"
                for k in METRICS
            )
            details.append(
                f"<tr><td>{escape(trial['profile'])}</td><td>{trial['repeat']}</td><td>{escape(trial['status'])}</td><td>{trial.get('successful_variants', 0)} / {trial.get('total_variants', case['expected_variants'])}</td>{values}</tr>"
            )
        energy_rows = []
        for group in [r for r in summary if r["case_id"] == case["id"]]:
            differences = group["energy_differences_vs_cpu1"]
            cells = "".join(
                f"<td>{max(abs(v[key]) for v in differences):.4f}</td>"
                if differences
                else "<td>—</td>"
                for key in ("delta_e", "delta_h", "delta_g")
            )
            energy_rows.append(
                f"<tr><td>{escape(group['profile'])}</td><td>{len(differences)}</td>{cells}</tr>"
            )
        energy_details = (
            "<details><summary>Energy differences from one core</summary>"
            "<p>Maximum absolute differences in kcal/mol, matched by repeat and stereochemical equation. "
            "Different devices or thread counts can select different conformers.</p>"
            "<div class='scroll'><table><thead><tr><th>Configuration</th><th>Compared variants</th>"
            "<th>|ΔE difference|</th><th>|ΔH difference|</th><th>|ΔG difference|</th></tr></thead>"
            f"<tbody>{''.join(energy_rows)}</tbody></table></div></details>"
        )
        columns = "<th>Calculation (s)</th><th>Cache writing (ms)</th><th>Thermo-data writing (ms)</th><th>Other (s)</th><th>Total (s)</th>"
        tables.append(
            f"<section><h2>Route {escape(case['route_id'])}, step {case['step']}</h2><p>{escape(case['description'])} · {len(case['expected_species'])} unique species · {case['expected_variants']} variants</p>"
            f"<div class='scroll'><table><thead><tr><th>Configuration</th><th>Coverage</th>{columns}<th>Speed vs 1 core</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<details><summary>Individual trials</summary><div class='scroll'><table><thead><tr><th>Configuration</th><th>Repeat</th><th>Status</th><th>Variants</th>{columns}</tr></thead><tbody>{''.join(details)}</tbody></table></div></details>"
            f"{energy_details}<details><summary>Balanced equation</summary><code>{escape(case['balanced_reaction'])}</code></details></section>"
        )
    configuration_labels = []
    for profile in manifest["profiles"]:
        if profile.get("device", "").startswith("cuda"):
            configuration_labels.append(
                f"one GPU with {profile['threads']} host CPU cores"
            )
        elif "threads" in profile:
            configuration_labels.append(f"{profile['threads']} physical CPU cores")
        else:
            configuration_labels.append(profile["id"])
    configuration_text = escape(", ".join(configuration_labels))
    wanted = len(manifest["cases"]) * len(manifest["profiles"]) * manifest["repeats"]
    passed = sum(t["status"] == "ok" for t in trials)
    css = """body{font:15px system-ui,sans-serif;color:#182839;background:#f4f6f8;margin:0}main{max-width:1450px;margin:36px auto;padding:0 20px}header,section{background:white;padding:24px;margin-bottom:22px;border:1px solid #dce2e8;border-radius:6px}header>div{display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:26px}h2{font-size:20px}p{line-height:1.5}button,summary{cursor:pointer}button{padding:10px 18px;background:white;border:1px solid #aab8c7;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:12px 9px;border-bottom:1px solid #dce2e8;vertical-align:top}thead{background:#edf2f7}small{display:block;color:#5a6878;font-size:11px;margin-top:4px;white-space:nowrap}.scroll{overflow:auto}details{margin-top:16px}code{overflow-wrap:anywhere}dialog{max-width:950px;width:85vw;border:1px solid #aab8c7;border-radius:6px}dialog::backdrop{background:#18283980}dialog pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}dialog>div{display:flex;justify-content:space-between;align-items:center}.status{font-weight:600}"""
    js = """const opener=document.querySelector('#manifest-open'),dialog=document.querySelector('#manifest');opener.onclick=()=>dialog.showModal();document.querySelector('#manifest-close').onclick=()=>dialog.close();dialog.addEventListener('close',()=>opener.focus());"""
    return (
        f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Apatinib CPU/GPU thermochemistry benchmark</title><style>{css}</style><main><header><div><h1>Apatinib thermochemistry benchmark</h1><button id='manifest-open' aria-haspopup='dialog' aria-controls='manifest'>Manifest</button></div>"
        f"<p class='status'>{escape(manifest['status'])} · {len(trials)} / {wanted} trials recorded · {passed} successful</p>"
        f"<p>One process per configuration: {configuration_text}. Fresh molecular-results cache for every reaction and repetition; startup is measured separately.</p>"
        f"<p>Target: {manifest['repeats']} repetitions per reaction and configuration. Medians, ranges and means summarize the recorded trials. Calculation uses synchronized wall-clock timing. Writing includes JSON serialization, buffered file close and atomic replacement; no forced fsync. Failed trials remain visible; speedups require complete successful coverage.</p>"
        "<p>Gas-phase RRHO · 298.15 K · 101325 Pa · symmetry number 1 · ten conformers per stereoisomer. CPU/GPU conformers can differ; raw results include energy differences and conformer outcomes.</p></header>"
        + "".join(tables)
        + f"</main><dialog id='manifest' aria-labelledby='manifest-title'><div><h2 id='manifest-title'>Benchmark manifest</h2><button id='manifest-close' autofocus>Close</button></div><pre>{escape(json.dumps(manifest, indent=2))}</pre></dialog><script>{js}</script></html>"
    )


def export_benchmark(output, *, publish=True):
    from scripts.benchmark_runtime import FORMAT_VERSION, fingerprint
    from synplan.utils.files import write_json_atomic

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    trials, startups, identities = [], [], {}
    strict = manifest.get("format_version") == FORMAT_VERSION
    for profile in manifest["profiles"]:
        for repeat in range(1, manifest["repeats"] + 1):
            block = output / profile["id"] / f"repeat-{repeat}"
            executions = {}
            startup_paths = (
                sorted((block / "startups").glob("*.json"))
                if strict
                else [block / "startup.json"]
            )
            for startup_path in startup_paths:
                if not startup_path.exists():
                    continue
                record = json.loads(startup_path.read_text(encoding="utf-8"))
                if strict and record["status"] == "ready":
                    expected = manifest["execution_identities"][profile["id"]]
                    key = fingerprint(expected)
                    if (
                        record["execution_identity"] != expected
                        or record["execution_fingerprint"] != key
                    ):
                        raise ValueError(
                            f"Startup execution identity mismatch: {startup_path}"
                        )
                    executions[record["execution_id"]] = key
                identity = record.pop("calculator", None)
                if identity is not None:
                    identities[json.dumps(identity, sort_keys=True)] = identity
                startups.append({**record, "repeat": repeat})
            if strict and (block / "complete.json").exists():
                if any(
                    not (block / c["id"] / "trial.json").exists()
                    for c in manifest["cases"]
                ):
                    raise ValueError(f"Completed block has missing trials: {block}")
            for case in manifest["cases"]:
                directory = block / case["id"]
                path = directory / "trial.json"
                if not path.exists():
                    failed = sorted(directory.glob("attempt-*/timings.json"))
                    if not failed:
                        continue
                    path = failed[-1]
                record = json.loads(path.read_text(encoding="utf-8"))
                if strict:
                    key = executions.get(record.get("execution_id"))
                    if key is None or key != record.get("execution_fingerprint"):
                        raise ValueError(
                            f"Trial has no matching execution provenance: {path}"
                        )
                if "thermochemistry" not in manifest:
                    annotation_path = (
                        directory
                        / f"attempt-{record.get('attempt', 1):03}"
                        / "thermochemistry.json"
                    )
                    if annotation_path.is_file():
                        annotation = json.loads(
                            annotation_path.read_text(encoding="utf-8")
                        )
                        provenance = annotation.get("calculation", {})
                        manifest["thermochemistry"] = {
                            "method": provenance.get("method"),
                            "reference_conventions": provenance.get(
                                "reference_conventions", {}
                            ),
                            "reaction_energy_units": "kcal/mol",
                            "reaction_sign": "products minus reactants",
                            "conformer_selection": "One valid minimum per stereoisomer; no ensemble averaging.",
                        }
                trials.append({**record, "profile": profile["id"], "repeat": repeat})
    manifest["startups"] = startups
    manifest["calculators"] = list(identities.values())
    summary = aggregate(manifest, trials)
    result = {"manifest": manifest, "summary": summary, "trials": trials}
    if not publish:
        return result
    write_json_atomic(output / "benchmark.json", result)
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "execution_id",
            "execution_fingerprint",
            "case_id",
            "route_id",
            "step",
            "profile",
            "repeat",
            "status",
            "successful_variants",
            "total_variants",
            "molecular_calculations",
            "cache_hits",
            *METRICS,
            "thermo_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trials)
    (output / "benchmark.html").write_text(
        report_html(manifest, summary, trials), encoding="utf-8"
    )
    return result
