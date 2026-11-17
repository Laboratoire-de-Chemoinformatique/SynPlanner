#!/usr/bin/env python3
"""Assess saved routes on CPU, checkpointing each reaction for exact-run resume.

A completed or interrupted identical run resumes from its reaction checkpoints.
Use a new output directory to retry expected molecular failures; successful
molecules can still be shared through --cache. Unexpected errors are re-raised
and leave a partial report with explicit run-aborted steps.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from synplan.chem.reaction.routes import Route
from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.molecular import file_digest
from synplan.chem.utils import mapped_smiles
from synplan.utils.files import read_json, write_json_atomic

REACTION_CHECKPOINT_VERSION = 2


class IncompatibleCheckpointError(ValueError):
    """An existing run belongs to different inputs or calculation settings."""


def load_routes(path):
    """Single route collection or one target's collection, preserving saved IDs."""
    saved = read_json(path)
    if isinstance(saved, dict) and len(saved) == 1:
        candidate = next(iter(saved.values()))
        if isinstance(candidate, dict) and "type" not in candidate:
            saved = candidate
    items = saved.items() if isinstance(saved, dict) else enumerate(saved)
    labels, routes = [], []
    for key, tree in items:
        labels.append(str(key))
        routes.append(Route.from_json(tree))
    return labels, routes


def _aborted(reaction, detail):
    return {
        "version": 1,
        "status": "failed",
        "original_reaction": mapped_smiles(reaction),
        "balanced_reaction": None,
        "variants": [],
        "molecules": {},
        "failure": {
            "category": "errored",
            "stage": "run",
            "reason": "run_aborted",
            "detail": detail,
        },
        "calculation": {},
    }


def export_results(routes, labels, annotations, output, manifest):
    from synplan.utils.visualisation import thermochemistry_report_html

    annotated = AIMNetReactionAssessor.annotate_routes(routes, annotations)
    write_json_atomic(
        output / "routes_aimnet.json",
        {
            label: route.to_json()
            for label, route in zip(labels, annotated, strict=True)
        },
    )
    thermochemistry_report_html(
        annotated, output / "routes_aimnet.html", route_labels=labels, manifest=manifest
    )
    return annotated


def run_assessment(
    input_path, output, config, cache_path, *, verify_cache=False, workers=1, threads=4
):
    input_path, output = Path(input_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels, routes = load_routes(input_path)
    reactions = [step.reaction for route in routes for step in route]
    service = AIMNetReactionAssessor(config, cache_path=cache_path)
    config = service.config
    completed = []
    started = perf_counter()
    manifest = {
        "source": {
            "file": str(input_path.resolve()),
            "sha256": file_digest(input_path),
            "route_ids": labels,
            "routes": len(routes),
            "steps": len(reactions),
        },
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": "running",
        "timing_scope": {
            "elapsed_seconds": "Current invocation through assessment, before cache verification and report export.",
            "statistics.elapsed_seconds": "Assessor loop for newly uncheckpointed reactions in this invocation.",
            "statistics.calculation_seconds": "Cumulative time waiting for molecular calculations across resumed sessions; CPU workers run concurrently.",
        },
    }
    sidecar = input_path.with_suffix(".provenance.json")
    if sidecar.exists():
        manifest["source"]["collection_provenance"] = read_json(sidecar)
    job_path = output / "checkpoint.json"
    if (
        job_path.exists()
        and read_json(job_path).get("version") != REACTION_CHECKPOINT_VERSION
    ):
        raise IncompatibleCheckpointError(
            "Legacy reaction checkpoints: use a new output directory; validated molecular caches may be reused."
        )
    if not job_path.exists() and any((output / "reactions").glob("*.json")):
        raise IncompatibleCheckpointError(
            "Reaction checkpoints have no job identity; use a new output directory."
        )
    worker_calculator = None
    try:
        # Resolve models before accepting checkpoints from an earlier process.
        if workers > 1:
            from scripts.aimnet_workers import CPUMolecularWorkers

            worker_calculator = CPUMolecularWorkers(
                config, output / "molecular_jobs", workers, threads
            )
            service.calculator = worker_calculator
        identity = service.calculator.identity()
        repository = Path(__file__).resolve().parents[1]
        package = repository / "synplan/chem/reaction"
        sources = [
            *sorted((package / "score").glob("*.py")),
            package / "curation/rebalancing.py",
            Path(__file__).resolve(),
            repository / "scripts/aimnet_workers.py",
        ]
        fingerprint_data = {
            "version": REACTION_CHECKPOINT_VERSION,
            "source_sha256": manifest["source"]["sha256"],
            "config": config.model_dump(mode="json"),
            "calculator": identity,
            "code": {str(p.relative_to(repository)): file_digest(p) for p in sources},
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True).encode()
        ).hexdigest()
        checkpoint = output / "reactions"
        checkpoint.mkdir(exist_ok=True)
        job_path = output / "checkpoint.json"
        if job_path.exists() and read_json(job_path)["fingerprint"] != fingerprint:
            raise IncompatibleCheckpointError(
                "Checkpoint input, settings, models, or calculation code changed; use a new output directory."
            )
        write_json_atomic(job_path, {"fingerprint": fingerprint, **fingerprint_data})
        carry = {}
        for index, reaction in enumerate(reactions):
            path = checkpoint / f"{index:05}.json"
            if not path.exists():
                break
            record = read_json(path)
            if record["reaction"] != mapped_smiles(reaction):
                raise IncompatibleCheckpointError(
                    f"Checkpoint reaction mismatch at {index}"
                )
            completed.append(record["annotation"])
            carry = record["stats"]
        offset = len(completed)
        previous_summary = output / "run_summary.json"
        if previous_summary.exists():
            previous = read_json(previous_summary)
            manifest["previous_runs"] = [
                *previous.get("previous_runs", []),
                {
                    key: previous[key]
                    for key in (
                        "started_utc",
                        "run_status",
                        "elapsed_seconds",
                        "resumed_reactions",
                        "execution",
                    )
                    if key in previous
                },
            ]
        manifest["resumed_reactions"] = offset
        manifest["calculator"] = identity
        manifest["config"] = config.model_dump(mode="json")
        print(
            f"{len(routes)} routes, {len(reactions)} steps; resuming after {offset} steps",
            flush=True,
        )
        manifest["execution"] = {
            "device": config.device,
            "workers": workers,
            "threads_per_worker": threads,
        }
        if worker_calculator is not None:
            unique, scheduled = worker_calculator.prime(
                reactions[offset:], service.cache
            )
            print(
                f"{unique} unique species; {scheduled} scheduled across {workers} CPU workers",
                flush=True,
            )

        def checkpoint_reaction(index, annotation):
            stats = dict(service.last_stats)
            for key in (
                "cache_hits",
                "cache_invalidations",
                "molecular_calculations",
                "molecular_failures",
                "calculation_seconds",
            ):
                stats[key] = stats.get(key, 0) + carry.get(key, 0)
            write_json_atomic(
                checkpoint / f"{offset + index:05}.json",
                {
                    "reaction": mapped_smiles(reactions[offset + index]),
                    "annotation": annotation,
                    "stats": stats,
                },
            )
            count = offset + index + 1
            if count % 10 == 0 or count == len(reactions):
                print(
                    f"{count}/{len(reactions)} steps; {stats['molecular_calculations']} molecules calculated, {stats['cache_hits']} cache hits, {stats['molecular_failures']} molecular failures",
                    flush=True,
                )

        tail = service.assess_reactions(
            reactions[offset:], silent=True, on_reaction=checkpoint_reaction
        )
        completed.extend(tail)
        stats = dict(service.last_stats)
        for key in (
            "cache_hits",
            "cache_invalidations",
            "molecular_calculations",
            "molecular_failures",
            "calculation_seconds",
        ):
            stats[key] += carry.get(key, 0)
        stats.update(
            routes=len(routes),
            reaction_occurrences=len(reactions),
            unique_species=len(
                {
                    species
                    for row in completed
                    for variant in row["variants"]
                    for side in ("reactants", "products")
                    for species in variant.get(side, [])
                }
            ),
            successful_reactions=sum(r["status"] == "ok" for r in completed),
            skipped_reactions=sum(r["status"] == "skipped" for r in completed),
            total_variants=sum(len(r["variants"]) for r in completed),
            successful_variants=sum(r.get("successful_variants", 0) for r in completed),
        )
        manifest.update(
            run_status="complete",
            statistics=stats,
            elapsed_seconds=perf_counter() - started,
        )
        if verify_cache:
            # Retry only successful reactions: failed molecules are intentionally
            # not molecular-cache entries. Verify all successful species via a new assessor.
            eligible = [
                (rxn, row)
                for rxn, row in zip(reactions, completed, strict=True)
                if row["status"] == "ok"
            ]
            warm = AIMNetReactionAssessor(
                config, cache_path, calculator=service.calculator
            )
            original_calculate = service.calculator.calculate

            def unexpected(smiles):
                raise AssertionError(
                    f"Successful cached molecule recalculated: {smiles}"
                )

            service.calculator.calculate = unexpected
            try:
                repeated = warm.assess_reactions([rxn for rxn, _ in eligible])
            finally:
                service.calculator.calculate = original_calculate
            assert warm.last_stats["molecular_calculations"] == 0
            assert [r["variants"] for r in repeated] == [
                r["variants"] for _, r in eligible
            ]
            manifest["cached_rerun"] = {
                **warm.last_stats,
                "scope": "all previously successful reactions",
            }
        export_results(routes, labels, completed, output, manifest)
        write_json_atomic(output / "run_summary.json", manifest)
        return manifest
    except IncompatibleCheckpointError:
        # Refusing a different job must not overwrite its completed report.
        raise
    except BaseException as exc:
        # Preserve the original exception and its full traceback in the job log.
        (output / "error.log").write_text(traceback.format_exc())
        manifest.update(
            run_status="aborted",
            error={"type": type(exc).__name__, "detail": str(exc)},
            elapsed_seconds=perf_counter() - started,
        )
        completed.extend(service.completed_reactions)
        completed = completed[: len(reactions)]
        completed.extend(
            _aborted(r, f"{type(exc).__name__}: {exc}")
            for r in reactions[len(completed) :]
        )
        try:
            export_results(routes, labels, completed, output, manifest)
        except Exception:
            (output / "report_error.log").write_text(traceback.format_exc())
        write_json_atomic(output / "run_summary.json", manifest)
        raise
    finally:
        if workers > 1 and hasattr(service.calculator, "close"):
            service.calculator.close()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--aimnet-model", type=Path)
    parser.add_argument("--loqi-model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-cache", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Isolated CPU molecular workers; defaults to one",
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    result = run_assessment(
        args.input,
        args.output,
        AIMNetConfig(
            device=args.device,
            aimnet_checkpoint=args.aimnet_model,
            loqi_checkpoint=args.loqi_model,
        ),
        args.cache,
        verify_cache=args.verify_cache,
        workers=args.workers,
        threads=args.threads,
    )
    print(json.dumps(result.get("statistics", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
