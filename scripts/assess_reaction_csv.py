"""Stream a reaction CSV into a resumable SQLite thermochemistry registry.

Run with python -m scripts.assess_reaction_csv --help. The occurrences table
preserves input order, source records and annotations. Molecules are stored
once per calculation method in the molecules table.
"""

import csv
import gzip
import json
import sqlite3
from argparse import ArgumentParser
from collections import Counter
from contextlib import closing
from pathlib import Path
from time import perf_counter

from chython import smiles
from chython.containers import ReactionContainer
from chython.exceptions import ValenceError

from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.aimnet import cache
from synplan.chem.reaction.score.aimnet.molecular import file_digest


def assess_csv(
    source, database, config, *, equation_column, curated_column=None, calculator=None
):
    source, database = Path(source), Path(database)
    if source.resolve() == database.resolve() or database.suffix not in {
        ".sqlite",
        ".sqlite3",
        ".db",
    }:
        raise ValueError("Use a separate SQLite output file")
    if curated_column:
        config = config.model_copy(update={"balance_policy": "validate"})
    assessor = AIMNetReactionAssessor(config, database, calculator=calculator)
    package = Path(cache.__file__).parent
    manifest = {
        "source_sha256": file_digest(source),
        "equation_column": equation_column,
        "curated_column": curated_column,
        "config": assessor.config.model_dump(mode="json"),
        "calculator": assessor.run_identity(),
        "implementation": {
            p.name: file_digest(p)
            for p in (
                *sorted(package.glob("*.py")),
                Path(__file__),
                package.parent.parent / "curation/rebalancing.py",
            )
        },
    }
    run = cache._fingerprint(manifest)
    manifest["source_path"] = str(source.resolve())
    opener = gzip.open if source.suffix == ".gz" else open
    stats = Counter(
        {
            "resumed": 0,
            "processed": 0,
            "molecular_calculations": 0,
            "calculation_seconds": 0.0,
            "molecular_write_seconds": 0.0,
            "reaction_write_seconds": 0.0,
        }
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    with (
        opener(source, "rt", encoding="utf-8-sig", newline="") as inp,
        closing(sqlite3.connect(database, timeout=30)) as db,
    ):
        reader = csv.DictReader(inp)
        required = {c for c in (equation_column, curated_column) if c is not None}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"Missing required CSV columns: {required - set(reader.fieldnames or ())}"
            )
        with db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY, manifest TEXT NOT NULL, complete INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS occurrences (campaign TEXT, ordinal INTEGER, source_record TEXT NOT NULL, annotation TEXT NOT NULL, calc_seconds REAL, molecule_write_seconds REAL, PRIMARY KEY(campaign, ordinal))"
            )
            db.execute(
                "INSERT OR IGNORE INTO campaigns VALUES (?,?,0)",
                (run, json.dumps(manifest, allow_nan=False)),
            )
        for index, row in enumerate(reader):
            if db.execute(
                "SELECT 1 FROM occurrences WHERE campaign=? AND ordinal=?", (run, index)
            ).fetchone():
                stats["resumed"] += 1
                continue
            text = row[curated_column or equation_column]
            try:
                reaction = smiles(text) if text else None
                if not isinstance(reaction, ReactionContainer):
                    raise ValueError(
                        "Expected a reaction equation with reactants and products"
                    )
            except (ValueError, ValenceError) as exc:
                annotation = {
                    "status": "failed",
                    "original_reaction": row[equation_column],
                    "variants": [],
                    "failure": {
                        "category": "errored",
                        "stage": "input",
                        "reason": "invalid_reaction",
                        "detail": str(exc),
                    },
                }
                metrics = {}
            else:
                annotation = assessor.assess_reactions([reaction])[0]
                metrics = assessor.last_stats
                # Calculation and source metadata are shared by every occurrence.
                for part in [
                    annotation,
                    *annotation.get("additional_phases", {}).values(),
                ]:
                    calculation = part.pop("calculation", {})
                    identity = calculation.get("identity")
                    part["molecular_method"] = (
                        cache._fingerprint(identity) if identity else None
                    )
                    records = part.pop("molecules", {})
                    part["species"] = list(records)
                    derived = {s: m for s, m in records.items() if m.get("corrections")}
                    if derived:
                        part["derived_species"] = {
                            s: {
                                k: v
                                for k, v in m.items()
                                if k
                                in {
                                    "energy_eV",
                                    "enthalpy_eV",
                                    "gibbs_eV",
                                    "reference_base_species",
                                    "corrections",
                                }
                            }
                            for s, m in derived.items()
                        }
            for key in (
                "molecular_calculations",
                "calculation_seconds",
                "molecular_write_seconds",
            ):
                stats[key] += metrics.get(key, 0)
            started = perf_counter()
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO occurrences VALUES (?,?,?,?,?,?)",
                    (
                        run,
                        index,
                        json.dumps(row),
                        json.dumps(annotation, allow_nan=False),
                        metrics.get("calculation_seconds", 0),
                        metrics.get("molecular_write_seconds", 0),
                    ),
                )
            stats["reaction_write_seconds"] += perf_counter() - started
            stats["processed"] += 1
        with db:
            db.execute("UPDATE campaigns SET complete=1 WHERE id=?", (run,))
    return {"campaign": run, **stats}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--equation-column", required=True)
    parser.add_argument(
        "--curated-column",
        help="Use a supplied equation verbatim and validate conservation",
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = AIMNetConfig.from_yaml(str(args.config)) if args.config else AIMNetConfig()
    print(
        json.dumps(
            assess_csv(
                args.source,
                args.database,
                config,
                equation_column=args.equation_column,
                curated_column=args.curated_column,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
