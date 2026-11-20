"""Compare a published scalar column with the matching calculated phase in a CSV campaign."""

import json
import sqlite3
from argparse import ArgumentParser
from collections import Counter
from contextlib import closing
from math import isfinite
from pathlib import Path

QUANTITIES = {
    "delta_e": "kcal/mol",
    "delta_h": "kcal/mol",
    "delta_g": "kcal/mol",
    "delta_s_cal_mol_K": "cal/mol/K",
}


def compare(database, *, reference_column, phase, quantity="delta_g", campaign=None):
    if phase not in {"gas", "cpcm_water"} or quantity not in QUANTITIES:
        raise ValueError("Unknown phase or thermochemical quantity")
    counts = Counter()
    difference_sum = absolute_sum = maximum = 0.0
    worst = None
    with closing(
        sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        if campaign is None:
            ids = db.execute("SELECT id FROM campaigns LIMIT 2").fetchall()
            if len(ids) != 1:
                raise ValueError("Select a campaign ID from the campaigns table")
            campaign = ids[0][0]
        entry = db.execute(
            "SELECT manifest,complete FROM campaigns WHERE id=?", (campaign,)
        ).fetchone()
        if entry is None:
            raise ValueError("Unknown campaign ID")
        manifest = json.loads(entry[0])
        primary = manifest["config"].get("phase", "gas")
        for ordinal, raw, annotation in db.execute(
            "SELECT ordinal,source_record,annotation FROM occurrences WHERE campaign=? ORDER BY ordinal",
            (campaign,),
        ):
            source, row = json.loads(raw), json.loads(annotation)
            if reference_column not in source:
                raise ValueError(
                    f"Reference column {reference_column!r} missing at ordinal {ordinal}"
                )
            text = source[reference_column]
            reference = float(text) if text is not None and str(text).strip() else None
            if reference is not None and not isfinite(reference):
                raise ValueError(f"Nonfinite reference at ordinal {ordinal}")
            part = (
                row
                if primary == phase
                else row.get("additional_phases", {}).get(phase, {})
            )
            variants = part.get("variants", [])
            counts["rows"] += 1
            if len(variants) > 1:
                counts["ambiguous_multiple_variants"] += 1
                continue
            prediction = (
                variants[0].get(quantity)
                if variants and variants[0]["status"] == "ok"
                else None
            )
            if prediction is None or reference is None:
                category = (
                    "both_missing"
                    if prediction is None and reference is None
                    else (
                        "missing_calculation"
                        if prediction is None
                        else "missing_reference"
                    )
                )
                counts[category] += 1
                continue
            error = prediction - reference
            if not isfinite(error):
                raise ValueError(f"Nonfinite residual at ordinal {ordinal}")
            counts["compared"] += 1
            difference_sum += error
            absolute_sum += abs(error)
            if worst is None or abs(error) > maximum:
                maximum, worst = abs(error), ordinal
    n = counts["compared"]
    return {
        "campaign": campaign,
        "complete": bool(entry[1]),
        "input_sha256": manifest["source_sha256"],
        "reference_column": reference_column,
        "phase": phase,
        "quantity": quantity,
        "units": QUANTITIES[quantity],
        "counts": dict(counts),
        "mean_signed_error": difference_sum / n if n else None,
        "mean_absolute_error": absolute_sum / n if n else None,
        "maximum_absolute_error": maximum if n else None,
        "worst_input_ordinal": worst,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--reference-column", required=True)
    parser.add_argument("--phase", choices=["cpcm_water", "gas"], required=True)
    parser.add_argument("--quantity", choices=QUANTITIES, default="delta_g")
    parser.add_argument("--campaign")
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
