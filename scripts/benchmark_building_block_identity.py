"""Benchmark Chython building-block identities without introducing RDKit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from chython import smiles
from chython.containers import MoleculeContainer

from synplan.chem.building_blocks import molecule_to_inchikey
from synplan.chem.precursor import Precursor
from synplan.chem.utils import safe_canonicalization


def _sample(path: Path, limit: int) -> tuple[MoleculeContainer, ...]:
    molecules = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        columns = [
            name for name in reader.fieldnames or () if name.casefold() == "smiles"
        ]
        if len(columns) != 1:
            raise ValueError(f"{path}: expected exactly one SMILES column")
        column = columns[0]
        for row in reader:
            try:
                molecule = smiles(row[column], ignore=True, ignore_stereo=False)
                if not isinstance(molecule, MoleculeContainer):
                    continue
                molecule = safe_canonicalization(molecule, clean_stereo=False)
                molecule_to_inchikey(molecule)
            except Exception:
                continue
            molecules.append(molecule)
            if len(molecules) == limit:
                break
    if len(molecules) < limit:
        raise ValueError(f"{path}: found only {len(molecules)} valid molecules")
    return tuple(molecules)


def _measure(operation, repeats: int) -> list[float]:
    timings = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        timings.append(perf_counter() - started)
    return timings


def benchmark(path: Path, sample_size: int, repeats: int) -> dict:
    molecules = _sample(path, sample_size)
    smiles_stock = frozenset(str(molecule) for molecule in molecules)
    inchikey_stock = frozenset(molecule_to_inchikey(molecule) for molecule in molecules)
    precursors = tuple(
        Precursor(molecule, canonicalize=False) for molecule in molecules
    )
    for precursor in precursors:
        _ = precursor.inchi_key

    operations = {
        "canonical_smiles_lookup": lambda: sum(
            str(molecule) in smiles_stock for molecule in molecules
        ),
        "chython_inchikey_generation_and_lookup": lambda: sum(
            molecule_to_inchikey(molecule) in inchikey_stock for molecule in molecules
        ),
        "cached_precursor_inchikey_lookup": lambda: sum(
            precursor.inchi_key in inchikey_stock for precursor in precursors
        ),
    }
    results = {}
    for name, operation in operations.items():
        timings = _measure(operation, repeats)
        results[name] = {
            "seconds": timings,
            "mean_seconds": mean(timings),
            "stdev_seconds": stdev(timings) if len(timings) > 1 else 0.0,
            "mean_microseconds_per_query": mean(timings) * 1_000_000 / sample_size,
        }
    return {
        "input": str(path),
        "sample_size": sample_size,
        "repeats": repeats,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_size <= 0 or args.repeats <= 0:
        parser.error("--sample-size and --repeats must be positive")

    result = benchmark(args.catalogue, args.sample_size, args.repeats)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
