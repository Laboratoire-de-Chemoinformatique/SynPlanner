"""Per-dataset success rate and accuracy for the missing-molecule imputer.

Runs against SynRBL's validation set (10.1186/s13321-024-00875-4) with that
paper's metric definitions:

* success     = balanced results / reactions that were not already balanced
* accuracy    = exact matches / balanced results.  A result with no expected
  reaction to check against counts as wrong, as SynRBL's benchmark does.
* end-to-end  = success x accuracy = exact matches / reactions needing a fix.
  The one number to compare against the paper.

Usage::

    uv run python scripts/rebalance_bench.py <validation_set.csv> [limit]
"""

import ast
import csv
import sys
import time
from collections import defaultdict

from chython import smiles

from synplan.chem.data.rebalancing import (
    RebalancingError,
    reaction_imbalance,
    rebalance_reaction,
)
from synplan.chem.utils import strip_reaction_mapping

csv.field_size_limit(10**7)


def reference(text: str):
    """Parse a ground-truth reaction written for RDKit.

    ``[HH]`` is dihydrogen — inside brackets a trailing H is a hydrogen count.
    chython reads it as a lone hydrogen atom and drops the second one, so it is
    respelled before either side is compared.
    """
    return smiles(text.replace("[HH]", "[H][H]"))


def _canonical(molecule) -> str:
    molecule = molecule.copy()
    molecule.remap({n: i for i, n in enumerate(molecule, 1)})
    try:
        molecule.kekule()
        molecule.thiele()
    except Exception:
        pass
    return str(molecule)


def _side_key(molecules) -> list[str]:
    """Canonical SMILES of one side, with free hydrogen folded to an atom count.

    The reference spells free hydrogen both ways — 309 rows use ``[HH]`` and 327
    use ``[H].[H]`` — so comparing that connectivity would measure SynRBL's own
    inconsistency rather than the chemistry.
    """
    named, hydrogen = [], 0
    for molecule in molecules:
        if set(molecule.brutto) == {"H"}:
            hydrogen += molecule.brutto["H"]
        else:
            named.append(_canonical(molecule))
    if hydrogen:
        named.append(f"H*{hydrogen}")
    return sorted(named)


def key(reaction) -> str:
    """Engine-consistent comparison key, never a SMILES compared across engines."""
    left = _side_key(reaction.reactants)
    right = _side_key(reaction.products)
    return ".".join(left) + ">>" + ".".join(right)


def bench(rows, add_redox_agents: bool = False, unmapped: bool = False):
    """Run the imputer over ``(dataset, reaction, expected)`` triples."""
    stats: dict = defaultdict(
        lambda: {"n": 0, "balanced": 0, "solved": 0, "scored": 0, "correct": 0}
    )
    failures: dict = defaultdict(int)
    for dataset, reaction_smiles, expected in rows:
        counts = stats[dataset]
        counts["n"] += 1
        try:
            reaction = smiles(reaction_smiles)
        except Exception as exc:
            failures[f"parse: {type(exc).__name__}"] += 1
            continue
        if unmapped:
            strip_reaction_mapping(reaction)
        if not reaction_imbalance(reaction):
            counts["balanced"] += 1
            continue
        try:
            result = rebalance_reaction(reaction, add_redox_agents=add_redox_agents)
        except RebalancingError as exc:
            failures[str(exc)[:60]] += 1
            continue
        except Exception as exc:
            failures[f"{type(exc).__name__}: {str(exc)[:50]}"] += 1
            continue
        counts["solved"] += 1
        if not expected:
            continue
        counts["scored"] += 1
        try:
            if key(result) == key(reference(expected)):
                counts["correct"] += 1
        except Exception:
            pass
    return stats, failures


def report(stats, failures, elapsed: float) -> None:
    print(
        f"{'dataset':<24}{'N':>6}{'bal':>6}{'success%':>10}{'acc%':>8}"
        f"{'end-to-end%':>13}"
    )
    total = dict.fromkeys(("n", "balanced", "solved", "scored", "correct"), 0)
    for dataset, counts in sorted(stats.items()):
        for field in total:
            total[field] += counts[field]
        rows = [dataset, counts["n"], counts["balanced"], *_rates(counts)]
        print("{:<24}{:>6}{:>6}{:>10.2f}{:>8.2f}{:>13.2f}".format(*rows))
    print(
        "{:<24}{:>6}{:>6}{:>10.2f}{:>8.2f}{:>13.2f}".format(
            "TOTAL", total["n"], total["balanced"], *_rates(total)
        )
    )
    print(f"{elapsed:.1f}s ({total['n'] / elapsed:.0f} rxn/s)")
    if failures:
        print("\nunsolved, most common first:")
        for message, count in sorted(failures.items(), key=lambda x: -x[1])[:12]:
            print(f"  {count:>6}  {message}")


def _rates(counts) -> tuple[float, float, float]:
    """Success, accuracy and their product, on SynRBL's definitions."""
    unbalanced = counts["n"] - counts["balanced"]
    success = 100 * counts["solved"] / unbalanced if unbalanced else 0.0
    # a solved reaction with nothing to check against counts as wrong
    accuracy = 100 * counts["correct"] / counts["solved"] if counts["solved"] else 0.0
    return success, accuracy, success * accuracy / 100


def load(path: str, limit: int | None = None):
    with open(path, encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if limit:
        records = records[:limit]
    return [
        (
            ast.literal_eval(record["datasets"])[0],
            record["reaction"],
            record["expected_reaction"],
        )
        for record in records
    ]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    redox = "--redox" in sys.argv[1:]
    unmapped = "--unmapped" in sys.argv[1:]
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    rows = load(positional[0], int(positional[1]) if len(positional) > 1 else None)
    start = time.monotonic()
    stats, failures = bench(rows, add_redox_agents=redox, unmapped=unmapped)
    report(stats, failures, time.monotonic() - start)


if __name__ == "__main__":
    main()
