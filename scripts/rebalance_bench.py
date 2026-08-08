"""Per-dataset success rate and accuracy for the missing-molecule imputer.

Runs against SynRBL's validation set (10.1186/s13321-024-00875-4).  That set is
not an independent answer key: `expected_reaction` is SynRBL's own output that a
reviewer accepted, so agreement with it is the most this can measure.  Two kinds
of row cannot measure even that and are dropped from the accuracy denominator
rather than counted as losses:

* no `expected_reaction` at all (327 rows) — unwinnable by construction.
* an `expected_reaction` that does not itself balance (250 rows) — the answer
  key fails the task it defines.  Every row whose reference spells free oxygen
  `[O]` lands here, because chython reads that as water.

SynRBL's own benchmark counts both as wrong.  Reproduce that with `--synrbl`.

* success     = balanced results / reactions that were not already balanced
* accuracy    = exact matches / results with a usable reference
* end-to-end  = success x accuracy

Usage::

    uv run python scripts/rebalance_bench.py <validation_set.csv> [limit]
        [--redox] [--unmapped] [--synrbl]
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
        pass  # already aromatic, or a ring that will not kekulise; compare as is
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


def usable_reference(expected: str):
    """The reference to score against, or ``None`` where it cannot be one.

    A reference that does not itself balance is not an answer to a balancing
    task; scoring against it measures nothing but SynRBL's own defects.
    """
    if not expected:
        return None
    try:
        parsed = reference(expected)
    except Exception:
        return None
    return None if reaction_imbalance(parsed) else parsed


def bench(
    rows,
    add_redox_agents: bool = False,
    unmapped: bool = False,
    synrbl_metric: bool = False,
):
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
            reaction = smiles(str(reaction))
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
        expected_reaction = usable_reference(expected)
        if expected_reaction is None:
            if not synrbl_metric:
                continue  # nothing here can be right or wrong
            counts["scored"] += 1
            continue  # SynRBL's metric: no usable reference counts as wrong
        counts["scored"] += 1
        try:
            if key(result) == key(expected_reaction):
                counts["correct"] += 1
        except Exception as exc:  # a reference chython cannot canonicalise
            failures[f"compare: {type(exc).__name__}"] += 1
    return stats, failures


def report(stats, failures, elapsed: float) -> None:
    template = "{:<24}{:>6}{:>6}{:>7}{:>10.2f}{:>8.2f}{:>13.2f}"
    print(
        f"{'dataset':<24}{'N':>6}{'bal':>6}{'ref':>7}{'success%':>10}{'acc%':>8}"
        f"{'end-to-end%':>13}"
    )
    total = dict.fromkeys(("n", "balanced", "solved", "scored", "correct"), 0)
    for dataset, counts in sorted(stats.items()):
        for field in total:
            total[field] += counts[field]
        print(
            template.format(
                dataset,
                counts["n"],
                counts["balanced"],
                counts["scored"],
                *_rates(counts),
            )
        )
    print(
        template.format(
            "TOTAL", total["n"], total["balanced"], total["scored"], *_rates(total)
        )
    )
    print(f"{elapsed:.1f}s ({total['n'] / elapsed:.0f} rxn/s)")
    if failures:
        print("\nunsolved, most common first:")
        for message, count in sorted(failures.items(), key=lambda x: -x[1])[:12]:
            print(f"  {count:>6}  {message}")


def _rates(counts) -> tuple[float, float, float]:
    """Success, accuracy and their product."""
    unbalanced = counts["n"] - counts["balanced"]
    success = 100 * counts["solved"] / unbalanced if unbalanced else 0.0
    accuracy = 100 * counts["correct"] / counts["scored"] if counts["scored"] else 0.0
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
    synrbl_metric = "--synrbl" in sys.argv[1:]
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    rows = load(positional[0], int(positional[1]) if len(positional) > 1 else None)
    start = time.monotonic()
    stats, failures = bench(
        rows,
        add_redox_agents=redox,
        unmapped=unmapped,
        synrbl_metric=synrbl_metric,
    )
    report(stats, failures, time.monotonic() - start)


if __name__ == "__main__":
    main()
