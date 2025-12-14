#!/usr/bin/env python
"""
=============================================================================
SAScore Benchmark: Plot Results
=============================================================================

This script visualizes benchmark results and compares with reference methods
from the literature.

QUICK START
-----------

1. First, run the benchmark (if not already done):

   uv run sascore-benchmark

2. Plot the results:

   uv run sascore-plot

   Or directly:
   uv run python scripts/sascore_bench/plot_results.py

   To specify a specific results folder:
   uv run sascore-plot /path/to/benchmark_results_combined_policy/<timestamp>

OUTPUT
------

The script generates:
  - benchmark_comparison.png: Bar chart comparing solve rates
  - benchmark_comparison.pdf: Same chart in PDF format
  - Console summary table with all methods

REFERENCE DATA
--------------

Comparison includes data from Roucairol et al. paper:
  - AiZynthFinder
  - SynPlanner (standard)
  - SynPlanner fulltime
  - NMCS 2 top 10
  - pBFS / pBFS 500
  - pNMCS / pNMCS 500

=============================================================================
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference data from Roucairol et al. paper (solve rates in %)
REFERENCE_DATA = {
    "AiZynthFinder": {2: 66, 3: 31, 4: 7, 5: 5},
    "SynPlanner": {2: 74, 3: 35, 4: 14, 5: 1},
    "SynPlanner fulltime": {2: 79, 3: 43, 4: 20, 5: 3},
    "NMCS 2 top 10": {2: 86, 3: 60, 4: 28, 5: 11},
    "pBFS": {2: 86, 3: 60, 4: 28, 5: 5},
    "pBFS 500": {2: 89, 3: 65, 4: 29, 5: 9},
    "pNMCS": {2: 87, 3: 67, 4: 33, 5: 12},
    "pNMCS 500": {2: 85, 3: 68, 4: 34, 5: 12},
}

# SAScore range mapping
SASCORE_MAPPING = {
    "1.5_2.5": 2,
    "2.5_3.5": 3,
    "3.5_4.5": 4,
    "4.5_5.5": 5,
}


def load_benchmark_results(results_folder: Path) -> dict:
    """Load benchmark results from CSV files."""
    results = {}

    for sascore_dir in results_folder.iterdir():
        if not sascore_dir.is_dir() or not sascore_dir.name.startswith("sascore_"):
            continue

        stats_file = sascore_dir / "stats.csv"
        if not stats_file.exists():
            continue

        # Extract SAScore range from folder name
        sascore_range = sascore_dir.name.replace("sascore_", "")
        if sascore_range not in SASCORE_MAPPING:
            continue

        sascore_category = SASCORE_MAPPING[sascore_range]

        # Read CSV and calculate solve rate
        df = pd.read_csv(stats_file)
        total = len(df)
        solved = df["solved"].sum()
        solve_rate = (solved / total * 100) if total > 0 else 0

        results[sascore_category] = {
            "total": total,
            "solved": solved,
            "solve_rate": solve_rate,
        }

    return results


def plot_comparison(benchmark_results: dict, output_path: Path = None):
    """Create comparison bar chart."""

    # SAScore categories to plot
    categories = [2, 3, 4, 5]
    x = np.arange(len(categories))

    # Methods to include (reference + our results) - ordered for comparison
    # Put Combined Policy right after SynPlanner fulltime
    methods = [
        "AiZynthFinder",
        "SynPlanner",
        "SynPlanner fulltime",
        "Combined Policy",  # Our method - placed here for comparison
        "NMCS 2 top 10",
        "pBFS",
        "pBFS 500",
        "pNMCS",
        "pNMCS 500",
    ]
    n_methods = len(methods)

    # Bar width and positions
    width = 0.08
    offsets = np.linspace(
        -(n_methods - 1) / 2 * width, (n_methods - 1) / 2 * width, n_methods
    )

    # Colors matching the paper style (ordered to match methods list)
    colors = [
        "#87CEEB",  # AiZynthFinder - light blue
        "#FFB6C1",  # SynPlanner - light pink
        "#F5DEB3",  # SynPlanner fulltime - wheat
        "#FF0000",  # Combined Policy - red (our method)
        "#808080",  # NMCS 2 top 10 - gray
        "#4B0082",  # pBFS - dark purple
        "#90EE90",  # pBFS 500 - light green
        "#6495ED",  # pNMCS - cornflower blue
        "#FFA07A",  # pNMCS 500 - light salmon
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot bars for each method
    for i, method in enumerate(methods):
        values = []
        for cat in categories:
            if method == "Combined Policy":
                val = benchmark_results.get(cat, {}).get("solve_rate", 0)
            else:
                val = REFERENCE_DATA[method].get(cat, 0)
            values.append(val)

        bars = ax.bar(
            x + offsets[i],
            values,
            width,
            label=method,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.annotate(
                    f"{int(val)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    # Customize plot
    ax.set_xlabel("SAScore", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title(
        "Retrosynthesis Benchmark: Combined Policy vs Reference Methods", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Add horizontal line at 100%
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()

    return fig


def print_summary(benchmark_results: dict):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 70)

    categories = [2, 3, 4, 5]

    # Header
    header = f"{'Method':<25}" + "".join([f"{'SAS ' + str(c):<12}" for c in categories])
    print(header)
    print("-" * 70)

    # Reference methods
    for method, data in REFERENCE_DATA.items():
        row = f"{method:<25}"
        for cat in categories:
            row += f"{data.get(cat, 0):<12}"
        print(row)

    # Our results
    row = f"{'Combined Policy':<25}"
    for cat in categories:
        val = benchmark_results.get(cat, {}).get("solve_rate", 0)
        row += f"{val:.0f}{'*':<11}"
    print(row)

    print("=" * 70)
    print("* Our results (Combined filtering + ranking policy)")


def main():
    """Main entry point for the plotting script."""
    # Find results folder
    results_root = Path("benchmark_results_combined_policy")

    if len(sys.argv) > 1:
        results_folder = Path(sys.argv[1])
    else:
        # Use most recent results
        if not results_root.exists():
            print(f"Error: Results folder not found: {results_root}")
            print("Run the benchmark first:")
            print("  uv run sascore-benchmark")
            sys.exit(1)

        subfolders = sorted(results_root.iterdir(), reverse=True)
        if not subfolders:
            print("Error: No benchmark results found.")
            sys.exit(1)

        results_folder = subfolders[0]

    print(f"Loading results from: {results_folder}")

    # Load benchmark results
    benchmark_results = load_benchmark_results(results_folder)

    if not benchmark_results:
        print("Error: No benchmark results found in folder.")
        sys.exit(1)

    # Print summary
    print_summary(benchmark_results)

    # Create plot
    output_path = results_folder / "benchmark_comparison.png"
    plot_comparison(benchmark_results, output_path)

    # Also save as PDF
    pdf_path = results_folder / "benchmark_comparison.pdf"
    plot_comparison(benchmark_results, pdf_path)


if __name__ == "__main__":
    main()
