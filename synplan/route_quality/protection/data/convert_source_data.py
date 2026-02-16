"""Convert source protection-strategy dataset files to SynPlanner format.

Source: Westerlund, A. M. et al. "Toward lab-ready AI synthesis plans with
protection strategies and route scoring." ChemRxiv, 2025.
https://doi.org/10.26434/chemrxiv-2025-gdrr8

This script reads the original tab-delimited files and produces:
  - competing_groups.yaml   (FG SMARTS patterns)
  - halogen_groups.yaml     (halogen SMARTS patterns)
  - incompatibility_matrix.tsv  (FG x FG matrix, tab-delimited with row labels)
  + copies supporting files (templates, label mapping)

Usage:
    python convert_source_data.py <source_dir> [--output-dir <dir>]

Where <source_dir> is the path to the "Datasets for protection strategies..."
directory containing the original .txt / .csv / .json files.
"""

import argparse
import csv
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# YAML representer for OrderedDict (preserves insertion order)
# ---------------------------------------------------------------------------

def _represent_ordereddict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, _represent_ordereddict)


# ---------------------------------------------------------------------------
# Halogen family derivation
# ---------------------------------------------------------------------------

_HALOGEN_FAMILIES = ["Fluoride", "Chloride", "Bromide", "Iodide", "Triflate"]


def _derive_halogen_family(name: str) -> str:
    """Derive the halogen family from the pattern name.

    E.g. "AcidX-Fluoride_SaturatedAliphatic" -> "fluoride"
    """
    for fam in _HALOGEN_FAMILIES:
        if fam in name:
            return fam.lower()
    return "unknown"


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def convert_competing_groups(src_path: Path, dst_path: Path) -> int:
    """Convert protection_reactive_function_SMARTS.txt -> competing_groups.yaml.

    TSV columns (no header row):
        0: category    (e.g. "AminoAcid")
        1: subcategory (e.g. "NonProlineAlphaAminoAcid")
        2: name        (e.g. "NonProlineAlphaAminoAcid_unprotected")
        3: smarts      (detection SMARTS)
        4: template_smarts (SMARTS with :1/:2 atom mapping)
        5: modification_type ("label" or "clip")
        6: example     (example SMILES)
        7: numeric     (integer)

    Output YAML structure (same as current loader expects):
        category:
          - name: ...
            smarts: ...
            template_smarts: ...
            modification_type: ...
            example: ...
    """
    groups = OrderedDict()
    count = 0

    with open(src_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 8:
                continue
            category = row[0]
            name = row[2]
            smarts_str = row[3]
            template_smarts = row[4]
            mod_type = row[5]
            example = row[6]

            entry = OrderedDict()
            entry["name"] = name
            entry["smarts"] = smarts_str
            entry["template_smarts"] = template_smarts
            entry["modification_type"] = mod_type
            entry["example"] = example

            if category not in groups:
                groups[category] = []
            groups[category].append(entry)
            count += 1

    header = (
        "# Reactive functional group SMARTS patterns for protection-group analysis.\n"
        "#\n"
        "# Converted from Westerlund et al., ChemRxiv, 2025.\n"
        "# https://doi.org/10.26434/chemrxiv-2025-gdrr8\n"
        "#\n"
        "# Organised by reactivity category. Each entry has:\n"
        "#   name              - unique label, also used as row/column key in the incompatibility matrix\n"
        "#   smarts            - SMARTS string for functional group detection\n"
        "#   template_smarts   - SMARTS with atom mapping for protection template application\n"
        "#   modification_type - 'label' (mark only) or 'clip' (remove leaving group)\n"
        "#   example           - example molecule SMILES\n"
        "#\n"
        f"# Total: {count} patterns in {len(groups)} categories.\n"
        "\n"
    )
    with open(dst_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.dump(dict(groups), fh, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=200)

    return count


def convert_halogen_groups(src_path: Path, dst_path: Path) -> int:
    """Convert halogen_reactive_function_SMARTS.txt -> halogen_groups.yaml.

    Same TSV format as competing groups. Family derived from name.

    Output YAML structure (same as current loader expects):
        pattern_name:
          smarts: ...
          family: ...
          template_smarts: ...
          modification_type: ...
          example: ...
    """
    halogens = OrderedDict()
    count = 0

    with open(src_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 8:
                continue
            name = row[2]
            smarts_str = row[3]
            template_smarts = row[4]
            mod_type = row[5]
            example = row[6]
            family = _derive_halogen_family(name)

            entry = OrderedDict()
            entry["smarts"] = smarts_str
            entry["family"] = family
            entry["template_smarts"] = template_smarts
            entry["modification_type"] = mod_type
            entry["example"] = example

            halogens[name] = entry
            count += 1

    header = (
        "# Halogen SMARTS patterns for competing halogen site detection.\n"
        "#\n"
        "# Converted from Westerlund et al., ChemRxiv, 2025.\n"
        "# https://doi.org/10.26434/chemrxiv-2025-gdrr8\n"
        "#\n"
        "# Grouped by halogen family for same-family competing detection.\n"
        "#   smarts            - SMARTS string for halogen detection\n"
        "#   family            - halogen family (fluoride/chloride/bromide/iodide/triflate)\n"
        "#   template_smarts   - SMARTS with atom mapping\n"
        "#   modification_type - 'clip' (remove leaving group)\n"
        "#   example           - example molecule SMILES\n"
        "#\n"
        f"# Total: {count} patterns.\n"
        "\n"
    )
    with open(dst_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.dump(dict(halogens), fh, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=200)

    return count


def convert_incompatibility_matrix(src_path: Path, dst_path: Path) -> int:
    """Convert protection_SMARTS_incompatibility.csv -> incompatibility_matrix.tsv.

    Source TSV format:
        Row 0: header with N FG names (tab-separated, no row label column)
        Rows 1-N: N tab-separated integer values (0/1/2)
        Row order matches header order (row[i] = FG name[i])

    Output TSV format (with explicit row labels):
        First row: empty cell + N column names (tab-separated)
        Data rows: row_name + N values (tab-separated)
    """
    with open(src_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows = list(reader)

    header = rows[0]
    n_cols = len(header)
    data_rows = rows[1:]

    if len(data_rows) != n_cols:
        print(f"  WARNING: matrix has {n_cols} columns but {len(data_rows)} data rows")

    with open(dst_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        # Header row: empty first cell + column names
        writer.writerow([""] + header)
        # Data rows: row name + values
        for i, data_row in enumerate(data_rows):
            if i >= n_cols:
                break
            row_name = header[i]
            writer.writerow([row_name] + data_row[:n_cols])

    return n_cols


def copy_supporting_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy protection_group_templates.csv and reactive_function_label_mapping.json."""
    for name in ("protection_group_templates.csv", "reactive_function_label_mapping.json"):
        src = src_dir / name
        dst = dst_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {name}")
        else:
            print(f"  WARNING: {name} not found in source directory")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Westerlund dataset to SynPlanner YAML")
    parser.add_argument("source_dir", help="Path to Westerlund dataset directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same directory as this script)")
    args = parser.parse_args()

    src_dir = Path(args.source_dir)
    dst_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent

    if not src_dir.is_dir():
        parser.error(f"Source directory not found: {src_dir}")

    print(f"Source: {src_dir}")
    print(f"Output: {dst_dir}")
    print()

    # 1. Competing groups (FGs)
    fg_src = src_dir / "protection_reactive_function_SMARTS.txt"
    fg_dst = dst_dir / "competing_groups.yaml"
    n_fg = convert_competing_groups(fg_src, fg_dst)
    print(f"competing_groups.yaml: {n_fg} FG patterns")

    # 2. Halogen groups
    hal_src = src_dir / "halogen_reactive_function_SMARTS.txt"
    hal_dst = dst_dir / "halogen_groups.yaml"
    n_hal = convert_halogen_groups(hal_src, hal_dst)
    print(f"halogen_groups.yaml: {n_hal} halogen patterns")

    # 3. Incompatibility matrix
    mat_src = src_dir / "protection_SMARTS_incompatibility.csv"
    mat_dst = dst_dir / "incompatibility_matrix.tsv"
    n_mat = convert_incompatibility_matrix(mat_src, mat_dst)
    print(f"incompatibility_matrix.tsv: {n_mat} x {n_mat}")

    # 4. Supporting files
    copy_supporting_files(src_dir, dst_dir)

    # 5. Sanity check: FG names in SMARTS should match matrix header
    with open(fg_dst, "r", encoding="utf-8") as fh:
        fg_data = yaml.safe_load(fh)
    fg_names = set()
    for entries in fg_data.values():
        for e in entries:
            fg_names.add(e["name"])

    with open(mat_dst, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        mat_header = next(reader)  # first cell is empty, rest are column names
        mat_names = set(mat_header[1:])

    in_matrix_not_smarts = mat_names - fg_names
    in_smarts_not_matrix = fg_names - mat_names

    print()
    if in_matrix_not_smarts:
        print(f"WARNING: {len(in_matrix_not_smarts)} names in matrix but not in SMARTS: {in_matrix_not_smarts}")
    if in_smarts_not_matrix:
        print(f"WARNING: {len(in_smarts_not_matrix)} names in SMARTS but not in matrix: {in_smarts_not_matrix}")
    if not in_matrix_not_smarts and not in_smarts_not_matrix:
        print(f"OK: all {len(fg_names)} FG names match between SMARTS and matrix")

    print("\nDone.")


if __name__ == "__main__":
    main()
