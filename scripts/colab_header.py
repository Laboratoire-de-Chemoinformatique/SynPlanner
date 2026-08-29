"""Give every tutorial a Colab badge and an install cell, or refresh the ones it has.

Colab opens any notebook straight from GitHub, so a tutorial needs nothing but a
link to itself and a cell that installs SynPlanner when — and only when — it is
running there. Run this after adding a tutorial:

    uv run python scripts/colab_header.py            # write
    uv run python scripts/colab_header.py --check    # fail if anything is missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = "Laboratoire-de-Chemoinformatique/SynPlanner"

#: The badge opens the notebook from this ref and the install takes the package
#: from it, so a reader gets a notebook and a package built from one tree. It is
#: ``main`` rather than the last tag because the tutorials use API as soon as it
#: lands: a released package could not run them.
REF = "main"
TUTORIALS = Path(__file__).resolve().parent.parent / "tutorials"

#: How the two generated cells are recognised on a rerun.
BADGE_MARK = "colab-badge.svg"
INSTALL_MARK = "# Colab setup"

INSTALL = """# Colab setup — this cell does nothing when you run the notebook locally.
import subprocess
import sys

if "google.colab" in sys.modules:
    subprocess.run(
        [
            "pip",
            "install",
            "-q",
            "git+https://github.com/{repo}.git@{ref}",
        ],
        check=True,
    )
    print("SynPlanner installed. If an import below fails, restart the runtime.")
"""


def badge(notebook: str) -> str:
    """The markdown badge that opens ``notebook`` in Colab."""

    url = f"https://colab.research.google.com/github/{REPO}/blob/{REF}/tutorials/{notebook}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/{BADGE_MARK})]({url})"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def update(path: Path) -> bool:
    """Add or refresh the header of one notebook. True when the file changed."""

    notebook = json.loads(path.read_text())
    cells = notebook["cells"]
    before = json.dumps(notebook, sort_keys=True)

    # the badge rides on the first markdown cell, so it renders at the top on
    # GitHub too; a notebook that opens with code gets one of its own
    at = next(
        (i for i, cell in enumerate(cells) if cell["cell_type"] == "markdown"), None
    )
    if at is None or at > 1:
        cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": []})
        at = 0
    title = cells[at]
    source = "".join(title["source"])
    line = badge(path.name)
    if BADGE_MARK in source:
        source = "\n".join(
            line if BADGE_MARK in existing else existing
            for existing in source.split("\n")
        )
    elif source.strip():
        heading, _, rest = source.partition("\n")
        source = f"{heading}\n\n{line}\n{rest}"
    else:
        source = line
    title["source"] = source.splitlines(keepends=True)

    # the install has to run before the first import, so it sits right after the badge
    install = INSTALL.format(repo=REPO, ref=REF)
    existing = next(
        (i for i, cell in enumerate(cells) if INSTALL_MARK in "".join(cell["source"])),
        None,
    )
    if existing is not None:
        cells.pop(existing)
    cells.insert(at + 1, code_cell(install))

    if json.dumps(notebook, sort_keys=True) == before:
        return False
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report what is missing, write nothing"
    )
    args = parser.parse_args(argv)

    stale = []
    for path in sorted(TUTORIALS.glob("*.ipynb")):
        if args.check:
            source = path.read_text()
            if BADGE_MARK not in source or INSTALL_MARK not in source:
                stale.append(path.name)
        elif update(path):
            print(f"updated {path.name}")
    if args.check and stale:
        print("no Colab header: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
