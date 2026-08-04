#!/usr/bin/env python
"""One-command release: bump the version and update CHANGELOG + docs switcher."""

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
SWITCHER = ROOT / "docs" / "_static" / "switcher.json"


def uv(*args: str) -> str:
    """Run a uv subcommand in the repo root and return its stdout."""
    result = subprocess.run(
        ["uv", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def next_version(part: str) -> str:
    out = uv("version", "--bump", part, "--dry-run")  # "name 1.5.1 => 1.5.2"
    if "=>" not in out:
        sys.exit(f"could not parse new version from: {out!r}")
    return out.split("=>")[-1].strip()


def last_released_version() -> str:
    """Newest ``vX.Y.Z`` git tag.

    The compare links and the switcher point at published tags, so the previous
    version has to come from the tags. ``pyproject.toml`` may hold a version that
    was bumped but never tagged, which would link to a tag that does not exist.
    """
    tags = subprocess.run(
        ["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*", "--sort=-v:refname"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not tags:
        sys.exit("no vX.Y.Z tag found; cannot derive the previous released version")
    return tags[0].removeprefix("v")


def update_changelog(text: str, new: str, prev: str, today: str) -> str:
    if f"## [{new}]" in text:
        sys.exit(f"CHANGELOG.md already has a [{new}] section")
    marker = "## [Unreleased]\n"
    if marker not in text:
        sys.exit("CHANGELOG.md: '## [Unreleased]' heading not found")
    text = text.replace(marker, f"{marker}\n## [{new}] - {today}\n", 1)

    base = re.search(r"\[Unreleased\]:\s*(https?://\S+?)/compare/", text)
    if not base:
        sys.exit("CHANGELOG.md: could not find the [Unreleased] footer link")
    repo = base.group(1)
    unreleased = f"[Unreleased]: {repo}/compare/v{new}...HEAD"
    text = re.sub(r"\[Unreleased\]:\s*\S+", unreleased, text, count=1)
    link = f"[{new}]: {repo}/compare/v{prev}...v{new}"
    return text.replace(f"{unreleased}\n", f"{unreleased}\n{link}\n", 1)


def format_switcher(entries: list[dict]) -> str:
    lines = ["["]
    for i, e in enumerate(entries):
        obj = '{{"name": {}, "version": {}, "url": {}}}'.format(
            json.dumps(e["name"]), json.dumps(e["version"]), json.dumps(e["url"])
        )
        lines.append("  " + obj + ("," if i < len(entries) - 1 else ""))
    lines.append("]")
    return "\n".join(lines) + "\n"


def update_switcher(text: str, new: str, prev: str) -> str:
    entries = json.loads(text)
    stable = next(i for i, e in enumerate(entries) if e.get("version") == "stable")
    entries[stable]["name"] = f"{new} (stable)"
    if not any(e.get("version") == f"v{prev}" for e in entries):
        entries.insert(
            stable + 1, {"name": prev, "version": f"v{prev}", "url": f"/en/v{prev}/"}
        )
    return format_switcher(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["patch", "minor", "major"])
    parser.add_argument(
        "--dry-run", action="store_true", help="preview without writing"
    )
    args = parser.parse_args()

    current = uv("version", "--short")
    prev = last_released_version()
    new = next_version(args.part)
    today = date.today().isoformat()

    if current != prev:
        print(
            f"note: pyproject.toml is at {current} but the last tag is v{prev}. "
            f"Links point at v{prev}; fold any [{current}] CHANGELOG section into "
            f"[{new}] by hand."
        )

    changelog = update_changelog(CHANGELOG.read_text(), new, prev, today)
    switcher = update_switcher(SWITCHER.read_text(), new, prev)

    if args.dry_run:
        print(f"[dry-run] {current} => {new}")
        print(f"  pyproject.toml / uv.lock  {current} -> {new}")
        print(f"  CHANGELOG.md              [Unreleased] -> [{new}] - {today}")
        print(f"  switcher.json             {prev} archived, {new} (stable)")
        print("  docker_images.rst         auto (Sphinx |release|)")
        return

    uv("version", "--bump", args.part, "--no-sync")
    CHANGELOG.write_text(changelog)
    SWITCHER.write_text(switcher)

    print(f"Bumped {current} -> {new}")
    print("Updated: pyproject.toml, uv.lock, CHANGELOG.md, switcher.json")
    print("\nNext (review the diff, then):")
    print("  git add -A")
    print(f'  git commit -m "chore: release v{new}"')
    print(f"  git tag v{new}")
    print("  git push && git push --tags")


if __name__ == "__main__":
    main()
