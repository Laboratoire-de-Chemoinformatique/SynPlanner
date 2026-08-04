"""Verify the shipped agent skill still matches the code it describes.

The skill and its task index are hand-written on purpose — generating them
would strip the ordering and defaults that make them useful. This test does
the other half: it fails as soon as they name something that no longer exists,
so a rename forces a human to decide what the new sentence says.

Everything here is static (``ast``); nothing imports ``synplan``, so the check
stays fast and never touches torch or chython.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / "skills" / "synplanner-usage"
SKILL_MD = SKILL_DIR / "SKILL.md"
TASKS_MD = SKILL_DIR / "references" / "tasks.md"
PKG = REPO_ROOT / "synplan"

pytestmark = pytest.mark.skipif(
    not SKILL_MD.is_file(), reason="agent skill not present in this checkout"
)


def _text() -> str:
    return SKILL_MD.read_text(encoding="utf-8") + TASKS_MD.read_text(encoding="utf-8")


def _code_spans(text: str) -> list[str]:
    """Everything inside single backticks."""
    return re.findall(r"`([^`\n]+)`", text)


# --------------------------------------------------------------------------- #
# What the package actually provides
# --------------------------------------------------------------------------- #


def _module_names() -> set[str]:
    out = set()
    for path in PKG.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        out.add(".".join(parts))
    return out


def _defined_symbols() -> set[str]:
    """Top-level function, class and assignment names across the package."""
    out = set()
    for path in PKG.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - would fail the real suite first
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, ast.Assign):
                out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


def _cli_commands() -> set[str]:
    """Names from @synplan.command(name="...") in the click group."""
    tree = ast.parse((PKG / "interfaces" / "cli.py").read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "command":
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    out.add(kw.value.value)
    return out


def _config_keys() -> set[str]:
    """YAML keys across configs/, so config knobs are not mistaken for symbols."""
    out = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.add(str(k))
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for path in (REPO_ROOT / "configs").glob("*.yaml"):
        walk(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return out


def _keyword_arguments() -> set[str]:
    """Parameter names across the package — `route_scorer`, `standardize`, ..."""
    out = set()
    for path in PKG.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in (*a.args, *a.posonlyargs, *a.kwonlyargs):
                    out.add(arg.arg)
    return out


def _packaging_tokens() -> set[str]:
    """Extras and dependency names from pyproject — `wandb`, `rdkit`, `cu126`, ..."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"?([A-Za-z][\w-]*)"?\s*=\s*\[', text, re.M)) | set(
        re.findall(r'"([A-Za-z][\w-]*)[><=~\[]', text)
    )


def _doc_page_stems() -> set[str]:
    """Page names such as `ten_minutes`, `migration`, `installation`."""
    return {p.stem for p in (REPO_ROOT / "docs").rglob("*.rst")}


# Keys of the dict returned by download_preset. They come from the preset YAML
# hosted on HuggingFace, so they cannot be derived from this repository.
PRESET_KEYS = {"building_blocks", "reaction_rules", "ranking_policy", "value_network"}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_frontmatter_matches_spec():
    """name must match the directory; both required fields within length caps."""
    head = SKILL_MD.read_text(encoding="utf-8").split("---")[1]
    meta = yaml.safe_load(head)
    assert meta["name"] == SKILL_DIR.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
    assert len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    if "compatibility" in meta:
        assert len(meta["compatibility"]) <= 500


def test_skill_body_within_recommended_length():
    """The spec recommends keeping SKILL.md under 500 lines."""
    assert len(SKILL_MD.read_text(encoding="utf-8").splitlines()) < 500


@pytest.mark.parametrize("dotted", sorted(set(re.findall(r"\bsynplan(?:\.[a-z_]+)+", _text()))))
def test_named_module_exists(dotted):
    """Every synplan.x.y path named in the skill resolves to a real module."""
    modules = _module_names()
    if dotted in modules:
        return
    # may be module + symbol, e.g. synplan.chem.reaction.routes.quality.scorer
    parent, _, leaf = dotted.rpartition(".")
    assert parent in modules and leaf in _defined_symbols(), (
        f"{dotted!r} names neither a module nor a symbol in an existing module. "
        f"If it moved, update the skill text — see docs/user_guide/migration.rst."
    )


@pytest.mark.parametrize("name", sorted(_cli_commands()))
def test_cli_command_documented_or_known(name):
    """Every CLI command exists; catches renames in either direction."""
    assert name in _text() or name in {"ord_convert"}, (
        f"CLI command {name!r} is not mentioned in the skill or task index."
    )


def test_named_configs_exist():
    """Every configs/*.yaml filename mentioned is present."""
    available = {p.name for p in (REPO_ROOT / "configs").glob("*.yaml")}
    named = {
        s for s in _code_spans(_text()) if s.endswith(".yaml") and "*" not in s
    }
    missing = named - available
    assert not missing, f"skill names missing config files: {sorted(missing)}"


def test_named_docs_pages_exist():
    """Doc paths such as `methods/planning` resolve to a real .rst file."""
    sections = ("methods", "configuration", "user_guide", "get_started")
    named = set(re.findall(rf"\b({'|'.join(sections)})/([a-z_]+)\b", _text()))
    missing = [
        f"{s}/{p}"
        for s, p in named
        if not (REPO_ROOT / "docs" / s / f"{p}.rst").is_file()
    ]
    assert not missing, f"skill links to missing docs pages: {sorted(missing)}"


def test_named_tutorials_exist():
    """Tutorial notebooks referenced by number and name exist."""
    available = {p.stem for p in (REPO_ROOT / "docs" / "user_guide").glob("*.ipynb")}
    named = set(re.findall(r"\b(\d{2}_[A-Za-z_]+)\b", _text()))
    missing = named - available
    assert not missing, f"skill references missing tutorials: {sorted(missing)}"


def test_named_python_symbols_exist():
    """Backticked identifiers are real symbols, config keys, or CLI commands."""
    known = (
        _defined_symbols()
        | _config_keys()
        | _cli_commands()
        | _keyword_arguments()
        | _packaging_tokens()
        | _doc_page_stems()
        | PRESET_KEYS
    )
    candidates = {
        s
        for s in _code_spans(_text())
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{3,}", s) and not s.startswith("synplan")
    }
    missing = sorted(s for s in candidates if s not in known)
    assert not missing, (
        f"skill names identifiers not found anywhere in synplan/, configs/, or the "
        f"CLI: {missing}. Rename in the skill, or add to the allowlist if they are "
        f"prose rather than code."
    )
