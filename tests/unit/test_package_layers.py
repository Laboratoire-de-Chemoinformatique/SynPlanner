"""Rule 1 of ``docs/development/package_layout.rst``: imports point down only."""

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "synplan"

# Which top-level packages each layer is allowed to import from.
ALLOWED = {
    "utils": set(),
    "chem": {"utils"},
    "mcts": {"chem", "ml", "utils"},
    "ml": {"chem", "mcts", "utils"},
    "interfaces": {"chem", "mcts", "ml", "utils"},
}

# Modules that still break the rule. This set may shrink, never grow.
KNOWN_VIOLATIONS = {
    # Assembly code: builds policies, value networks and reactors from a config.
    # Belongs in `interfaces`, moves once the CLI stops importing it as a util.
    "utils/loading.py",
    # Route and tree rendering; belongs under `chem/reaction/routes`.
    "utils/visualisation/routes.py",
    "utils/visualisation/clustering.py",
    # `parse_reaction(check_atom_mapping=...)` calls chem.utils through a
    # function-level import placed there to dodge this rule. The check is
    # chemistry and belongs in `chem`, not in the file handler.
    "utils/files.py",
}


def _is_type_checking_block(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _imported_packages(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # An import under `if TYPE_CHECKING:` creates no runtime edge, so it cannot
    # produce a cycle. Annotating across a layer is allowed; calling is not.
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_block(node):
            node.body = []
    packages = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "synplan" and len(parts) > 1 and parts[1] in ALLOWED:
                packages.add(parts[1])
    return packages


def test_imports_point_down():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE)
        layer = relative.parts[0]
        if layer not in ALLOWED or relative.as_posix() in KNOWN_VIOLATIONS:
            continue
        forbidden = _imported_packages(path) - ALLOWED[layer] - {layer}
        if forbidden:
            offenders.append(f"{relative.as_posix()} -> {sorted(forbidden)}")
    assert offenders == [], "\n".join(offenders)


def test_known_violations_still_exist():
    """A stale entry hides a rule that is already satisfied."""
    for name in sorted(KNOWN_VIOLATIONS):
        path = PACKAGE / name
        assert path.exists(), f"{name} is gone; drop it from KNOWN_VIOLATIONS"
        layer = pathlib.PurePosixPath(name).parts[0]
        forbidden = _imported_packages(path) - ALLOWED[layer] - {layer}
        assert forbidden, f"{name} no longer violates rule 1; drop it"
