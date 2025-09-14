try:
    # Python 3.10+: standard library source of installed distribution version
    from importlib.metadata import PackageNotFoundError, version as _dist_version
except Exception:  # pragma: no cover - extremely unlikely on supported Pythons
    _dist_version = None  # type: ignore[assignment]
    PackageNotFoundError = Exception  # type: ignore[assignment]


def _read_version_from_pyproject():
    """Best-effort fallback to read version from pyproject.toml in dev mode.

    This is used when the package is imported directly from the repository
    without being installed as a distribution, so importlib.metadata cannot
    find the installed package metadata.
    """
    try:
        from pathlib import Path
        import re

        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if not pyproject_path.exists():
            return None

        text = pyproject_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r'^\s*version\s*=\s*["\']([^"\']+)["\']\s*$', text, re.MULTILINE
        )
        if match:
            return match.group(1)
    except Exception:
        return None
    return None


def _get_version():
    package_dist_name = "SynPlanner"
    if _dist_version is not None:
        try:
            return _dist_version(package_dist_name)
        except PackageNotFoundError:
            pass

    fallback = _read_version_from_pyproject()
    return fallback if fallback is not None else "0.0.0+unknown"


__version__ = _get_version()


# Expose selected symbols lazily to avoid importing heavy dependencies
__all__ = ["Tree", "__version__"]


def __getattr__(name):
    if name == "Tree":
        from .mcts import Tree  # local import for lazy loading

        return Tree
    if name == "Node":
        from .mcts import Node  # allow optional top-level access

        return Node
    raise AttributeError(f"module 'synplan' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["Tree", "__version__"])
