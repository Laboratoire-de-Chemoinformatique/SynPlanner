"""Route post-processing tools.

This package owns route-level operations that run after route generation:
RouteCGR construction, route clustering, route IO, depiction, analysis, and
route quality scoring.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "ProtectionRouteScorer": ("synplan.routes.quality.scorer", "ProtectionRouteScorer"),
    "RouteCGRContainer": ("synplan.routes.route_cgr", "RouteCGRContainer"),
    "RouteDynamicBond": ("synplan.routes.route_cgr", "RouteDynamicBond"),
    "RouteScorer": ("synplan.routes.quality.scorer", "RouteScorer"),
    "cluster_routes": ("synplan.routes.clustering", "cluster_routes"),
    "compose_all_route_cgrs": ("synplan.routes.route_cgr", "compose_all_route_cgrs"),
    "compose_all_sb_cgrs": ("synplan.routes.route_cgr", "compose_all_sb_cgrs"),
    "compose_route_cgr": ("synplan.routes.route_cgr", "compose_route_cgr"),
    "compose_sb_cgr": ("synplan.routes.route_cgr", "compose_sb_cgr"),
    "export_tree_to_csv": ("synplan.routes.io", "export_tree_to_csv"),
    "export_tree_to_json": ("synplan.routes.io", "export_tree_to_json"),
    "extract_reactions": ("synplan.routes.route_cgr", "extract_reactions"),
    "make_dict": ("synplan.routes.io", "make_dict"),
    "make_json": ("synplan.routes.io", "make_json"),
    "read_routes_csv": ("synplan.routes.io", "read_routes_csv"),
    "read_routes_json": ("synplan.routes.io", "read_routes_json"),
    "subcluster_all_clusters": ("synplan.routes.clustering", "subcluster_all_clusters"),
    "write_routes_csv": ("synplan.routes.io", "write_routes_csv"),
    "write_routes_json": ("synplan.routes.io", "write_routes_json"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'synplan.routes' has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_EXPORTS])
