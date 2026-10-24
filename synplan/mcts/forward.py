"""Assembly of optional forward validation for planning and explicit finalization."""

from copy import deepcopy
from pathlib import Path
from time import perf_counter

from synplan.mcts.config import ForwardValidationConfig


class ForwardValidationError(RuntimeError):
    """A failed execution, never a negative chemical prediction."""


def load_forward_validation(config=None, results_root="."):
    """Return resolved settings and one shared assessor, or two Nones.

    A path is resolved from the caller's working directory; the planning CLI
    resolves its reference relative to the planning YAML before calling here.
    Model/cache paths in a YAML are relative to that YAML.
    """
    if config is None:
        return None, None
    if not isinstance(config, ForwardValidationConfig):
        config = ForwardValidationConfig.from_yaml(config)
    if config.forward_val_type is None:
        return None, None
    if config.forward_val_path is None:
        raise ValueError("Enabled forward validation requires forward_val_path")
    if config.cache_path is None:
        config = config.model_copy(
            update={"cache_path": Path(results_root).resolve() / "forward_cache.json"}
        )
    from synplan.chem.reaction.routes.quality.forward import ForwardReactionAssessor
    from synplan.ml.inference.chemformer import OnnxChemformer

    started = perf_counter()
    model = OnnxChemformer(
        config.forward_val_path, device=config.device, threads=config.threads
    )
    assessor = ForwardReactionAssessor(
        model, config.cache_path, batch_size=config.batch_size, top_k=config.top_k
    )
    assessor.initialization_seconds = perf_counter() - started
    return config, assessor


def finalize_forward_routes(tree, assessor, mode, *, silent=True, combine_search=False):
    """Explicitly assess completed routes and persist data on their tree nodes.

    Search itself and read-only route reconstruction never call this function.
    Terminal stereo proposals are annotated too, without changing their status.
    """
    from synplan.chem.reaction.routes.quality.scorer import ForwardRouteScorer
    from synplan.chem.reaction.routes.route import Route

    if mode not in ("on-the-fly", "post-processing"):
        raise ValueError("Expected on-the-fly or post-processing forward validation")
    if tree.config.direction != "retro":
        raise ValueError(
            "Forward validation currently supports retrosynthetic search only"
        )
    started = perf_counter()
    routes = tree.routes()
    solved_count = len(routes)
    routes += [
        Route.from_tree(tree, node_id)
        for node_id in tree.proposal_nodes
        if node_id not in tree.winning_nodes
    ]
    if mode == "post-processing":
        routes = assessor.assess(routes, silent=silent)
    routes = ForwardRouteScorer(combine_search=combine_search).assess(
        routes, silent=silent
    )
    for route in routes:
        for step in route.steps:
            tree.nodes[step.origin.tree_node_id].forward_prediction = deepcopy(
                step.forward_prediction
            )
        tree.nodes[route.provenance.tree_node_id].route_assessment = deepcopy(
            route.assessment
        )
    tree.forward_metadata.update(
        {
            "mode": mode,
            "top_k": assessor.top_k,
            "model_fingerprint": assessor.fingerprint,
            "model_identity": assessor.identity,
        }
    )
    tree.forward_stats["postprocessing_seconds"] = perf_counter() - started
    if mode == "post-processing":
        tree.forward_stats.update(assessor.last_stats)
    return routes[:solved_count]
