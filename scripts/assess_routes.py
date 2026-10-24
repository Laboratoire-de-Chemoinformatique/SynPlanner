#!/usr/bin/env python3
"""Assess a target-keyed route JSON/JSON.GZ collection without reranking."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from synplan.chem.reaction.routes import Route
from synplan.chem.reaction.routes.quality import ForwardReactionAssessor
from synplan.ml.inference.chemformer import OnnxChemformer
from synplan.utils.json import read_json, write_json_atomic


def assess_collection(collection, assessor, *, silent=True):
    """Overlay annotations on a copy, preserving every existing JSON field."""
    collection = deepcopy(collection)
    routes, nodes = [], {}

    def remember(reaction, node):
        nodes[id(reaction)] = node

    for target_routes in collection.values():
        trees = (
            target_routes.values() if isinstance(target_routes, dict) else target_routes
        )
        for tree in trees:
            # Reaction reassessment invalidates any aggregate from an earlier run.
            tree.pop("assessment", None)
            routes.append(Route.from_json(tree, on_step=remember))
    for route in assessor.assess(routes, silent=silent):
        for step in route.steps:
            nodes[id(step.reaction)].setdefault("meta", {})["forward_prediction"] = (
                step.forward_prediction
            )
    return collection


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Portable model bundle directory"
    )
    parser.add_argument(
        "--model-type", choices=("chemformer", "chimera"), default="chemformer"
    )
    parser.add_argument("--backend", choices=("onnx", "ckpt"), default="onnx")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, help="Default: output directory/forward_cache.json"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cache = args.cache or args.output.parent / "forward_cache.json"
    if len({p.resolve() for p in (args.input, args.output, cache)}) != 3:
        parser.error("Input, output and cache must be separate files")
    if args.model_type == "chimera":
        from synplan.ml.inference.chimera import ForwardChimera

        model = ForwardChimera(
            args.model, device=args.device, threads=args.threads, backend=args.backend
        )
    else:
        if args.backend != "onnx":
            parser.error("Chemformer currently supports only the ONNX backend")
        model = OnnxChemformer(args.model, device=args.device, threads=args.threads)
    assessor = ForwardReactionAssessor(
        model, cache_path=cache, batch_size=args.batch_size
    )
    annotated = assess_collection(read_json(args.input), assessor, silent=args.quiet)
    write_json_atomic(args.output, annotated)


if __name__ == "__main__":
    main()
