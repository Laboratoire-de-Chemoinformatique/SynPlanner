#!/usr/bin/env python3
"""Score already forward-annotated route collections, without model inference."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from synplan.chem.reaction.routes import Route
from synplan.chem.reaction.routes.quality import ForwardRouteScorer
from synplan.utils.json import read_json, write_json_atomic


def collection_routes(collection):
    """Yield display IDs and source trees in the original target/route order."""
    multiple_targets = len(collection) > 1
    for target, target_routes in collection.items():
        items = (
            target_routes.items()
            if isinstance(target_routes, dict)
            else enumerate(target_routes, 1)
        )
        for route_id, tree in items:
            label = f"{target} / {route_id}" if multiple_targets else str(route_id)
            yield label, tree


def score_collection(collection, scorer, *, silent=True):
    """Overlay only root assessments; preserve all existing collection metadata."""
    collection = deepcopy(collection)
    trees = [tree for _, tree in collection_routes(collection)]
    routes = scorer.assess((Route.from_json(tree) for tree in trees), silent=silent)
    for tree, route in zip(trees, routes, strict=True):
        tree["assessment"] = route.assessment
    return collection


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path, help="Optional self-contained HTML report")
    parser.add_argument("--combine-search", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    paths = [args.input, args.output] + ([args.html] if args.html else [])
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Input, output and HTML must be separate files")
    scorer = ForwardRouteScorer(combine_search=args.combine_search)
    result = score_collection(read_json(args.input), scorer, silent=args.quiet)
    write_json_atomic(args.output, result)
    if args.html:
        from synplan.utils.visualisation import routes_report_html

        records = list(collection_routes(result))
        args.html.parent.mkdir(parents=True, exist_ok=True)
        routes_report_html(
            [Route.from_json(tree) for _, tree in records],
            str(args.html),
            prices=False,
            route_labels=[label for label, _ in records],
        )


if __name__ == "__main__":
    main()
