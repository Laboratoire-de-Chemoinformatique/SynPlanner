"""A finished search as data: the node graph, the winning ids, the statistics.

The record says what happened; it cannot resume a search. The policy network,
the reaction rules and the evaluation function are machinery, not data, and the
building-block catalogue belongs to the run's configuration -- it is referenced
by name there, never embedded here.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from chython import smiles as read_smiles

from synplan.chem.precursor import Precursor
from synplan.chem.reaction.routes.io.json import molecule_key
from synplan.mcts.node import Node
from synplan.mcts.tree import Tree

if TYPE_CHECKING:
    from chython.containers import MoleculeContainer

__all__ = [
    "SEARCH_RECORD_SCHEMA",
    "SearchRecord",
    "read_search_record",
    "write_search_record",
]

#: Versioned identifier for the search-record file. Bump when the shape changes.
SEARCH_RECORD_SCHEMA = "synplan-tree/1"


@dataclass
class SearchRecord:
    """A search read back from a file: its nodes, its routes, its statistics.

    Not a :class:`~synplan.mcts.tree.Tree` -- it has no policy, no rules and no
    evaluator, so it cannot search. It answers the questions that need only the
    node graph, and :meth:`routes` rebuilds the routes the search found.

    :param target: The molecule the search was run on.
    :param nodes: The search's nodes, by id, as the search left them.
    :param stats: ``to_stats_dict()`` plus ``branching_profile`` and
        ``routes_found_at`` -- the counters the node graph cannot recompute.
        :meth:`winning_rule_ranks` is recomputed, not stored.
    """

    target: MoleculeContainer
    nodes: dict[int, Node]
    parents: dict[int, int]
    children: dict[int, set[int]]
    winning_nodes: list[int]
    stats: dict

    # The Tree readouts that touch nodes and parents only, bound from the real
    # class: a record has to answer them exactly as the search did.
    route_to_node = Tree.route_to_node
    route_score = Tree.route_score
    route_details = Tree.route_details
    synthesis_route = Tree.synthesis_route
    winning_rule_ranks = Tree.winning_rule_ranks
    routes = Tree.routes

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"<SearchRecord {self.target}: {len(self.nodes)} nodes, "
            f"{len(self.winning_nodes)} routes>"
        )


def _open(file_path: str | PathLike[str], mode: str):
    """``.gz`` writes and reads itself; anything else is plain text."""

    opener = gzip.open if str(file_path).endswith(".gz") else open
    return opener(file_path, mode, encoding="utf-8")


def write_search_record(tree: Tree, file_path: str | PathLike[str]) -> Path:
    """Write ``tree`` as a search record, gzipped when the path ends in ``.gz``.

    The routes are not written: a route is a path through the node graph, so
    :meth:`SearchRecord.routes` rebuilds them from the nodes rather than the
    file carrying a second copy of them. Node values are written unrounded, so
    the record scores a route with the number the search scored it with.

    :return: The path written.
    """

    molecules: dict[str, int] = {}

    def index(precursor: Precursor) -> int:
        key = molecule_key(precursor.molecule)
        return molecules.setdefault(key, len(molecules))

    nodes = [
        {
            "id": node_id,
            "parent": tree.parents[node_id],
            "depth": node.depth,
            "visits": node.visit,
            "value": node.total_value,
            "init": node.init_value,
            "prob": node.prob,
            "rule": node.rule_key,
            "rank": node.policy_rank,
            "new": [index(precursor) for precursor in node.new_precursors],
            # the unexpanded precursors: the first is the one the node was
            # expanding, an empty list is a solved node, and the whole list is
            # the route's unresolved leaves
            "expand": [index(precursor) for precursor in node.precursors_to_expand],
        }
        for node_id, node in sorted(tree.nodes.items())
    ]
    record = {
        "schema": SEARCH_RECORD_SCHEMA,
        "target": molecule_key(tree.nodes[1].curr_precursor.molecule),
        "molecules": list(molecules),
        "nodes": nodes,
        "winning": list(tree.winning_nodes),
        "stats": {
            # only what the node graph cannot recompute: winning_rule_ranks is
            # a third of the file and every number in it is on the nodes
            **tree.to_stats_dict(),
            "branching_profile": tree.branching_profile(),
            "routes_found_at": tree.stats.routes_found_at,
        },
    }
    with _open(file_path, "wt") as file:
        json.dump(record, file)
    return Path(file_path)


def _rule(key: str | None) -> tuple[str | None, int | None]:
    """``rule_source`` and ``rule_id`` back out of a ``rule_key``."""

    source, _, rule_id = (key or "").rpartition(":")
    if not source:
        return None, None
    return source, int(rule_id)


def read_search_record(file_path: str | PathLike[str]) -> SearchRecord:
    """Read a search record back into nodes its routes can be rebuilt from.

    Molecules are interned in the file, but each occurrence comes back as its
    own object: a route is held together by which molecule *object* a step makes
    and the next one consumes, and a convergent route makes the same molecule
    twice.

    :raises ValueError: if the file is not a search record this version reads,
        or a node claims a precursor no ancestor of it produced.
    """

    with _open(file_path, "rt") as file:
        raw = json.load(file)
    if raw.get("schema") != SEARCH_RECORD_SCHEMA:
        raise ValueError(
            f"{file_path} is {raw.get('schema')!r}, not a {SEARCH_RECORD_SCHEMA} "
            "search record"
        )

    molecules = [read_smiles(smiles) for smiles in raw["molecules"]]
    nodes: dict[int, Node] = {}
    parents: dict[int, int] = {}
    children: dict[int, set[int]] = {}
    unexpanded: dict[int, tuple[int, ...]] = {}

    for entry in raw["nodes"]:  # a node is always written after its parent
        node_id, parent_id = entry["id"], entry["parent"]
        new = tuple(
            Precursor(molecules[i].copy(), canonicalize=False) for i in entry["new"]
        )
        # what this node could still have to expand: what its parent handed on,
        # then what this node itself made
        pool = list(zip(entry["new"], new))
        if parent_id:
            leftover = zip(unexpanded[parent_id][1:], nodes[parent_id].next_precursor)
            pool = [*leftover, *pool]
        expand = []
        for wanted in entry["expand"]:
            # take the first unclaimed precursor spelling it: which occurrence of
            # two equal ones is which is not written down, and swapping them
            # swaps interchangeable subtrees
            slot = next((s for s, (i, _) in enumerate(pool) if i == wanted), None)
            if slot is None:
                raise ValueError(
                    f"node {node_id} has {raw['molecules'][wanted]} to expand, but "
                    "no ancestor of it produced that molecule"
                )
            expand.append(pool.pop(slot)[1])

        rule_source, rule_id = _rule(entry["rule"])
        nodes[node_id] = Node(
            precursors_to_expand=tuple(expand),
            new_precursors=new,
            visit=entry["visits"],
            depth=entry["depth"],
            prob=entry["prob"],
            init_value=entry["init"],
            total_value=entry["value"],
            rule_id=rule_id,
            rule_source=rule_source,
            rule_key=entry["rule"],
            policy_rank=entry["rank"],
        )
        parents[node_id] = parent_id
        unexpanded[node_id] = tuple(entry["expand"])
        children.setdefault(node_id, set())
        if parent_id:
            children.setdefault(parent_id, set()).add(node_id)

    stats = dict(raw["stats"])
    # JSON has no int keys and no tuples; the search's statistics have both
    stats["branching_profile"] = {
        int(depth): profile for depth, profile in stats["branching_profile"].items()
    }
    stats["routes_found_at"] = [tuple(found) for found in stats["routes_found_at"]]
    return SearchRecord(
        target=nodes[1].curr_precursor.molecule,
        nodes=nodes,
        parents=parents,
        children=children,
        winning_nodes=list(raw["winning"]),
        stats=stats,
    )
