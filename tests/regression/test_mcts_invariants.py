"""Structural invariants for the MCTS data structures and search loop.

These tests pin behaviour that must hold regardless of model weights,
reaction rules, or specific molecule inputs:

1. Every ``Node`` must expose ``next_precursor`` after construction, even
   for solved/terminal nodes. The constructor sets the attribute only
   in the non-empty branch (``node.py:26-30``), so reading the attribute
   on a solved node raises ``AttributeError`` silently in any future code
   path that touches it.

2. ``run_search`` must contain ``mol_from_smiles(target_smi)`` *inside* the
   per-target try/except. One bad SMILES in the input file must not abort
   the entire batch. This is asserted statically against the source
   because running MCTS in CI is too heavy.

Both tests are invariants of the source-and-API contract, not of any
particular search outcome — they do not need updating as model weights or
rule sets evolve.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from synplan.mcts.node import Node

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_solved_node_has_next_precursor_attribute():
    """A Node built with an empty ``precursors_to_expand`` (solved/terminal)
    must still expose ``next_precursor``.

    The current ``Node.__init__`` only assigns ``next_precursor`` in the
    non-empty branch. Any caller that reads ``node.next_precursor`` on a
    solved node raises ``AttributeError`` — silent until it bites.
    """
    node = Node(precursors_to_expand=(), new_precursors=())
    # The cheapest invariant: the attribute exists, regardless of value.
    assert hasattr(node, "next_precursor"), (
        "Node built with empty precursors_to_expand has no next_precursor "
        "attribute. Reading it raises AttributeError. Set "
        "self.next_precursor = () in the empty-branch of Node.__init__."
    )


def test_run_search_wraps_mol_from_smiles_in_try_except():
    """``mol_from_smiles`` must be called *inside* the per-target
    try/except in ``run_search``. A bad SMILES line in the input must not
    abort the whole batch.

    The static check inspects the AST of ``run_search``: every call to
    ``mol_from_smiles`` whose argument is a per-target loop variable must
    appear inside a ``try`` block.
    """
    src_path = REPO_ROOT / "synplan" / "mcts" / "search.py"
    assert src_path.exists(), f"missing {src_path}"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    run_search_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_search":
            run_search_fn = node
            break
    if run_search_fn is None:
        pytest.skip("run_search not found in mcts/search.py")

    # Find every call to mol_from_smiles and check it's inside a try block.
    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.try_stack: list[ast.Try] = []
            self.uncaught_calls: list[ast.Call] = []

        def visit_Try(self, n):
            self.try_stack.append(n)
            self.generic_visit(n)
            self.try_stack.pop()

        def visit_Call(self, n):
            func = n.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name == "mol_from_smiles" and not self.try_stack:
                self.uncaught_calls.append(n)
            self.generic_visit(n)

    v = _Visitor()
    v.visit(run_search_fn)
    assert not v.uncaught_calls, (
        "run_search has mol_from_smiles call(s) outside any try/except. "
        f"Lines: {[c.lineno for c in v.uncaught_calls]}. One malformed "
        "SMILES in the input aborts the whole batch (subsequent valid "
        "targets are silently skipped, output CSV/JSON is truncated)."
    )
