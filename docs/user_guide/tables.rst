.. _tables:

================================
Tables that draw molecules
================================

Rules, synthons and search statistics are all naturally tabular, and a notebook
reading them wants to see the structures rather than the SMILES.
:class:`~synplan.utils.frames.ChemFrame` wraps a pandas frame and depicts any
column holding a chython object, keeping the object itself in the cell.

Anything with a ``depict()`` method draws, which covers every chython
container: molecules, reactions, CGRs, synthons and SMARTS queries. Adding a new
kind of table needs no change to the class::

    from synplan.utils.frames import ChemFrame

    ChemFrame(
        [{"id": r.name, "reaction": r, "steps": len(r)} for r in reactions],
        depict_columns=["reaction"],
    )

Three constructors ship with it:

``rules_frame()`` (:mod:`synplan.chem.synthon.frames`)
    One row per synthon disconnection, the rule depicted, with its id, name,
    kind, provenance, reaction name, and the reagents it forms. Called with no
    argument it gives every shipped record; passing the rules from
    ``synthon_priority_rules()`` restricts it to those actually loaded.

``synthons_frame(dag, stock=None)`` (:mod:`synplan.chem.synthon.frames`)
    One row per synthon per fragmentation pathway, the synthon depicted,
    ordered by pathway availability. Pass a stock to fill ``in_stock``; without
    one the column is empty rather than uniformly false.

``tree_stats_frame(trees)`` (:mod:`synplan.utils.frames`)
    One row of :meth:`~synplan.mcts.tree.Tree.to_stats_dict` per tree. Accepts a
    single tree, an iterable, or a ``{run name: tree}`` mapping, which is what a
    two-run comparison wants. It returns plain pandas, because nothing in the
    statistics is depictable.

Working with one
--------------------------------

Ordinary pandas verbs work through delegation, and a call returning a frame is
re-wrapped so the depiction survives::

    rules = rules_frame()
    rules[rules.df["kind"] == "ring"]          # still a ChemFrame, still draws
    rules.head(10)                             # so is this

``.df`` gives the plain frame, and is what ``.groupby()`` and ``.str`` need. Two
things to know about it: the objects in it are the same objects, not copies, so
mutating a molecule there mutates it everywhere; and every depiction is redrawn
each time the frame displays, so a drawn table is for tens of rows. The view
stops at ``max_display_rows`` (20) and says how many it left out; read hundreds
through ``.df`` instead.
