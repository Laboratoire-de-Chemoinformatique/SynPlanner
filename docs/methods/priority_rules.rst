.. _priority_rules:

================================
Priority rules
================================

A *priority rule* is a curated retrosynthetic SMARTS pattern that you want
the planner to try ahead of the learned policy on every node. Typical
use cases:

- A few hand-picked Ugi 4CR templates when synthesising peptide-like targets.
- A deprotection rule set for fully protected intermediates.
- A small in-house ring-formation library you trust more than the policy on
  a specific scaffold class.

Priority rules sit *outside* the policy network: they bypass policy
ranking, enter sibling selection with a strong UCB prior, and are tracked
under their own counters in :class:`~synplan.mcts.tree.TreeStats`.

Passing priority rules to ``Tree``
----------------------------------

Priority rules are passed as a mapping of named sets:

.. code-block:: python

    from synplan.mcts.tree import Tree
    from synplan.mcts.config import TreeConfig

    Tree(
        target=...,
        config=TreeConfig(use_priority=True),
        priority_rules={
            "ugi": ugi_rules,
            "boc_deprotection": boc_rules,
        },
        ...
    )

Each set's name becomes the ``rule_source`` label on every child it
produces and gets its own counter pair under
``tree.stats.per_priority_source[<name>]``.

Validation rules:

- The reserved name ``"policy"`` is rejected (raises ``ValueError``).
- Empty rule lists are rejected: either populate the set or remove
  the key.
- Non-empty string keys are required.
- Setting ``config.use_priority=True`` without supplying ``priority_rules``
  raises ``ValueError`` rather than running silently with no priority
  effect.

Mechanism
---------

Priority rules match by chython substructure isomorphism
(``pattern < molecule``) and enter expansion with ``prob=1.0``. The
existing per-product fragment-count multiplier in
:meth:`~synplan.mcts.tree.Tree._add_child_if_new` then applies the
relation

.. math::

   \mathrm{scaled\_prob} = \mathrm{prob} \times n_{\text{qualifying fragments}}

so a 4-fragment priority disconnect (e.g. an Ugi 4CR-style template that
produces four valid precursor fragments) enters UCB with prior 4. This
is intentional: curated multi-fragment disconnects are designed to
dominate sibling selection over single-fragment policy children.

If you observe ``Node.prob`` values larger than 1.0, this is expected
behaviour when a priority set introduces non-policy rules; see the
priority semantics section above.

Iterated application
--------------------

``TreeConfig.priority_rule_multiapplication`` (alias kwarg
``apply_reaction_rule(multirule=True, rm_dup=True)``) repeatedly
applies the same rule to its own product set until no new tuple is
produced. The flag fires for *every* priority source in the mapping,
not for the policy.

The intended use is bulk transformations such as stripping every
Boc/Cbz/etc. protective group from a fully protected substrate in one
expansion step.

SMARTS dialect note
-------------------

Priority rules are loaded through ``CanonicalRetroReactor.from_smarts``, which parses
patterns with chython's aromaticity perception. This differs from
RDKit's. Patterns authored against RDKit may match unexpectedly under
chython.

The :func:`synplan.chem.reaction.rules.parse_priority_rules` helper
reports a SMARTS as broken via chython *and* runs an RDKit fallback
parse so you can distinguish "broken pattern" from "dialect mismatch".
Validate priority rules on a known target before scaling up.

Per-source statistics
---------------------

``Tree.stats`` (a :class:`~synplan.mcts.tree.TreeStats` dataclass)
exposes:

- ``policy_rules_tried`` / ``policy_rules_succeeded``: policy-only
  counters.
- ``priority_rules_tried`` / ``priority_rules_succeeded``: aggregate
  across all priority sources.
- ``per_priority_source[<set_name>].tried`` and ``.succeeded``: the
  per-set breakdown.

``Tree.to_stats_dict()`` flattens ``per_priority_source`` into the
output dict for CSV/JSON export. Per-route priority usage stats
(``n_routes_with_priority``, ``fraction_routes_with_priority``) treat
*any* non-policy step as priority.

Rule provenance on nodes and routes
-----------------------------------

Every child carries:

- ``rule_source``: either the priority set name or
  :data:`~synplan.mcts.tree.POLICY_SOURCE_NAME` (``"policy"``).
- ``rule_key``: collision-safe identifier formatted as
  ``<source>:<id>`` so priority and policy IDs never collide in
  serialization.
- ``policy_rank``: exact 1-indexed Top-N position from the expansion
  function, or ``None`` for priority children.

Route SVG, JSON, and RDKit exports propagate this metadata, and route
SVGs annotate rule keys and policy ranks alongside molecules.

Ring-forming rules
------------------

``synthon_priority_rules()`` loads the shipped synthon disconnections as a
priority set. It carries both the acyclic disconnections and the ring-forming
ones, so a planning run can propose building a heterocycle rather than only
buying a pre-formed version of it.

A ring rule needs a reagent form written by hand. The acyclic rules derive one:
the loader spells a leaving group onto each labelled atom, which turns the
disconnection into a pair of orderable compounds. That decoration works on one
atom, and a heterocyclisation cuts two bonds, so the derived form names the
wrong compound class -- a triazole comes apart into a triazene and a styrene
where the reaction consumes an azide and an alkyne.

Each supported ring record therefore ships a ``retro_smarts`` naming the
reagents its reaction actually consumes, and the loader uses that string
verbatim rather than capping anything::

    [n;D3;+0:1]1[n;+0;D2:2][n;+0;D2:3][c:4][c:5]1
        >> [N:1]=[N+:2]=[N-:3].[C:4]#[C:5]

69 of the 76 ring records carry one, giving a default set of 108 rules. The
remaining seven are excluded because the reagent they would emit cannot be
isolated: an enamine that tautomerises to the imine, a beta-halo thiol that is
really an episulfide precursor, an N-unsubstituted hydrazonoyl halide. A ring
record without a ``retro_smarts`` is skipped rather than loaded uncapped.

Provenance differs across the set. The 39 acyclic rules are ``human``, curated
from the reference implementation; the 69 ring rules are ``llm``, authored in
this repository and not yet signed off by a chemist. ``rules_frame()``
(:mod:`synplan.chem.synthon.frames`) shows the split, and
``docs/development/chemist_review`` is the review queue.

Tutorial
--------

Tutorial 13 (`Priority Rules`_) walks through curating a small Ugi
priority set, running planning with and without it, comparing
``per_priority_source`` counters, and rendering route SVGs that surface
priority hits.

.. _Priority Rules: ../user_guide/13_Priority_Rules.ipynb

What the ring rules measurably buy
----------------------------------

The ring rules add routes rather than reach. Measured on 25 heterocyclic targets
at a fixed iteration budget, adding the 69 ring rules to the 39 acyclic ones
takes the route count from 2627 to 2867 and solves no target the acyclic set
could not already solve.

Searching for a target the ring rules genuinely unlock -- unsolvable with the
acyclic priority set, solvable with the ring set added -- turned up exactly one
in 465 real targets screened, drawn from the SAScore benchmark, its hardest
bin, and a drug set:

======================================  =======  ==============
set                                     targets  ring unlocks
======================================  =======  ==============
marketed drugs                                6               0
hand-picked heterocyclic                     25               0
SAScore benchmark 7.5-8.5 (alkaloids)        26               0
SAScore benchmark 4.5-7.5                    77               1
======================================  =======  ==============

The one is a macrocyclic kinase inhibitor at SAScore 5.36, opened by ``R17.93``
at its C-N bond -- a retro-reductive-amination giving one acyclic precursor
carrying both the aldehyde and the amine, which is how that class is made.

The reason the number is so low is that the policy network is trained on USPTO,
which is full of common heterocyclisations. The policy already proposes click
triazoles, Fischer indoles and pyrazole condensations, so the ring rules mostly
duplicate coverage the search already had. They earn their place by finding
convergent routes the policy ranks poorly, not by rescuing unsolvable targets.

Reagent availability is not a quality signal here. Of the 76 ring records, 32
emit two catalogue hits, 24 one and 16 neither. A precursor that is not
purchasable is an ordinary node for the planner to expand further, so the split
says how often a rule terminates a branch in one step, not how often it is
right.

Authoring a reagent form
----------------------------------

A hand-authored ``retro_smarts`` fails in three ways that leave no error behind,
which is why ``synplan.chem.synthon.rules.validate_retro`` exists and why
``expected_reagents`` is recorded per rule:

- **An unmapped atom on the right-hand side.** chython's patcher accepts it and
  returns the *intact* target plus a free fragment. Writing ``[O]`` instead of
  ``[O:20]`` on a benzimidazole retro gives back the benzimidazole and a
  formaldehyde, which is plausible, purchasable and wrong.
- **A rule that does not open the ring.** The transform applies, the products
  parse, and the heterocycle is still there.
- **The right transform on the wrong atoms.** Tautomerism moves a map number:
  chython canonicalises an N-H azole to the tautomer that parks the substituted
  ring carbon next to the NH, so a rule can emit an alpha-bromo aldehyde where
  phenacyl bromide was intended. Automorphism does the same on a ring with two
  interchangeable positions -- acetaldoxime plus benzonitrile becomes
  benzaldoxime plus acetonitrile.

The first two are mechanical and the validator refuses them. The third is not:
both forms break the ring with unique map numbers and parse cleanly, so the
validator compares emitted products against ``expected_reagents`` per product
set, and flags a target whose tautomers are degenerate for a human to read.
