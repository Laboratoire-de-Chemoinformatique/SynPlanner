.. _synthons:

================================
Synthons
================================

A *synthon* is a valence-complete fragment whose atoms carry a reaction-centre
label saying how that atom will react. ``synplan.chem.synthon`` is a native port
of Synt-On (SynthI), described in Zabolotna *et al.*, "SynthI: A New Open-Source
Tool for Synthon-Based Library Design", *J. Chem. Inf. Model.* 2022, 62(9),
2151-2163, `doi:10.1021/acs.jcim.1c00754
<https://doi.org/10.1021/acs.jcim.1c00754>`_.

The reference stores the label as an RDKit atom-map number and does its work
with string surgery on SMILES. This port makes it a graph property on a forked
chython ``SynthonContainer``, serialised inside the bracket as ``[NH2_nuc]``.

Seven components carry over: a building-block classifier (147 ordered classes
over 2401 SMARTS), building-block synthonisation (147 rule programs, 389 steps,
four execution strategies), fragmentation, recombination, Bemis-Murcko scaffolds
after protecting-group removal, the rule of two, and positional analogue
scanning. The CLI covers them as ``bb_classifying``, ``bb_synthonizing``,
``synthon_fragment``, ``synthon_enumerate`` and ``bb_scaffolds``, all configured
by ``configs/synthonisation.yaml``.

Fragmentation
----------------------------------

``Fragmenter.fragment`` cuts a target into synthons and returns a disconnection
DAG. Pass the stock at construction or the availability figures mean nothing.
Availability is the fraction of the *target's* atoms coming from stocked
synthons, not the fraction of slots filled, so a pathway can score well and
still have a slot nothing fills; ``availability_denominator`` switches the
denominator.

``best_available()`` orders pathways by that score. Its first entry is not
guaranteed to be fully fillable — walk down the list until one yields products,
because a pathway with every slot non-empty and a pathway with an unfillable
slot both return nothing and look identical from outside.

Enumeration
----------------------------------

``Enumerator.enumerate_library`` grows products from a whole stock with no
target. ``Enumerator.enumerate_analogues`` fills the slots of one target's
pathway. Only the second is reachable from the CLI. Joins are keyed by a 29-pair
compatibility table.

Ring closure is expressible here although the reference excludes it by design.
A ring synthon is a fragment of the *product*, carrying product bond orders — a
triazole cuts to a triazene and a styrene rather than to an azide and an alkyne
— so no bond order is rewritten at join time. The missing primitive was
``close_ring``, which draws the second bond after ``join`` has already merged
the two fragments. ``ring_closure_sizes`` controls which ring sizes may close,
and emptying it restores acyclic-only behaviour in the enumerator; the
fragmenter still cuts with ring rules unless they are deselected too.

The knobs that silently disable what you asked for are listed in
:doc:`/configuration/synthonisation`.

The disconnection rules
----------------------------------

Two id blocks. ``R16`` holds the families that shipped with the reference (13
rules); ``R17.1``-``R17.93`` holds the rules authored here (63), one contiguous
range per curation lane: pyrrole/thiophene/indole ``R17.1``-``R17.20``,
multi-heteroatom azoles ``R17.30``-``R17.36``, thiazole/oxazole/imidazole
``R17.40``-``R17.50``, azines ``R17.55``-``R17.62``, benzo-fused azoles
``R17.70``-``R17.74``, saturated N/O/S rings ``R17.80``-``R17.93``. Every rule
pins the ring heteroatom's charge, which no shipped ring rule did.

Four defects in the original nine were fixed, each of which changes an answer:

- chython's ``D`` counts **heavy** neighbours, so ``[n;D3]`` never matched an
  N-unsubstituted azole and ``R16.1``-``R16.4`` and ``R16.8`` silently missed
  every N-H parent. chython refuses ``[n;D3,h1:1]``, so each affected family
  ships as an ``a`` (N-substituted) / ``b`` (N-H) twin pair.
- ``R16.6``'s left-hand side is mirror-symmetric while its right-hand side is
  not, and chython's automorphism filter dedupes on the set of matched atoms, so
  one of the two Kröhnke disconnections was lost. ``R16.6a``/``R16.6b`` spell
  both orientations.
- ``R16.4`` corrupted the brutto formula on 49% of a drug-like sample.
- ``R16.6a/b``, ``R16.7``, ``R16.9`` and ``R17.58`` disconnected N-H azinones,
  because canonicalisation rewrites an N-H azinone to the aromatic hydroxy-azine
  before any rule sees it. Guards alpha and gamma to the ring nitrogen refuse
  them while leaving plain azines untouched.

Guards applied family-wide: a four-token N-substituent idiom with the Fokin
sulfonyl-azide and Pellizzari hydrazide exemptions, ``;R1`` on the ring
heteroatom of the monocyclic and saturated rules, and a carbocyclic-fusion
requirement on the indole rules that stops them claiming azaindoles and
7-deazapurines. The generic saturated-ring residual ``R17.93`` now fires on
17.7% of drug-like molecules rather than 33.3%, with no acetal, aminal or
anomeric carbon left as its electrophile.

Provenance and what is not covered
----------------------------------

Every rule records where it came from: 78 converted from the reference, 76 ring
rules authored in this repository and not yet signed off by a chemist. Ten more
are held out of ``rules.json`` entirely, each gated on a chemist ruling;
``docs/development/chemist_review`` is the queue and names the open question
against each. Holding two furan rules leaves the port with no rule for a plain
2,5-dialkylfuran, which is declared rather than hidden.

Beside ``provenance`` each rule carries a nullable ``reaction_name``, what it
``forms``, its ``reagents``, and the ids it ``supersedes``. Null is kept as the
honest answer for a transformation class that names no single named reaction:
the free-text ``name`` alone could not be read as one, because it followed two
incompatible conventions.

``rules.json`` ships 154 records — 39 acyclic disconnections, their 39
macrocyclic twins, and 76 ring rules. ``rule_mode: use_all`` loads the 115
non-macrocyclic ones, and the macrocyclic twins are added only when the target
carries a ring larger than 11 atoms.

The pipeline is stereo-blind. ``safe_canonicalization`` discards stereocentres,
so any route through a ring rule is racemic. ``smirks_stereo`` and
``stereo_spec`` are recorded for all 96 curated rules but the stock is keyed on
flat structures. Promote ``StereoDiscardedWarning`` to an error to refuse
stereo-bearing input rather than racemise it silently.

Corpus coverage
----------------------------------

``classify_coverage`` answers whether a mapped reaction builds a bond that one
of the 39 acyclic disconnections already breaks, and ``synthon_coverage`` splits
a reaction file on that answer. The use is corpus preparation: a one-step policy
learns nothing from reactions the curated rules already provide. On a 100k
mapped USPTO sample, 37.9% is covered at roughly 1 ms per reaction.

The atom mapping gives the formed bond and each rule gives its broken bond. The
reactant-side leaving groups are checked against the rule's ``_label`` tokens,
which is the only label-aware step available — ``QueryElement.__eq__`` never
consults ``_label``, so the substructure match cannot be. Coverage is
disconnection-level rather than mechanism-level: a reductive amination counts as
covered by ``R3.1``.

The eight rules that name their nucleophile require that element on the reactant
side. Read as the bare "no halide left this atom", ``nuc`` is satisfied by an
arene doing nothing, and Friedel-Crafts acylations were being absorbed by
``R10.2`` (906 hits down to 118), C-H functionalisations by ``R12.5`` (443 to
124) and enolate acylations by ``R10.1`` (698 to 290).

Using the rules for planning
----------------------------------

``synthon_priority_rules()`` returns the disconnections as an MCTS priority set.
See :doc:`/methods/priority_rules` for how they enter the search, and
"Ring-forming rules" there for why ring records carry a hand-authored reagent
form.

On a 40-target FDA-2020 sample at a fixed iteration budget the acyclic set
solves 30/40 against a policy-only 23/40 (McNemar p=0.0156), and 92.8% of the
winning routes contain at least one synthon step.

A second design was built, measured on the same sample and reverted: keeping the
children as labelled synthons and checking them against a separate synthon stock
solved 21/40, below policy-only, because a synthon child that misses the synthon
stock is an absorbing dead end — 39.3% of expansions — where a plain child that
misses the ordinary stock is handed back to the policy.
