.. _rebalancing:

================================
Reaction rebalancing
================================

Most recorded reactions do not balance. A patent or an ELN entry names the
reagents a chemist cared about and leaves out the base, the solvent, the
counterion and whatever left as gas, so the product side carries atoms the
reactant side never accounts for. Rule extraction reads those gaps as
chemistry: an atom that appears from nowhere becomes part of the pattern, and
the rule learned from it proposes precursors that cannot make the product.

Rebalancing adds the missing molecules back before extraction sees the
reaction.

What the step does
----------------------------------

The imputer works in two passes. Missing carbon is recovered first, as a
substructure of the reactants, which keeps a leaving group attached to the
fragment it came from. The remaining deficit — hydrogen, halogen, oxygen,
charge — is covered from a table of small molecules.

Where the reaction carries atom mapping, the CGR names which bonds break, so a
reagent the reaction never touched is carried through whole instead of being
cut apart. Without mapping the step still runs; it just has less to work with.

An imputed species is rejected when it leaves an atom with an unsatisfied
valence. That is what separates ``Mg(OH)Br`` from a bare ``Mg``, and an
oxidation balanced with a peroxide from one balanced with atomic oxygen, which
chython cannot hold and which comes back off disk unbalanced.

The step is off by default. Turn it on with ``rebalance_reaction_config`` in
the standardization config, and read
:doc:`/configuration/standardization` for the options: naming the reagent
behind a redox step, refusing an answer that invents free hydrogen, dropping
products the reactants cannot have made, and ignoring the mapping.

What it achieves
----------------------------------

Measured on SynRBL's validation set (5032 reactions, 467 of them already
balanced) with ``scripts/rebalance_bench.py``:

============================  ======  ==========  ======  ============
dataset                            N   success%     acc%   end-to-end%
============================  ======  ==========  ======  ============
Jaworski                         637      98.51    88.57         87.25
USPTO_diff                      1587      99.43    95.91         95.36
USPTO_random_class               717      99.86    90.12         89.99
USPTO_unbalance_class            540     100.00    92.02         92.02
golden_dataset                  1551      98.12    88.60         86.94
**TOTAL**                       5032      99.10    92.08         91.25
============================  ======  ==========  ======  ============

``success`` is the share of unbalanced reactions the step balanced at all.
``accuracy`` is the share of those answers matching the reference.

What the numbers do not say
----------------------------------

**The reference is not an answer key.** ``expected_reaction`` in that
validation set is SynRBL's own output that a reviewer accepted. Agreement with
it is the most this benchmark can measure, so 92% means "agrees with SynRBL on
92% of rows it can be scored on", not "is right 92% of the time". A systematic
error both tools share would score as a win.

**Two kinds of row cannot be scored at all** and are reported apart rather than
counted as losses: 327 rows carry no reference, and 250 rows carry one that
does not itself balance. Every reference spelling free oxygen as ``[O]`` lands
in the second group, because chython reads that as water. SynRBL's own
benchmark counts both as wrong; ``--synrbl`` reproduces that metric and gives
80.70% accuracy over 4524 rows instead of 92.08% over 3965.

**The confidence score does not separate right answers from wrong ones.** Every
answer carries one, and ``min_confidence`` refuses those below a threshold, but
raising the threshold discards correct answers at close to the rate it discards
incorrect ones. Use it to control how much imputation you accept, not as a
filter for correctness.

**Roughly one reaction in a hundred gets no answer.** The failures cluster:
either no reactant fragment holds the missing carbons, or no combination of the
small-molecule table covers the remaining element deficit — boron, silicon,
selenium and osmium show up here, as do multi-element deficits with a charge
imbalance.

When to use it
----------------------------------

Turn it on when unbalanced reactions in a corpus are producing rules that
propose impossible precursors, and when a wrong-but-balanced reaction is less
costly to you than an unbalanced one. Leave it off when the corpus is already
balanced, or when downstream work treats every reactant as literally recorded.
