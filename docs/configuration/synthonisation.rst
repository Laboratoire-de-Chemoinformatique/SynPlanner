.. _synthonisation_config:

================================
Synthons
================================

One configuration file drives every synthon workflow: building-block
classification and synthonisation, target fragmentation, recombination, analogue
scanning, scaffold analysis and coverage classification. Each CLI command reads
the same file and uses the parameters that apply to it.

Download example configuration
--------------------------------

- GitHub: `configs/synthonisation.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/synthonisation.yaml>`_

Quickstart (CLI)
----------------

The stock comes first: fragmentation and enumeration read synthons, not raw
building blocks.

.. code-block:: bash

   synplan bb_synthonizing \
     --config configs/synthonisation.yaml \
     --input building_blocks.smi \
     --output synthons.smi

   synplan synthon_fragment \
     --config configs/synthonisation.yaml \
     --input targets.smi \
     --stock synthons.smi \
     --output pathways.tsv

See :doc:`/user_guide/cli_interface` for the full command sequence and the input
formats each command accepts.

**Configuration file**

.. code-block:: yaml

    keep_protecting_groups: false
    ignore_solvents: true
    max_components: 4
    max_stages: 5
    max_rc_per_fragment: 3
    max_pathways: 10000
    rule_mode: use_all
    rules_selection: R1-R13
    fragments_to_ignore: []
    availability_denominator: target
    max_reacted_synthons: 6
    max_products: 1000
    mw_lower: 100.0
    mw_upper: 1000.0
    ring_closure_sizes: [5, 6, 7]
    find_analogues: false
    similarity_threshold: -1.0
    pas_removal_direction: true
    ro2_filtration: false
    ro2_variant: paper
    strict_availability: false
    write_audit_files: false
    audit_overwrite: error

**Building-block synthonisation**

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Parameter                          Description
    ================================== =================================================================================
    keep_protecting_groups             Keeps protecting groups on the synthonised building block instead of removing them. The reference calls this ``keepPG``.
    ignore_solvents                    Drops recognised solvents from a multi-component record rather than synthonising them.
    max_components                     Largest number of components a building-block record may have before it is rejected.
    ================================== =================================================================================

**Fragmentation**

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Parameter                          Description
    ================================== =================================================================================
    max_stages                         Depth of the disconnection DAG: how many successive cuts a pathway may make.
    max_rc_per_fragment                Largest number of reaction-centre labels one fragment may carry.
    max_pathways                       Width bound on the search. The reference bounds depth only, so a highly decomposable target could expand without limit.
    rule_mode                          ``use_all`` cuts with every non-macrocyclic rule. ``include_only`` and ``exclude_some`` apply ``rules_selection``. ``one_by_one`` reproduces the reference's stepwise walk: the first level stops after the first rule that matched, and below it no pathway looks back past the rule that made the synthon or continues past the first rule that cuts it.
    rules_selection                    Rule ids or ranges, comma separated — ``R1``, ``R1.2``, ``R1.2-R1.4``. Read only when ``rule_mode`` is not ``use_all``. A range is a slice of the ordered rule list, and a selector matching no rule is an error rather than a silently empty run. The shipped ``R1-R13`` names no ring rule.
    ring_closure_sizes                 Ring sizes a heterocyclisation may close. 5 and 6 cover every azole and azine, 7 the diazepines. An empty list disables ring closure in the enumerator and drops the ring rules from the fragmenter, restoring acyclic-only behaviour exactly.
    fragments_to_ignore                SMILES of fragments that must never be produced by a cut.
    availability_denominator           Whether a pathway's availability rate is measured against the whole ``target`` or against that ``pathway``.
    ================================== =================================================================================

The shipped ``rules.json`` holds 154 records: 39 acyclic disconnections, their 39
macrocyclic twins, and 76 ring-forming rules. ``use_all`` loads the 115
non-macrocyclic ones; the macrocyclic twins are added only when the target has a
ring larger than 11 atoms. Every rule records its provenance, and the ring rules
authored in this repository have not been signed off by a chemist — see
:doc:`/development/chemist_review`.

**Enumeration and analogues**

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Parameter                          Description
    ================================== =================================================================================
    max_reacted_synthons               Largest number of synthons that may be joined into one product.
    max_products                       Bound on how many products are returned, not on how much work is done. The reference calls this ``desiredNumberOfNewMols``.
    mw_lower / mw_upper                Molecular-weight window a product must fall in. ``mw_lower`` above ``mw_upper`` is rejected at construction.
    find_analogues                     Widens each slot of a fragmentation pathway to the synthon's positional analogues — the step that turns the paper's Library1 into Library2.
    similarity_threshold               Tanimoto threshold for an additional similarity route into the same slot. ``-1`` disables it.
    pas_removal_direction              Enables the positional-analogue removal direction. The reference's own removal branch is unsatisfiable, so this direction is reachable only here.
    ro2_filtration                     Restricts the stock to what the rule of two calls reagent-like.
    ro2_variant                        ``paper`` reproduces the published numbers, which come from an implementation that does not apply the reference's own documented corrections. ``corrected`` is label-aware and discounts each attachment point.
    strict_availability                All-or-nothing veto: one slot the stock cannot fill kills the whole pathway. Off, an empty slot is a real answer.
    ================================== =================================================================================

Stereochemistry is not carried through: the stock is keyed on flat structures and
canonicalisation discards stereocentres, so any route through a ring rule is
racemic. Promote ``StereoDiscardedWarning`` to an error to refuse stereo-bearing
input rather than racemise it silently.

**Audit bundle**

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Parameter                          Description
    ================================== =================================================================================
    write_audit_files                  Writes ``fallback.smi``, ``fallback.tsv``, ``errors.tsv``, ``summary.json`` and ``run.log`` beside the primary output. Use a dedicated directory per command: the sidecar names are fixed. ``synthon_coverage`` accepts the parameter but writes no sidecars.
    audit_overwrite                    ``error`` refuses to overwrite an earlier bundle; ``replace`` replaces it.
    ================================== =================================================================================

**Runtime**

``num_workers`` and ``time_budget_s`` are accepted by ``SynthonConfig`` but are
deliberately absent from the shipped file: ``num_workers`` defaults to the number
of CPUs on the machine running the job, and ``time_budget_s`` defaults to no
limit. Set them in your own copy when a run needs bounding.
