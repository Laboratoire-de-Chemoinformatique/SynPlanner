.. _standardization:

================================
Reaction standardization
================================

This page explains how to do a reaction standardization in ``SynPlanner``.

Reaction mapping
--------------------------------

Reaction atom-to-atom (AAM) mapping in SynPlanner is performed with GraphormerMapper,
an algorithm for AAM based on a transformer neural network adopted for the direct processing of molecular graphs
as sets of atoms and bonds, as opposed to SMILES/SELFIES sequence-based approaches, in combination with the
Bidirectional Encoder Representations from Transformers (BERT) network. The graph transformer serves to extract molecular
features that are tied to atoms and bonds. The BERT network is used for chemical transformation learning.
In a benchmarking study [https://doi.org/10.1021/acs.jcim.2c00344], GraphormerMapper achieved 89.5% correctly
mapped reactions on the "Golden" benchmarking data set, compared to 84.5% for IBM RxnMapper.

Two ways to map
~~~~~~~~~~~~~~~

Both run the same GraphormerMapper model and give the same mapping. They differ
in how much work they do per call.

**chython's built-in**, one reaction at a time:

.. code-block:: python

   from chython import smiles

   rxn = smiles("BrC1=CN=CC=C1.C1=CC=C(C2CCNCC2)C=C1>>...")
   rxn.reset_mapping()

Convenient inside a script that is already holding reactions in memory. It runs
one reaction per call with no batching, so throughput is what a single forward
pass gives you.

**SynPlanner's pipeline**, batched and streaming:

.. code-block:: bash

   synplan reaction_mapping --input reactions.smi --output mapped.smi

.. code-block:: python

   from synplan.chem.data.mapping import MappingConfig, map_reactions_from_file

   map_reactions_from_file(MappingConfig(), "reactions.smi", "mapped.smi")

This batches inference, streams the file in chunks, parallelises parsing across
processes, and writes failures to a separate error file instead of stopping.

Choosing between them, and hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mapping is the one curation stage where a GPU changes what is practical.

- **Up to roughly ten thousand reactions**, CPU is fine. Either route works;
  reach for whichever fits the code you are already writing.
- **Beyond that, and certainly at millions**, use the pipeline on a GPU. A
  per-reaction loop at that scale is not a slow run, it is an impractical one.

``MappingConfig`` selects the device, auto-detecting when left unset:

.. table::
   :widths: 20 15 65

   =================== ========= ==========================================================
   Parameter           Default   Meaning
   =================== ========= ==========================================================
   ``device``          ``None``  ``"cuda"``, ``"mps"``, ``"cpu"``, or auto-detect
   ``batch_size``      16        Reactions per forward pass; raise it on a large GPU
   ``chunk_size``      5000      Lines read per streaming chunk
   ``no_amp``          False     Disable mixed precision (it is used on cuda and mps)
   ``worker_timeout``  120       Seconds per chunk before it is skipped and logged failed
   =================== ========= ==========================================================

Apple Silicon (``mps``) is supported and sits between CPU and a discrete GPU.
Mixed precision applies on ``cuda`` and ``mps`` only.

Atom-mapping enforcement at readers
------------------------------------

Every downstream stage in SynPlanner (filtering, rule extraction, policy
and value training, retrosynthetic search) relies on atom-to-atom mapping
to compose CGRs and pattern-match rules. Unmapped or partially mapped
input is silently miscomputed because there is no shared atom identity
between reactants and products, so atom numbering is essentially random.

The :func:`synplan.utils.files.parse_reaction` reader and the
:func:`synplan.utils.loading.load_reaction_rules` SMARTS loader accept a
``check_atom_mapping`` flag with three values:

- ``"off"``: no check (use only when input is known mapped or the
  caller is explicitly mapping the data, e.g. the mapping pipeline).
- ``"reject_unmapped"``: raise on reactions whose reactant and product
  sides share no atom numbers. Default for ``load_reaction_rules``;
  rule SMARTS with leaving/incoming groups (partial maps) still load.
- ``"reject_partial"``: additionally raise on partial maps. Useful
  when curating training data that needs full mapping coverage.

The status is recorded on ``rxn.meta["mapping_status"]`` so worker
processes can route partially-mapped reactions to audit logs instead
of failing the whole batch.

.. warning::
    ``parse_reaction`` defaults to ``check_atom_mapping="off"``, and none of the
    ``synplan reaction_standardizing`` / ``reaction_filtering`` / ``rule_extracting``
    commands expose the flag. **The data-preparation CLIs do not verify atom mapping.**
    Unmapped input is parsed with sequential atom numbering, which makes reactant and
    product atoms unrelated; the resulting CGR marks nearly every bond as dynamic.
    In practice such records are rejected by ``DynamicBondsFilter`` — reported as an
    ordinary filter rejection, with nothing pointing at the real cause — and if that
    filter is not enabled they flow straight into rule extraction.

    Run ``synplan reaction_mapping`` first, or verify mapping yourself by calling
    ``parse_reaction(..., check_atom_mapping="reject_unmapped")`` over a sample of the
    input. A filtration run whose rejections are overwhelmingly ``DynamicBondsFilter``
    is the signature of unmapped input, not of bad chemistry.

Reaction standardization
--------------------------------

The reaction data are standardized using an original protocol for reaction data curation
published earlier [https://doi.org/10.1002/minf.202100119]. This protocol includes two layers:
standardization of individual molecules (reactants, reagents, products) and reaction standardization.
Steps for standardization of individual molecules include functional group standardization, aromatization/kekulization,
and valence checking.
The reaction standardization layer includes reaction role assignment, reaction equation balancing,
and atom-to-atom mapping fixing. The duplicate reactions and erroneous reactions are removed.

The current available reaction standardizers in ``SynPlanner``:

- Reaction mapping (reaction atom mapping using ``chython`` from ``chytorch``)
- Reaction mapping fix (fix reaction mapping in reaction when needed and possible)
- Functional groups standardization (standardization of functional groups)
- Kekule / Aromatic form conversion (conversion between Kekule and Aromatic form when needed)
- Atom valence validation (check atom valences)
- Isotope validation (check and clean isotope atoms when possible)
- Reagents validation (remove reagents from reaction)
- Unchanged parts validation (remove unchanged parts in reaction)
- Hydrogen manipulation (remove hydrogen atoms)
- Ions splitting (split ions in reaction when possible)
- Reaction rebalancing (add the molecules an unbalanced reaction does not account for)
- Duplicate reaction removal (remove duplicate reactions)

Standardization order
--------------------------------

Reaction standardization uses a fixed internal order. The configuration turns
individual steps on or off and supplies parameters, but users do not need to
arrange the steps manually. This keeps the common curation path robust for
large reaction corpora where a small ordering mistake can create many false
errors.

The default order applies Kekule conversion and functional-group normalization
before chemical validation. Reagents are then removed before atom valence
validation because reagents are omitted downstream and can include species
whose valence should not reject an otherwise valid transformation. Aromatic
conversion is applied after valence-sensitive checks so final standardized
records and duplicate detection use a consistent aromatic representation.
