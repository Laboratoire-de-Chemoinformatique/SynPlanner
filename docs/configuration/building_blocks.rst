.. _building_block_configuration:

============================
Building-block preparation
============================

Ordinary building blocks terminate MCTS branches. They are separate from the
labelled :class:`~synplan.chem.synthon.stock.SynthonStock` used by target
fragmentation and library enumeration.

Use :class:`~synplan.chem.building_blocks.config.BuildingBlockPreparationConfig`
to prepare a catalogue. The defaults preserve the two-path command's historical
role while making output deterministic and stereochemistry-preserving:

.. literalinclude:: ../../configs/building_blocks_preparation.yaml
   :language: yaml

The audited full-pipeline example enables conservative deprotection and a full
Standard InChIKey stock:

.. literalinclude:: ../../configs/building_blocks_full_pipeline.yaml
   :language: yaml

Input
-----

``input_format`` may be ``auto``, ``smi``, ``smiles``, ``cxsmiles``, ``sdf``,
``csv``, or ``tsv``. Headerless SMI/CXSMILES puts the complete chemistry in the
first field and separates optional provenance with TAB. Headered tables require
one case-insensitive ``SMILES`` or ``CXSMILES`` column; ``smiles_column`` may
name an explicit column. Existing compressed CSV/TSV inputs are supported.

Deprotection
------------

``deprotect`` is opt-in. ``conservative`` applies 84 reviewed transformations;
``aggressive`` adds 11 broader rules. ``replace`` writes the resulting planner
form, whereas ``append`` retains both the protected and changed deprotected
forms.

When deprotection is enabled, a protected canonical sidecar is always produced
for :doc:`/api_reference/synplan.chem.synthon`. The protected file—not the deprotected planner
stock—is the supported input to ``bb_classifying`` and ``bb_synthonizing``.

Identity outputs
----------------

``write_inchikey_stock`` writes full 27-character Standard InChIKeys. Standard
InChI is retained in the identity reference but is not a Tree identity mode.
Optional path fields override the derived output names. For an output named
``building_blocks.smi``, the derived files are ``building_blocks_protected.smi``,
``building_blocks.inchikey``, ``building_blocks_identity.tsv``,
``building_blocks_duplicates.tsv``, ``building_blocks_collisions.tsv``, and
``building_blocks_stereo.tsv``.

``identity_reference_file`` records the source index, input and canonical SMILES,
Standard InChI, full InChIKey, InChI return code/warnings, output origin, and
emission status. The identity and stereo reports omit the larger ``source_info``
payload; duplicate reports retain source indexes but omit first/duplicate source-info
columns. Collision reports preserve every collapsed source relationship.

Auditing and overwrite
----------------------

``write_audit_files`` adds ``fallback.smi``, ``fallback.tsv``, ``errors.tsv``,
``summary.json``, and ``run.log``. ``audit_overwrite: error`` refuses an
existing completed or partial bundle. ``replace`` retains completed files while
the replacement is staged and validated; ``summary.json`` is promoted last.

``num_workers: null`` selects the available CPU count with an eight-worker cap.
``batch_size`` controls ordered process-pool chunks. Parallel preparation keeps at
most one pending batch per worker and periodically recycles workers, so peak input
queue memory is bounded by roughly ``num_workers * batch_size`` records.
``run.log.partial`` is flushed at start and every 10,000 records while a run is in
progress; the terminal also shows a tqdm progress bar. It becomes ``run.log`` when
the validated bundle is committed. Library defaults keep
auditing and InChIKey output disabled.

Planning stock format
---------------------

Stock decoding is configured independently from tree-search behavior:

.. code-block:: yaml

   building_blocks:
     identity_format: auto  # auto | smiles | inchikey
     chemistry_column: null
     delimiter: null

``identity_format`` describes either canonical SMILES or full Standard
InChIKeys. Raw InChI is reference data only and is not accepted as a stock
encoding. With ``auto``, the complete file is validated and no format setting is required.
``chemistry_column`` may select an
otherwise non-standard header in CSV/TSV input, and ``delimiter`` overrides
the suffix-derived table delimiter.

The loader returns a typed stock using either canonical SMILES or full Standard
InChIKeys. :class:`~synplan.utils.config.TreeConfig` receives that resolved
stock and does not own file-format settings.
