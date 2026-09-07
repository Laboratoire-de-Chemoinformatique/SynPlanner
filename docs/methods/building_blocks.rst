====================================
InChIKey building-block catalogues
====================================

SynPlanner supports vendor-aware JSON, gzip-compressed JSON and SQLite catalogues for
retrosynthetic planning.
It uses Chython Standard InChIKeys throughout; the catalogue and MCTS identity
path do not convert molecules through RDKit.

Loading a prepared catalogue
===========================

Distribute the prepared ``building_blocks.json.gz`` to avoid repeating
standardization and InChI generation on each machine. The normal loader detects
the format and retains all stereo identities and vendor prices:

.. code-block:: python

    from synplan.utils.loading import load_building_blocks

    building_blocks = load_building_blocks("building_blocks.json.gz")

On first use, the file streams into an indexed SQLite cache; no extracted JSON
copy is needed. Subsequent runs open that cache without loading the catalogue
into Python objects. The same process reuses the immutable catalogue object.
The prepared file stores full InChIKeys and canonical SMILES; loading does not
parse either into molecules. Retain the source TSV separately when original
vendor SMILES are needed.

The default cache directory is ``$XDG_CACHE_HOME/synplanner/building_blocks``
(or ``~/.cache/synplanner/building_blocks``). The lower-level
``load_building_block_catalogue(path, cache_dir=...)`` accepts an explicit cache
directory. Changes to the source file, adjacent ``meta.yaml``, preparation
version or Chython version select a new cache. Source checks on subsequent
loads use filesystem metadata, so they do not rehash a large file. Old cache
files can be removed when no workers reference them.

For a portable prepared release, create the database explicitly and keep its
versioned filename stable:

.. code-block:: python

    from synplan.chem.utils import standardize_building_blocks

    standardize_building_blocks("building_blocks.json.gz", "bb-v1.sqlite")
    building_blocks = load_building_blocks("bb-v1.sqlite")

Prepared conversion reuses the supplied full identities, SMILES, stereo flags
and offers; it never recomputes their chemistry. Invalid prepared records reject
the whole build. SQLite metadata records the schema/preparation versions, source
SHA-256, release identity, record counts and original ``meta.yaml`` text. A
downloaded SQLite release opens directly and does not need the JSON source.
Compress it for transfer if useful, then decompress once before opening it.

Preparation
===========

The public Python function and CLI command are the same operation:

.. code-block:: python

    from synplan.chem.utils import standardize_building_blocks

    standardize_building_blocks("building_blocks.tsv", "building_blocks.json")

For large vendor updates, prepare a compressed distribution file using multiple
CPU processes:

.. code-block:: bash

    synplan building_blocks_standardizing \
      --input building_blocks.tsv \
      --output building_blocks.json.gz \
      --num-workers 8

The Python function accepts the same ``num_workers=8`` option. The default is one
process. Batches are merged in source order so duplicate handling and error
reports are identical for sequential and parallel preparation. In a Python
script, call parallel preparation inside ``if __name__ == "__main__":``.

The output extension selects the behavior. ``.json`` and ``.json.gz`` expect a
TSV or TSV.GZ with exactly one case-insensitive ``SMILES`` column and one or more
``*_ppg`` columns. Other supported molecular output formats retain the legacy
canonical-SMILES standardization path.

``.sqlite`` accepts either that raw vendor TSV/TSV.GZ or prepared JSON/JSON.GZ.
The same CLI and ``num_workers`` option apply. Raw TSV preparation computes
identities once and merges duplicate offers on disk. Calling
``load_building_blocks("vendor.tsv", num_workers=8)`` also prepares and caches
TSV files with ``*_ppg`` headers automatically, preserving their offers.

JSON preparation parses and canonicalizes with stereo enabled, removes the
``_ppg`` suffix from vendor names, and omits blank and zero prices. A negative,
non-finite, or non-numeric price rejects its complete row. Duplicate full
InChIKeys retain the first canonical SMILES and merge the minimum positive
price per vendor. This is deliberate: Standard InChI may merge some tautomeric
representations, and the initial implementation does not preserve their
alternative SMILES.

Every invalid row is omitted and recorded in ``<output>.errors.tsv``. If at least one row succeeds, all valid
records are published atomically and the function returns the output path. If no
row succeeds, the error report is written, an existing output is left
untouched, and preparation raises ``ValueError``. A clean run removes a stale
error report.

The on-disk shape is:

.. code-block:: json

    {
      "LFQSCWFLJHTTHZ-UHFFFAOYSA-N": {
        "smiles": "CCO",
        "vendors": {"LN": 13.0},
        "has_stereo": false
      }
    }

Runtime catalogue and MCTS identity
===================================

``load_building_block_catalogue()`` returns a read-only
``SQLiteBuildingBlockCatalogue`` implementing ``BuildingBlockCatalogue`` (a
mapping interface). The mapping key is the first 14 InChIKey
characters and each value is a tuple containing every matching
:class:`~synplan.chem.building_blocks.BuildingBlock`. Every record retains its
complete InChIKey, vendor offers, and stereo flag; a bucket never chooses an
arbitrary stereoisomer.

Each reader uses an 8 MiB SQLite page cache and an LRU of 8,192 prefix buckets,
including misses. Records are decoded only on lookup. ``len(catalogue)`` gives
the number of prefixes; ``catalogue.record_count`` gives the number of full
identities. Explicit iteration and mapping equality visit the entire stock;
tree initialization, matching, costing and report generation do not.

Prepare once in the parent before launching parallel searches. Spawned processes
can share the database file; each process/thread opens its own read-only
connection and has its own bounded caches. Pass the database path to each worker
and call ``load_building_blocks(path)`` there; do not serialize the catalogue or
live connection. Save routes as JSON and catalogues as SQLite. The constructor
accepts ``expected_cache_id`` when a worker must enforce a particular release.
Do not share an open connection through
``fork``; use ``spawn`` or reload stock inside the worker. ``catalogue.close()``
releases the current thread's reader; a later lookup reopens it.

``match_building_blocks(catalogue, inchikey)`` always returns the complete
connectivity-prefix bucket. Full InChIKeys and ``has_stereo`` remain catalogue
metadata used by stereo-compatible stock selection.

MCTS preserves stereo. The first 14 InChIKey characters retrieve candidates;
Chython then checks identity and specified tetrahedral, E/Z and allene requirements.
Wrong and unspecified configurations cannot fulfill a specified request.
Relative groups and mixture records require assessment. Small molecules also
require actual stock records. Existing stereo-free policy weights use a separate
connectivity projection; this projection never decides stock membership.

Each finalized ``Precursor`` generates its Chython InChIKey at most once.
Repeated checks reuse the result for that precursor and immutable catalogue.
If Chython cannot generate an InChIKey for a malformed aromatic precursor,
the failed attempt is cached, a warning is logged, and the precursor is
conservatively treated as not purchasable instead of aborting the search.
Legacy SMILES/SDF/CSV and TSV without vendor headers, and
``Tree(building_blocks=set(...))`` callers continue to use canonical-SMILES
membership. Vendor-aware catalogues are restricted to retrosynthesis; forward search
keeps the legacy path.

Route costs
===========

Routes remain detached and immutable. Pass the same catalogue used by MCTS
when a cost is needed:

.. code-block:: python

    tree.run()
    routes = tree.routes()
    cost = routes[0].calculate_cost(building_blocks)

The method considers compatible records and honors the full record selected by
the route audit. An opposite isomer's cheaper offer cannot price the route. Repeated
leaves are counted as molar equivalents. It assumes one equivalent per leaf and
100% reaction yield, and treats each catalogue number as an unnormalised raw
price per gram. Missing and catalogue-present-but-unpriced leaves are reported
separately; an incomplete route has ``null`` complete totals and retains its
partial priced totals.

For CLI planning with a vendor-aware catalogue, SynPlanner writes
``route_costs.json``. Because tree node IDs restart for every target, this
sidecar is keyed first by input target SMILES and then by route tree-node ID.
Existing route JSON and visualisation schemas are unchanged.

HTML route reports also show the selected purchased structure, full InChIKey,
vendors and price per gram. Offers come from the detached route's
``selected_stock`` metadata, including after JSON export/import: rendering needs
no database connection. Prices retain catalogue units; no currency is inferred.

Synthonizer adapter
===================

``BBSynthoniser.synthonise_building_block(block)`` delegates through the
existing component-aware ``synthonise_smiles(block.smiles)`` path. Vendor and
identity metadata remain on the ``BuildingBlock``. MCTS does not invoke the
Synthonizer, and this catalogue adds no protection or stereo provenance.

Measured identity cost
======================

``scripts/benchmark_building_block_identity.py`` is a reproducible,
Chython-only benchmark. On 1,000 valid rows sampled from the combined
``all-bb-2026-06`` TSV, three repeats measured these means on the development
machine:

.. list-table::
   :header-rows: 1

   * - Operation
     - Mean per query
   * - Canonical SMILES set lookup on an already-finalized molecule
     - 0.254 microseconds
   * - Chython InChIKey generation plus lookup
     - 521.267 microseconds
   * - Cached ``Precursor`` InChIKey lookup
     - 0.103 microseconds

InChI generation is therefore the material cost and must happen once per
finalized precursor. The immutable dictionary lookup after caching is not the
bottleneck. Re-run the benchmark on deployment hardware rather than treating
these wall-clock values as universal.
