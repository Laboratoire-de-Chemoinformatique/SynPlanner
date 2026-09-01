====================================
InChIKey building-block catalogues
====================================

SynPlanner supports a vendor-aware JSON catalogue for retrosynthetic planning.
It uses Chython Standard InChIKeys throughout; the catalogue and MCTS identity
path do not convert molecules through RDKit.

Preparation
===========

The public Python function and CLI command are the same operation:

.. code-block:: python

    from synplan.chem.utils import standardize_building_blocks

    standardize_building_blocks("building_blocks.tsv", "building_blocks.json")

.. code-block:: bash

    synplan building_blocks_standardizing \
      --input building_blocks.tsv \
      --output building_blocks.json

The output extension selects the behavior. ``.json`` expects exactly one
case-insensitive ``SMILES`` column and one or more ``*_ppg`` columns. Other
supported molecular output formats retain the legacy canonical-SMILES
standardization path.

JSON preparation parses and canonicalizes with stereo enabled, removes the
``_ppg`` suffix from vendor names, and omits blank and zero prices. A negative,
non-finite, or non-numeric price rejects its complete row. Duplicate full
InChIKeys retain the first canonical SMILES and merge the minimum positive
price per vendor. This is deliberate: Standard InChI may merge some tautomeric
representations, and the initial implementation does not preserve their
alternative SMILES.

The complete catalogue is published atomically only if every row succeeds.
Failures are collected in ``building_blocks.json.errors.tsv`` and an existing
JSON output is left untouched.

The on-disk shape is:

.. code-block:: json

    {
      "LFQSCWFLJHTTHZ-UHFFFAOYSA-N": {
        "smiles": "CCO",
        "vendors": {"LN": 13.0},
        "has_stereo": false
      }
    }

Runtime indexes and MCTS identity
=================================

``load_building_block_indexes()`` streams the JSON and returns two immutable
indexes:

* ``building_blocks_by_inchikey`` maps each complete key to one
  :class:`~synplan.chem.building_blocks.BuildingBlock`.
* ``building_block_candidates`` maps the first 14 key characters to every
  matching record. A bucket never chooses an arbitrary stereoisomer.

The target selects one policy for the complete search. If it has any explicit
atom or bond stereo, MCTS requires full-key matches. Without explicit target
stereo, MCTS uses the first 14 characters and accepts every corresponding
candidate. The relaxed policy intentionally also collapses isotope and
protonation information.

Each finalized ``Precursor`` generates its Chython InChIKey at most once.
Subsequent exact checks use that cached complete key, while relaxed checks slice
its prefix. Legacy SMILES/SDF/CSV/TSV stock and
``Tree(building_blocks=set(...))`` callers continue to use canonical-SMILES
membership. JSON catalogues are restricted to retrosynthesis; forward search
keeps the legacy path.

Route costs
===========

Routes remain detached and immutable. Pass only the candidate index when a
cost is needed:

.. code-block:: python

    tree.run()
    routes = tree.routes()
    cost = routes[0].calculate_cost(building_block_candidates)

The target stereo determines exact versus relaxed candidate selection just as
it does in MCTS. Repeated leaves are counted as molar equivalents. The method
selects the cheapest positive vendor offer, assumes one equivalent per leaf and
100% reaction yield, and treats each catalogue number as an unnormalised raw
price per gram. Missing and catalogue-present-but-unpriced leaves are reported
separately; an incomplete route has ``null`` complete totals and retains its
partial priced totals.

For CLI planning with a JSON catalogue, SynPlanner writes
``route_costs.json``. Because tree node IDs restart for every target, this
sidecar is keyed first by input target SMILES and then by route tree-node ID.
Existing route JSON and visualisation schemas are unchanged.

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
