.. _building_block_method:

=======================================
Building-block identity and preparation
=======================================

Ordinary-stock membership is the terminal condition for a retrosynthetic
precursor. SynPlanner represents it with
:class:`~synplan.chem.building_blocks.stock.BuildingBlockStock`, independently
of the labelled Synthon stock used for analogue enumeration.

Identity model
--------------

A SMILES stock uses stereo-preserving canonical SMILES. An InChIKey stock uses
the full Standard InChIKey generated directly from the prepared molecule:

.. code-block:: text

   Chython molecule copy
       -> RDKit molecule without atom maps
       -> MolToInchiKey
       -> full Standard InChIKey

Preparation reports generate Standard InChI separately from the same RDKit
molecule and retain its warnings as provenance; only the directly generated
complete key is used for membership. Regression tests require the direct key to
equal the key derived from that Standard InChI. The first 14-character connectivity block is deliberately
not used because it collapses stereoisomers.

Defined tetrahedral and double-bond stereo survives target parsing,
canonicalization, precursor construction, reactor cleanup, tree hashing, and
stock lookup. Atom mapping does not contribute to external identity. Salts are
identified as complete records, and isotopes remain significant.

Preparation stages
------------------

Each source record is parsed with its metadata, standardized on a copy, and
written in stable input order. Optional deprotection reaches a bounded fixed
point under either the conservative or aggressive rule set. Final planner
candidates are deduplicated, while structured reports retain every
source-to-candidate and source-to-identity relation.

When deprotection is enabled, preparation has two intentional products:

* the ordinary planner stock follows ``replace`` or ``append``;
* the Synthon input remains the canonical protected catalogue.

This prevents a preparation policy from silently changing the set of labelled
Synthons available to fragmentation or enumeration.

Stock membership
----------------

For InChIKey stocks, each immutable planning ``Precursor`` lazily caches its
full key after the first membership check. The cache is not attached to mutable
Chython containers, where an in-place chemistry or stereochemistry change could
make it stale. The typed stock owns key generation and membership. The
small-molecule shortcut remains an MCTS policy in :meth:`synplan.chem.precursor.Precursor.is_building_block`;
it does not add fictitious keys to the catalogue.

The same typed-stock identity is used by Tree and rollout evaluation and is consulted
by route extraction, visualisation, RDKit conversion, RouteCGR processing, GUI
planning, and reinforcement search. Legacy sets are adapted as trusted prepared
SMILES by default or detected as full-InChIKey stocks. Pass ``canonicalize=True``
to :func:`synplan.chem.building_blocks.stock.coerce_building_block_stock` when a
raw SMILES iterable needs normalization. Raw-InChI stocks and pickles are
rejected.

For configuration and CLI usage, see :doc:`/configuration/building_blocks`
and :doc:`/user_guide/cli_interface`. For the labelled-fragment subsystem,
see :doc:`/api_reference/synplan.enumeration.synthon`.
