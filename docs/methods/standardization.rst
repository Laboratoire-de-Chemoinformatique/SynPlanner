.. _standardization:

================================
Reaction standardization
================================

This page explains how to do a reaction standardization in ``SynPlanner``.

Reaction mapping
--------------------------------

Reaction atom-to-atom (AAM) mapping in SynPlanner is performed with GraphormerMapper,
a new algorithm for AAM based on a transformer neural network adopted for the direct processing of molecular graphs
as sets of atoms and bonds, as opposed to SMILES/SELFIES sequence-based approaches, in combination with the
Bidirectional Encoder Representations from Transformers (BERT) network. The graph transformer serves to extract molecular
features that are tied to atoms and bonds. The BERT network is used for chemical transformation learning.
In a benchmarking study, it was demonstrated [https://doi.org/10.1021/acs.jcim.2c00344] that GraphormerMapper
is superior to the state-of-the-art IBM RxnMapper algorithm in the “Golden” benchmarking data set
(total correctly mapped reactions 89.5% vs. 84.5%).

Reaction standardization
--------------------------------

The reaction data are standardized using an original protocol for reaction data curation
published earlier [https://doi.org/10.1002/minf.202100119]. This protocol includes two layers:
standardization of individual molecules (reactants, reagents, products) and reaction standardization.
Steps for standardization of individual molecules include functional group standardization, aromatization/kekulization,
valence checking, hydrogens manipulation, cleaning isotopes, and radicals, etc.
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
- Reaction rebalancing (rebalancing reaction)
- Duplicate reaction removal (remove duplicate reactions)

