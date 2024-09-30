.. _standardization:

================================
Reaction standardization
================================

This page explains how to do a reaction standardization in SynPlanner.

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

The current available reaction standardizers in SynPlanner:

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Reaction standardizer              Description
    ================================== =================================================================================
    reaction_mapping_config            Maps atoms of the reaction using chython (chytorch)
    functional_groups_config           Standardization of functional groups
    kekule_form_config                 Transform molecules to Kekule form when possible
    check_valence_config               Check atom valences
    implicify_hydrogens_config         Remove hydrogen atoms
    check_isotopes_config              Check and clean isotope atoms when possible
    split_ions_config                  Split ions in reaction when possible
    aromatic_form_config               Transform molecules to aromatic form when possible
    mapping_fix_config                 Fix atom-to-atom mapping in reaction when needed and possible
    unchanged_parts_config             Remove unchanged parts in reaction
    small_molecules_config             Remove small molecule from reaction
    remove_reagents_config             Remove reagents from reaction
    rebalance_reaction_config          Rebalance reaction
    duplicate_reaction_config          Remove duplicate reactions
    ================================== =================================================================================
