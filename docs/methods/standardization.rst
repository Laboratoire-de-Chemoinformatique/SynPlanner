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
- Reaction rebalancing (rebalancing reaction)
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
