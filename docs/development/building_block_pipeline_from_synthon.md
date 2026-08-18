# Building-block pipeline after the Synthon port

- Status: implementation specification and migration record
- Baseline branch: `feature/synton`
- Baseline commit: `6f3b2c3`
- Branch point: `f4ce6fd` on `feature/synton`
- Successor branch: `feat/building-block-pipeline`
- Main ancestor: `v1.6.0` / `38f929a`

## Purpose

This document defines the ordinary building-block pipeline that follows the
native `synplan.enumeration.synthon` port. It records the ownership boundary, identity
contract, stereochemical migration, preparation artifacts, and remaining
planner integration.

The branch is named `feature/synton` (without the second `h`), while the
Python package and chemistry term are *synthon*. Archived building-block commits
are behavior references only. They predate the Synthon port and must not be
replayed mechanically.

## Ordinary stocks and Synthon stocks solve different problems

An ordinary building-block stock answers whether a concrete precursor is
purchasable and therefore whether an MCTS branch is solved. A
`SynthonStock` maps labelled reaction fragments to source building blocks for
fragmentation, analogue search, and library enumeration.

The current MCTS integration deliberately adapts the 39 default Synthon
disconnections to capped ordinary-molecule priority rules. Resulting children
are checked against the ordinary stock; a `SynthonStock` is not stored in
`Tree`.

```text
raw vendor catalogue
        |
        v
ordinary building-block preparation
  - stereo-preserving canonicalization
  - optional conservative/aggressive deprotection
  - direct full Standard InChIKey generation
  - separate Standard InChI reference data
  - deduplication, provenance, and audits
        |
        +--> protected canonical SMILES ------> BBClassifier / BBSynthoniser
        |
        +--> planner canonical SMILES --------> MCTS / rollout / routes
        |
        +--> full InChIKey stock -------------> MCTS / rollout / routes
        |
        +--> identity reference TSV
                  |
                  +-- canonical SMILES
                  +-- Standard InChI
                  +-- full Standard InChIKey
                  +-- source metadata and warnings

protected canonical SMILES
        |
        v
Synthon stock -------------------------------> fragment / enumerate / analogues
```

## What the Synthon baseline already provides

`synplan.enumeration.synthon` contains:

- `BBClassifier`: 147 ordered classes and 2,401 SMARTS;
- `BBSynthoniser`: component-aware building-block-to-synthon conversion;
- `SynthonStock`: labelled synthon SMILES to source building blocks;
- `Fragmenter`: target disconnection DAGs and stock-aware pathway ranking;
- `Enumerator`: direct and analogue library enumeration;
- positional analogue search, rule-of-two filtering, scaffold extraction, and
  leaving-group capping;
- five configured CLIs with optional audited output bundles;
- `synthon_priority_rules()` for ordinary-molecule MCTS priority expansion.

A fragmentation `Pathway` is a reagent set, not an ordered synthetic route. It
has no intermediates, step order, or yields and does not replace SynPlanner
route construction or export.

The configured enumeration-slot gap recorded in the original draft is closed
at this baseline: both regular and audited paths load `SynthonStock` with the
effective `SynthonConfig` and call `stock.slots(synthons, config)`. Analogue
expansion and Ro2 filtration therefore follow the selected settings. In strict
availability mode an empty/filtered slot rejects the pathway; in non-strict
mode the enumerator may fall back to the pathway synthon.

## Resolved design decisions

1. **Planner identity is typed.** `BuildingBlockStock` owns either canonical
   SMILES or full Standard InChIKeys. Legacy sets remain accepted as SMILES
   stocks.
2. **InChI is reference data, not a stock format.** Standard InChI is retained
   in identity reports as auditable provenance generated from the same RDKit
   molecule as each direct full key, but stock inputs are canonical SMILES or full Standard InChIKeys only.
3. **The complete stock is validated.** Auto-detection never trusts only a
   suffix or first row, and mixed/malformed identities fail.
4. **Planning preserves defined stereo.** R/S and E/Z structures remain
   distinct through target parsing, `Precursor`, tree hashing/deduplication,
   reactor products, and full-key stock membership.
5. **The protected canonical artifact feeds Synt-On.** Deprotected structures
   affect only the ordinary planner stock. They never silently enlarge the
   Synthon action space.
6. **Full keys are used.** The first 14-character connectivity block is not
   sufficient for purchasing/stock membership.
7. **Stock format and preparation are separate from `SynthonConfig`.**
   `BuildingBlockPreparationConfig` owns catalogue processing.

## AiZynth-style identity

AiZynthFinder uses the complete 27-character InChIKey for equality and stock
membership. Its connectivity-only comparison is a separate opt-in helper and
is not used by stock queries.

SynPlanner uses a direct stock-identity path:

```text
stereo-preserved Chython MoleculeContainer copy
    -> RDKit molecule (atom mapping removed)
    -> MolToInchiKey
    -> full Standard InChIKey
```

The public helpers are:

```python
molecule_to_inchi(molecule) -> str
inchi_to_inchi_key(inchi) -> str
molecule_to_inchi_key(molecule) -> str
molecule_identity(molecule) -> MoleculeIdentity
```

`MoleculeIdentity` separately retains canonical SMILES, Standard InChI, full
key, return code, and warnings from the same prepared RDKit molecule. Conversion
uses `rdkit.Chem.rdinchi`:

- return codes 0 and 1 are accepted;
- the InChI must start with `InChI=1S/`;
- stock keys are generated directly with `MolToInchiKey` and validated
  completely;
- reference InChI is generated separately, and direct/InChI-derived keys must
  agree in tests;
- atom mappings are removed without mutating the source;
- complete salts and isotopes are retained;
- valid warnings such as proton adjustment or metal disconnection are
  preserved in reports;
- parse, conversion, non-Standard, empty, and malformed results fail
  explicitly.

Direct RDKit molecule-to-key output must equal the generated
InChI-to-InChIKey result in regression tests.

## Typed ordinary stock

```python
BuildingBlockStock(
    keys=frozenset(...),
    identity_format="smiles" | "inchikey",
    canonicalize=False,
)
```

The stock exposes `key_for_molecule`, `contains_molecule`,
`without_molecule`, length, and iteration. The MCTS `min_mol_size`
shortcut remains in `Precursor.is_building_block()`; it must not be
represented as real stock membership.

Direct construction and `coerce_building_block_stock()` trust SMILES keys as
already prepared by default, avoiding a second chemistry pass over large legacy
stocks. Set `canonicalize=True` explicitly for raw SMILES iterables. The file
loader keeps its separate `BuildingBlockStockLoadConfig.standardize` policy.

`load_building_block()` accepts:

```python
load_building_block(
    path,
    config=BuildingBlockStockLoadConfig(identity_format="auto"),
)
```

- canonicalizable SMILES/CXSMILES;
- full Standard InChIKeys;
- `auto` detection;
- plain `.smi`, `.smiles`, and `.inchikey`;
- SDF;
- headered CSV/TSV and supported compressed tables.

Tables require exactly one selected case-insensitive `SMILES`, `CXSMILES`,
or `InChIKey` column. Empty, ambiguous, mixed, malformed, or
non-Standard files fail. HDF5/PyTables remains out of scope.

`load_building_blocks()` remains a legacy `frozenset` facade.

## Stereochemical migration

Historical `safe_canonicalization()` removes stereo. It keeps that default for
unrelated data workflows, while a copy-safe `clean_stereo=False` path is used
for planning and stock identity.

Planning now must:

- parse targets without deleting defined stereo;
- preserve stereo in `Precursor`;
- use `fix_stereo()`, rather than wholesale `clean_stereo()`, after reactor
  and CGR reconstruction;
- distinguish R/S and E/Z states in tree hashes and deduplication;
- remove atom mappings only for external identity generation.

This is intentionally behavior-changing. Stereochemical catalogues prepared
under the old policy should be regenerated. Old SMILES APIs remain callable,
but an old stereo-stripped stock cannot recover information it discarded.

## Preparation and deprotection

```python
prepare_building_blocks(...) -> BuildingBlockPreparationResult
standardize_building_blocks(input_file, output_file) -> str
```

The second function is the defaults-only compatibility wrapper.

The feature-local strict configuration is:

```yaml
input_format: auto
smiles_column: SMILES

deprotect: false
deprotect_policy: conservative
deprotect_output: replace

write_inchikey_stock: false
protected_output_file: null
inchikey_file: null
identity_reference_file: null
duplicates_file: null
collisions_file: null
stereo_file: null

write_audit_files: false
audit_overwrite: error
num_workers: null
batch_size: 500
```

Unknown keys and invalid dependencies fail. `num_workers: null` resolves
automatically with an eight-worker cap. The CLI accepts either a YAML
configuration or explicit processing flags, never both.

The deprotection taxonomy is feature-local and contains 95 programs: 84
conservative rules and 11 additional aggressive rules. The archived rules used
the pre-1.6 chython `xN` heteroatom-neighbour primitive. Their current
representation translates it to `yN`; `zN` remains hybridization. Every
translated positive exemplar and reviewed decoy is a regression contract.

Removal is iterative with a visited-state and maximum-pass guard. `replace`
writes the deprotected planner form; `append` retains both protected and
changed deprotected planner forms. `append` is catalogue enrichment, not a
claim that both forms are separately purchasable.

## Output contract

- `--output`: deduplicated stereo-preserving canonical-SMILES planner stock;
- `<stem>_protected.smi`: protected canonical Synt-On input whenever
  deprotection is enabled;
- `<stem>.inchikey`: optional deduplicated full-key planner stock;
- `<stem>_identity.tsv`: every source/candidate identity and emission status;
- `<stem>_duplicates.tsv`, `<stem>_collisions.tsv`, and
  `<stem>_stereo.tsv`: structured duplicate, identity-collision, and stereo
  reports.

When deprotection is disabled, `synthon_input` points to the primary output.
When enabled, it always points to the protected sidecar, independently of
`replace` or `append`.

The identity reference retains the source sequence and input chemistry,
canonical SMILES, Standard InChI, full key, return code/warnings, output origin,
and emission status. Deduplication never discards the source-index-to-identity
relationship. To keep full-catalogue reports compact, identity and stereo rows omit
`source_info`; duplicate rows omit `first_source_info` and
`duplicate_source_info` while retaining both source indexes.

## Audited outputs

Reusable record/provenance helpers live in `synplan.utils.audit`. Feature
status and report schemas remain under `synplan.chem.building_blocks`.
Existing Synthon CLI contracts are unchanged.

With `write_audit_files: true`, preparation also writes:

- `fallback.smi`;
- `fallback.tsv`;
- `errors.tsv`;
- `summary.json`;
- `run.log`.

Every artifact is staged as `.partial`. Input and taxonomy hashes are captured
before processing and verified unchanged before publication. Counts, input
partitions, line counts, and artifact hashes are validated; `summary.json` is
promoted last as the commit marker. `audit_overwrite: error` refuses existing
final or partial artifacts without touching them. `replace` removes stale
partials but retains the prior completed bundle until validation succeeds.

## Planner and route consumers

Tree and its rollout evaluator must use equivalent typed-stock identities.
`contains_molecule()` is the sole membership operation in:

- tree expansion and solved-node detection;
- rollout evaluation;
- planning CLI and `run_search`;
- GUI planning;
- reinforcement target removal and tree search;
- route extraction and visualisation;
- RDKit route conversion;
- both RouteCGR building-block selection paths;
- tree-aware route JSON `in_stock` flags.

The sibling planning `building_blocks` section accepts
`identity_format: auto | smiles | inchikey`. This source configuration is
parsed into
`BuildingBlockStockLoadConfig`; it is not part of `TreeConfig`. Standard InChI
appears only in preparation identity reports. SMILES input remains a typed,
stereo-preserving canonical-SMILES stock in the loaded Tree.

`Tree` accepts a typed stock or legacy set/frozenset. Pickle migration converts
current v1.6 frozensets into SMILES stocks while full-key sets remain
InChIKey stocks. Raw InChI and mixed or malformed legacy values fail
clearly. New round trips retain the identity format.
`Tree.save_pickle()` is unchanged and `TreeWrapper` is not restored.

Synthon priority-rule source, ID/name, policy rank, and route-arrow labels must
survive every stock migration.

## Input ownership

Format-level readers and provenance primitives belong in `synplan.utils`.
Chemical policy belongs under:

```text
synplan/chem/building_blocks/
    __init__.py
    config.py
    identity.py
    stock.py
    preparation.py
    deprotection.py
    reports.py
    data/protective_rules.tsv
```

Do not recreate a generic 700-line `molecule/io.py`, and do not place
ordinary-stock deprotection in route-quality protection or Synthon modules.

The preparation input contract supports SMI/SMILES/CXSMILES, SDF, and headered
CSV/TSV. Headerless molecule records use TAB for metadata so a complete
CXSMILES field is not truncated. Named table provenance, including vendor and
pricing columns, is retained in the reference output. Input duplicates remain
independently auditable even though the planner stock is deduplicated.

## Port provenance

Reconstruct against the Synthon baseline and record `Ported-from:` trailers:

| Archived commit | Behavior adapted |
| --- | --- |
| `098cbd4` | deprotection programs and transformation behavior |
| `74efd02` | ordinary-stock identity and planner/route integration, corrected to full Standard InChIKeys |
| `9030efc` | strict preparation configuration |
| `537be03` | YAML-configured CLI and config/flag exclusion |
| `06a872c` | usage guidance, presets, and migration concepts |

Do not replay merge `5cd3973`, old `cgr_opt` integration, obsolete v1.5.2
documentation, or generic molecule-I/O ownership.

## Validation gates

Identity and stock tests cover known values and direct-RDKit equivalence; R/S
and E/Z distinction; atom-map independence; salts/component order; aromatic
forms; tautomers; charges/zwitterions; isotopes; accepted warnings; hard
failures; complete-file validation; and raw-InChI normalization.

Preparation tests cover all 95 positive rule references, conservative versus
aggressive policy, decoys, overlaps, fixed-point/idempotence/cycle guards,
protected Synt-On input invariance, duplicate/collision provenance,
deterministic multiprocessing, and atomic overwrite/failure behavior.

Integration tests require equivalent solved routes from corresponding SMILES
and InChIKey stocks, legacy set compatibility, old/new pickle migration, and
agreement among Tree, rollout, reinforcement, GUI, RouteCGR, route JSON,
visualisation, and RDKit exports. Existing Synthon classification,
synthonisation, fragmentation, and enumeration fixtures remain unchanged.

Repository gates are Ruff, type checking, `git diff --check`, focused and full
pytest, Sphinx, and a non-CI full-catalogue run against
`all-bb-2026-06` in a separate experiment directory.

## Deferred decisions

Automatic planning-CLI construction of `synthon_priority_rules()` remains a
separate opt-in feature decision. It does not alter ordinary-stock identity or
the preparation-to-Synt-On boundary defined here.
