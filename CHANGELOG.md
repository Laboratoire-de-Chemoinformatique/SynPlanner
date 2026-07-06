# Changelog

All notable changes to SynPlanner are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.6.0] - 2026-08-03

### Added

- Added `synplan.chem.reaction.rules.symmetry` with
  `needs_decollapsed_matches()` for detecting reaction SMARTS where a
  non-identity left-hand-side automorphism changes the right-hand-side product
  rule patch.

### Changed

- `load_reaction_rules()` now parses TSV SMARTS once and, by default, disables
  Chython's `automorphism_filter` only for rules that need decollapsed symmetric
  matches. Callers can opt out with `decollapse_symmetric_matches=False`.
- TSV rule loading now builds `CanonicalRetroReactor` instances directly from the
  parsed reactant and product query patterns, preserving the existing pickle
  loading behavior unchanged.

### Fixed

- Fixed symmetric SMARTS rule loading so valid precursor orientations are
  retained when symmetric target-atom matches produce different product-side
  results, including external-fragment differences, target-target bond changes,
  and target atom-state changes.

## [1.5.2] - 2026-06-05

### Added

- Added `synplan.chem.reaction.routes` as the public route post-processing layer for route
  clustering, route quality, RouteCGR utilities, route I/O, analysis, depiction,
  and notebook plotting.
- Added typed RouteCGR and route-export contracts, including
  `RouteCGRBuildResult`, `RouteExportResult`, route diagnostics, and strict
  export failure handling while preserving the existing JSON and CSV shapes.
- Added `synplan.chem.reaction.routes.representation` to group RouteCGR builder, container,
  depiction, state, hashing, and native deconvolution helpers.
- Added route-aware RouteCGR composition with `route_order` metadata on dynamic
  atoms and bonds, transient bonds for bonds formed and later removed during a
  synthetic route, and compact per-step labels for reconstructing mapped route
  reactions from the RouteCGR itself.
- Added exact RouteCGR hashing in `synplan.chem.reaction.routes.representation.hash`, including a
  fast WL bucket hash followed by exact canonical confirmation for shared
  buckets.
- Added RouteCGR comparison helpers for identifying overlapping and unique route
  IDs across route dictionaries whose route IDs do not need to match.
- Added RouteCGR analysis helpers for selected building-block lookup and real
  versus supporting pseudo-reactant usage statistics.
- Added native RouteCGR deconvolution APIs (`routes_dict_from_route_cgrs()` and
  `reactions_from_route_cgr()`).
- Added `synplan.chem.reaction.routes.clustering` as a package, with clustering logic in
  `synplan.chem.reaction.routes.clustering.core` and route marker helpers in
  `synplan.chem.reaction.routes.leaving_groups`.
- Added `synplan.chem.reaction.rules` and `synplan.chem.reaction.rules.representation`
  as the public homes for rule analysis, extraction, priority rules, and Chython
  QueryCGR/Morgan rule representations.
- Added MHN ranking policy support (`architecture: mhn_ranking`) with QueryCGR
  rule fingerprints by default, optional QueryCGR rule-graph encoding with GPS,
  bounded runtime rule-association caches, dynamic runtime rule-set scoring, and
  checkpoint migration for the redesigned policy networks.
- Added `synplan mhn_network_tuning` for fine-tuning an already trained MHN
  ranking checkpoint on new rule policy data.
- Added policy/network modules for pure policy wrappers, graph embedders,
  checkpoint loading, featurization caches, and Lightning-based supervised
  training utilities.
- Added MHN ranking configuration, policy documentation, CLI documentation, and
  Tutorial 14 for MHN training/tuning workflows.
- Added strict Chython-to-RDKit SMARTS conversion with reverse diagnostics and
  semantic-loss reporting for strict round-trip validation.
- Added the `mhnreact_rdkit` rule-fingerprint mode for MHN ranking, using the
  original RDKit path-fingerprint template encoding.
- Added `Tree.save_pickle(path)` for direct tree persistence and the
  `synplan-convert-protection-data` command for protection source-data conversion.
- Added API documentation for `synplan.chem.reaction.routes`.

### Changed

- Requires `chython-synplan` 1.100, pinned exactly. Its SMARTS parser now reads
  92% of real-world patterns instead of 57%, and three primitives follow
  Daylight rather than chython's own reading: an unmarked charge is
  unconstrained, `A` is any aliphatic atom where `*` is any atom, and `x` is
  ring connectivity where the heteroatom count it used to mean is now `y`.
  Rules written against the old meanings of `A` and `x` need updating.
- Supports Python 3.14, which chython 1.100 is the first release to ship wheels
  for. Python 3.15 is out of reach until torch, rdkit, numpy and chytorch
  publish cp315 wheels.
- Verifies chython ring-connectivity queries against RDKit's `AtomRingBondCount`
  during rule conversion instead of passing them through unchecked; only
  `rings_count` and heteroatom queries stay unverifiable.
- Dropped the `exclude-newer` resolution cutoff; the exact `chython-synplan` pin
  is what holds the lock steady now.
- Moved route clustering, route analysis, route depiction, route I/O, notebook
  plotting, route-quality scoring, and RouteCGR implementation to
  `synplan.chem.reaction.routes`.
- Moved reaction-rule analysis, extraction, priority-rule parsing, and rule
  representation helpers under `synplan.chem.reaction.rules`.
- Moved MCTS expansion wrappers into `synplan.mcts.policy`, separating generic
  policy selection from MHN-specific dynamic rule-association preparation.
- Split ML graph embedders, policy networks, checkpoint loading, featurization,
  and training helpers into focused submodules under `synplan.ml`.
- Updated internal imports, tests, tutorials, Colab notebooks, API docs, and
  migration notes to use the new route, rule, policy, and ML module layout.
- Centralized route parent-chain traversal for tree, visualisation, and RDKit
  route conversion while preserving their existing output formats.
- `compose_route_cgr()` now uses a fast default path that returns the RouteCGR
  without eagerly reconstructing reactions; callers that need the debug
  `reactions_dict` can request it explicitly or deconvolute from the RouteCGR.
- RouteCGR composition now assumes route-level atom mapping: the same atom-map
  number must identify the same atom throughout a route, while independently
  mapped local reaction steps are outside the supported input contract.
- MHN light prediction keeps the rollout-friendly integer rule-count API; dynamic
  MHN rule associations are prepared by the full `predict_reaction_rules()` path.

### Fixed

- Four shipped configuration files could not be loaded at all, because every
  config model forbids unknown keys and nothing checked that the files parse:
  the two combined-policy configs carried `priority_rules_fraction`, which
  belongs to a different model, `mhn_ranking_policy_training.yaml` used
  `rule_encoder_type` where the field is `rule_embedding_type`, and
  `planning_value.yaml` was unreachable because the planning CLI merged the
  `node_evaluation` section into the tree config. Loading every shipped config
  is now a test.
- Protecting-group selection cannot protect aldehydes, ketones, alpha-halo
  ketones or phthalimide-protected amines. Nine of the twenty-five published
  templates attach at two points, which acetals and dithianes need because they
  replace the carbonyl double bond rather than substituting onto it, and the
  product builder adds a single bond and deletes nothing. Eighteen reactive
  functions allow only those templates, so they have no protection option at
  all. The table is unchanged and the loader now says this once at startup
  rather than failing quietly at each call.
- The functional-group patterns in `extraction_functional_groups.yaml` used `A`
  to mean any atom, which chython 1.100 reads as any *aliphatic* atom, so they
  silently stopped matching aromatic neighbours such as benzaldehyde and
  styrene. They now use `*`.
- Route JSON export now recovers precursor nodes whose producing fragment is
  structurally identical to the consuming reactant even when atom-number overlap
  with the final target is absent, and drops malformed routes instead of emitting
  JSON `null` children.
- QueryCGR rule fingerprints and QueryCGR rule graphs preserve important query
  constraints, including degree, hydrogen-count, ring-size, set-valued atom
  labels, and dynamic bond semantics.
- Runtime MHN rule-association caches are bounded and keyed by the ordered rule
  SMARTS plus the rule representation contract.

### Removed

- Removed deprecated route compatibility namespaces: `synplan.routes`,
  `synplan.routes.quality`, and `synplan.route_quality`.
- Retained `synplan.chem.reaction_routes` as a deprecated forwarding namespace
  for the historical `main`-branch route module paths.
- Removed `TreeWrapper`; use `Tree.save_pickle(path)` instead.
- Removed the old flat `synplan.mcts.expansion` and `synplan.ml.networks.policy`
  modules in favor of the new policy/network package layout.

## [1.5.1] - 2026-06-04

### Fixed

- Fixed score assignment for priority-rule nodes during evaluation.

## [1.5.0] - 2026-05-16

> Migration guide: see [docs/user_guide/migration.rst](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/docs/user_guide/migration.rst).
> Priority rules concept page: see [docs/methods/priority_rules.rst](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/docs/methods/priority_rules.rst).

### ⚠️ Backwards-incompatible

- `apply_reaction_rule` default `top_reactions_num` raised from 3 → 5;
  pin the old behaviour with `apply_reaction_rule(..., top_reactions_num=3)`.
- Per-node state moved from nine `Tree.nodes_*` parallel dicts onto
  `Node` attributes (e.g. `tree.nodes_depth[nid]` → `tree.nodes[nid].depth`).
  Old reads raise `AttributeError` with a migration hint.
- `Tree.stats` is now a `TreeStats` dataclass — use attribute access,
  not subscript or `.get()`. `Tree.to_stats_dict()` is unchanged.
- `EvaluationStrategy.evaluate_node` signature collapses
  `(node, node_id, nodes_depth, nodes_prob)` into
  `(node, node_id, nodes: dict[int, Node])`.
- Pickled `Tree` instances from 1.4.x partially fail on the migrated
  surfaces (`tree.stats.*`, new `Node` provenance fields). Re-run the
  search to recover full functionality.
- YAML `key:` (null) for nested standardization / filtering configs now
  enables the step with defaults instead of silently disabling it. To
  disable, omit the key entirely.
- `ReactorConfig` no longer exposes `fix_aromatic_rings` or
  `fix_tautomers`. `CanonicalRetroReactor` always forces
  `fix_aromatic_rings=False` and runs the inline kekule + thiele +
  tautomer-fix pipeline in its `_patcher`; tautomer fixing inside
  that inline call relies on chython's default `fix_tautomers=True`.
- `load_reaction_rules` now defaults to `check_atom_mapping="reject_unmapped"`:
  SMARTS rules without atom maps are rejected with an error naming the
  offending TSV row. Pass `check_atom_mapping="off"` to restore the old
  behaviour.

### Added

- Atom-mapping validator wired into the reader chokepoint
  (`synplan.utils.files.parse_reaction`) and the SMARTS rule loader
  (`synplan.utils.loading.load_reaction_rules`), with helpers in
  `synplan.chem.utils`:
  `reaction_mapping_status`, `reaction_string_mapping_status`,
  `is_reaction_atom_mapped`, `assert_reaction_atom_mapped`, and the
  `AtomMappingCheck` literal type (`"off" | "reject_unmapped" | "reject_partial"`).
  Tagged reactions get `meta["mapping_status"]` for downstream routing.

- Priority-rule support for MCTS expansion: a mapping of named SMARTS
  rule sets passed to `Tree(..., priority_rules={"ugi": ..., ...})`,
  tried ahead of the learned policy on every node. Each set gets its
  own counter pair in `tree.stats.per_priority_source[<name>]`.
  Reserved name `"policy"` is rejected; `use_priority=True` without
  `priority_rules` raises.
- Optional iterated rule application via
  `TreeConfig.priority_rule_multiapplication` and the new
  `apply_reaction_rule(multirule=True, rm_dup=True)` kwargs — useful
  for stripping every protective group of a given kind in one step.
- Source-specific rule counters in `tree.stats` /
  `tree.to_stats_dict()`: `policy_rules_tried/succeeded`,
  `priority_rules_tried/succeeded` (aggregate), and the per-set
  `per_priority_source[<set_name>]` breakdown.
- Rule provenance on tree nodes and route outputs: `rule_source`,
  collision-safe `rule_key` (formatted as `<source>:<id>`), and exact
  1-indexed `policy_rank`.
- Route SVG labels for rule keys and policy ranks, opt-in
  partial-route rendering with `allow_unsolved`, and JSON-route SVG
  rendering that can display stored rule metadata.
- Tutorial 13 for the priority-rule workflow.
- `reactor_validation` knob on `RuleExtractionConfig` (default `True`): skip
  rules whose reactor cannot reconstruct the original products. The flag is
  tagged on each rule and counted in the extraction summary.
- `single_product_only`, `ignore_stereo`, and `worker_timeout_per_reaction`
  exposed on `RuleExtractionConfig` and `configs/rules_extraction.yaml`.

### Changed
- Bumped `chython-synplan` floor to `>=1.95`, which carries upstream fixes
  for: multi-component query mapping against single-component targets (Bug
  #3), `[C]` uppercase aliphatic strictening against aromatic carbons (Bug
  #4), and aromatic snapshot/restore protection in the reactor (Bug #6).
- Tree expansion now tracks exact policy Top-N rank from the expansion function
  and enforces the configured top-rules limit during rule iteration.
- Policy rule selection in `expansion.py:_predict_rules_common` uses
  `torch.topk` instead of full `torch.sort` over the rule-probability
  tensor. Same Top-K output; ~10% per-iter speed-up in MCTS on 17k-rule
  rule sets.
- `apply_reaction_rule` now skips computing the canonical-SMILES dedup
  key when neither `rm_dup` nor `multirule` is enabled — a no-op for
  performance (chython caches the SMILES anyway) but cleaner control flow.
- Route, RDKit, JSON, SVG, and tree-stat exports now carry source-aware rule
  metadata through serialized and rendered route outputs.
- Multi-product reactions and rules-all-filtered-out are now traceable
  through the per-reaction audit TSV (`<rules_path_base>.audit.tsv`) in
  addition to the summary counters.

### Fixed
- Winning rule rank reporting now uses stored policy ranks when available instead
  of approximating ranks from sibling probabilities.
- Route JSON export now attaches rule metadata to the matching retrosynthetic
  step order and avoids closure state leakage while building nested route nodes.
- Route SVG and RDKit route extraction now derive paths from node IDs, preserving
  priority/policy metadata and respecting `min_mol_size` during building-block
  checks.
- Repeated route SVG renders now clear stale molecule labels and statuses before
  applying route-specific annotations.
- Worked around chython's SMARTS writer emitting CXSMILES extension
  blocks (`|...|`) mid-string between disconnected fragments by
  stripping any CXSMILES block before tokenisation in the mapping
  validator.
- Regression tests now exercise the canonical-key invariant via two
  SMILES with different mapping offsets instead of
  `QueryCGRContainer.remap`, which was broken in chython ≤1.94 (its
  override forwarded an unsupported `copy=` kwarg to `Graph.remap`).

## [1.4.4] - 2026-05-04

### Changed
- Improved standardization, filtering, and rule extraction pipeline robustness
  with worker-serialized results, stable CGR deduplication keys, visible progress
  reporting, and explicit stale-worker cleanup.
- Improved policy dataset preparation with safetensors-backed cache reuse,
  parallel preprocessing progress, nested result directory creation, and a
  stratified split that avoids duplicate-product validation leakage.
- Made optional remote logger integrations installable through extras instead
  of core dependencies: `SynPlanner[litlogger]`, `SynPlanner[wandb]`,
  `SynPlanner[mlflow]`, or `SynPlanner[loggers]`.
- Configured `ty` rules for the dynamic chython, RDKit, PyTorch, and
  NumPy typing surface while keeping unresolved-reference checks enabled.
- Documented updated CLI flags, policy logger settings, GPS embedder
  configuration, PR review acceptance guidelines, and new shared pipeline/cache
  helper modules.

### Fixed
- Rule extraction summary now always reports failed reaction counts.
- Ranking dataset cache loading now iterates safetensors keys correctly.
- Deduplication now fails fast if worker-computed dedup keys are unavailable.
- Standardization ion-splitting warnings now use the module logger.
- Reaction standardization now preserves mapped SMI source columns in
  successful output rows and error reports, applies a fixed canonical chemistry
  order for enabled standardizers, and excludes failed reactions from the
  standardized output when `ignore_errors` is enabled.

## [1.4.3] - 2026-03-19

### Changed

#### Parallelization
- **Removed Ray dependency entirely** — all parallel pipelines now use
  `ProcessPoolExecutor` via the new `process_pool_map_stream` utility
- `process_pool_map_stream` enhanced with `ordered` mode (submission-order
  yield), per-future `timeout`, `initializer`/`initargs` for non-picklable
  worker state, `max_tasks_per_child` (Python 3.11+), and `on_timeout` callback
- New `graceful_shutdown()` context manager for SIGTERM/SIGINT handling with
  automatic signal handler restoration

#### Data Pipeline
- Standardization, filtering, rule extraction, and ML preprocessing pipelines
  migrated from Ray to `process_pool_map_stream` with initializer pattern
- Writer-side CGR dedup: `hash(~rxn)` (condensed graph of reaction hash) for
  mechanism-level reaction deduplication — 8 bytes per entry in memory
- New shared result types: `ProcessResult`, `ErrorEntry`, `FilteredEntry`,
  `PipelineSummary` in `synplan.chem.data.reaction_result`

#### Compatibility
- Removed `from __future__ import annotations` from all modules (Dagster
  compatibility)
- Forward references quoted for self-referencing return types

### Removed
- `ray` dependency removed from `pyproject.toml`
- `init_ray_logging()` removed from `synplan.utils.logging`
- `DedupActor` Ray actor removed

### Added
- 10 unit tests for `process_pool_map_stream` and `graceful_shutdown`
  (`tests/unit/utils/test_parallel.py`)
- 8 unit tests for `ProcessResult`, `PipelineSummary`, and CGR dedup
  (`tests/unit/chem/data/test_pipeline.py`)

## [1.4.2] - 2026-03-15

### Added

#### ORD (Open Reaction Database) Support
- New `synplan/utils/ord/` package for reading ORD `.pb` Dataset files via protobuf
  (`dataset_pb2.py`, `reaction_pb2.py`) without depending on `ord-schema`
- `iter_ord_reactions()` iterator for lazy ORD `.pb` file parsing
- `convert_ord_to_smiles()` utility for batch ORD-to-SMILES conversion
- `synplan ord_convert` CLI command for converting ORD `.pb` files to `.smi`
- `ReactionReader` and `RawReactionReader` now accept `.pb` files natively
- `_ORDReadAdapter` for transparent ORD reading through the existing `Reader` protocol
- 367-line test suite (`test_ord_reader.py`) covering ORD parsing

#### Configuration
- `ReactorConfig` pydantic model for typed Reactor construction parameters
  (`automorphism_filter`, `delete_atoms`, `one_shot`, `fix_aromatic_rings`,
  `fix_tautomers`) with `to_reactor_kwargs()` serialization
- `load_reaction_rules()` now accepts optional `reactor_config` parameter

### Changed

#### Rule Extraction
- Rule deduplication now uses CGR (condensed graph of reaction) instead of
  `ReactionContainer` hashing — correctly preserves query-level atom annotations
  (neighbors, hybridization) when rules contain `QueryContainer` molecules
- `_update_rules_statistics()` and `sort_rules()` updated to use `cgr_to_rule`
  mapping for CGR-based dedup
- `process_completed_batch()` receives `cgr_to_rule` dict

#### Docker
- Added `.dockerignore` to exclude `.git`, `.venv`, `docs`, `tests`, `tutorials`,
  build caches, and data directories from Docker build context

#### Dependencies
- Added `protobuf>=4.21` to core dependencies (ORD `.pb` support)
- Added `grpcio-tools>=1.78.0` to dev dependencies (protobuf code generation)

#### Fixes & Cleanup
- `depict_settings()` calls updated to module-level function (was
  `MoleculeContainer.depict_settings()`)
- `routes_clustering_report` / `routes_subclustering_report`: safer target SMILES
  lookup with `.get()` fallback instead of direct key access
- Removed unused imports: `yaml` from `filtering.py` / `standardizing.py`, `os` from `cli.py`, `Any` from `mapping.py`
- Import order cleanup (ruff/black formatting)

## [1.4.1] - 2026-03-03

### Fixed
- Coordinate bonds that break `mol_to_pyg` graph conversion are now removed via
  `remove_coordinate_bonds(keep_to_terminal=False)` before kekulization across
  6 call sites (`rdkit_compat.py`, `reaction.py`, `extraction.py`, `utils.py` ×2,
  `preprocessing.py` ×2)

### Changed

#### Documentation
- Replaced `.nblink` files with direct symlinks to tutorial notebooks (removed
  `nbsphinx_link` dependency)
- Version switcher now uses `READTHEDOCS_CANONICAL_URL` for correct multi-version
  docs hosting
- ReadTheDocs build switched from `jobs` to `commands` with explicit `uv run sphinx`
- Cleaned up `conf.py` comments and removed `nbsphinx_link` from extensions

### Infrastructure
- Bumped version to 1.4.1 in `pyproject.toml` and `uv.lock`

## [1.4.0] - 2026-03-03

> **This is a major breaking release.** SynPlanner now uses `chython-synplan` as its
> sole cheminformatics backend, replacing CGRtools and minimizing RDKit to an optional
> scoring dependency. **All pretrained models must be retrained** — chython produces
> different canonical SMILES, atom features, and reaction products than CGRtools for the
> same inputs. **Results from previous SynPlanner versions are not reproducible.**

### Added

#### Protection Strategy Scoring (NEW MODULE)
- New `synplan/chem/reaction/routes/quality/` module implementing the competing-sites scoring framework
  from Westerlund et al. (ChemRxiv, 2025)
- `FunctionalGroupDetector` with 102 SMARTS patterns across 18 reactivity categories
- `HalogenDetector` with 140 SMARTS patterns across 5 halogen families
- CGR-based `ReactionClassifier` with broad (4-category) and detailed (12-category)
  reaction type classification
- `IncompatibilityMatrix` with 3-level severity (compatible / competing / incompatible)
- `RouteScanner` for per-step competing functional group interaction detection
- `CompetingSitesScore` with worst-per-step S(T) formula for route quality scoring
- `ProtectionRouteScorer` integrated directly with `Tree` for automatic post-search
  route re-ranking based on functional group selectivity
- `ProtectionConfig` dataclass with YAML serialization
- Full test suite: 69 unit tests across 4 test modules

#### Search Algorithms
- `CombinedPolicyNetworkFunction` for weighted filtering + ranking logit combination
  with configurable `ranking_weight` and `temperature` parameters
- New evaluation strategies: `RDKitEvaluationStrategy`, `PolicyEvaluationStrategy`
- Stochastic mode for `RolloutSimulator` (probability-weighted rule sampling)
- Tree pruning via redundant expansion state caching (`enable_pruning` config)
- `predict_reaction_rules_light()` for lightweight rollout rule prediction

#### Data Pipeline
- `RawReactionReader` for lazy batch processing of raw SMILES/RDF strings
- Distributed SMILES parsing across Ray workers (was main-thread bottleneck)
- `BaseStandardizer` abstract class with template method pattern
- `StandardizationError` with safe pickling for Ray workers
- `STANDARDIZER_REGISTRY` for declarative standardizer configuration
- `DuplicateReactionStandardizer` with Ray `DedupActor` for cluster-wide dedup
- `DedupActor` Ray actor for cluster-wide unique reaction tracking
- 4 new reaction filters: `MultiCenterFilter`, `WrongCHBreakingFilter`,
  `CCsp3BreakingFilter`, `CCRingBreakingFilter`
- `ignore_errors` mode with structured TSV error files for all data pipelines
- Categorized error taxonomy (`_DATA_ERROR_STAGES`, `_DATA_ERROR_TYPES`)
  distinguishing data noise from pipeline bugs
- `parse_reaction()` with format auto-detection (SMILES / RDF)
- `load_rule_index_mapping_tsv()` for new TSV rule format

#### Infrastructure
- `download_preset()` for structured preset downloads from HuggingFace
  (replaces deprecated `download_all_data()`)
- HuggingFace data moved to `Laboratoire-De-Chemoinformatique/SynPlanner-data`
- Preset YAML manifests (e.g., `presets/synplanner-article.yaml`)
- TSV building blocks format support (`.tsv`, `.tsv.gz`)
- CUDA 12.6 and 12.8 extras (`--extra cu126`, `--extra cu128`)
- Python 3.13 and 3.14 support (`>=3.10,<3.15`)
- Multi-stage Docker builds with `uv sync --locked`
- `HEALTHCHECK` directive for GUI Docker image
- Cross-platform CI matrix (3 OS x 4 Python versions)
- `uv build --wheel` + `uv publish` for PyPI/TestPyPI releases
- `--ignore-errors`, `--error-file`, `--batch_size` CLI options on all processing commands
- `synplan download_preset` CLI command

#### Tutorials & Documentation
- Tutorial 00: Welcome to Chython (chython onboarding for new users)
- Tutorial 01: Coming from RDKit (migration guide with 35+ operation cheat sheet)
- Tutorial 07: Protection Scoring (end-to-end with capivasertib, 128 routes)
- Tutorial 08: Combined Ranking Filtering Policy (dual policy tuning)
- Tutorial 09: NMCS Algorithms (Nested Monte Carlo Search guide)
- API docs for `synplan.chem.reaction.routes.quality` module
- 5 new user guide pages linked from docs index

#### Configs
- `combined_ranking_filtering_policy.yaml` — combined policy network config
- `planning_combined_policies.yaml` — planning with combined filtering + ranking
- `planning_value.yaml` — GCN value network evaluation config
- `rules_extraction.yaml` — fine-grained atom info retention for rule extraction
- `extraction_functional_groups.yaml` — FG-aware extraction with 26 SMARTS patterns

#### Testing
- 80+ new unit and integration tests
- `test_clustering_visualization_e2e.py` — 27+ tests covering full clustering pipeline
- `test_loading.py` — building blocks loading with CSV, gzip, and TSV
- SAScore benchmark suite (`scripts/sascore_bench/`) with configurable YAML and plotting

### Changed

#### Chemistry Backend Migration (BREAKING)
- **ALL** CGRtools imports replaced by chython equivalents across the entire codebase
- `chython-synplan[racer-default]>=1.93` replaces both `cgrtools-stable` and the
  git-pinned chython fork
- RDKit isolated to optional `synplan/chem/rdkit_utils.py` for SA score calculations
- Module-level `smiles_parser` singleton removed; each module imports `chython.smiles`
- Bridge functions `cgrtools_to_chython_molecule()` and `chython_query_to_cgrtools()`
  deleted

#### Reaction Rule Format (BREAKING)
- Rules output changed from pickle to **SMARTS TSV** (human-readable,
  version-controllable, portable)
- TSV columns: `rule_smarts`, `popularity`, `reaction_indices`
- Legacy pickle still loadable with automatic conversion via
  `_convert_cgrtools_query_container()`
- `load_reaction_rules()` returns `tuple` (immutable, cached) instead of `list`

#### Reactor API (BREAKING)
- Reactor constructed with explicit `patterns=`, `products=`, `delete_atoms=False`
- Reactants unpacked with `*reactants` instead of passed as a list
- `molecule_substructure_as_query()` replaces CGRtools' `as_query=True` API
  using `QueryElement.from_atom()` with explicit `neighbors`, `hydrogens`,
  `ring_sizes` flags

#### MCTS Architecture (BREAKING)
- `evaluation_function` parameter type changed from `ValueNetworkFunction` to
  `EvaluationStrategy`
- `tree.policy_network` renamed to `tree.expansion_function`
- `tree.value_network` removed; replaced by `tree.evaluator`
- `tree.building_blocks` is now `frozenset` (immutable)
- `tree.reaction_rules` is now `tuple` (immutable)
- `evaluation_type` string dispatch replaced by typed evaluation config objects
- `value_network_path` parameter removed from `run_search()`; use
  `evaluation_config`

#### Data Pipeline
- Ray workers receive raw SMILES strings instead of parsed `ReactionContainer` objects
- `extract_rules()` returns `tuple[list, bool]` instead of `list`
- `sort_rules()` returns `tuple[list, dict]`; `single_product_only` parameter removed
- `filter_reaction()` returns 3-tuple `(bool, ReactionContainer | None, str | None)`
- `clean_atom()` no longer manages `hybridization` attribute
- `depict_settings` is now a module-level function, not a class method

#### Dependencies
- `cgrtools-stable==4.2.13` removed
- `chython` git pin replaced by `chython-synplan[racer-default]>=1.93`
- `chytorch-synplan>=1.70` (was `>=1.69`)
- `chytorch-rxnmap-synplan>=1.7` (was `>=1.6`)
- `rdkit>=2023.9.1` (relaxed from `>2025.3.5`)
- CUDA extras: `--extra cuda` replaced by `--extra cu126` / `--extra cu128`

#### Other
- `download_all_data()` deprecated in favor of `download_preset()`
- Type annotations modernized: `Dict`, `List`, `Union` -> `dict`, `list`, `|`
- `tqdm` -> `tqdm.auto` for notebook compatibility
- All existing tutorials (Steps 2-6) rewritten for chython-synplan

### Fixed
- Product validation now copies molecule before `kekule()` to prevent mutation
- `RankingPolicyDataset`: `if rule_id:` -> `if rule_id is None:` (was silently
  skipping rule index 0)
- Variable-shadowing bug in `_expand_node` (`for new_precursor in new_precursor`)
- `InvalidAromaticRing` exception now caught alongside `KeyError` and `IndexError`
- Reactor no longer deletes atoms by default (`delete_atoms=False`)
- Windows path handling
- CUDA/PyTorch resolution in CI
- GUI and CI fixes
- Visualisation bugs

### Breaking Changes Summary

> **Data & Reproducibility**: All pretrained models, reaction rules (pickle format),
> and building block files from previous versions produce **different results** with
> v1.4.0. Users must:
> 1. Re-extract reaction rules (now saved as SMARTS TSV)
> 2. Retrain all policy and value networks
> 3. Re-standardize building blocks
>
> The root cause is that chython produces different canonical SMILES, different atom
> feature vectors, different Kekulization, and different reaction products compared to
> CGRtools. While the 11-dimensional atom feature schema is unchanged, the underlying
> values differ for aromaticity perception, ring detection, and hydrogen counting.

| Breaking Change | Migration Path |
|---|---|
| CGRtools imports | Replace with `chython` equivalents |
| Pickle reaction rules | Re-extract rules (outputs SMARTS TSV) or load legacy pickle (auto-converted) |
| `ValueNetworkFunction` as Tree arg | Use `EvaluationStrategy` subclass |
| `evaluation_type` string config | Use typed config objects (`ValueNetworkEvaluationConfig`, etc.) |
| `tree.policy_network` | Use `tree.expansion_function` |
| `tree.value_network` | Use `tree.evaluator` |
| `tree.building_blocks` mutation | Filter before Tree init (`frozenset`) |
| `value_network_path` in `run_search()` | Use `evaluation_config` parameter |
| `--extra cuda` | Use `--extra cu126` or `--extra cu128` |
| `download_all_data()` | Use `download_preset()` |
| Pretrained models | Retrain — feature vectors differ |
| HuggingFace repo | Data moved to `SynPlanner-data` repo |

## [1.3.2] - 2025-12-14

### Added
- NMCS and LazyNMCS tutorials (`09_NMCS_Algorithms`)
- Combined ranking and filtering policy tutorial (`08_Combined_Ranking_Filtering_Policy`)
- SAScore benchmark scripts and result plotting
- Support for loading SMILES from CSV files

### Changed
- Moved build system from Poetry to uv

### Fixed
- PyPI publishing pipeline (`--skip-existing` flag)
- Black formatting

## [1.3.1] - 2025-11-13

### Fixed
- Streamlit GUI rerun error

## [1.3.0] - 2025-11-13

### Added
- NMCS (Nested Monte Carlo Search) and LazyNMCS search algorithms
- Best-first, breadth-first, and beam search strategies
- Parallel building block loading
- Unified evaluation function loading (`load_evaluation_function`)
- `silent` parameter for suppressing tree search progress output
- Clustering bug fix and improved test coverage

### Changed
- Search algorithms separated from the tree into dedicated modules
- Evaluation system refactored: unified node evaluation via `EvaluationService`
- Tree configuration updated: evaluation function now part of `TreeConfig`
- Rule extraction configuration updated
- Simplified Docker setup
- Removed single-core/single-worker legacy logic

### Fixed
- NMCS algorithm correctness fixes
- UCT formula after algorithm separation
- `mol_to_pyg` performance (removed unnecessary molecule copy)
- SAScore division-by-zero edge case with UCT
- Tree config backward compatibility

## [1.2.1] - 2025-09-15

### Changed
- Updated dependencies
- Improved README and documentation

## [1.2.0] - 2025-08-13

### Added
- Route clustering by strategic bonds (contributed by Almaz Gilmullin)
- Streamlit-based graphical user interface
- Route clustering CLI command
- Integration tests for clustering workflow
- HTML clustering report generation

### Changed
- Refactored route CGR representation (`SB-CGR`)
- Refactored visualisation module
- Enhanced GUI session state management

## [1.1.2] - 2025-05-11

### Changed
- Updated dependency versions

## [1.1.1] - 2025-05-11

### Added
- RxnMapper integration for atom-to-atom mapping

### Fixed
- PyPI publishing configuration and dependencies

## [1.1.0] - 2025-05-04

### Added
- Initial CI pipeline and tests
- Cross-platform dependency resolution

### Changed
- Refactored standardization pipeline
- Updated NumPy compatibility

## [1.0.0] - 2024-12-20

### Added
- Initial public release
- MCTS-based retrosynthetic planning with rollout evaluation
- Reaction data curation pipeline (standardization and filtration)
- Reaction rule extraction from reaction databases
- Ranking policy network training (GCN-based)
- HTML route visualisation
- CLI interface (`synplan` command)
- Docker images for CLI and GUI

[Unreleased]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.4.4...v1.5.0
[1.4.4]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.4.3...v1.4.4
[1.4.3]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.4.2...v1.4.3
[1.4.2]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.3.2...v1.4.0
[1.3.2]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/releases/tag/v1.0.0
