# Changelog

All notable changes to SynPlanner are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.5.0] - 2026-05-16

> Migration guide: see [docs/user_guide/migration.rst](docs/user_guide/migration.rst).
> Priority rules concept page: see [docs/methods/priority_rules.rst](docs/methods/priority_rules.rst).

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
- New `synplan/route_quality/` module implementing the competing-sites scoring framework
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
- API docs for `synplan.route_quality` module
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

[Unreleased]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.5.0...HEAD
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
