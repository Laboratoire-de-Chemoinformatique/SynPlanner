# Changelog

All notable changes to SynPlanner are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/compare/v1.4.0...HEAD
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
