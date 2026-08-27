# Changelog

All notable changes to SynPlanner are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Route scoring is a post-search step, not something the tree holds. `Tree` no longer
  takes `route_scorer` (it raises `TypeError`), and `Tree.route_score` returns the
  search's own number, always cheap. Rank with
  `ProtectionRouteScorer.from_config().rank(routes)`, which hands them back best first.
  A scorer stands behind one number, `score`; `rank` orders by it, and whether that
  number is built out of the search's is the scorer's own business, not the base
  class's. For this one it is `search_score * S(T)`, so it needs routes a search
  produced; `competing_sites_score` is S(T) alone, for routes read from a file.
  **`tree.routes()` now returns search order where a `route_scorer` used to make it
  protection order.** On a 330-route celecoxib tree the
  call goes from 33.3s to 0.01s, because the protection scan was 95% of it and nobody
  had asked for it.

- **`best_route_score` in `tree_search_stats.csv` is the raw search score now.** With a
  `route_scorer` configured it used to be the protection-blended number, so the column
  is silently rescaled across this release and is not comparable with an earlier run's.
  Nothing else in the CSV changed.

- `RouteScorer.rescore` is `RouteScorer.ranking_score`: nothing is re-scored, it reads
  the search's number off the route and weights it by `score`, and it reads as what it
  is at the `key=` it is passed to. It takes a `Route` and reads the search score off
  it, instead of being handed one; `score` takes a `Route` too. Both were passed raw
  reaction tuples.

- `rank` no longer lets a route with no search behind it win by default. A search score
  the route does not carry used to stand in as `1.0`, which is not neutral -- on a
  330-route celecoxib tree the real search scores run 0.048 to 0.311, so one route read
  back out of a file ranked first of 331 whatever its quality. A list where nothing
  carries a search score now ranks on `score` alone, and a list mixing the two raises
  `ValueError` rather than picking a winner on two scales. `ranking_score` raises on a
  route with no search score for the same reason. Its docstring stops advertising the
  mix it could not do.

### Removed

- `CompetingSitesScore.rank_routes`, which had no caller inside SynPlanner and blended
  by `(1 - w) * normalised_search + w * S(T)` -- an additive formula that is not the
  paper's. The paper (Westerlund et al., 2025) re-ranks "by using the default state
  score, weighted by the competing sites score", which is the plain
  `search_score * S(T)` that `ProtectionRouteScorer` does. Use `rank`, which takes
  `Route` objects rather than `{route_id: {step_id: ReactionContainer}}` and hands back
  routes rather than score tuples.

- `ProtectionRouteScorer(weight=...)`, whose `(1 - w) + w * S(T)` softening is nowhere
  in the paper and which nothing ever set below its `1.0` default, where it is
  arithmetically the plain product.

- `ProtectionConfig.score_weight` and `ProtectionConfig.enable_reranking`. The weight
  was `rank_routes`'s, and the flag was never read by anything. A YAML carrying either
  key is now rejected rather than silently ignored.

### Fixed

- The GUI's "Generate full HTML report" button lost its protection ranking when the
  scorer came off the `Tree`, and the report came out in search order with nobody told.
  It ranks again, behind the button where the wait already sits under a spinner.

- The route lists under the drawings in `routes_clustering_report` and
  `routes_subclustering_report` were numbered in the tree's step order while the discs
  in the drawing above them were numbered in the route's, so "Step 3" in the text and
  the disc marked 3 were different reactions on a convergent route -- 21 of the 330
  routes of a celecoxib tree. The Tree-sourced half of both reports now reads the
  route's own steps; the JSON-sourced half, which `synplan clustering` uses, still
  does not.

- The functional-group detector cached its matches by a SMILES that carries no atom
  numbers while the matches carried them, so two structurally identical molecules
  numbered differently received each other's atom numbers -- indices the queried
  molecule does not contain. The cache now stores positions in the canonical SMILES
  order and fills in the queried molecule's numbering on the way out. On a 330-route
  celecoxib tree the old cache mis-answered 112 of 6105 lookups and changed the
  reacting group identified on 10 routes; on a larger one it mis-answered 228 and
  hid real penalties, reporting S(T) 1.00 for routes that score 0.875, which moved
  the ranking from position 35 down. Whether it changed a score at all depends on
  the tree.

## [1.7.0] - 2026-08-25

### Added

- Added ring-forming disconnections to the priority rule set, so a planning run can
  propose building a heterocycle rather than only buying one. 69 of the 76 ring records
  ship a hand-authored `retro_smarts` naming the reagents its reaction consumes, taking
  the default set from 39 rules to 108. See `docs/methods/priority_rules.rst`,
  "Ring-forming rules".

- Added `synplan.utils.frames.ChemFrame`, a pandas frame that depicts any column holding
  a chython object, with `rules_frame`, `synthons_frame` and `tree_stats_frame` built on
  it. See `docs/user_guide/tables.rst`.

- Added a building-block check before the search in `run_search`, reported as
  `target_in_stock` in the statistics CSV and on the console. See
  `docs/methods/planning.rst`, "Targets that are already purchasable".

- Added `ProductsTruncatedWarning`, raised when `max_products` ends an enumeration
  before the walk completes, so a truncated depth-first result is not mistaken for a
  sample of the library. See `docs/configuration/synthonisation.rst`.

- Added `synplan.chem.reaction.curation.rebalancing`, which adds the molecules an unbalanced
  reaction is missing without needing atom mapping. Missing carbon is recovered
  as a substructure of the reactants, and the remaining deficit is covered from
  a table of small molecules. Where the reaction is mapped, the CGR names the
  bonds that break, so reagents it never touched are carried through whole
  instead of being cut apart. Imputed species are rejected when an atom is left
  with an unsatisfied valence, which is what distinguishes Mg(OH)Br from a bare
  `[Mg]Br`, and species that cannot survive in a flask are broken into what they
  vent.
  See `docs/methods/rebalancing.rst` for the measured results and their limits.

  Exposed as the `rebalance_reaction_config` standardization step, with options
  to name the reagent behind a redox step rather than balance it with loose
  hydrogen, to refuse an answer that invents free hydrogen the record cannot
  account for, to drop the products the reactants cannot have made, and to read
  or ignore the atom mapping. A reagent the record names leaves as its spent
  form rather than as whatever covers the arithmetic. Every answer carries a
  confidence score, and `min_confidence` refuses the ones that fall below it.

- Added `scripts/rebalance_bench.py`, which measures success rate and accuracy
  against SynRBL's validation set. Rows whose reference is missing, or does not
  itself balance, are reported apart rather than counted as losses.

- Added `TreeConfig.direction`, which selects between retrosynthetic and forward search
  and validates that the tree and the rollout evaluator score the same finish line —
  in forward mode `building_blocks` is the goal to reach rather than the stock, and a
  configuration where only one of the two carries it is now rejected instead of
  silently searching toward one target and rewarding another. `apply_reaction_rule`
  gains `co_reactants`, without which a bimolecular rule run forward is handed one
  structure and yields nothing. Partner selection is not implemented, so the tree does
  not yet pass co-reactants during expansion.

- The test suite runs in parallel. `pytest-xdist` joins the dev group and CI runs
  `pytest -n auto --dist loadscope`, taking CI from 224s to 79s and a local no-coverage run from
  120s to 40s. `loadscope` keeps each module's tests on one worker so module-scoped fixtures — the
  clustering integration pipeline in particular — are built once rather than once per worker. No
  test was deleted, skipped or weakened to get there; the end-to-end planner and clustering tests
  that catch real regressions all still run.

- `test_mcts_imports_do_not_depend_on_route_import_order` spawns its six fresh interpreters
  concurrently instead of serially, 10.8s to 2.5s. Each snippet still gets its own untouched
  interpreter, which is the whole point of the test; they simply no longer queue behind each
  other's torch import.

#### Synthons (Synt-On port)

- Added `synplan.chem.synthon`, a native port of Synt-On (SynthI; Zabolotna et al., *J.
  Chem. Inf. Model.* 2022, 62(9), 2151-2163). Seven components, none cut: building-block
  classification and synthonisation, fragmentation, recombination, Bemis-Murcko
  scaffolds, the rule of two, and positional analogue scanning. CLI: `bb_classifying`,
  `bb_synthonizing`, `synthon_fragment`, `synthon_enumerate`, `bb_scaffolds`, configured
  by `configs/synthonisation.yaml`. See `docs/methods/synthons.rst`.
- Added opt-in audited output bundles to the five audited synthon CLI workflows. Set
  `write_audit_files: true` in `SynthonConfig` to write `fallback.smi`, `fallback.tsv`,
  `errors.tsv`, `summary.json` and `run.log` beside the primary output, with an exact
  success/fallback input partition and atomic replacement. `audit_overwrite` defaults to
  `error`; use a dedicated output directory per command. Python return values and CLI
  messages are unchanged. See `docs/configuration/synthonisation.rst`.
- Added synthon coverage of a mapped reaction: `classify_coverage` answers whether a
  reaction builds a bond one of the 39 acyclic disconnections already breaks, and the
  `synthon_coverage` CLI splits a reaction file on that answer. Its use is corpus
  preparation. 37.9% of a 100k mapped USPTO sample is covered. See
  `docs/methods/synthons.rst`, "Corpus coverage".
- The shipped synthon disconnections can be used as an MCTS priority-rule set.
  `synthon_priority_rules()` returns them under the source name `"synthon"` as
  `run_search(priority_rules=...)` input. On a 40-target FDA-2020 sample at a fixed
  iteration budget this solves 30/40 against a policy-only 23/40 (McNemar p=0.0156). A
  synthon-stock design was built, measured at 21/40 and reverted. See
  `docs/methods/synthons.rst`, "Using the rules for planning".
- Added heterocyclisation support to synthon enumeration: ring-forming disconnections, a
  `ring_pairs` table and `SynthonConfig.ring_closure_sizes`. The reference excludes ring
  closure by design; it is expressible here because a ring synthon carries product bond
  orders, so no bond order is rewritten at join time. This changes default enumeration
  output — across ten drug targets unique products go from 21 to 28. See
  `docs/methods/synthons.rst`, "Enumeration".
- Curated the heterocyclisation block from 9 ring rules to 76, in two id blocks: `R16`
  keeps the shipped families and `R17.1`-`R17.93` holds the new ones. **This changes
  enumeration output and rule ids** — the nine `R16.1`-`R16.9` selectors are gone, so a
  `rules_selection` naming them must be rewritten. Four defects in the shipped nine are
  fixed, each of which changes an answer, and ten curated rules are held out pending
  chemist review. See `docs/methods/synthons.rst`, "The disconnection rules".
- Every shipped disconnection now records where it came from and what it is.
  `provenance` is `human` for the 78 rules converted from the reference and `llm` for the
  76 ring rules authored here, so the half no chemist has signed off is identifiable
  rather than inferred from a rule id. Beside it each rule carries `reaction_name`,
  `forms`, `reagents` and `supersedes`. See `docs/methods/synthons.rst`, "Provenance and
  what is not covered".
- `rebalance_reaction_config` now imputes the molecules a reaction is missing
  rather than round-tripping it through a CGR. The old step could only
  redistribute atoms the reaction already had, and only when the mapping was
  good enough to build a CGR from, so it left every genuinely unbalanced
  reaction unbalanced.

- `rebalance_reaction_config` now runs after `remove_reagents_config` rather
  than before it. Reagent removal moves spectators out of the reactants and
  products, which unbalances whatever was balanced first.

- An oxidation recorded without its oxidant is now balanced with a peroxide
  rather than a bare oxygen atom. chython cannot hold atomic oxygen, so an
  answer spelled that way balanced in memory and came back off disk unbalanced.

- Naming the reagent behind a plain loss of hydrogen is now gated on
  `add_redox_agents`, as its documentation always said. By default the hydrogen
  is left loose, which is the honest way to say the record was written short.

- Rebalancing now withdraws the atom mapping where imputation leaves atoms on
  only one side. Nothing establishes which invented atom on one side is which on
  the other, and numbering them apart let a CGR compose while naming the wrong
  reaction centre. A caller that needs a mapping maps the balanced reaction
  itself.

- Package layout is now written down and enforced. `docs/development/package_layout.rst` states
  seven rules — four import layers, configuration beside its domain, verbs for pipeline stages and
  nouns for things, no package named `data`, adapters beside their target named after their source,
  entry points only in `interfaces`, and flat packages until roughly ten modules — plus a table
  saying where a new use case belongs. `tests/unit/test_package_layers.py` asserts the layer graph;
  three modules under `utils` are recorded there as known violations so the debt stays visible and
  cannot grow.

  Applying those rules moved several things. `synplan.synthon` is now `synplan.chem.synthon`: it
  imports from `chem` and nothing in `chem` imports it back, so it was always a leaf on top of the
  chemistry, not a peer of it. Inside it, `enumeration` became `enumerate` and `reactor` became
  `transformer` (which also ends the collision with `chem.reaction.reactor`), `data` became `rules`
  because it holds the shipped rule files, and `authoring` became `rules.validate` next to the
  converter that should call it. The command layer left the package: `cli` and `audit` are now
  `synplan.interfaces.synthon_commands` and `synplan.interfaces.synthon_audit`. The priority-rule
  adapter moved to `synplan.chem.reaction.rules.synthon`, beside the loader it feeds.

  `synplan.chem.data` became `synplan.chem.reaction.curation` — it holds no data, only the
  standardizing, filtering, mapping and rebalancing pipeline, and the old name collided with the
  synthon package's genuine data directory.

  `synplan.utils.config` was 767 lines holding every package's configuration. `TreeConfig` and the
  evaluation configs are now in `synplan.mcts.config`, the policy, value and tuning configs in
  `synplan.ml.config`, `ReactorConfig` in `synplan.chem.reaction.config` and `RuleExtractionConfig`
  in `synplan.chem.reaction.rules.config`; `synplan.utils.config` keeps only `BaseConfigModel` and
  `NestedConfigContainer`, which makes `utils` a leaf again.

  Every old import path still works, resolved lazily through a module-level `__getattr__` so that
  the shims do not reintroduce the import edges the layer rule forbids, and each emits a
  `DeprecationWarning` naming its replacement.

- `synplan.utils.parallel` no longer imports torch at module level. `select_device` is its only
  consumer and now imports it when called, so the process pool helpers and `default_num_workers`
  are torch-free. The synthon package took the framework purely to reach
  `min(os.cpu_count() or 4, cap)`, which cost 366 ms of a 501 ms import; importing
  `synplan.chem.synthon.config` now takes 143 ms and pulls no torch at all.

- `chython-synplan` is pinned to 1.104, which adds the `Synthon` atom family,
  `SynthonContainer`, the `_token` bracket field in SMILES and SMARTS, `!rN` as a real
  excluded-ring-sizes constraint, and the depiction fixes for synthon structures.

### Fixed

- `in_stock` in the exported routes now means purchasable. The route exporter set the flag from
  a node's position in the tree, marking every leaf available and every intermediate not, so the
  file `--export_routes` writes contradicted `extracted_routes.json` on the same run.

- Decomposing a RouteCGR dropped the implicit hydrogen on an aromatic nitrogen, so every azole
  came back as an unmatchable `c1cnc2...` and the documented building-block lookup silently found
  none of the routes using one.

- The protection scanner reported a pair it had never assessed as `compatible`. A missing matrix
  row, a missing column, or a reaction whose reacting group could not be identified now report
  `unknown`. Scores are unchanged; only the label was wrong.

- The solvent strip list warned about its own discarded stereochemistry at import, before any
  user molecule existed, which fired on `--help`.

- Documentation in six places said a `route_scorer` makes routes come back re-ranked. Nothing
  reorders: `winning_nodes` keeps discovery order and neither exporter sorts, so the export is
  byte-identical with and without one. Rank on `tree.route_score` yourself.

- Rebalancing no longer caps an open bond with a halide on a carbon that already
  holds two oxygens. That invented carbonate halides, which balance perfectly
  and do not exist. An acyl halide has one oxygen on that carbon and is
  unaffected.

- Rebalancing no longer leaves a reaction whose CGR cannot be composed. Imputed
  species are parsed fresh, so they started at atom 1 and landed on top of the
  reaction's own numbering, and on mapped input every later standardization step
  then failed.

- The fragment search no longer lets hydrogen outvote heavy atoms when reading
  where a bond broke. Counting it equally picked a second acid out of an ester
  rather than the alcohol that actually left.

- A halogenation written without its halogen is now balanced as elemental
  halogen going in and the hydrogen halide coming out, rather than as the acid
  going in and loose hydrogen venting.

- `RebalanceReactionStandardizer.from_config` no longer binds its options
  positionally, where reordering either signature would have misbound them
  silently.

- The ORD reader now chooses the desired product within each outcome rather than
  across the whole record, so a second outcome that flags nothing keeps its
  products and their yields.

- `split_ions_config` no longer fails on every ionic reaction. It read a
  molecule's total charge off an attribute chython-synplan no longer has.

- Nine more tests asserted a shape rather than a behaviour, each confirmed by mutating the code they
  cover and watching them stay green. The MHN ranking network had no behavioural test at all: collapsing
  every rule to one identical association vector — leaving the model unable to rank anything, which is its
  entire job — passed all 46 tests in the suite's largest file. It now asserts the ranking contract, that
  two rules are scored apart and that permuting the rule rows permutes the logits. `classify_reaction_type_detailed`
  was verified only by `rtype in (...)` tuples that included the `"other"` fallback, so stubbing the whole
  function to return `"other"` passed; each case now asserts the specific label its reaction must produce.
  `Tree.to_stats_dict` was checked by key name, so returning every value zeroed passed. Three
  `RouteScanner` tests looped over `interactions` without asserting the list was non-empty, so returning
  `[]` passed; they are now one test asserting the exact interaction. `test_compose_route_cgr_tree_based_invalid_route_id`
  never called the function it named. `Precursor` construction was asserted with `is not None`, so dropping
  canonicalisation passed. Retro-amidation asserted only that some product came back, so emitting an amine
  instead of a carboxylic acid passed.

- `classify_reaction_type_detailed` documented an `'acylation'` return value it cannot produce — the more
  specific amide and ester branches always match first — and the changelog advertised the classifier as
  12-category. Both now say 11, which is what the function actually returns. Behaviour is unchanged; the
  loose tests were what let the wrong count ship.

- Four tests asserted a shape rather than a behaviour and passed against broken code. Each was
  rewritten and then re-checked by re-applying the mutation that had slipped through.
  `test_audit_run_publishes_consistent_sidecars_and_summary` verified every artifact's provenance
  hash by calling the same `sha256_file` that had written it, so stubbing that function to a
  constant left all 83 audit tests green; it now compares against an independent
  `hashlib.sha256`. `test_juxtaposed_recursive_primitives_are_anded` counted semicolons in the
  translated SMARTS, so emitting a malformed pattern passed; it now checks that the ported query
  accepts an amine and rejects both an amide and a sulfonamide, which is what the AND means.
  `test_no_standardization` asserted only the return type, so dropping the standardization step
  entirely passed all 21 tests in the file; it now asserts that the flags change the result.
  `test_in_stock_flags` asserted the flag list's length, so inverting every flag passed; it now
  asserts the flags.

### Removed

- Removed the `cgr_connected_components_config` filter. Everything it caught was
  already caught by `multi_center_config` or `dynamic_bonds_config`, it is in
  neither shipped config, and against rebalanced reactions it is actively wrong:
  a dissociated leaving group is an atom with no bonds, so it is always its own
  component.

- Removed the `compete_products_config` filter. It compared reagents against
  products, so it only saw a competing product after atom mapping and reagent
  removal had already chosen between them, and reagent removal discards the
  loser outright. `rebalance_reaction_config`'s `drop_competing_products`
  replaces it, reading the imbalance instead and needing no mapping.

## [1.6.0] - 2026-08-03

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
  `PipelineSummary` in `synplan.chem.reaction.curation.reaction_result`

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
- CGR-based `ReactionClassifier` with broad (4-category) and detailed (11-category)
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
