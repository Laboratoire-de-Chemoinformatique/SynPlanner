# Review - Alex

## Scope

Reviewed the current dirty worktree, including the untracked revision module, tests/fixtures and `tutorials/08a_Protection_Handling.ipynb`, against main/HEAD `53ffed21794e829cd660b63effaaf4351e10cf95`. Focus: newcomer discovery, prerequisites and the Route revision learning path. No code, notebook or test edits. Kept the requested simplified notebook workflow and fixed search limits as constraints, not review complaints.

## Findings

### Onboarding Blockers

1. **[P2, introduced] The new tutorial cannot be reached from its advertised filename.** `tutorials/README.md:29` links to `route_pg_support.ipynb`; `docs/methods/routes.rst:175` gives the same obsolete filename. The actual added file is `tutorials/08a_Protection_Handling.ipynb`. Verified with `Path.is_file()`: advertised file is absent, actual file exists. Following the tutorial index yields a missing page. Update these two references to the actual filename; no notebook restructuring is needed.

2. **[P2, introduced] The new setup has no documented path to its required prepared catalogue.** `tutorials/08a_Protection_Handling.ipynb:89-97` defaults to `synplan_data/building_blocks/all-bb-2026-06/building_blocks.json`, and lines 114-117 immediately load the rules/model/catalogue. Setup prose at lines 29-31 only names environment overrides: it does not explain where to obtain or prepare these files. A user who follows the linked tutorial 08's `download_preset("synplanner-gps")`, or the current documented `synplanner-gps-mcule-molport` preset, does not obtain that default catalogue path; the latter explicitly supplies `building_blocks/mcule-molport-2026-08/building_blocks.json.gz` (`docs/get_started/data_download.rst:13-24`). The new notebook has no `download_preset`, preparation step, or setup link. Consequently a clean installed checkout cannot reach planning without independently discovering and adapting resource preparation. Minimal fix: add a short prerequisite link/command for preparing the intended catalogue and setting `SYNPLAN_BUILDING_BLOCKS_JSON`, plus the model/rule acquisition pointer. Preserve the intended catalogue and fixed search parameters. Alternatively support the documented preset explicitly if that is the intended example input. The README's universal Colab/setup promise at `tutorials/README.md:3-9` is pre-existing; adding this notebook without a setup cell makes the new entry an unmentioned exception, so clarify that exception if setup remains manual.

### Important Clarity Issues

None beyond the actionable setup issue above. The new Route documentation explains input immutability, accepted-insertion semantics, step indexing, unresolved added reagents, unsupported cases and graph-only validation sufficiently for this review scope.

### Suggestions

No additional requested changes. The direct `tree.routes()` workflow, concise before/after drawings and explicit no-suitable-repair outcome are useful as written. Export workflows, stereo-status detail, CGR plots and atom labels are not necessary onboarding fixes.

## Docs And Tests Reviewed

- `tutorials/README.md`; full source and saved output types of `08a_Protection_Handling.ipynb`; setup of existing `08_Protection_Scoring.ipynb`.
- `docs/methods/routes.rst` changes and `docs/get_started/data_download.rst`; resource references in README and building-block documentation.
- New `ProtectionRouteReviser` public models, configuration and execution flow; Route reconciliation/serialization documentation changes.
- `tests/unit/route_quality/test_revision.py` examples covering construction, carry, mappings, diagnostics, limits and round trips; `tests/regression/test_boceprevir_pg_revision.py` recorded-route example.
- Read-only filesystem/source probes verified the broken filename and absence of resource preparation/download setup. Did not rerun MCTS or assert that fresh planning deterministically reproduces the saved route.

## Missing Explanations

Only the concrete resource acquisition/preparation pointer in finding 2. Manual environment overrides are already explained.

## Open Questions

None blocking this review. Choosing whether the new notebook is explicitly local/manual or should satisfy the index's Colab promise is a small documentation/setup choice; neither requires changing the Route API or search.
