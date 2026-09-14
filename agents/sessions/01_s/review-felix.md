# Review - Felix

## Scope

Second-pass review of the current dirty worktree against HEAD/origin/main `53ffed21794e829cd660b63effaaf4351e10cf95`, including untracked protection revision code and tests. Read `agents/felix.md` first and all four completed reports before inspecting code. Challenged attribution, severity, acceptance boundaries, and uncovered failure assumptions. No application, test, fixture, or notebook changes; this report is the only file written. No fresh MCTS planning.

## First-Pass Reports Reviewed

- `agents/sessions/01_s/review-jack.md`
- `agents/sessions/01_s/review-margaret.md`
- `agents/sessions/01_s/review-alex.md`
- `agents/sessions/01_s/review-jerome.md`

## Findings

### Newly Found Blockers

None reproduced. The existing evidence supports one representation compatibility defect and two bounded onboarding fixes. It does not support claiming an introduced atom-mapping, stereochemistry, SB-CGR chemistry, or graph-connectivity blocker.

### Confirmed Concerns

1. **[P2, introduced] Exact RouteCGR identity depends on the input adapter.** Confirm Jack's Important finding and Jerome's corresponding risk; count once. Independently loaded `chain(True)` from `tests/unit/route_quality/test_revision.py`, composed its native RouteCGR, deconvoluted with `reactions_from_route_cgr`, and rebuilt through `build_route_cgr({0: reactions}, 0)`. `route_cgrs_equal` returned **False**, while `mapped_cgr_state` equality returned **True**. The explanation is supported by `representation/route_cgr.py:822–823,869–870`: native Routes supply explicit dependency depths, while dictionaries retain `_Fold`'s other depth calculation at `:521–523`. Existing exact hash/equality incorporates this metadata. This is a real public representation regression even on an unrevised route, so ordinary passing chemical-invariant assertions do not dismiss it. Its P2 severity is appropriate; the evidence does not establish corruption of the mapped chemistry.

2. **[P2 for tutorial delivery, introduced] Tutorial discovery and prerequisites need bounded documentation fixes.** Confirm the scope of Alex's two findings: `tutorials/README.md:29` and `docs/methods/routes.rst:175` advertise `route_pg_support.ipynb`, while the actual notebook is `tutorials/08a_Protection_Handling.ipynb`; the notebook's required prepared catalogue also lacks the acquisition/preparation pointer documented in Alex's reproducible setup analysis. These prevent following the advertised beginner workflow, not use of the installed revision API. Keep the two fixes distinct from a runtime chemistry blocker, and preserve the explicitly requested short notebook workflow.

3. **[Non-blocking interpretation limitation] Score improvement does not prove fewer total conflicts.** Independently reproduced Margaret's esterification example through the public default reviser with `max_revisions_per_route=1`: accepted **True**, score **0.5 → 0.8333333333333334**. Summing worst-per-step penalties from detailed scans gives **(0.5, 1 processed step) → (0.5, 3 processed steps)**. `_candidate_score` at `revision.py:366–385` genuinely checks the original targeted interaction and configured positive improvement, so the example does not bypass that stated acceptance contract. The new notebook sentence “a higher score means fewer problems” overstates the metric. It should describe the normalized score rather than an absolute count. The denominator limitation comes from the existing scorer; it is newly relevant when route length changes.

### Disputed Or Overstated Concerns

- **Do not escalate Margaret's chemistry boundaries into introduced blockers.** The existing N/O SEM row selection can produce a false class label and should remain visible as a metadata defect; the report correctly attributes the behavior to existing candidate expansion. Unsupported acetonide, missing stoichiometric byproducts, and absent reaction-condition validation are not evidence that the new acceptance boundary silently approves an unsupported two-attachment graph. `revision.py:408` explicitly rejects failed reagent mapping, and `docs/methods/routes.rst:171` describes graph proposals.
- **Missing stereo tests are not demonstrated stereo loss.** I added no tests to the repository, but ran an independent accepted esterification probe carrying an E/Z double bond through protection. One E/Z requirement was present, and its sign survived in the changed original transformation. An analogous carbon-13 substrate was also accepted under the existing mandatory mapped-chemistry checks. Combined with the existing tetrahedral test, this weakens a blanket claim that graft/canonicalization necessarily destroys stereo. It does not establish coverage of all stereogenic or charged sites.
- **Do not turn optional architecture advice into required redesign.** Jerome's shared reconciliation owner and explicit detailed scan are justified by the current responsibilities. Repeated scanning is a possible optimization, not an observed correctness failure. Existing `Route.stereo_status` at `route.py:355–362` checks route context, so merely retaining a detached stereo summary is not evidence that changed routes silently retain a fulfilled status.
- User-requested omissions of exports, assertions, stereo output, and CGR drawings are not tutorial defects.

### Suggestions

Resolve the exact identity contract and the two small tutorial setup/discovery issues without broad refactoring. Correct the score interpretation sentence; changing the underlying acceptance policy is a separate scientific/product decision.

## Missed Tests Or Evidence

- Exact native RouteCGR → reaction dictionary → RouteCGR hash/equality assertions for at least three sequential steps, before and after insertion. Current chemical-state assertions exclude the metadata responsible for the confirmed defect.
- Keep an explicit conflict-count example beside score interpretation; length normalization can conceal unchanged absolute penalty.
- Broad charged/salt, stereogenic attachment, additional attachment-family, and selective-condition cases remain coverage limits, not reproduced blockers.

Independent probes used `PYTHONPATH=.` and `/data1/agilmullin/almaz/SynPlanner/.venv/bin/python` from this worktree. The inline probe loaded the existing esterification/chain fixtures, compared exact versus chemical identity, computed detailed-scan penalty totals, and exercised E/Z and carbon-13 variants. All four probes completed normally with the results above. I did not repeat the full regression suite; root reports 397 passed, 1 xfailed, and clean lint/type/notebook checks, while the first-pass reports document their focused executions.

## Open Questions

No new clarification is required to complete the requested review. The outstanding representation choice is consistent depth/history semantics across supported adapters. Treat that as one explicit compatibility issue; do not claim the chemistry-preservation contract failed on the evidence available.
