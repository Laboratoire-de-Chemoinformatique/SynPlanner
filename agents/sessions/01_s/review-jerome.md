# Review - Jerome

## Scope

Reviewed the current dirty worktree against HEAD/origin/main `53ffed21794e829cd660b63effaaf4351e10cf95`: reconciliation and RouteCGR entry points, Route serialization, protection revision, scanner/scorer/planner interfaces, configuration, and associated new tests/documentation. Review only; this report is the only file written. The requested simplified `tutorials/08a_Protection_Handling.ipynb` workflow does not require exports or additional infrastructure.

## Findings

### Architectural Blockers

No additional architectural blocker found. The representation compatibility issue below is confirmed by Jack and should be resolved before declaring the input adapters interchangeable. It does not demonstrate failure of the requested mapped chemical invariants.

### Important Design Risks

**[P2] Exact RouteCGR identity now depends on the adapter used to construct it (same finding as Jack; count once).** `representation/route_cgr.py:822-823,869-870` supplies explicit dependency depths for native Routes, while `_Fold` at `route_cgr.py:521-523` still derives depths with the existing overlapping-atom heuristic for dictionaries. The existing exact identity contract includes those depths (`representation/hash.py:142-149,163-170`). Therefore this adapter difference crosses into equality, hashing, and deduplication rather than remaining a depiction detail.

Jack's concrete probe uses `chain(True)` from `tests/unit/route_quality/test_revision.py`, builds its native RouteCGR, deconvolutes with `reactions_from_route_cgr()`, and rebuilds through the dictionary adapter. `route_cgrs_equal()` returns false; atom 4 changes depth from `{3}` to `{2}` even though mapped chemistry is preserved. See `review-jack.md` for the runnable reproduction and observed hashes. I checked the dependency and hash paths rather than duplicating his execution.

Keep the resolution bounded to this representation contract. The fix needs consistent retained/derived depth semantics across this roundtrip; moving reconciliation into another module or replacing the revision API would not address it. Do not simply remove depth from existing exact hashes without an explicit compatibility decision.

### Suggestions

- **Optional: reuse scan data if candidate scanning becomes a measured bottleneck.** `quality/protection/revision.py:366` performs a detailed candidate scan; `revision.py:382` then calls `CompetingSitesScore.score_route()`, which repeats the scan at `quality/protection/scorer.py:68` and reads coverage through `last_processed_steps` at line 83. A small internal score-from-scan helper could make coverage, targeted-interaction validation, and score use one result. This is an efficiency/maintenance suggestion, not a demonstrated correctness failure; the current serial execution is coherent.
- **Optional documentation clarification only.** `ProtectionRevisionAction` says all references are final positions at `revision.py:49`, although `original_step_id` is deliberately an input position. `RevisedRoute` says detached at line 79, although the explicitly tested no-action result returns the original object at lines 339-340. The public methods document unchanged no-action behavior, and tests require it; clarify these short docstrings rather than changing that behavior.

## Boundaries Reviewed

- **Mapping ownership:** `_reconcile_route` in the existing representation module is a reasonable minimal owner. Route JSON export, RouteCGR composition, and revision all reach it through `build_route_cgr`. Explicit identity links prevent chemically identical branch occurrences from being merged. There is no architectural need to reintroduce the deleted reconciliation module.
- **Scanner compatibility:** `scan_route(..., detailed=True)` has a concrete purpose: exact site handles and explicit processed/skipped coverage used when accepting revisions. Default tuple callers remain supported. The richer return object is not gratuitous abstraction, and deleting it would require equivalent information elsewhere. Literal overloads could improve static ergonomics later, but are not a delivery requirement.
- **Revision boundaries:** graph changes stay in the reviser; site identification stays in the scanner, candidate ranking in the planner, and route interpretation in Route/representation. Constructor injection remains available; `from_config()` is a convenience factory. No runtime model download, network requirement, export layer, or new persistence system was introduced.
- **Data lifecycle:** accepted candidates use detached copies; no-action results intentionally preserve original identity. Reaction metadata, selected products, provenance, unresolved stock state, and dependency links are explicitly carried through reconstruction. Existing stereo reassessment remains the relevant mechanism after route changes.
- **Failure boundaries:** candidate exceptions become candidate diagnostics; failed/incomplete composition or scans prevent acceptance. Unsupported shared protected intermediates and byproduct-only sites are explicit limitations rather than silent graph approximations.

## Missing Architecture Tests Or Checks

- Add an exact hash/equality roundtrip assertion for a route with at least three sequential steps, both before and after protection insertion. The current `mapped_cgr_state()` assertions intentionally omit history metadata and cannot detect the confirmed adapter mismatch.
- Specify/test whether an already globally mapped Route and its reaction dictionary must yield identical exact RouteCGR identity. This is the same representation issue, not a second finding.

I reviewed existing tests without creating or changing tests. Root reports the complete suite passed: 397 passed, 1 xfailed; lint, type, and notebook syntax/schema checks also passed. Jack independently reports 282 focused tests passed. These checks establish substantial chemistry coverage but do not cover the exact identity mismatch above.

## Open Questions

No user clarification is needed to proceed with the minimal requested architecture. The remaining design decision is the established depth/identity contract across native Route and dictionary/deconvolution adapters; resolve it explicitly while preserving the new dependency-aware atom reconciliation.
