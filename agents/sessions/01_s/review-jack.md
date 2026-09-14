# Review - Jack

## Scope

Reviewed the current dirty worktree against `origin/main` / HEAD `53ffed21794e829cd660b63effaaf4351e10cf95`, including untracked `quality/protection/revision.py`, revision/reconciliation tests, and the recorded Boceprevir fixture. Focus: revision correctness, mapping and identity, scanner integration, and existing Route/RouteCGR API behavior. Application code, tests, and notebooks were not edited.

## Findings

### Blocking

No confirmed violation of the requested atom-ID, mapped SB-CGR chemistry, input-preservation, or producer/consumer-identity contracts in the exercised accepted revisions. The multi-step insertion and real Boceprevir regression pass. There is one confirmed existing-API regression below that should be resolved before treating the representations as interchangeable.

### Important

**[P2] RouteCGR exact identity changes on deconvolution/recomposition and through the existing reactions-dict adapter.**

- Location: `synplan/chem/reaction/routes/representation/route_cgr.py:822–823,869–870`; exposed to existing callers by `synplan/chem/reaction/routes/route.py:657–659`.
- The new Route path stamps dependency-derived depths onto `route_order`, while the unchanged dictionary path computes depths using any shared product/reactant atom. A linear three-step route therefore gets depth 3 for its first step through `Route.route_cgr()`, but depth 2 through the dictionary adapter. Reconstructing the dictionary with `reactions_from_route_cgr()` also loses the original depths. `route_order` participates in existing exact hashes/equality, so the identical synthesis stops deduplicating with its reconstructed or dictionary representation. This affects ordinary routes even without revision; it is not just a changed hash caused by adding protection steps.
- Concrete reproducer (from repository root):

  ```python
  import runpy
  from synplan.chem.reaction.routes.representation import (
      build_route_cgr, reactions_from_route_cgr,
  )
  from synplan.chem.reaction.routes.representation.hash import (
      route_cgr_hash, route_cgrs_equal,
  )
  route = runpy.run_path("tests/unit/route_quality/test_revision.py")["chain"](True)
  cgr = route.route_cgr()
  rebuilt = build_route_cgr({0: reactions_from_route_cgr(cgr)}, 0).cgr
  print(route_cgrs_equal(cgr, rebuilt))
  print(cgr.atom(4).route_order, rebuilt.atom(4).route_order)
  print(route_cgr_hash(cgr), route_cgr_hash(rebuilt))
  ```

  Observed: `False`, `{3} {2}`, and hashes `939755f3666613afc5824307bcd0434c01dd8b0bfd136eae30b8ab74e2cadf04` versus `ee6f4899b71d105e4a105c5d515a4a5f1504d8269e91e380754cf9503511a683`.
- The accepted five-step revised version also compares unequal to its reconstructed CGR: protection depth `{5}` and original-transformation depths `{4}`/`{3}` collapse to `{2}`. The full runnable probe is `/tmp/jack_roundtrip_probe.py`.
- Attribution: the overlapping-atom depth heuristic is pre-existing; the new regression is introducing a different depth convention for Route while retaining the existing dictionary convention and routing the existing `Route.route_cgr()` API through the new path. HEAD's Route method called the dictionary path directly.
- The existing new roundtrip tests miss this because `mapped_cgr_state()` deliberately excludes route metadata; the one additional assertion checks `route_atom_step_states`, which also does not detect the depth loss. This finding is about exact RouteCGR identity/API compatibility, not altered mapped SB-CGR chemistry.

### Suggestions

No cosmetic or broad rewrite suggestions. Prefer a bounded resolution of the confirmed representation-contract mismatch above.

## Tests Reviewed

Executed with `/data1/agilmullin/almaz/SynPlanner/.venv/bin/python`:

```text
python -m pytest tests/unit/chem/reaction/routes tests/unit/route_quality tests/regression/test_boceprevir_pg_revision.py -q
282 passed, 2 warnings in 83.32s
```

Coverage includes baseline mapping collisions, identical convergent branches, selected non-first products, two carried transformations, deprotection before a reactive step, multiple actions, reagent stock verdicts, metadata, input detachment, one tetrahedral stereo example, incomplete scans, explicit rejected candidates, and the five-to-seven-step Boceprevir revision.

Also ran `/tmp/jack_roundtrip_probe.py` to test full hash/equality and inspect atom `route_order` on both original and revised routes.

## Missing Tests

- Exact `route_cgrs_equal()` / `route_cgr_hash()` equality after RouteCGR → reaction dictionary → RouteCGR for a linear route of at least three steps, both before and after protection insertion.
- Explicit behavior contract for `Route.route_cgr()` versus `build_route_cgr({0: route.reactions_dict}, 0)` on an already globally mapped route.

## Open Questions

Is changing existing RouteCGR hash/depth semantics and making them depend on the input API intentionally authorized? The requested emphasis on preserving existing behavior suggests it should not happen implicitly. Chemistry preservation itself passed the focused tests; no suspected chemistry issue is promoted to a blocker here.
