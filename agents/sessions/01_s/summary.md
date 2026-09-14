# Dirty-worktree review

Reviewed `route_pg_support` against HEAD/local `origin/main` at `53ffed21794e829cd660b63effaaf4351e10cf95`, including untracked feature files. The notebook on disk is `tutorials/08a_Protection_Handling.ipynb`. Jack, Margaret, Alex and Jerome completed first-pass reviews; Felix read all four and independently challenged their conclusions. Only review reports were written.

## Confirmed findings

1. **P2 — RouteCGR exact identity changes across composition adapters.** Native Route composition records dependency-derived depths, while dictionary composition derives different depths from atom overlap (`synplan/chem/reaction/routes/representation/route_cgr.py:822–823,869–870`, legacy path `:521–523`). On the three-step fixture, RouteCGR → deconvolution → dictionary recomposition changes atom 4's `route_order` from `{3}` to `{2}`. Exact equality and hashes consequently differ. The accepted revised route has the same problem. Mapped chemistry remains equal. Jack, root and Felix independently reproduced the discrepancy. This is the main code issue to resolve; align retained/derived depth semantics across the supported paths and cover exact identity in regression checks. See [Jack](review-jack.md), [Jerome](review-jerome.md) and [Felix](review-felix.md).

2. **P2 — The renamed notebook has stale references.** `tutorials/README.md:29` and `docs/methods/routes.rst:175` still reference the absent `route_pg_support.ipynb`. Update those references to `08a_Protection_Handling.ipynb`. See [Alex](review-alex.md).

3. **P2 — Tutorial resource setup is incomplete for a clean checkout.** The notebook defaults to a prepared `building_blocks/all-bb-2026-06/building_blocks.json` without explaining how to obtain or prepare it. The documented download presets do not populate that default path. Add a short resource acquisition/preparation pointer and explain the existing path override; retain the simple workflow and search limits. See [Alex](review-alex.md) and [Felix](review-felix.md).

## Other confirmed concerns

- **Score wording overstates the metric.** The notebook says a higher score means fewer problems. A reproduced accepted revision raises the score from 0.5 to 0.8333 while the worst-per-step penalty total remains 0.5: the denominator grows from one step to three. The targeted original conflict is reduced, so the stated acceptance contract holds, but fewer total conflicts do not follow. Correct the interpretation; changing the scoring policy is a separate decision. See [Margaret](review-margaret.md) and [Felix](review-felix.md).
- **Inherited SEM metadata mismatch.** Shared template label 9 can describe an oxygen-attached SEM group as N-SEM. The graph is O-SEM; the class label is incorrect. Existing candidate expansion already has this problem. It is not an introduced atom-mapping failure. See [Margaret](review-margaret.md).

No introduced atom-ID, mapped SB-CGR chemistry, dependency-identity or stereo-preservation blocker was reproduced in the exercised cases. This does not establish reaction-condition feasibility. The reviewers found no need for a broad architecture rewrite; repeated scans and two docstring clarifications are optional maintenance work.

## Verification

- Relevant route, quality, export/import, integrity, stereo and Boceprevir regression suite: **397 passed, 1 expected failure, 6 warnings**, 94.18 seconds. The expected failure concerns pre-existing stereo-corpus accounting.
- Ruff checks on the route modules, new tests and notebook: passed.
- Type checks on changed route, representation, scanner and revision modules: passed.
- Notebook schema and syntax: passed; 15 code cells and zero retained error outputs.
- `git diff --check`: passed.
- Focused review probes reproduced original/revised exact-identity mismatches and the score-normalization example. The saved Boceprevir repair passed; all seven distinct claimed PG choices passed its graph checks. Additional E/Z and isotope probes passed.
- No fresh MCTS search was run during this review. Recorded notebook outputs and the saved route regression were reviewed.

## Individual reports

- [Jack — correctness and API behavior](review-jack.md)
- [Margaret — chemistry](review-margaret.md)
- [Alex — tutorial usability](review-alex.md)
- [Jerome — architecture](review-jerome.md)
- [Felix — independent challenge pass](review-felix.md)
