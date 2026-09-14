# Review - Margaret

## Scope

Reviewed the current dirty worktree against HEAD/origin/main `53ffed21794e829cd660b63effaaf4351e10cf95`, including untracked `quality/protection/revision.py`, both Boceprevir data fixtures, revision tests and `tutorials/08a_Protection_Handling.ipynb`. Focus: chemical graphs, mapped atoms, attachment/removal, persistence, stereochemistry and the distinction between graph validation and synthesis feasibility. No application code, tests, fixtures or notebook were edited.

## Findings

### Blocking

**No introduced chemical-correctness blocker reproduced within the requested single-attachment graph-template scope.** The focused revision suite passes, the recorded Boceprevir example reproduces, and all seven distinct protecting groups claimed in its notebook pass the implemented graph checks. This is not evidence of experimental feasibility.

### Important

**Non-blocking metadata defect inherited from existing candidate expansion: O-SEM can be reported as N-SEM.** `synplan/chem/reaction/routes/quality/protection/revision.py:422` and `:529` copy the selected row's class names into reaction metadata and actions without checking that they match the attachment atom. `strategy.py:210` already expanded every row for each allowed numeric label before this change; label 9 contains both N-SEM and O-SEM rows, N first (`data/protection_group_templates.csv:7`, `:10`). New `ranked_candidates` at `strategy.py:216` reuses that behavior.

Reproduction using only the public API: use the mapped esterification reaction below, and supply a `ProtectingGroupClassifier` whose `rank` returns `[g for g in candidates if g.label == 9]`. With one revision, the result is accepted, attaches through **oxygen 8**, and reports `protection_class='5.1.8 N-SEM protection'`, `deprotection_class='6.1.18 N-SEM deprotection'`. The added O–C SEM graph is correct; the nitrogen-specific scientific label is false and survives serialization. On Boceprevir oxygen 33, both table rows likewise pass and make the same O-SEM chemistry. This is a pre-existing label-selection defect exposed through the new result metadata, not an introduced attachment or atom-identity failure.

```text
[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]
```

### Domain Risks

1. **An increased score does not establish fewer total selectivity conflicts.** Notebook line 377 says “a higher score means fewer problems.” The unchanged score formula at `quality/protection/scorer.py:89` divides worst-per-step penalties by processed reaction count; the new acceptance check at `revision.py:382` compares that score after inserting two steps. On the esterification above, default revision is accepted with O-Piv and score **0.5 → 0.8333333333**. Before revision there is one competing alcohol at oxygen 8 in the esterification. Afterwards there is one competing alcohol at oxygen 5 in the inserted protection. Total recorded penalty remains **0.5**; the denominator changes from one to three. The targeted original interaction is removed, so this meets the literal implemented acceptance contract. It does not demonstrate an overall reduction in unresolved chemical selectivity. The result exposes the remaining interaction. This is a scientific interpretation/metric limitation, not a reproduced violation of the specified score check.

2. **Graph persistence is not protecting-group survival or selective deprotection under conditions.** `revision.py:490` carries the group until a later center intersects the detected functional-group atom set, and `:497` reconstructs the original endpoint as a deprotection product. Neither checks reagents/conditions, relative lability of other groups, epimerization, yield or competing reactions. In the recorded example, an O-Piv ester is drawn across two transformations and removed before alcohol oxidation; graph consistency alone does not establish that a selective laboratory procedure exists. The documentation at `docs/methods/routes.rst:171` explicitly limits this to graph proposals, which is the appropriate scope. The notebook provides no independent feasibility assessment.

3. **These reactions are not full stoichiometric equations.** The new protection construction (`revision.py:432`) includes the example introducing reagent but no leaving-group product; deprotection (`revision.py:497`) contains only the protected substrate and original endpoint. For recorded Boceprevir O-Piv, all 13 heavy atoms of pivalic anhydride are allocated IDs 58–70, six atoms form the carried pivaloyl group, and the other reagent atoms disappear from the protection products. The released group is absent from the deprotection products. This accords with the pre-existing principal-product template convention in `strategy.py:protection_pair`; mapped SB-CGR equality should not be interpreted as atom-balanced reaction chemistry.

4. **The O-acetonide table defect is pre-existing and does not currently escape the new acceptance boundary.** `data/protection_group_templates.csv:17` uses the single-attachment fragment `C[C:512]C` under an acetonide name; that fragment does not encode two oxygen attachments. On the recorded alcohol the new code returns `protection_reagent_mapping_failed` at `revision.py:408`, so it is diagnostic-only with the bundled reagent. Duplicate-map carbonyl/Phth labels 11, 12, 13, 14 and 18 are rejected by the existing loader. These are not newly introduced accepted carbonyl approximations.

### Suggestions

None requested; review only. The observations above distinguish non-blocking limitations from acceptance-contract defects and do not prescribe code changes.

## Chemistry Cases Reviewed

- Ran `/data1/agilmullin/almaz/SynPlanner/.venv/bin/python -m pytest -q tests/unit/route_quality/test_revision.py tests/regression/test_boceprevir_pg_revision.py`: **20 passed**, 2 pre-existing deprecation warnings, 53.74 seconds.
- Coverage exercised by those tests includes a mapped esterification with oxygen-4 water byproduct; PG carry through two original transformations; removal before oxidation; identical convergent branches; repeated revisions and disjoint added IDs; selected product not first; unsupported ketone protection; global atom-ID collisions involving non-target oxygen/chlorine; metadata and JSON/RouteCGR round trips; and preservation of an existing tetrahedral stereocenter.
- Independently replayed recorded Boceprevir route 7614: five original steps become seven, score 0.7 → 13/14, anchor oxygen 33, O-Piv protection at final step 1, protected transformations 2 and 3, deprotection 4 (all zero-based). The original-to-final map is `{0: 0, 1: 2, 2: 3, 3: 5, 4: 6}`. Existing mapped atoms and mapped SB-CGR chemistry remain unchanged; PG/reagent IDs begin at 58, above original maximum 57.
- Independently enumerated every allowed row for that site's recorded interaction and called `_build_candidate`, `_validate` and `_candidate_score`: Piv, TES, MOM, N-SEM-labelled/O-SEM graph, O-SEM, TBS, TIPS and TMS all returned validation `None`, score `0.9285714285714286`, and carry positions `(2, 3)`. These are **seven distinct groups**; the duplicated SEM rows explain eight passing table rows. O-acetonide returned `protection_reagent_mapping_failed`. Thus notebook lines 2617–2619 are reproducible for the recorded fixture, though the notebook itself does not rerun that candidate-by-candidate audit.
- Reviewed recorded run metadata: 300 is the iteration **limit**, not the realized count. The fixture reports 287 iterations and 120.4 seconds under a 120-second configured limit. No fresh search/model execution or experimental validation was claimed by this review.

## Missing Chemistry Tests

These are coverage limits, not reproduced blockers:

- Full acceptance tests for allowed amine, carboxylic acid, phenol and heteroaromatic N attachment families; the new main regression evidence is alcohol protection.
- Assert class-name/site agreement for numeric labels with N/O variants.
- Absolute conflict counts alongside a length-normalized score, including the demonstrated case where the conflict is moved into protection.
- Charged/salt, isotope, explicit-hydrogen, E/Z and stereogenic-site cases. `mapped_cgr_state` at `revision.py:111` compares element/isotope/charge/radical and bond orders but does not itself compare stereochemistry. The dedicated existing-stereo test passes, and I did not reproduce a stereo-loss bug.
- Conditions and balanced byproducts are outside the current model and are not established by these tests.

## Open Questions

None blocks the requested graph-template review. Whether acceptance should mean a lower total chemical-risk burden, rather than satisfying the explicitly stated targeted-interaction and normalized-score checks, is a separate product/scientific policy decision.
