"""Material observations and scoped chemistry assessments, separate from graphs.

These records carry supplied evidence. They do not predict selectivity, invent
mixture ratios or make a resolution step executable merely by naming it.
"""

import json
from copy import deepcopy
from hashlib import sha256


def reaction_context(reaction) -> str:
    """Versioned dependency key; substrate, products, agents and procedure matter."""
    payload = {
        "reactants": sorted(str(m) for m in reaction.reactants),
        "products": sorted(str(m) for m in reaction.products),
        "agents": sorted(str(m) for m in reaction.reagents),
        "conditions": reaction.meta.get("conditions"),
        "procedure": reaction.meta.get("procedure"),
        "materials": reaction.meta.get("materials"),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def attach_stereo_evidence(
    reaction,
    *,
    source: str,
    observation: dict,
    assessment: str = "unreviewed",
    reviewer: str | None = None,
    reason: str | None = None,
):
    """Attach source evidence or an explicit chemist assessment to this context.

    Observations may include ee/er/dr/E:Z, composition, measurement stage, yield
    basis, isolated fractions, a failed procedure or workup stability. Values
    stay supplied observations; missing values remain unknown. Accepting a
    structural strategy does not certify its experimental purity.
    """
    if not source.strip():
        raise ValueError("stereo evidence requires a source")
    if assessment not in {"unreviewed", "accepted", "rejected"}:
        raise ValueError("assessment must be unreviewed, accepted or rejected")
    if assessment != "unreviewed" and (not reviewer or not reason):
        raise ValueError("a scoped chemistry assessment requires reviewer and reason")
    record = {
        "schema": 1,
        "source": source,
        "observation": deepcopy(observation),
        "assessment": assessment,
        "reviewer": reviewer,
        "reason": reason,
        "context": reaction_context(reaction),
    }
    reaction.meta.setdefault("stereo_evidence", []).append(record)
    return deepcopy(record)


def assessed_evidence(reaction) -> list[dict]:
    """Keep observations when context changes, reopening their applicability."""
    context = reaction_context(reaction)
    return [
        {
            **deepcopy(record),
            "applicability": "current"
            if record.get("context") == context
            else "needs_reassessment",
        }
        for record in reaction.meta.get("stereo_evidence", ())
    ]


def chemistry_assessment(reaction) -> str:
    """A matching explicit review may discharge a strategy obligation."""
    decisions = {
        r.get("assessment")
        for r in assessed_evidence(reaction)
        if r["applicability"] == "current"
        and r.get("reviewer")
        and r.get("reason")
        and r.get("source")
    }
    if "accepted" in decisions and "rejected" in decisions:
        return "conflicting"
    if "rejected" in decisions:
        return "rejected"
    return "accepted" if "accepted" in decisions else "unreviewed"


def apply_chemistry_assessment(assessment, reaction):
    """Review supports only assessed transformations, never broken mapping/stock."""
    decision = chemistry_assessment(reaction)
    if decision == "accepted":
        dischargeable = {
            "requires_stereo_forming_step",
            "requires_explicit_resolution_or_inversion_strategy",
        }
        assessment["obligations"] = [
            o for o in assessment["obligations"] if o["reason"] not in dischargeable
        ]
        for event in assessment["events"]:
            if event.get("reason") in dischargeable:
                event.update(
                    event="reviewed_transformation", basis="scoped_chemist_assessment"
                )
    return decision


def review_stereo_route(
    route, step_index, *, source, observation, reviewer, reason, catalogue=None
):
    """Return a reviewed copy; callers explicitly choose the procedure and step.

    Original observations and the input route remain intact. Mapping uncertainty,
    contradictory configurations and missing stock cannot be approved away.
    """
    from dataclasses import replace

    if route.stereo is None or route.stereo_status == "needs_reassessment":
        raise ValueError("audit the current route before reviewing its stereo strategy")
    reviewed = deepcopy(route)
    original_target = reviewed.stereo["original_target"]
    step = reviewed.steps[step_index]
    attach_stereo_evidence(
        step.reaction,
        source=source,
        observation=observation,
        assessment="accepted",
        reviewer=reviewer,
        reason=reason,
    )
    remaining = []
    for obligation in reviewed.stereo.get("obligations", ()):
        belongs = obligation.get("step") == step_index or (
            step.origin and obligation.get("tree_node_id") == step.origin.tree_node_id
        )
        if not (
            belongs
            and obligation["reason"]
            in {
                "requires_stereo_forming_step",
                "requires_explicit_resolution_or_inversion_strategy",
            }
        ):
            remaining.append(obligation)
    if not remaining and reviewed.connectivity_solved:
        from frozendict import frozendict

        from synplan.chem.building_blocks import BuildingBlock
        from synplan.chem.reaction.routes.stereo import audit_stereo_inheritance

        if catalogue is None:
            buckets = {}
            for leaf in reviewed.leaves():
                record = leaf.meta.get("selected_stock")
                if record:
                    block = BuildingBlock(
                        record["smiles"],
                        record["inchikey"],
                        frozendict(record["vendors"]),
                        True,
                    )
                    buckets.setdefault(block.inchikey[:14], []).append(block)
            catalogue = frozendict({k: tuple(v) for k, v in buckets.items()})
        audit = audit_stereo_inheritance(
            reviewed,
            reviewed.target,
            catalogue,
            allow_unconstrained_target=True,
            mapping_sources={i: "recorded_route_mapping" for i in range(len(reviewed))},
        )
        if audit.supported:
            reviewed = audit.route
        else:
            remaining.extend(audit.issues)
    from synplan.chem.stereo import StereoObligations

    status = (
        (
            "could_not_be_assessed"
            if StereoObligations().extend(remaining).unassessed
            else "strategy_needed"
        )
        if remaining
        else ("fulfilled" if reviewed.connectivity_solved else "pending")
    )
    summary = route_stereo_summary(
        reviewed, original_target=original_target, status=status, obligations=remaining
    )
    summary["basis"] = "scoped_chemist_assessment_and_inherited_constraints"
    summary["selectivity_evidence_status"] = "chemist_reviewed_for_recorded_context"
    return replace(reviewed, stereo=summary)


def route_context(route) -> str:
    """Invalidate route-level fulfillment after any structural/procedure edit."""
    from synplan.chem.utils import mapped_smiles

    def digest(record):
        record = json.loads(json.dumps(record))
        return sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def multiset(records):
        total, count = 0, 0
        for record in records:
            total = (total + int(digest(record), 16)) % (1 << 256)
            count += 1
        return count, hex(total)

    # Independent branches can change traversal order on JSON import. Include
    # their actual producer/consumer links instead of relying on list positions.
    steps = [
        digest(
            [
                reaction_context(s.reaction),
                mapped_smiles(s.reaction),
                str(s.product),
                s.reaction.meta.get("stereo_evidence", []),
            ]
        )
        for s in route.steps
    ]
    producers = {id(s.product): key for s, key in zip(route.steps, steps)}
    edges = (
        [key, format(mol, "m"), producers.get(id(mol))]
        for s, key in zip(route.steps, steps)
        for mol in s.reaction.reactants
    )
    payload = [
        str(route.target),
        producers.get(id(route.target)),
        multiset(steps),
        multiset(edges),
    ]
    # Leaves can be emitted in a different order by the SMILES/JSON writer.
    # An additive digest retains multiplicity in linear work without sorting
    # every material record. Normalize JSON keys before hashing map metadata.
    material_sum, material_count = 0, 0
    for leaf in route.leaves():
        material = [str(leaf), leaf.meta.get("selected_stock")]
        material = json.loads(json.dumps(material))
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
        material_sum = (
            material_sum + int.from_bytes(sha256(encoded.encode()).digest())
        ) % (1 << 256)
        material_count += 1
    payload.append((material_count, hex(material_sum)))
    payload = json.loads(json.dumps(payload))
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def route_stereo_summary(
    route,
    *,
    original_target=None,
    status="could_not_be_assessed",
    obligations=(),
    audit=None,
):
    return {
        "schema": 1,
        "original_target": str(original_target or route.target),
        "stereo_status": status,
        "selectivity_evidence_status": "not_evaluated",
        "obligations": deepcopy(list(obligations)),
        "first_responsible": deepcopy(next(iter(obligations), None)),
        "context": route_context(route),
        "audit": deepcopy(audit),
        "basis": "structural_inheritance_from_explicit_stock"
        if status == "fulfilled"
        else None,
    }
