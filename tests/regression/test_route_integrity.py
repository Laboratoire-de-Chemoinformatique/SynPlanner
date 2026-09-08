"""Route records must preserve chemistry, review scope and bounded work."""

from collections import Counter
from dataclasses import replace
from itertools import permutations

import pytest
from chython import smiles
from chython.containers import ReactionContainer

from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.routes.representation.hash import route_cgr_hash
from synplan.chem.reaction.routes.route import StepOrigin
from synplan.chem.stereo_evidence import review_stereo_route, route_stereo_summary


def test_route_children_match_identity_and_multiplicity():
    reaction = smiles("CCO.CCO.CC(=O)O>>CCOC(C)=O")
    expected = Counter(str(m) for m in reaction.reactants)
    children = [
        dict(type="mol", smiles=str(m), in_stock=True) for m in reaction.reactants
    ]
    node = dict(
        type="mol",
        smiles=str(reaction.products[0]),
        children=[
            dict(type="reaction", smiles=format(reaction, "m"), children=children)
        ],
    )
    for order in permutations(children):
        node["children"][0]["children"] = order
        route = Route.from_json(node)
        assert route.solved
        assert Counter(str(m) for m in route.steps[0].reaction.reactants) == expected
    node["children"][0]["children"] = children[1:]
    route = Route.from_json(node)
    assert not route.solved
    assert Counter(str(m) for m in route.steps[0].reaction.reactants) == expected
    extra = dict(
        type="mol",
        smiles="CCCCO",
        children=[dict(type="reaction", smiles="CCCCBr.O>>CCCCO", children=[])],
    )
    node["children"][0]["children"] = [*children, extra]
    with pytest.raises(ValueError, match="molecule identity"):
        Route.from_json(node)
    node["children"][0]["children"] = children
    node["smiles"] = "CCCCO"
    with pytest.raises(ValueError, match="route product"):
        Route.from_json(node)


def test_review_without_node_ids_only_discharges_its_own_step():
    a, b, c = map(smiles, ("CC=O", "CCO", "CCOC"))
    route = Route(
        (
            Step(ReactionContainer([a], [b]), b, StepOrigin()),
            Step(ReactionContainer([b], [c]), c, StepOrigin()),
        ),
        unresolved={str(a)},
    )
    obligations = [
        dict(step=i, tree_node_id=None, reason="requires_stereo_forming_step")
        for i in range(2)
    ]
    route = replace(
        route,
        stereo=route_stereo_summary(
            route,
            original_target=str(c),
            status="strategy_needed",
            obligations=obligations,
        ),
    )
    reviewed = review_stereo_route(
        route,
        0,
        source="procedure",
        observation={},
        reviewer="chemist",
        reason="step 0 only",
    )
    assert reviewed.stereo["obligations"] == [obligations[1]]
    assert route.stereo["obligations"] == obligations


@pytest.mark.parametrize(
    "text", ["N(c1ccccc1)(c1ccccc1)c1ccccc1", "C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1"]
)
def test_symmetric_route_hash_refuses_unbounded_exact_work(text, monkeypatch):
    import synplan.chem.reaction.routes.representation.hash as hashing

    mol = smiles(text)
    cgr = ~ReactionContainer([mol], [mol.copy()])

    def unexpected_enumeration(*args):
        pytest.fail("must bound the factorial work before allocating permutations")

    monkeypatch.setattr(hashing, "permutations", unexpected_enumeration)
    with pytest.raises(ValueError, match="5000 atom orders"):
        route_cgr_hash(cgr)
