"""Forward metadata survives JSON without changing shared route chemistry."""

from copy import deepcopy
from dataclasses import replace

from chython import inchi_key, smiles

from synplan.chem.reaction.routes.route import Route, RouteProvenance, Step


def annotated_route():
    reaction = smiles("CC.N>>CCO")
    reaction.meta["existing"] = {"confidence": 0.2}
    prediction = {
        "product_inchikey": inchi_key(reaction.products[0], ignore_stereo=True),
        "rank": 4,
        "score": 0.25,
        "alert": False,
    }
    return Route(
        (Step(reaction, reaction.products[0], forward_prediction=prediction),),
        provenance=RouteProvenance(search_score=0.8, policy_log_likelihood=-2.5),
        assessment={"forward_score": 0.25},
    )


def test_annotations_provenance_and_legacy_json():
    source = annotated_route()
    tree = source.to_json()
    restored = Route.from_json(tree)
    exported = restored.to_json()
    assert exported["assessment"] == tree["assessment"]
    assert exported["provenance"] == tree["provenance"]
    assert restored.provenance == source.provenance
    assert restored.assessment == source.assessment
    assert restored.steps[0].forward_prediction == source.steps[0].forward_prediction
    restored.assessment["forward_score"] = 0.7
    assert tree["assessment"]["forward_score"] == 0.25
    legacy = deepcopy(tree)
    legacy.pop("assessment")
    legacy.pop("provenance")
    legacy["children"][0]["meta"].pop("forward_prediction")
    restored = Route.from_json(legacy)
    assert restored.assessment is None
    assert restored.provenance.search_score is None
    assert restored.steps[0].forward_prediction is None


def test_annotation_is_separate_from_chemistry_and_source_metadata():
    source = annotated_route()
    tree = source.to_json()
    observed = []
    restored = Route.from_json(
        tree, on_step=lambda reaction, node: observed.append((reaction, node))
    )
    assert observed == [(restored.steps[0].reaction, tree["children"][0])]
    assert observed[0][1] is tree["children"][0]
    assert (
        restored.steps[0].reaction.meta["existing"]
        == source.steps[0].reaction.meta["existing"]
    )
    assert source.steps[0].reaction.meta == {"existing": {"confidence": 0.2}}
    assert "forward_prediction" not in restored.steps[0].reaction.meta
    assert "reason" not in restored.steps[0].forward_prediction
    cleared = replace(
        source, steps=(replace(source.steps[0], forward_prediction=None),)
    )
    assert "forward_prediction" not in cleared.to_json()["children"][0]["meta"]
    assert tree["children"][0]["meta"]["forward_prediction"]["rank"] == 4
