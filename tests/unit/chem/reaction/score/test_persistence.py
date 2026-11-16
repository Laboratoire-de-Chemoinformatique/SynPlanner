"""Thermochemistry stays on steps through exports and alongside forward scores."""

from copy import deepcopy
from dataclasses import replace

from chython import inchi_key, smiles

from scripts.assess_routes import assess_collection
from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.routes.quality import (
    ForwardReactionAssessor,
    ForwardRouteScorer,
)
from synplan.chem.reaction.score import AIMNetReactionAssessor

from .test_aimnet import Calculator, route


class ForwardModel:
    def prediction_identity(self, **kwargs):
        return {"model": "test"}

    def predict(self, inputs, **kwargs):
        return [
            {
                "predictions": ["CC(=O)OCC"],
                "canonical_products": ["CC(=O)OCC"],
                "product_inchikeys": [
                    inchi_key(smiles("CC(=O)OCC"), ignore_stereo=True)
                ],
                "log_likelihoods": [0.0],
                "ranking_scores": [1.0],
                "augmentation_count": 1,
            }
            for _ in inputs
        ]


def test_both_assessment_orders_and_forward_reassessment(tmp_path):
    source = route()
    source = replace(source, assessment={"custom": {"keep": True}})
    thermo = AIMNetReactionAssessor(cache_path=None, calculator=Calculator())
    forward = ForwardReactionAssessor(ForwardModel(), tmp_path / "forward.json")
    scorer = ForwardRouteScorer()
    first = thermo.assess(scorer.assess(forward.assess([source])))[0]
    second = scorer.assess(forward.assess(thermo.assess([source])))[0]
    assert first.to_json() == second.to_json()
    reassessed = forward.assess([first])[0]
    assert "forward_score" not in reassessed.assessment
    assert reassessed.assessment["aimnet"] == first.assessment["aimnet"]
    assert reassessed.assessment["custom"] == {"keep": True}
    assert reassessed.steps[0].thermochemistry == first.steps[0].thermochemistry
    collection = {"target": [first.to_json()]}
    updated = assess_collection(collection, forward)
    assert (
        updated["target"][0]["assessment"]["aimnet"]
        == collection["target"][0]["assessment"]["aimnet"]
    )
    assert (
        updated["target"][0]["children"][0]["meta"]["thermochemistry"]
        == first.steps[0].thermochemistry
    )


def test_branched_step_annotations_roundtrip():
    ethanol, acid, ester = map(smiles, ("CCO", "CC(=O)O", "CC(=O)OCC"))
    left = Step(Reaction([smiles("CC=O"), smiles("[H][H]")], [ethanol]), ethanol)
    right = Step(
        Reaction([smiles("CC(=O)Cl"), smiles("O")], [acid, smiles("Cl")]), acid
    )
    root = Step(Reaction([ethanol, acid], [ester, smiles("O")]), ester)
    source = Route((root, right, left))
    result = AIMNetReactionAssessor(cache_path=None, calculator=Calculator()).assess(
        [source]
    )[0]
    saved = result.to_json()
    restored = Route.from_json(saved)
    actual = {
        s.thermochemistry["original_reaction"]: s.thermochemistry for s in restored
    }
    expected = {
        s.thermochemistry["original_reaction"]: s.thermochemistry for s in result
    }
    assert actual == expected
    assert restored.assessment == result.assessment
    assert all("thermochemistry" not in s.reaction.meta for s in restored)
    restored.steps[0].thermochemistry["status"] = "edited"
    assert all(s.thermochemistry["status"] != "edited" for s in result)


def test_mapped_export_keeps_thermochemistry():
    reaction = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5].[OH2:4]"
    )
    source = Route((Step(reaction, reaction.products[0]),))
    assessed = AIMNetReactionAssessor(cache_path=None, calculator=Calculator()).assess(
        [source]
    )[0]
    before = deepcopy(assessed.steps[0].thermochemistry)
    saved = assessed.to_json(reconcile_atom_mapping=True)
    assert Route.from_json(saved).steps[0].thermochemistry == before
    assert assessed.steps[0].thermochemistry == before


def test_legacy_json_has_no_thermochemistry():
    restored = Route.from_json(route().to_json())
    assert restored.steps[0].thermochemistry is None
    assert restored.assessment is None


def test_forward_invalidation_preserves_unrelated_flat_assessment():
    from synplan.chem.reaction.routes.quality.forward import without_forward_assessment

    original = {"aggregation": "custom", "total_steps": 7, "nested": {"value": 1}}
    retained = without_forward_assessment(original)
    assert retained == original
    retained["nested"]["value"] = 2
    assert original["nested"]["value"] == 1
