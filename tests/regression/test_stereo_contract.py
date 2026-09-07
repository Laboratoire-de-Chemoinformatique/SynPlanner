"""Stereo contract regressions: source reactions and explicitly labeled controls."""

import json
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pytest
from chython import smiles
from chython.containers import ReactionContainer
from chython.files.SDFrw import ESDFWrite, SDFRead
from frozendict import frozendict

from synplan.chem.building_blocks import BuildingBlock, molecule_to_inchikey
from synplan.chem.building_blocks.stereo import compatible_records
from synplan.chem.mapping import MappingBudgetExceeded, bounded_mappings, mapping_budget
from synplan.chem.precursor import Precursor
from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.reaction.routes.route import Route
from synplan.chem.reaction.rules.config import RuleExtractionConfig
from synplan.chem.reaction.rules.extraction import (
    _make_extracted_rule_record,
    extract_rules,
    molecule_substructure_as_query,
)
from synplan.chem.stereo import _requirements, _sign, stereo_events
from synplan.chem.stereo_evidence import assessed_evidence, attach_stereo_evidence
from synplan.chem.utils import in_atom_order, mol_from_smiles, safe_canonicalization
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import EvaluationStrategy, RolloutSimulator
from synplan.mcts.tree import Tree
from synplan.utils.files import split_smiles_record

# Lactic acid / crotonic acid structures and an allene API control. These are
# representation controls, not claims of experimental allene synthesis.
STEREO = ["C[C@H](O)C(=O)O", "C/C=C/C(=O)O", "CC=[C@]=CC"]


def catalogue(*entries):
    buckets = defaultdict(list)
    for text, price in entries:
        mol = safe_canonicalization(smiles(text))
        key = molecule_to_inchikey(mol)
        buckets[key[:14]].append(
            BuildingBlock(
                str(mol), key, frozendict({"supplier": price}), bool(_requirements(mol))
            )
        )
    return frozendict({key: tuple(value) for key, value in buckets.items()})


@pytest.mark.parametrize("text", STEREO)
def test_default_preservation_and_query_positive_negative(text):
    mol = mol_from_smiles(text)
    (req,) = _requirements(mol)
    prepared = Precursor(mol).molecule
    assert _sign(prepared, req) == req.sign
    ordered = in_atom_order(mol)
    assert _sign(ordered, req) == req.sign
    query = molecule_substructure_as_query(mol, mol)
    from synplan.chem.mapping import bounded_query

    query = bounded_query(query)
    assert list(query.get_mapping(mol))
    opposite = mol.copy()
    if req.kind == "double_bond":
        opposite.bond(*req.atoms)._stereo = not opposite.bond(*req.atoms).stereo
    else:
        opposite.atom(req.atoms[0])._stereo = not opposite.atom(req.atoms[0]).stereo
    opposite.flush_cache()
    assert not list(query.get_mapping(opposite))
    unspecified = mol.copy()
    unspecified.clean_stereo()
    assert not list(query.get_mapping(unspecified))
    stock = catalogue((str(opposite), 1), (str(mol), 9), (str(unspecified), 0.1))
    matches = compatible_records(mol, stock)
    assert len(matches) == 1
    assert matches[0].vendors["supplier"] == 9
    assert not Precursor(mol).is_building_block(catalogue((str(opposite), 1)), 100)


@pytest.mark.parametrize("group", ["a:1,3", "o1:1,3", "&1:1,3"])
def test_enhanced_group_cxsmiles_v3000_and_stock_semantics(group):
    text = f"C[C@H](O)[C@H](N)C(=O)O |{group}|"
    assert split_smiles_record(text + "\tsource")[0] == text
    assert split_smiles_record(text + " source")[0] == text
    molecule = mol_from_smiles(text)
    output = StringIO()
    writer = ESDFWrite(output)
    writer.write(molecule)
    restored = next(iter(SDFRead(StringIO(output.getvalue()), ignore=False)))
    assert str(restored) == str(molecule)
    stock = catalogue((text, 1))
    assert bool(compatible_records(molecule, stock)) == group.startswith("a:")


def test_unsupported_atropisomer_annotation_is_not_silently_erased():
    with pytest.raises(ValueError, match="unsupported"):
        mol_from_smiles("Cc1ccccc1-c1ccccc1C |wU:1.6|")


def test_budget_interrupts_rejected_matches_not_only_yielded_products():
    from chython import smarts

    query = smarts("[C:1]-[C:2]-[C:3]-[C:4]-[N:5]")
    with mapping_budget(5), pytest.raises(MappingBudgetExceeded):
        list(bounded_mappings(query, smiles("CCCCCCCCCCCC")))


class FixedEvaluator(EvaluationStrategy):
    def _evaluate_node(self, node, node_id, nodes):
        return 1.0


class RulesPolicy:
    def predict_reaction_rules(self, precursor, reaction_rules):
        yield from (
            (1 / len(reaction_rules), rule, i) for i, rule in enumerate(reaction_rules)
        )


def tree_for(target, rule, stock, **config):
    return Tree(
        target=mol_from_smiles(target),
        reaction_rules=(rule,),
        building_blocks=stock,
        config=TreeConfig(
            max_iterations=20,
            max_time=10,
            max_depth=3,
            min_mol_size=0,
            silent=True,
            **config,
        ),
        expansion_function=RulesPolicy(),
        evaluation_function=FixedEvaluator(),
    )


@pytest.mark.parametrize(
    "algorithm", ["uct", "breadth_first", "best_first", "beam", "nmcs", "lazy_nmcs"]
)
def test_created_lactic_acid_center_is_proposal_not_solved_or_success_reward(algorithm):
    # Representation control: pyruvate reduction creates a lactic acid center.
    # No catalyst/selectivity evidence is supplied.
    rule = CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]=[O:2]")
    tree = tree_for(
        STEREO[0], rule, catalogue(("CC(=O)C(=O)O", 1)), algorithm=algorithm
    )
    tree.run()
    assert not tree.winning_nodes
    assert tree.proposal_nodes
    node = tree.nodes[tree.proposal_nodes[0]]
    assert node.is_terminal() and not node.is_solved()
    assert tree._get_node_value(tree.proposal_nodes[0]) == 0
    route = Route.from_tree(tree, tree.proposal_nodes[0])
    assert route.connectivity_solved and not route.solved
    assert route.stereo_status == "strategy_needed"
    assert route.to_json()["stereo"]["first_responsible"]


def test_strict_mode_excludes_new_center_and_rollout_cannot_reward_it():
    rule = CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]=[O:2]")
    stock = catalogue(("CC(=O)C(=O)O", 1))
    tree = tree_for(STEREO[0], rule, stock, stereo_mode="strict")
    tree.run()
    assert len(tree.nodes) == 1
    rollout = RolloutSimulator(RulesPolicy(), (rule,), stock, 0, 3)
    assert rollout.simulate_precursor(Precursor(smiles(STEREO[0]))) < 1


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("text", STEREO)
def test_reactor_direct_and_cgr_preserve_remote_stereo(text, rebuild):
    # Append a mapped terminal hydroxy group away from the stereo environment.
    mol = smiles(text)
    next_atom = max(mol) + 1
    mol.add_atom("C", next_atom)
    mol.add_atom("O", next_atom + 1)
    mol.add_bond(next_atom, next_atom + 1, 1)
    original = _requirements(mol)
    rule = CanonicalRetroReactor.from_smarts("[C;h3:1]-[O;h1:2]>>[C:1]=[O:2]")
    groups = list(apply_reaction_rule(mol, rule, rebuild_with_cgr=rebuild))
    assert groups
    for products in groups:
        for req in original:
            owner = next(p for p in products if p.has_atom(req.atoms[0]))
            assert _sign(owner, req) == req.sign


def test_real_patent_step_search_inheritance_export_price_and_invalidation():
    from synplan.utils.visualisation import routes_report_html

    cases = json.loads(
        (Path(__file__).parents[1] / "data/stereo/routes.json").read_text()
    )
    source = cases["source:n5-00611"]["tree"]["children"][0]
    reaction = smiles(source["smiles"])
    (target,) = reaction.products
    rule = CanonicalRetroReactor(
        (molecule_substructure_as_query(target, target),),
        reaction.reactants,
        delete_atoms=False,
    )
    stock = catalogue(*((str(m), 3) for m in reaction.reactants))
    tree = tree_for(str(target), rule, stock, algorithm="breadth_first")
    tree.run()
    assert tree.winning_nodes, tree.stereo_diagnostics
    (route,) = tree.routes()
    assert route.stereo_status == "fulfilled"
    restored = Route.from_json(route.to_json())
    assert restored.solved and restored.stereo_status == "fulfilled"
    assert molecule_to_inchikey(restored.target) == molecule_to_inchikey(target)
    assert all(m.meta.get("selected_stock") for m in restored.leaves())
    page = routes_report_html([restored], None)
    assert "Stereo requirements fulfilled" not in page
    assert "Selectivity evidence:" not in page
    assert "Stereo:" not in page
    assert "Selected building block:" not in page
    route.steps[0].reaction.meta["procedure"] = "changed workup"
    assert route.stereo_status == "needs_reassessment" and not route.solved
    assert "Stereo needs reassessment after edits" in routes_report_html([route], None)


def test_source_annotation_gain_differs_from_creation_and_stereo_only_rules_survive():
    text = "[CH3:1][CH:2]([OH:3])[C:4](=[O:5])[OH:6]>>[CH3:1][C@H:2]([OH:3])[C:4](=[O:5])[OH:6]"
    reaction = smiles(text)
    assert stereo_events(reaction)[0]["event"] == "annotation_added"
    rules, _ = extract_rules(RuleExtractionConfig(reactor_validation=False), reaction)
    assert rules
    positive = _make_extracted_rule_record(rules[0])
    opposite = smiles(text.replace("[C@H:2]", "[C@@H:2]"))
    other, _ = extract_rules(RuleExtractionConfig(reactor_validation=False), opposite)
    assert positive.cgr_key != _make_extracted_rule_record(other[0]).cgr_key


def test_observations_survive_but_applicability_expires_when_context_changes():
    reaction = smiles("CC(=O)C(=O)O>>C[C@H](O)C(=O)O")
    attach_stereo_evidence(
        reaction,
        source="chemist-supplied procedure",
        observation={
            "er": "98:2",
            "stage": "isolated",
            "yield_basis": "desired isomer",
        },
        assessment="accepted",
        reviewer="chemist",
        reason="Reviewed this exact substrate and procedure",
    )
    assert assessed_evidence(reaction)[0]["applicability"] == "current"
    reaction.meta["procedure"] = "different catalyst"
    (evidence,) = assessed_evidence(reaction)
    assert evidence["applicability"] == "needs_reassessment"
    assert evidence["observation"]["er"] == "98:2"


@pytest.mark.parametrize("text", ["C[C@H](C)O", "C/C"])
def test_discarded_stereo_is_an_explicit_parse_failure(text):
    with pytest.raises(ValueError):
        mol_from_smiles(text)


@pytest.mark.parametrize("text", ["OCC[C@H](F)Cl", "OCC/C=C/C", "OCC=[C@]=CC"])
def test_complete_family_route_stock_record_and_cgr_roundtrip(text, tmp_path):
    import pickle
    from html import escape

    from synplan.chem.reaction.routes.representation.deconvolution import (
        reactions_from_route_cgr,
    )
    from synplan.chem.reaction.routes.representation.route_cgr import build_route_cgr
    from synplan.mcts.record import read_search_record, write_search_record
    from synplan.utils.visualisation import routes_report_html

    rule = CanonicalRetroReactor.from_smarts("[C;h2:1]-[O;h1:2]>>[C:1]-[Cl:3].[O:2]")
    stock = catalogue((text.replace("O", "Cl", 1), 4), ("O", 0.1))
    tree = tree_for(text, rule, stock, algorithm="breadth_first")
    tree.run()
    assert tree.winning_nodes, tree.stereo_diagnostics
    (route,) = tree.routes()
    assert route.solved
    record = read_search_record(write_search_record(tree, tmp_path / "search.json"))
    (restored,) = record.routes()
    assert restored.solved and restored.to_json()["stereo_status"] == "fulfilled"
    assert str(restored.target) == str(route.target)
    before = [format(step.reaction, "m") for step in restored]
    page = routes_report_html([restored], None)
    assert "Stereo:" not in page
    assert "Selected building block:" not in page
    assert escape(str(restored.steps[0].reaction)) in page
    if "@" in text:  # Chython draws both tetrahedral and allene stereo as wedges.
        assert ' Z"' in page or 'data-stereo="hashed"' in page
    assert [format(step.reaction, "m") for step in restored] == before
    cgr = build_route_cgr({0: route.reactions_dict}, 0, include_reactions=True).cgr
    for variant in (cgr, cgr.copy(), pickle.loads(pickle.dumps(cgr))):
        steps = reactions_from_route_cgr(variant)
        assert str(steps[0].products[0]) == str(route.target)
    remapped = cgr.remap({n: n + 100 for n in cgr}, copy=True)
    assert str(reactions_from_route_cgr(remapped)[0].products[0]) == str(route.target)
    # Per-step labels changed without a corresponding source stereo update.
    _, atom = next(cgr.atoms())
    before = atom.route_atom_step_states[1]
    atom.route_atom_step_states[1] = (before[0] + 1, *before[1:])
    with pytest.raises(ValueError, match="reassessment"):
        reactions_from_route_cgr(cgr)


def test_proposal_review_and_search_record_keep_outstanding_strategy(tmp_path):
    from synplan.chem.stereo_evidence import review_stereo_route
    from synplan.mcts.record import read_search_record, write_search_record
    from synplan.utils.visualisation import routes_report_html

    rule = CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]=[O:2]")
    tree = tree_for(
        STEREO[0], rule, catalogue(("CC(=O)C(=O)O", 1)), algorithm="breadth_first"
    )
    tree.run()
    record = read_search_record(write_search_record(tree, tmp_path / "proposal.json"))
    assert not record.winning_nodes and record.proposal_nodes
    route = Route.from_tree(record, record.proposal_nodes[0])
    assert route.stereo_status == "strategy_needed"
    page = routes_report_html([route], None)
    assert "Selectivity evidence:" not in page
    assert (
        "Required stereochemistry must be established at this step"
        in page.split('<div class="step">')[1]
    )
    assert "Required configuration is inherited" not in page
    reviewed = review_stereo_route(
        route,
        0,
        source="API review control",
        observation={"er": "not measured"},
        reviewer="chemist",
        reason="Supplied procedure for this exact substrate",
    )
    assert reviewed.solved, reviewed.stereo
    assert not route.solved
    repeat = tree_for(
        STEREO[0],
        rule,
        catalogue(("CC(=O)C(=O)O", 1)),
        algorithm="breadth_first",
        stereo_mode="strict",
        stereo_assessments=reviewed.steps[0].reaction.meta["stereo_evidence"],
    )
    repeat.run()
    assert repeat.winning_nodes
    reviewed.steps[0].reaction.meta["stereo_evidence"][0]["assessment"] = "rejected"
    assert not reviewed.solved
    reviewed.steps[0].reaction.meta["procedure"] = "different procedure"
    assert not reviewed.solved


def test_unmapped_source_cannot_assert_creation_or_inversion():
    from synplan.utils.files import parse_reaction

    reaction = parse_reaction("CC(=O)C(=O)O>>C[C@H](O)C(=O)O", "smi")
    assert {e["event"] for e in stereo_events(reaction)} == {"mapping_unresolved"}


@pytest.mark.parametrize("text", STEREO)
def test_stereo_rule_identity_survives_atom_renumbering(text):
    molecule = smiles(text)
    query = molecule_substructure_as_query(molecule, molecule)
    rule = ReactionContainer((query,), (query.copy(),))
    original = _make_extracted_rule_record(rule).cgr_key
    for mol in rule.molecules():
        mol.remap({n: 100 - n for n in mol})
    rule.flush_cache()
    assert _make_extracted_rule_record(rule).cgr_key == original


def test_completed_outcomes_survive_later_mapping_exhaustion():
    class InterruptedRule:
        def __call__(self, molecule):
            yield ReactionContainer((molecule,), (smiles("CC=O"),))
            raise MappingBudgetExceeded("control: later correspondence exhausted")

    diagnostics = []
    outcomes = list(
        apply_reaction_rule(smiles("CCO"), InterruptedRule(), diagnostics=diagnostics)
    )
    assert len(outcomes) == 1
    assert diagnostics[0]["reason"] == "mapping_budget_exceeded"


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("multicenter", [True, False])
def test_extraction_source_evidence_and_rule_manifest_roundtrip(
    workers, multicenter, tmp_path
):
    from synplan.chem.reaction.rules.extraction import extract_rules_from_reactions
    from synplan.utils.loading import load_reaction_rules

    records = json.loads(
        (Path(__file__).parents[1] / "data/regression/pr104_uspto.json").read_text()
    )
    source = records["stereo_nitrile_reduction"]["source_line"]
    input_file = tmp_path / "source.smi"
    input_file.write_text(source + "\n" + source + "\n")
    rules_file = tmp_path / "rules.tsv"
    extract_rules_from_reactions(
        RuleExtractionConfig(min_popularity=1, multicenter_rules=multicenter),
        str(input_file),
        str(rules_file),
        workers,
        1,
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "rules.stereo.jsonl").read_text().splitlines()
    ]
    assert {e["reaction_index"] for e in events} == {0, 1}
    assert all(e["source"]["source_0002"] == "US03950405" for e in events)
    assert all(e["events"] for e in events)
    manifest = json.loads((tmp_path / "rules.manifest.json").read_text())
    rules = load_reaction_rules(str(rules_file))
    assert rules and rules.vocabulary_digest == manifest["rules_sha256"]
    assert rules_file.read_text().splitlines()[1].split("\t")[1:] == ["2", "0,1"]


def test_new_vocabulary_cannot_relabel_old_policy_outputs():
    from types import SimpleNamespace

    from synplan.chem.reaction.rules.vocabulary import RuleLibrary
    from synplan.mcts.policy.template_based import LinearPolicy

    network = SimpleNamespace(architecture="linear", n_rules=1)
    network.eval = lambda: network
    wrapper = LinearPolicy(network)
    rules = RuleLibrary((object(),), "current vocabulary")
    with pytest.raises(ValueError, match="matching trained policy"):
        list(wrapper.predict_reaction_rules(Precursor(smiles("CCO")), rules))
    network.rule_vocabulary_digest = "current vocabulary"
    wrapper._predict_rules_common = lambda *args: None
    assert list(wrapper.predict_reaction_rules(Precursor(smiles("CCO")), rules)) == []


def test_rule_validation_budget_is_incomplete_not_contradicted(monkeypatch):
    from synplan.chem.reaction.rules import extraction

    molecule = smiles(STEREO[0])
    reaction = ReactionContainer((molecule,), (molecule.copy(),))
    query = molecule_substructure_as_query(molecule, molecule)
    rule = ReactionContainer((query,), (query.copy(),))
    monkeypatch.setattr(extraction, "_isomorphism_cost_estimate", lambda *args: 1e15)
    assert not extraction.validate_rule(rule, reaction)
    assert rule.meta["reactor_validation"] == "could_not_be_assessed"


def test_v3000_reaction_roundtrip_preserves_enhanced_groups():
    from chython.files.RDFrw import RDFRead

    from synplan.utils.stereo_io import RDFWrite

    molecule = smiles("C[C@H](O)[C@H](N)C(=O)O |&1:1,3|")
    reaction = ReactionContainer((molecule,), (molecule.copy(),))
    stream = StringIO()
    RDFWrite(stream).write(reaction)
    restored = next(iter(RDFRead(StringIO(stream.getvalue()), ignore=False)))
    assert str(restored) == str(reaction)
    assert [a.extended_stereo for _, a in restored.products[0].atoms()] == [
        a.extended_stereo for _, a in molecule.atoms()
    ]
    assert not compatible_records(restored.products[0], catalogue((str(molecule), 1)))


def test_multibranch_source_route_fulfillment_survives_json_order():
    from dataclasses import replace

    from synplan.chem.reaction.routes.stereo import audit_stereo_inheritance
    from synplan.chem.reaction.routes.stereo_io import read_stereo_route
    from synplan.chem.stereo_evidence import route_stereo_summary

    data = Path(__file__).parents[1] / "data/stereo"
    case = json.loads((data / "routes.json").read_text())["planner:n5-02115"]
    stock = catalogue(
        *((r["SMILES"], 1) for r in json.loads((data / "stock.json").read_text()))
    )
    route, sources = read_stereo_route(case["tree"], strip_stereo=True)
    audit = audit_stereo_inheritance(
        route, smiles(case["original_target"]), stock, mapping_sources=sources
    )
    assert audit.supported, audit.issues
    route = replace(
        audit.route, stereo=route_stereo_summary(audit.route, status="fulfilled")
    )
    restored = Route.from_json(route.to_json())
    assert restored.solved, restored.stereo_status
    restored.steps[0].reaction.meta["procedure"] = "changed"
    assert not restored.solved


@pytest.mark.parametrize("text", STEREO)
def test_json_rejects_conflicting_molecule_and_reaction_stereo(text):
    from synplan.chem.stereo import reaction_smiles

    molecule = smiles(text)
    reaction = ReactionContainer((molecule,), (molecule.copy(),))
    flat = molecule.copy()
    flat.clean_stereo()
    raw = {
        "type": "mol",
        "smiles": str(flat),
        "children": [
            {
                "type": "reaction",
                "smiles": reaction_smiles(reaction),
                "children": [
                    {"type": "mol", "smiles": str(molecule), "in_stock": True}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="inconsistent stereo"):
        Route.from_json(raw)


def test_enhanced_reaction_json_keeps_groups_unassessed():
    from synplan.chem.stereo import reaction_smiles

    text = "C[C@H](F)Cl |&1:1|"
    molecule = smiles(text)
    reaction = ReactionContainer((molecule,), (molecule.copy(),))
    raw = {
        "type": "mol",
        "smiles": str(molecule),
        "children": [
            {
                "type": "reaction",
                "smiles": reaction_smiles(reaction),
                "children": [
                    {"type": "mol", "smiles": str(molecule), "in_stock": True}
                ],
            }
        ],
    }
    route = Route.from_json(raw)
    assert not route.solved
    restored = Route.from_json(route.to_json())
    assert str(restored.target) == str(molecule)
    assert not restored.solved


@pytest.mark.parametrize("multicenter", [True, False])
def test_patent_rule_record_is_canonicalized_once(multicenter, monkeypatch):
    from hashlib import sha256
    from unittest.mock import Mock

    from synplan.chem.reaction.rules import extraction
    from synplan.utils.files import parse_reaction

    records = json.loads(
        (Path(__file__).parents[1] / "data/regression/pr104_uspto.json").read_text()
    )
    reaction = parse_reaction(records["stereo_nitrile_reduction"]["source_line"], "smi")
    canonicalize = Mock(wraps=extraction.canonical_query_cgr_key)
    monkeypatch.setattr(extraction, "canonical_query_cgr_key", canonicalize)
    (record,), skipped = extraction._extract_rules(
        RuleExtractionConfig(multicenter_rules=multicenter), reaction, as_records=True
    )
    assert not skipped and canonicalize.call_count == 1
    # Frozen from the pre-refactor US03950405 key and exact TSV SMARTS.
    assert sha256(
        (record.cgr_key + "\n" + record.rule_smarts).encode()
    ).hexdigest() == (
        "8fca1a7ab106bc7e2df1b3b9dadab178d77a7d7548e2582b5a20527c24b1c1b1"
    )


@pytest.mark.parametrize("text", [*STEREO, "C[C@H](O)[C@@H](O)C", "C[C@H](O)[C@H](O)C"])
@pytest.mark.parametrize("extra_stereo", [True, False])
def test_stock_signatures_match_materialized_alignments(
    text, extra_stereo, monkeypatch
):
    from unittest.mock import Mock

    from chython.containers import MoleculeContainer

    from synplan.chem.building_blocks.stereo import _record_molecule
    from synplan.chem.reaction.routes.stereo import (
        _mappings,
        _orientation_key,
        _stock_match,
    )
    from synplan.chem.stereo import _Unresolved

    stock = catalogue((text, 9))
    (record,) = next(iter(stock.values()))
    candidate = _record_molecule(record.smiles, record.inchikey)
    molecule = candidate.copy()
    molecule.remap({n: 100 - n for n in molecule})
    if extra_stereo:
        molecule.clean_stereo()
    before = format(molecule, "m"), format(candidate, "m"), dict(candidate.meta)
    required = _requirements(molecule)
    compatible = [
        m
        for m in _mappings(molecule, candidate, 256)
        if all(_sign(candidate, r.remap(m)) == r.sign for r in required)
    ]
    # Reference behavior before the refactor: copy and remap every alignment.
    versions = []
    for mapping in compatible:
        version = candidate.copy()
        version.remap({v: k for k, v in mapping.items()})
        versions.append(version)
    orientations = {_orientation_key(molecule, _requirements(v)) for v in versions}
    remap = Mock(side_effect=MoleculeContainer.remap)
    monkeypatch.setattr(
        MoleculeContainer, "remap", lambda self, *a, **kw: remap(self, *a, **kw)
    )
    if len(orientations) > 1:
        with pytest.raises(_Unresolved, match="stereo-distinct alignments"):
            _stock_match(molecule, required, stock, 256)
        assert remap.call_count == 0
    else:
        selected, detail = _stock_match(molecule, required, stock, 256)
        assert remap.call_count == 1
        assert format(selected, "m") == format(versions[0], "m")
        assert detail["compatible_mapping_count"] == len(compatible)
        assert detail["vendors"] == {"supplier": 9} and detail["price"] == 9
    assert before == (
        format(molecule, "m"),
        format(candidate, "m"),
        dict(candidate.meta),
    )


def test_stock_audit_shares_strict_preparation_and_keeps_pinned_record():
    from synplan.chem.building_blocks.stereo import _record_molecule
    from synplan.chem.reaction.routes.stereo import _stock_match
    from synplan.chem.stereo import _Unresolved

    # Kekule/aromatic encodings of the same 1-phenylethanol record.
    molecule = mol_from_smiles("C[C@H](O)c1ccccc1")
    key = molecule_to_inchikey(molecule)
    invalid = ["bad", "CC>>CC", "C[C@H](C)O", "C[C@@H](O)c1ccccc1"]
    valid = BuildingBlock("C[C@H](O)C1=CC=CC=C1", key, frozendict(pinned=7), True)
    bucket = (
        *(BuildingBlock(s, key, frozendict(cheap=1), True) for s in invalid),
        valid,
    )
    stock = frozendict({key[:14]: bucket})
    assert compatible_records(molecule, stock) == (valid,)
    selected, detail = _stock_match(molecule, _requirements(molecule), stock, 256)
    assert len(detail["rejected_candidates"]) == len(invalid)
    assert detail["smiles"] == valid.smiles and detail["price"] == 7
    assert str(selected) == str(_record_molecule(valid.smiles, key)) == str(molecule)
    molecule.meta["selected_stock"] = {"inchikey": key, "smiles": invalid[-1]}
    with pytest.raises(_Unresolved, match="no explicit compatible record"):
        _stock_match(molecule, _requirements(molecule), stock, 256)


def test_raw_native_mapping_keeps_compiled_query_and_shared_budget(monkeypatch):
    from chython import smarts

    from synplan.chem import mapping

    if not mapping._native_budget:
        pytest.skip("paired Chython native dispatch control")

    def no_wrap(*args, **kwargs):
        pytest.fail("native raw queries must retain their compiled caches")

    monkeypatch.setattr(mapping, "bounded_query", no_wrap)
    query, molecule = smarts("[$(CC)]"), smiles("CCC")
    first = list(bounded_mappings(query, molecule))
    assert len(first) == 3 and first == list(bounded_mappings(query, molecule))
    with mapping_budget(1), pytest.raises(MappingBudgetExceeded):
        list(bounded_mappings(query, molecule))


@pytest.mark.parametrize("spec", ["m", "!cm", "!xm", "!sm"])
def test_reaction_group_fallback_matches_native_formatting(spec, monkeypatch):
    from synplan.chem.stereo import reaction_smiles

    reaction = smiles("C[C@H](O)N.[CH3]>O>C[C@H](O)Cl |o1:1,&2:7|")
    encoded = reaction_smiles(reaction, spec)
    if getattr(ReactionContainer, "_supports_stereo_groups", False):
        original = ReactionContainer.__format__

        # Emulate the released reaction writer: keep radicals, omit groups.
        def released_format(self, fmt):
            import re

            return re.sub(r",?[&o]\d+:\d+(?:,\d+)*", "", original(self, fmt))

        monkeypatch.setattr(ReactionContainer, "__format__", released_format)
        monkeypatch.setattr(ReactionContainer, "_supports_stereo_groups", False)
        assert reaction_smiles(reaction, spec) == encoded
    if "!s" not in spec and "!x" not in spec:
        assert "o1:" in encoded and "&2:" in encoded and "^1:" in encoded


def test_native_ungrouped_reaction_formats_components_once(monkeypatch):
    from chython.containers import MoleculeContainer

    from synplan.chem.stereo import reaction_smiles

    if not getattr(ReactionContainer, "_supports_stereo_groups", False):
        pytest.skip("paired Chython writer capability control")
    reaction = smiles("CCO>>CC=O")
    original, calls = MoleculeContainer.__format__, []

    def tracked(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MoleculeContainer, "__format__", tracked)
    reaction_smiles(reaction)
    assert len(calls) == 2


def test_evidence_decisions_do_not_copy_observations_and_expire():
    from copy import deepcopy

    from synplan.chem.stereo_evidence import chemistry_assessment

    class NoCopy:
        def __deepcopy__(self, memo):
            pytest.fail("internal decision copied an observation")

    reaction = smiles("CC=O>>CCO")
    record = attach_stereo_evidence(
        reaction,
        source="control",
        observation={},
        assessment="accepted",
        reviewer="chemist",
        reason="control",
    )
    reaction.meta["stereo_evidence"][0]["observation"] = NoCopy()
    assert chemistry_assessment(reaction) == "accepted"
    rejected = deepcopy(record)
    rejected["assessment"] = "rejected"
    reaction.meta["stereo_evidence"].append(rejected)
    assert chemistry_assessment(reaction) == "conflicting"
    reaction.meta["stereo_evidence"].pop(0)
    assert chemistry_assessment(reaction) == "rejected"
    reaction.meta["procedure"] = "changed"
    assert chemistry_assessment(reaction) == "unreviewed"


def test_empty_assessment_index_skips_context_lookup(monkeypatch):
    from synplan.mcts import tree as tree_module

    tree = tree_for(
        STEREO[0],
        CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]=[O:2]"),
        catalogue(("CC(=O)C(=O)O", 1)),
        algorithm="breadth_first",
    )

    original = tree_module.Reaction

    def no_context(*args, **kwargs):
        assert "meta" in kwargs, (
            "empty evidence index must skip the assessment reaction"
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(tree_module, "Reaction", no_context)
    tree.run()
    assert tree.proposal_nodes and not tree.winning_nodes
    assert tree._get_node_value(tree.proposal_nodes[0]) == 0


@pytest.mark.parametrize("decision", ["accepted", "rejected", "conflicting", "stale"])
@pytest.mark.parametrize("strict", [True, False])
def test_indexed_evidence_controls_search_outcomes(decision, strict):
    rule = CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]=[O:2]")
    stock = catalogue(("CC(=O)C(=O)O", 1))
    initial = tree_for(STEREO[0], rule, stock, algorithm="breadth_first")
    initial.run()
    reaction = Route.from_tree(initial, initial.proposal_nodes[0]).steps[0].reaction
    decisions = (
        ["accepted", "rejected"]
        if decision == "conflicting"
        else ["accepted" if decision == "stale" else decision]
    )
    records = [
        attach_stereo_evidence(
            reaction,
            source="control",
            observation={},
            assessment=d,
            reviewer="chemist",
            reason="control",
        )
        for d in decisions
    ]
    if decision == "stale":
        records[0]["context"] = "expired"
    tree = tree_for(
        STEREO[0],
        rule,
        stock,
        algorithm="breadth_first",
        stereo_mode="strict" if strict else "proposal",
        stereo_assessments=records,
    )
    tree.run()
    assert bool(tree.winning_nodes) == (decision == "accepted")
    assert bool(tree.proposal_nodes) == (
        not strict and decision in {"conflicting", "stale"}
    )
    for node in tree.proposal_nodes:
        assert tree._get_node_value(node) == 0


def test_vocabulary_streaming_digest_rejects_content_edits(tmp_path, monkeypatch):
    from hashlib import sha256

    from synplan.chem.reaction.rules.vocabulary import file_digest, manifest_digest

    source = b"rule1\nrule2\n" * 100_000
    path = tmp_path / "rules.tsv"
    path.write_bytes(source)
    expected = sha256(source).hexdigest()
    path.with_suffix(".manifest.json").write_text(
        json.dumps({"schema": "synplan-rules/2", "rules_sha256": expected})
    )

    def no_read_bytes(*args):
        pytest.fail("vocabulary hashing must stream the complete file")

    monkeypatch.setattr(Path, "read_bytes", no_read_bytes)
    assert file_digest(path) == manifest_digest(path) == expected
    for edited in (source.replace(b"rule1", b"rule3"), b"rule2\nrule1\n" * 100_000):
        path.write_bytes(edited)
        with pytest.raises(ValueError, match="rule file changed"):
            manifest_digest(path)
