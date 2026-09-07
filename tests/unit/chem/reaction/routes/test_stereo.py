"""Real patent/planner routes with explicit catalogue records and fault controls."""

import json
import pickle
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pytest
from chython import smiles
from frozendict import frozendict

from synplan.chem.building_blocks import BuildingBlock, molecule_to_inchikey
from synplan.chem.reaction.routes.stereo import audit_stereo_inheritance
from synplan.chem.reaction.routes.stereo_io import read_stereo_route
from synplan.chem.stereo import stereo_requirements, stereo_sign, transfer_stereo

DATA = Path(__file__).resolve().parents[4] / "data" / "stereo"


@pytest.fixture(scope="module")
def fixtures():
    return json.loads((DATA / "routes.json").read_text())


@pytest.fixture(scope="module")
def stock():
    buckets = defaultdict(list)
    for row in json.loads((DATA / "stock.json").read_text()):
        buckets[row["InChIKey"][:14]].append(
            BuildingBlock(
                row["SMILES"], row["InChIKey"], frozendict(), "@" in row["SMILES"]
            )
        )
    return frozendict({key: tuple(value) for key, value in buckets.items()})


def run(case, stock, **kwargs):
    route, sources = read_stereo_route(case["tree"], strip_stereo=True)
    return route, audit_stereo_inheritance(
        route, smiles(case["original_target"]), stock, mapping_sources=sources, **kwargs
    )


@pytest.mark.parametrize(
    "name",
    ["source:n5-00611", "source:n5-02322", "planner:n5-01258", "planner:n5-02115"],
)
def test_multistep_inheritance_whole_route_and_stock(name, fixtures, stock):
    original, audit = run(fixtures[name], stock)
    assert audit.supported, audit.issues
    assert len(audit.route.steps) == len(original.steps) >= 4
    assert audit.assigned_stereo == "inferred_during_reconstruction"
    assert all(not stereo_requirements(s.product) for s in original.steps)
    assert molecule_to_inchikey(audit.route.target) == molecule_to_inchikey(
        smiles(fixtures[name]["original_target"])
    )
    made = {id(s.product): s.product for s in audit.route.steps}
    for step in audit.route.steps:
        for precursor in step.reaction.reactants:
            if id(precursor) in made:
                assert precursor is made[id(precursor)]
    leaf_entries = [r for r in audit.ledger if r["stage"] == "leaf"]
    assert len(leaf_entries) == len(audit.route.leaves())
    assert all(r["selected_stock"]["inchikey"] for r in leaf_entries)
    for mol, entry in zip(audit.route.leaves(), leaf_entries):
        assert molecule_to_inchikey(mol) == entry["selected_stock"]["inchikey"]
    assert not audit.route.unresolved


def test_input_objects_and_provenance_are_unchanged(fixtures, stock):
    case = fixtures["source:n5-02322"]
    route, sources = read_stereo_route(case["tree"], strip_stereo=True)
    target = smiles(case["original_target"])
    route.steps[0].reaction.name = "source reaction label"
    before = pickle.dumps((route, target, stock))
    audit = audit_stereo_inheritance(route, target, stock, mapping_sources=sources)
    assert audit.supported
    assert pickle.dumps((route, target, stock)) == before
    for a, b in zip(route.steps, audit.route.steps):
        assert a.origin == b.origin
        assert a.conditions == b.conditions
        assert a.reaction.meta == b.reaction.meta
        assert a.reaction.name == b.reaction.name
        assert a.product is not b.product


def test_real_wrong_catalogue_stereoisomers_are_rejected(fixtures, stock):
    _, audit = run(fixtures["planner:n5-00611"], stock)
    assert not audit.supported and audit.route is None
    assert audit.issues[0]["reason"] == "required_stock_unavailable"
    assert audit.issues[0]["leaf"] == 3
    assert "BSSCFNOPCQQJAZ" in audit.issues[0]["detail"]


@pytest.mark.parametrize("mode", ["opposite", "unspecified", "absent"])
def test_real_small_diol_cannot_use_wrong_or_unspecified_stock(mode, fixtures, stock):
    # Hexatomic (S)-butane-1,3-diol is an actual leaf, not a toy SMARTS.
    mol = smiles("C[C@H](O)CCO")
    from synplan.chem.precursor import is_purchasable

    assert len(mol) == 6
    assert not is_purchasable(mol, frozendict(), min_mol_size=6)
    prefix = molecule_to_inchikey(mol)[:14]
    replacement = dict(stock)
    if mode == "absent":
        replacement.pop(prefix)
    else:
        if mode == "opposite":
            mol = smiles("C[C@@H](O)CCO")
        else:
            mol.clean_stereo()
        replacement[prefix] = (
            BuildingBlock(
                str(mol), molecule_to_inchikey(mol), frozendict(), mode == "opposite"
            ),
        )
    _, audit = run(fixtures["source:n5-02322"], frozendict(replacement))
    assert not audit.supported
    assert any(
        i["reason"] == "required_stock_unavailable" and i["leaf"] == 1
        for i in audit.issues
    )


def test_achiral_small_leaf_also_requires_actual_stock(fixtures, stock):
    _, control = run(fixtures["source:n5-02322"], stock)
    achiral = next(
        x for x in control.ledger if x["stage"] == "leaf" and not x["requirements"]
    )
    reduced = dict(stock)
    reduced.pop(achiral["selected_stock"]["inchikey"][:14])
    _, audit = run(fixtures["source:n5-02322"], frozendict(reduced))
    assert not audit.supported
    assert any(i["leaf"] == achiral["leaf"] for i in audit.issues)


@pytest.mark.parametrize(
    "name,reason",
    [
        ("source:n5-04086", "requires_stereo_forming_step"),
        ("source:n5-02571", "requires_stereo_forming_step"),
        ("source:n5-00701", "requires_explicit_resolution_or_inversion_strategy"),
    ],
)
def test_real_stereo_forming_or_substitution_steps_abstain(
    name, reason, fixtures, stock
):
    _, audit = run(fixtures[name], stock)
    assert not audit.supported
    assert audit.issues[0]["reason"] == reason
    assert audit.issues[0]["step"] is not None
    assert audit.issues[0]["target_atoms"]


def test_real_symmetry_retains_uncertainty(fixtures):
    with pytest.raises(ValueError, match="stereo-distinct intermediate alignments"):
        read_stereo_route(fixtures["source:n5-08737"]["tree"], strip_stereo=True)


def test_original_target_connectivity_mismatch_abstains(fixtures, stock):
    _, audit = run(fixtures["planner:n5-01338"], stock)
    assert not audit.supported
    assert audit.issues[0]["reason"] == "mapping_or_representation_unresolved"


def test_corrupt_real_intermediate_is_rejected_before_linking(fixtures):
    tree = deepcopy(fixtures["source:n5-02322"]["tree"])
    tree["children"][0]["children"][0]["smiles"] = "C"
    with pytest.raises(ValueError, match="inconsistent adjacent"):
        read_stereo_route(tree, strip_stereo=True)


def test_missing_map_provenance_is_not_invented(fixtures, stock):
    case = fixtures["source:n5-02322"]
    route, _ = read_stereo_route(case["tree"], strip_stereo=True)
    audit = audit_stereo_inheritance(
        route, smiles(case["original_target"]), stock, mapping_sources={}
    )
    assert not audit.supported
    assert all(
        x["reason"] == "mapping_or_representation_unresolved" for x in audit.issues
    )


def test_conflicting_existing_configuration_is_not_overwritten(fixtures, stock):
    case = fixtures["source:n5-02322"]
    route, sources = read_stereo_route(case["tree"], strip_stereo=True)
    leaf = route.leaves()[1]
    n = next(iter(leaf.chiral_tetrahedrons))
    env = leaf.stereogenic_tetrahedrons[n]
    leaf.add_atom_stereo(n, env, True)
    first = audit_stereo_inheritance(
        route, smiles(case["original_target"]), stock, mapping_sources=sources
    )
    if first.supported:
        leaf.clean_stereo()
        leaf.add_atom_stereo(n, env, False)
    audit = audit_stereo_inheritance(
        route, smiles(case["original_target"]), stock, mapping_sources=sources
    )
    assert any(x["reason"] == "configuration_contradicted" for x in audit.issues)


def test_mapping_bound_abstains_instead_of_choosing(fixtures, stock):
    with pytest.raises(ValueError, match="exceeded"):
        read_stereo_route(
            fixtures["source:n5-00611"]["tree"], strip_stereo=True, max_mappings=1
        )


def test_catalogue_price_belongs_to_selected_record(fixtures, stock):
    prefix = molecule_to_inchikey(smiles("C[C@H](O)CCO"))[:14]
    enriched = dict(stock)
    enriched[prefix] = tuple(
        BuildingBlock(
            r.smiles,
            r.inchikey,
            frozendict(vendor=50.0 if "C[C@H]" in r.smiles else 1.0),
            r.has_stereo,
        )
        for r in stock[prefix]
    )
    _, audit = run(fixtures["source:n5-02322"], frozendict(enriched))
    entry = next(r for r in audit.ledger if r["stage"] == "leaf" and r["leaf"] == 1)
    assert audit.supported
    assert entry["selected_stock"]["price"] == 50.0


def test_real_uspto_cip_change_without_inversion():
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    case = json.loads((DATA / "cip_change.json").read_text())["cases"][0]
    reaction = smiles(case["source_line"].split()[0])
    product = next(m for m in reaction.products if m.has_atom(case["atom"]))
    precursor = next(m for m in reaction.reactants if m.has_atom(case["atom"]))
    req = next(r for r in stereo_requirements(product) if r.atoms == (case["atom"],))
    assert stereo_sign(precursor, req) == req.sign

    def cip(mol):
        rd_mol = Chem.MolFromSmiles(format(mol, "m"))
        rdCIPLabeler.AssignCIPLabels(rd_mol)
        atom = next(a for a in rd_mol.GetAtoms() if a.GetAtomMapNum() == case["atom"])
        return atom.GetProp("_CIPCode")

    assert cip(precursor) == "S"
    assert cip(product) == "R"
    reactants = tuple(m.copy() for m in reaction.reactants)
    for mol in reactants:
        mol.clean_stereo()
    slot = transfer_stereo(product, reactants, req)
    assert stereo_sign(reactants[slot], req) == req.sign


def test_real_uspto_double_bond_inheritance():
    case = json.loads((DATA / "cip_change.json").read_text())["cases"][0]
    reaction = smiles(case["source_line"].split()[0])
    product = next(m for m in reaction.products if m.has_atom(case["atom"]))
    reqs = [r for r in stereo_requirements(product) if r.kind == "double_bond"]
    assert len(reqs) == 2
    for req in reqs:
        reactants = tuple(m.copy() for m in reaction.reactants)
        for mol in reactants:
            mol.clean_stereo()
        slot = transfer_stereo(product, reactants, req)
        assert stereo_sign(reactants[slot], req) == req.sign


def test_stereo_lost_and_recreated_at_an_intermediate_is_unresolved(fixtures, stock):
    # A declared fault injection on the real diol route, not a literature route:
    # replace the earliest diol transformation by oxidation then reduction of
    # its stereocentre. Purchased chirality cannot discharge the later creation.
    from chython.containers import ReactionContainer

    from synplan.chem.reaction.routes.route import Route, Step

    case = fixtures["source:n5-02322"]
    original, sources = read_stereo_route(case["tree"], strip_stereo=True)
    leaf = original.leaves()[1]
    n = next(iter(leaf.chiral_tetrahedrons))
    o = next(k for k in leaf.neighbor_numbers(n) if leaf.atom(k).atomic_number == 8)
    ketone = leaf.copy()
    ketone.bond(n, o)._order = 2
    ketone.atom(n)._implicit_hydrogens = 0
    ketone.atom(o)._implicit_hydrogens = 0
    ketone.flush_cache()
    # Reparse for independently recomputed valence and stereo perception.
    ketone = smiles(format(ketone, "m"))
    starting = leaf.copy()
    oxidation = ReactionContainer([starting], [ketone])
    reduction = ReactionContainer([ketone], [leaf])
    augmented = Route((Step(oxidation, ketone), Step(reduction, leaf), *original.steps))
    sources = {
        i: "controlled_loss_recreation_on_patent_route"
        for i in range(len(augmented.steps))
    }
    audit = audit_stereo_inheritance(
        augmented, smiles(case["original_target"]), stock, mapping_sources=sources
    )
    assert not audit.supported
    responsible = next(
        i for i in audit.issues if i["reason"] == "requires_stereo_forming_step"
    )
    assert 0 < responsible["step"] < len(augmented.steps) - 1


def test_stereo_preserving_import_retains_source_configuration(fixtures):
    case = fixtures["source:n5-02322"]
    route, _ = read_stereo_route(case["tree"])
    assert molecule_to_inchikey(route.target) == molecule_to_inchikey(
        smiles(case["original_target"])
    )
    assert stereo_requirements(route.leaves()[1])


def test_stereo_preserving_import_rejects_conflicting_node(fixtures):
    tree = deepcopy(fixtures["source:n5-02322"]["tree"])
    reaction = smiles(tree["children"][0]["smiles"])
    product = reaction.products[0]
    req = stereo_requirements(product)[0]
    product.clean_stereo()
    product.add_atom_stereo(req.atoms[0], req.environment, not req.sign)
    tree["children"][0]["smiles"] = format(reaction, "m")
    with pytest.raises(ValueError, match="opposite mapped configuration"):
        read_stereo_route(tree)


def test_exports_read_live_stereo_status_once_and_recheck_edits(
    fixtures, stock, monkeypatch
):
    from dataclasses import replace

    from synplan.chem import stereo_evidence
    from synplan.chem.reaction.routes.route import Route
    from synplan.utils.visualisation import routes_report_html

    _, audit = run(fixtures["source:n5-02322"], stock)
    route = audit.route
    route = replace(
        route, stereo=stereo_evidence.route_stereo_summary(route, status="fulfilled")
    )
    status, calls = Route.stereo_status.fget, []

    def tracked(self):
        calls.append(self)
        return status(self)

    monkeypatch.setattr(Route, "stereo_status", property(tracked))
    exported = route.to_json()
    assert len(calls) == 1
    assert (
        exported["stereo_status"] == exported["stereo"]["stereo_status"] == "fulfilled"
    )
    calls.clear()
    assert '<div class="draw">' in routes_report_html([route], None)
    assert len(calls) == 1
    context = route.stereo["context"]
    restored = Route.from_json(exported)
    assert stereo_evidence.route_context(restored) == context
    for mutate in (
        lambda r: r.steps[0].reaction.meta.update(procedure="changed"),
        lambda r: r.steps[0].reaction.meta.update(
            stereo_evidence=[{"source": "changed"}]
        ),
        lambda r: r.leaves()[0].meta["selected_stock"].update(price=42),
        lambda r: r.target.clean_stereo(),
    ):
        changed = deepcopy(route)
        mutate(changed)
        assert changed.stereo_status == "needs_reassessment"
        assert changed.to_json()["stereo_status"] == "needs_reassessment"
