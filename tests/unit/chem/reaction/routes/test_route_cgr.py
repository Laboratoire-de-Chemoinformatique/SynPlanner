from types import SimpleNamespace

import pytest
from chython import smiles
from chython.containers import CGRContainer, ReactionContainer
from chython.containers.bonds import DynamicBond

from synplan.chem.reaction.routes.io import (
    make_dict,
    read_routes_csv,
    read_routes_json,
)
from synplan.chem.reaction.routes.representation import (
    compose_all_route_cgrs,
    compose_route_cgr,
    compose_sb_cgr,
    get_clean_mapping,
    routes_dict_from_route_cgrs,
)
from synplan.chem.reaction.routes.representation.container import RouteCGRContainer
from synplan.chem.reaction.routes.representation.state import RouteDynamicBond


class _MockRouteTree:
    def __init__(self, routes_dict):
        self._routes_dict = routes_dict
        self.winning_nodes = list(routes_dict)
        self.config = SimpleNamespace(min_mol_size=999)
        self.building_blocks = set()

    def synthesis_route(self, route_id):
        steps = self._routes_dict[route_id]
        return [steps[step_id] for step_id in sorted(steps)]


# --- Test Data ---
CSV_DATA = """route_id,step_id,smiles,meta
38,0,[CH2:20]([S:17][CH3:16])[S:21](=[O:22])[CH3:24].[S:70]([O:71][OH:18])(=[O:19])[O-:72]>>[O:18]=[S:17](=[O:19])([CH2:20][S:21](=[O:22])[CH3:24])[CH3:16].[S:70]([OH:71])[O-:72],
38,1,[O:18]=[S:17](=[O:19])([CH2:20][S:21](=[O:22])[CH3:24])[CH3:16].[c:60]1[c:61]([c:62][c:63]([c:64][c:65]1)[Cl:66])[C:67]([O:68][OH:69])=[O:23]>>[O:18]=[S:17](=[O:19])([CH3:16])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24].[c:60]1[c:61]([CH2:67][O:68][OH:69])[c:62][c:63]([c:64][c:65]1)[Cl:66],
38,2,[CH2:52]([CH3:53])[O:51][P:50]([O:54][CH2:55][CH3:56])([Cl:59])=[O:57].[O:18]=[S:17](=[O:19])([CH3:16])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24]>>[CH3:53][CH2:52][O:51][P:50]([O:54][CH2:55][CH3:56])([CH2:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24])=[O:57].[ClH:59],
38,3,[CH3:53][CH2:52][O:51][P:50]([O:54][CH2:55][CH3:56])([CH2:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24])=[O:57].[c:1]1[c:2][c:3][c:4][c:5][c:6]1[CH2:7][O:8][c:9]2[c:10]([CH:15]=[O:58])[c:11][c:12][c:13][c:14]2>>[CH3:53][CH2:52][O:51][PH:50]([O:54][CH2:55][CH3:56])=[O:57].[OH2:58].[c:1]1[c:2][c:3][c:4][c:5][c:6]1[CH2:7][O:8][c:9]2[c:14][c:13][c:12][c:11][c:10]2[CH:15]=[CH:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24],
38,4,[CH2:42]([CH3:43])[O:41][P:40]([O:44][CH2:45][CH3:46])([Cl:49])=[O:47].[c:1]1[c:2][c:3][c:4][c:5][c:6]1[CH2:7][O:8][c:9]2[c:14][c:13][c:12][c:11][c:10]2[CH:15]=[CH:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24]>>[CH2:42]([O:41][P:40]([O:44][CH2:45][CH3:46])([CH2:24][S:21](=[O:22])(=[O:23])[CH2:20][S:17](=[O:18])(=[O:19])[CH:16]=[CH:15][c:10]1[c:9]([O:8][CH2:7][c:6]2[c:1][c:2][c:3][c:4][c:5]2)[c:14][c:13][c:12][c:11]1)=[O:47])[CH3:43].[ClH:49],
38,5,[CH2:42]([O:41][P:40]([O:44][CH2:45][CH3:46])([CH2:24][S:21](=[O:22])(=[O:23])[CH2:20][S:17](=[O:18])(=[O:19])[CH:16]=[CH:15][c:10]1[c:9]([O:8][CH2:7][c:6]2[cH:1][cH:2][cH:3][cH:4][cH:5]2)[cH:14][cH:13][cH:12][cH:11]1)=[O:47])[CH3:43].[cH:35]1[cH:36][cH:37][cH:38][cH:39][c:34]1[CH2:33][O:32][c:31]2[c:26]([CH:25]=[O:48])[cH:27][cH:28][cH:29][cH:30]2>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH2:7][O:8][c:9]2[c:10]([cH:11][cH:12][cH:13][cH:14]2)[CH:15]=[CH:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH:24]=[CH:25][c:26]3[cH:27][cH:28][cH:29][cH:30][c:31]3[O:32][CH2:33][c:34]4[cH:35][cH:36][cH:37][cH:38][cH:39]4,
39,0,[CH2:20]([S:17][CH3:16])[S:21](=[O:22])[CH3:24].[c:70]1[c:71]([c:72][c:73]([c:74][c:75]1)[Cl:76])[C:77]([O:78][OH:19])=[O:18]>>[O:18]=[S:17](=[O:19])([CH2:20][S:21](=[O:22])[CH3:24])[CH3:16].[OH:78][CH2:77][c:71]1[c:70][c:75][c:74][c:73]([Cl:76])[c:72]1,
39,1,[O:18]=[S:17](=[O:19])([CH2:20][S:21](=[O:22])[CH3:24])[CH3:16].[c:60]1[c:61]([c:62][c:63]([c:64][c:65]1)[Cl:66])[C:67]([O:68][OH:69])=[O:23]>>[O:18]=[S:17](=[O:19])([CH3:16])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24].[c:60]1[c:61]([CH2:67][O:68][OH:69])[c:62][c:63]([c:64][c:65]1)[Cl:66],
39,2,[CH2:52]([CH3:53])[O:51][P:50]([O:54][CH2:55][CH3:56])([Cl:59])=[O:57].[O:18]=[S:17](=[O:19])([CH3:16])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24]>>[CH3:53][CH2:52][O:51][P:50]([O:54][CH2:55][CH3:56])([CH2:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24])=[O:57].[ClH:59],
39,3,[CH3:53][CH2:52][O:51][P:50]([O:54][CH2:55][CH3:56])([CH2:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24])=[O:57].[c:1]1[c:2][c:3][c:4][c:5][c:6]1[CH2:7][O:8][c:9]2[c:10]([CH:15]=[O:58])[c:11][c:12][c:13][c:14]2>>[CH3:53][CH2:52][O:51][PH:50]([O:54][CH2:55][CH3:56])=[O:57].[OH2:58].[c:1]1[c:2][c:3][c:4][c:5][c:6]1[CH2:7][O:8][c:9]2[c:14][c:13][c:12][c:11][c:10]2[CH:15]=[CH:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24],
39,4,[CH2:42]([CH3:43])[O:41][P:40]([O:44][CH2:45][CH3:46])([Cl:49])=[O:47].[c:1]1[c:2][c:3][c:4][c:5][c:6]1[CH2:7][O:8][c:9]2[c:14][c:13][c:12][c:11][c:10]2[CH:15]=[CH:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH3:24]>>[CH2:42]([O:41][P:40]([O:44][CH2:45][CH3:46])([CH2:24][S:21](=[O:22])(=[O:23])[CH2:20][S:17](=[O:18])(=[O:19])[CH:16]=[CH:15][c:10]1[c:9]([O:8][CH2:7][c:6]2[c:1][c:2][c:3][c:4][c:5]2)[c:14][c:13][c:12][c:11]1)=[O:47])[CH3:43].[ClH:49],
39,5,[CH2:42]([O:41][P:40]([O:44][CH2:45][CH3:46])([CH2:24][S:21](=[O:22])(=[O:23])[CH2:20][S:17](=[O:18])(=[O:19])[CH:16]=[CH:15][c:10]1[c:9]([O:8][CH2:7][c:6]2[cH:1][cH:2][cH:3][cH:4][cH:5]2)[cH:14][cH:13][cH:12][cH:11]1)=[O:47])[CH3:43].[cH:35]1[cH:36][cH:37][cH:38][cH:39][c:34]1[CH2:33][O:32][c:31]2[c:26]([CH:25]=[O:48])[cH:27][cH:28][cH:29][cH:30]2>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH2:7][O:8][c:9]2[c:10]([cH:11][cH:12][cH:13][cH:14]2)[CH:15]=[CH:16][S:17](=[O:18])(=[O:19])[CH2:20][S:21](=[O:22])(=[O:23])[CH:24]=[CH:25][c:26]3[cH:27][cH:28][cH:29][cH:30][c:31]3[O:32][CH2:33][c:34]4[cH:35][cH:36][cH:37][cH:38][cH:39]4,
"""


@pytest.fixture(scope="module")
def routes_data_csv_to_dict():
    """Provides reaction data loaded from the CSV string."""
    # with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_csv:
    #     tmp_csv.write(CSV_DATA)
    #     csv_file_path = tmp_csv.name*

    csv_file_path = "tests/data/routes_mol_1.csv"
    data = read_routes_csv(csv_file_path)
    return data


@pytest.fixture(scope="module")
def routes_data_json_to_dict():
    """Load reaction data from JSON into a nested dict via make_dict."""
    json_file = "tests/data/routes_mol_1.json"
    raw = read_routes_json(json_file)
    return make_dict(raw)


@pytest.fixture(scope="module")
def routes_data_tree(routes_data_csv_to_dict):
    return _MockRouteTree(routes_data_csv_to_dict)


@pytest.mark.parametrize(
    "routes_fixture", ["routes_data_csv_to_dict", "routes_data_json_to_dict"]
)
def test_compose_route_cgr_dict_based_single_route(routes_fixture, request):
    """Test compose_route_cgr with dict input for a valid route_id."""
    data = request.getfixturevalue(routes_fixture)
    print(data)
    route_id = 38
    result = compose_route_cgr(data, route_id)

    assert result is not None
    assert set(result) == {"cgr"}
    assert isinstance(result["cgr"], CGRContainer)
    assert isinstance(result["cgr"], RouteCGRContainer)

    restored_routes = routes_dict_from_route_cgrs({route_id: result["cgr"]})
    assert set(restored_routes[route_id]) == set(data[route_id])

    compat_result = compose_route_cgr(data, route_id, return_reactions_dict=True)
    assert "cgr" in compat_result and "reactions_dict" in compat_result
    assert isinstance(compat_result["reactions_dict"], dict)
    # Ensure all steps are present
    assert len(compat_result["reactions_dict"]) == len(data[route_id])
    for rxn in compat_result["reactions_dict"].values():
        assert isinstance(rxn, ReactionContainer)


@pytest.mark.parametrize(
    "routes_fixture", ["routes_data_csv_to_dict", "routes_data_json_to_dict"]
)
def test_compose_route_cgr_dict_based_invalid_route_id(routes_fixture, request):
    """compose_route_cgr should raise KeyError for invalid route_id."""
    data = request.getfixturevalue(routes_fixture)
    invalid_route_id = 999
    with pytest.raises(KeyError):
        compose_route_cgr(data, invalid_route_id)


def test_get_clean_mapping_preserves_non_identity_atom_mapping():
    current = smiles("[CH3:1][CH3:2]")
    target = smiles("[CH3:10][CH3:20]")

    assert list(current.get_mapping(target)) == [{1: 20, 2: 10}]
    assert get_clean_mapping(current, target) == {1: 20, 2: 10}
    assert get_clean_mapping(current, target, reverse=True) == {20: 1, 10: 2}


def test_compose_route_cgr_tree_based_single_route(routes_data_tree):
    """Test compose_route_cgr with a mock Tree input for a single route."""
    route_id_to_test = 38

    result = compose_route_cgr(routes_data_tree, route_id_to_test)

    assert result is not None
    assert set(result) == {"cgr"}
    assert isinstance(result["cgr"], CGRContainer)
    assert isinstance(result["cgr"], RouteCGRContainer)

    compat_result = compose_route_cgr(
        routes_data_tree,
        route_id_to_test,
        return_reactions_dict=True,
    )
    assert isinstance(compat_result["reactions_dict"], dict)


def test_compose_route_cgr_tree_based_invalid_route_id(routes_data_tree):
    """Test compose_route_cgr with dict input for an invalid route_id."""
    invalid_route_id = 998  # Assuming this ID is not in CSV_DATA
    print(set(routes_data_tree.winning_nodes))
    assert invalid_route_id not in set(routes_data_tree.winning_nodes)


def test_compose_sb_cgr_from_route_data(routes_data_csv_to_dict):
    """Test compose_sb_cgr with a CGR derived from actual route data."""
    route_id_to_test = 38
    composed_route_info = compose_route_cgr(routes_data_csv_to_dict, route_id_to_test)

    assert composed_route_info is not None
    original_route_cgr = composed_route_info["cgr"]
    assert isinstance(original_route_cgr, CGRContainer)

    sb_cgr = compose_sb_cgr(original_route_cgr)
    assert isinstance(sb_cgr, CGRContainer)

    assert len(sb_cgr) < len(original_route_cgr)


def test_compose_route_cgr_preserves_formed_then_broken_bond():
    """Preserved transient bonds are marked as DynamicBond(None, None)."""
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
            1: smiles("[CH3:1][CH3:2]>>[CH4:1]"),
        }
    }

    default_cgr = compose_route_cgr(routes, 1, preserve_transient_bonds=False)["cgr"]
    transient_cgr = compose_route_cgr(routes, 1, preserve_transient_bonds=True)["cgr"]

    assert isinstance(default_cgr, RouteCGRContainer)
    assert isinstance(transient_cgr, RouteCGRContainer)
    assert 2 not in default_cgr._bonds.get(1, {})
    assert transient_cgr.connected_components == [{1, 2, 3}]
    assert str(compose_sb_cgr(transient_cgr)) == str(compose_sb_cgr(default_cgr))

    bond = transient_cgr._bonds[1][2]
    assert isinstance(bond, DynamicBond)
    assert bond.order is None
    assert bond.p_order is None
    assert bond.route_order == 1
    assert bond.route_step_order == {1, 2}
    assert transient_cgr._atoms[1].route_order == {1, 2}
    assert transient_cgr._atoms[2].route_order == {1, 2}
    assert transient_cgr._atoms[1].route_step_order == {1, 2}
    assert transient_cgr._atoms[2].route_step_order == {1, 2}

    batch_cgr = compose_all_route_cgrs(
        routes, route_id=1, preserve_transient_bonds=True
    )[1]
    assert isinstance(batch_cgr, RouteCGRContainer)
    batch_bond = batch_cgr._bonds[1][2]
    assert batch_bond.order is None
    assert batch_bond.p_order is None


def test_compose_route_cgr_preserves_transient_bonds_by_default():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
            1: smiles("[CH3:1][CH3:2]>>[CH4:1]"),
        }
    }

    route_cgr = compose_route_cgr(routes, 1)["cgr"]

    assert isinstance(route_cgr, RouteCGRContainer)
    bond = route_cgr._bonds[1][2]
    assert bond.order is None
    assert bond.p_order is None


def test_compose_route_cgr_route_order_uses_route_depth_for_convergent_route():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:10]>>[CH3:1][CH3:2].[ClH:10]"),
            1: smiles("[CH3:3].[CH3:4][Cl:11]>>[CH3:3][CH3:4].[ClH:11]"),
            2: smiles(
                "[CH3:1][CH3:2].[CH3:3][CH3:4][Cl:12]>>"
                "[CH3:1][CH2:2][CH2:3][CH3:4].[ClH:12]"
            ),
        }
    }

    route_cgr = compose_route_cgr(routes, 1)["cgr"]

    assert route_cgr._bonds[2][3].route_order == 1
    assert route_cgr._bonds[1][2].route_order == 2
    assert route_cgr._bonds[3][4].route_order == 2
    assert route_cgr._bonds[2][3].route_step_order == {3}
    assert route_cgr._bonds[1][2].route_step_order == {1}
    assert route_cgr._bonds[3][4].route_step_order == {2}
    assert route_cgr._atoms[2].route_order == {1, 2}
    assert route_cgr._atoms[3].route_order == {1, 2}
    assert route_cgr._atoms[2].route_step_order == {1, 3}
    assert route_cgr._atoms[3].route_step_order == {2, 3}


def test_compose_route_cgr_route_order_covers_all_final_dynamic_bonds():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:10]>>[CH3:1][CH3:2].[ClH:10]"),
            1: smiles("[CH3:1][CH3:2].[CH3:3][Cl:11]>>[CH3:1][CH2:2][CH3:3].[ClH:11]"),
        }
    }

    route_cgr = compose_route_cgr(routes, 1)["cgr"]

    dynamic_bonds = [
        (atom1, atom2, bond)
        for atom1, atom2, bond in route_cgr.bonds()
        if bond.order != bond.p_order
    ]
    assert dynamic_bonds
    assert any(bond.order is None for _, _, bond in dynamic_bonds)
    assert all(isinstance(bond, RouteDynamicBond) for _, _, bond in dynamic_bonds)
    assert all(bond.route_order is not None for _, _, bond in dynamic_bonds)
    assert all(bond.route_step_order for _, _, bond in dynamic_bonds)


def test_compose_sb_cgr_syncs_copied_atom_state_after_charge_reduction():
    routes = {
        1: {
            0: smiles("[NH3+:1][CH3:2]>>[NH2:1][CH3:2]"),
        }
    }

    route_cgr = compose_route_cgr(routes, 1)["cgr"]
    sb_cgr = compose_sb_cgr(route_cgr)

    for atom_num, atom in sb_cgr.atoms():
        assert atom.charge == sb_cgr._charges[atom_num]
        assert atom.p_charge == sb_cgr._p_charges[atom_num]
        assert atom.is_radical == sb_cgr._radicals[atom_num]
        assert atom.p_is_radical == sb_cgr._p_radicals[atom_num]
    assert sb_cgr._atoms[1].charge == 0
    assert "+" not in str(sb_cgr)


def test_compose_sb_cgr_preserves_product_side_charge_delta():
    routes = {
        1: {
            0: smiles("[NH2:1][CH3:2]>>[NH3+:1][CH3:2]"),
        }
    }

    route_cgr = compose_route_cgr(routes, 1)["cgr"]
    sb_cgr = compose_sb_cgr(route_cgr)

    assert sb_cgr._charges[1] == 0
    assert sb_cgr._p_charges[1] == 1
    assert sb_cgr._atoms[1].charge == 0
    assert sb_cgr._atoms[1].p_charge == 1
    assert ">+" in str(sb_cgr)


def test_compose_sb_cgr_preserves_unchanged_charged_atoms_with_charge_delta():
    routes = {
        1: {
            0: smiles("[O-:1][CH2:2][NH2:3]>>[O-:1][CH2:2][NH3+:3]"),
        }
    }

    route_cgr = compose_route_cgr(routes, 1)["cgr"]
    sb_cgr = compose_sb_cgr(route_cgr)

    assert sb_cgr._charges[1] == -1
    assert sb_cgr._p_charges[1] == -1
    assert sb_cgr._atoms[1].charge == -1
    assert sb_cgr._atoms[1].p_charge == -1
    assert sb_cgr._charges[3] == 0
    assert sb_cgr._p_charges[3] == 1
    assert "O0>-" not in str(sb_cgr)


def _reaction_atom_maps(reaction):
    return {
        "reactants": sorted(sorted(molecule._atoms) for molecule in reaction.reactants),
        "products": sorted(sorted(molecule._atoms) for molecule in reaction.products),
    }


def test_route_cgr_has_native_deconvolution_labels_without_payload_attributes():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
        }
    }

    route_cgr = compose_route_cgr(routes, 1)["cgr"]

    assert not hasattr(route_cgr, "route_reconstruction_schema")
    assert not hasattr(route_cgr, "route_reaction_smiles")
    assert not hasattr(route_cgr, "route_reaction_metadata")
    assert not hasattr(route_cgr, "route_json")
    assert any(
        getattr(atom, "route_atom_step_states", None)
        for atom in route_cgr._atoms.values()
    )
    assert any(
        getattr(bond, "route_bond_step_states", None)
        for _, _, bond in route_cgr.bonds()
    )


def test_single_step_route_cgr_deconvolves_to_mapped_reaction():
    reaction = smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]")
    routes = {1: {0: reaction}}

    route_cgr = compose_route_cgr(routes, 1)["cgr"]
    restored = routes_dict_from_route_cgrs({1: route_cgr})

    expected = ReactionContainer.from_cgr(reaction.compose())
    assert set(restored[1]) == {0}
    assert _reaction_atom_maps(restored[1][0]) == _reaction_atom_maps(expected)


def test_multi_step_route_cgr_deconvolves_composed_reaction_atom_maps(
    routes_data_csv_to_dict,
):
    composed = compose_route_cgr(
        routes_data_csv_to_dict,
        38,
        preserve_transient_bonds=True,
        return_reactions_dict=True,
    )

    restored_routes = routes_dict_from_route_cgrs({38: composed["cgr"]})

    assert set(restored_routes[38]) == set(composed["reactions_dict"])
    assert all(
        _reaction_atom_maps(composed["reactions_dict"][step_id])
        == _reaction_atom_maps(restored_routes[38][step_id])
        for step_id in composed["reactions_dict"]
    )


def test_transient_route_cgr_deconvolves_formed_then_broken_bond():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
            1: smiles("[CH3:1][CH3:2]>>[CH4:1]"),
        }
    }

    composed = compose_route_cgr(
        routes,
        1,
        preserve_transient_bonds=True,
        return_reactions_dict=True,
    )
    restored_routes = routes_dict_from_route_cgrs({1: composed["cgr"]})

    assert set(restored_routes[1]) == {0, 1}
    assert all(
        _reaction_atom_maps(composed["reactions_dict"][step_id])
        == _reaction_atom_maps(restored_routes[1][step_id])
        for step_id in composed["reactions_dict"]
    )


def test_convergent_route_cgr_deconvolution_preserves_chronological_steps():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:10]>>[CH3:1][CH3:2].[ClH:10]"),
            1: smiles("[CH3:3].[CH3:4][Cl:11]>>[CH3:3][CH3:4].[ClH:11]"),
            2: smiles(
                "[CH3:1][CH3:2].[CH3:3][CH3:4][Cl:12]>>"
                "[CH3:1][CH2:2][CH2:3][CH3:4].[ClH:12]"
            ),
        }
    }

    composed = compose_route_cgr(routes, 1, return_reactions_dict=True)
    restored_routes = routes_dict_from_route_cgrs({1: composed["cgr"]})

    assert list(restored_routes[1]) == [0, 1, 2]
    assert all(
        _reaction_atom_maps(composed["reactions_dict"][step_id])
        == _reaction_atom_maps(restored_routes[1][step_id])
        for step_id in composed["reactions_dict"]
    )
