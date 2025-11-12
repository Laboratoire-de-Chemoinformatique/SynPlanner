import pytest

# === Tests for route_cgr.py functions ===
from synplan.chem.reaction_routes.route_cgr import compose_route_cgr, compose_sb_cgr
from synplan.chem.reaction_routes.io import (
    read_routes_csv,
    read_routes_json,
    TreeWrapper,
    make_dict,
)
from CGRtools.containers import (
    CGRContainer,
    ReactionContainer,
)


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
def routes_data_tree():
    mol_id = 1
    config = 1
    path = "tests/data/forest"
    tree = TreeWrapper.load_tree_from_id(mol_id, config, path)
    return tree


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
    assert "cgr" in result and "reactions_dict" in result
    assert isinstance(result["cgr"], CGRContainer)
    assert isinstance(result["reactions_dict"], dict)
    # Ensure all steps are present
    assert len(result["reactions_dict"]) == len(data[route_id])
    for rxn in result["reactions_dict"].values():
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


def test_compose_route_cgr_tree_based_single_route(routes_data_tree):
    """Test compose_route_cgr with a mock Tree input for a single route."""
    route_id_to_test = 38

    result = compose_route_cgr(routes_data_tree, route_id_to_test)

    assert result is not None
    assert "cgr" in result
    assert "reactions_dict" in result
    assert isinstance(result["cgr"], CGRContainer)
    assert isinstance(result["reactions_dict"], dict)


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
