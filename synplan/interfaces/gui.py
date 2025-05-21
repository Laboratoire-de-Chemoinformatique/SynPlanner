import base64
import pickle
import re
import uuid
import io
import zipfile

import pandas as pd
import streamlit as st
from CGRtools.files import SMILESRead
from streamlit_ketcher import st_ketcher
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars


from synplan.mcts.expansion import PolicyNetworkFunction
from synplan.mcts.search import extract_tree_stats
from synplan.mcts.tree import Tree
from synplan.chem.utils import mol_from_smiles
from synplan.chem.reaction_routes.route_cgr import *
from synplan.chem.reaction_routes.clustering import *
from synplan.utils.visualisation import (
    routes_clustering_report,
    routes_subclustering_report,
)


from synplan.utils.config import TreeConfig, PolicyNetworkConfig
from synplan.utils.loading import load_reaction_rules, load_building_blocks
from synplan.utils.visualisation import generate_results_html, get_route_svg


import psutil
import gc


disable_progress_bars("huggingface_hub")

smiles_parser = SMILESRead.create_parser(ignore=True)
DEFAULT_MOL = "c1cc(ccc1Cl)C(CCO)NC(C2(CCN(CC2)c3c4cc[nH]c4ncn3)N)=O"


# --- Helper Functions ---
def download_button(
    object_to_download, download_filename, button_text, pickle_it=False
):
    """
    Issued from
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False).encode("utf-8")

    try:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    )
    return dl_link


@st.cache_resource
def load_planning_resources_cached():  # Renamed to avoid conflict if main calls it directly
    building_blocks_path = hf_hub_download(
        repo_id="Laboratoire-De-Chemoinformatique/SynPlanner",
        filename="building_blocks_em_sa_ln.smi",
        subfolder="building_blocks",
        local_dir=".",
    )
    ranking_policy_weights_path = hf_hub_download(
        repo_id="Laboratoire-De-Chemoinformatique/SynPlanner",
        filename="ranking_policy_network.ckpt",
        subfolder="uspto/weights",
        local_dir=".",
    )
    reaction_rules_path = hf_hub_download(
        repo_id="Laboratoire-De-Chemoinformatique/SynPlanner",
        filename="uspto_reaction_rules.pickle",
        subfolder="uspto",
        local_dir=".",
    )
    return building_blocks_path, ranking_policy_weights_path, reaction_rules_path


# --- GUI Sections ---


def initialize_app():
    """1. Initialization: Setting up the main window, layout, and initial widgets."""
    st.set_page_config(page_title="SynPlanner GUI", page_icon="🧪", layout="wide")

    # Initialize session state variables if they don't exist.
    if "planning_done" not in st.session_state:
        st.session_state.planning_done = False
    if "tree" not in st.session_state:
        st.session_state.tree = None
    if "res" not in st.session_state:
        st.session_state.res = None
    if "target_smiles" not in st.session_state:
        st.session_state.target_smiles = (
            ""  # Initial value, might be overwritten by ketcher
        )

    # Clustering state
    if "clustering_done" not in st.session_state:
        st.session_state.clustering_done = False
    if "clusters" not in st.session_state:
        st.session_state.clusters = None
    if "reactions_dict" not in st.session_state:
        st.session_state.reactions_dict = None
    if "num_clusters_setting" not in st.session_state:  # Store the setting used
        st.session_state.num_clusters_setting = 10
    if "route_cgrs_dict" not in st.session_state:
        st.session_state.route_cgrs_dict = None
    if "r_route_cgrs_dict" not in st.session_state:
        st.session_state.r_route_cgrs_dict = None

    # Subclustering state
    if "subclustering_done" not in st.session_state:
        st.session_state.subclustering_done = False
    if "subclusters" not in st.session_state:  # Renamed from 'sub' for clarity
        st.session_state.subclusters = None

    # Download state (less critical now with direct download links)
    if "clusters_downloaded" not in st.session_state:  # Example, might not be needed
        st.session_state.clusters_downloaded = False

    if "ketcher" not in st.session_state:  # For ketcher persistence
        st.session_state.ketcher = DEFAULT_MOL

    intro_text = """
    This is a demo of the graphical user interface of
    [SynPlanner](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/).
    SynPlanner is a comprehensive tool for reaction data curation, rule extraction, model training and retrosynthetic planning.

    More information on SynPlanner is available in the [official docs](https://synplanner.readthedocs.io/en/latest/index.html).
    """
    st.title("`SynPlanner GUI`")
    st.write(intro_text)


def setup_sidebar():
    """2. Sidebar: Handling the widgets and logic within the sidebar area."""
    # st.sidebar.image("img/logo.png") # Assuming img/logo.png is available
    st.sidebar.title("Docs")
    st.sidebar.markdown("https://synplanner.readthedocs.io/en/latest/")

    st.sidebar.title("Tutorials")
    st.sidebar.markdown(
        "https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials"
    )

    st.sidebar.title("Paper")
    st.sidebar.markdown(
        "https://chemrxiv.org/engage/chemrxiv/article-details/66add90bc9c6a5c07ae65796"
    )

    st.sidebar.title("Issues")
    st.sidebar.markdown(
        "[Report a bug 🐞](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=%5BBUG%5D)"
    )


def handle_molecule_input():
    """3. Molecule Input: Managing the input area for molecule data."""
    st.header("Molecule input")
    st.markdown(
        """
        You can provide a molecular structure by either providing:
        * SMILES string + Enter
        * Draw it + Apply
        """
    )
    # Use st.session_state.ketcher to persist drawn molecule
    molecule_text_input = st.text_input(
        "SMILES:", value=st.session_state.ketcher, key="smiles_text_input_key"
    )

    smile_code_ketcher = st_ketcher(molecule_text_input, key="ketcher_widget")
    # col_kethcer, col_info = st.columns([0.8, 0.2])
    # with col_kethcer:
    #     smile_code_ketcher = st_ketcher(molecule_text_input, key="ketcher_widget")
    # with col_info:
    #     st.subheader("Synthetic Complexity")
    #     sascore = ()
    #     st.markdown(f"SAScore: {sascore}")
    #     syba_score = ()
    #     st.markdown(f"SYBA: {sascore}")


    current_smile_code = smile_code_ketcher # The output from ketcher is the definitive SMILES

    if (
        "target_smiles" in st.session_state
        and current_smile_code != st.session_state.target_smiles
    ):
        st.warning("Molecule structure changed. Please re-run planning.")
        st.session_state.planning_done = False
        st.session_state.clustering_done = False
        st.session_state.subclustering_done = False
        st.session_state.tree = None
        st.session_state.res = None
        st.session_state.clusters = None
        st.session_state.reactions_dict = None
        st.session_state.subclusters = None
        st.session_state.ketcher = current_smile_code

    return current_smile_code


def setup_planning_options():
    """4. Planning: Encapsulating the logic related to the "planning" functionality."""
    st.header("Launch calculation")
    st.markdown(
        """If you modified the structure, please ensure you clicked on `Apply` (bottom right of the molecular editor)."""
    )
    # This smile_code display will be updated if handle_molecule_input has run and returned a new smile_code
    # However, to display it correctly, we need the current smile_code from the session or input handler.
    # For simplicity, let's assume handle_molecule_input has updated st.session_state.ketcher
    st.markdown(
        f"The molecule SMILES is actually: ``{st.session_state.get('ketcher', DEFAULT_MOL)}``"
    )

    st.subheader("Planning options")
    st.markdown(
        """
        The description of each option can be found in the
        [Retrosynthetic Planning Tutorial](https://synplanner.readthedocs.io/en/latest/tutorial_files/retrosynthetic_planning.html#Configuring-search-tree).
        """
    )

    col_options_1, col_options_2 = st.columns(2, gap="medium")
    with col_options_1:
        search_strategy_input = st.selectbox(
            label="Search strategy",
            options=(
                "Expansion first",
                "Evaluation first",
            ),
            index=0,
            key="search_strategy_input",
        )
        ucb_type = st.selectbox(
            label="UCB type",
            options=("uct", "puct", "value"),
            index=0,
            key="ucb_type_input",
        )  # Fixed label
        c_ucb = st.number_input(
            "C coefficient of UCB",
            value=0.1,
            placeholder="Type a number...",
            key="c_ucb_input",
        )

    with col_options_2:
        max_iterations = st.slider(
            "Total number of MCTS iterations",
            min_value=50,
            max_value=1000,
            value=300,
            key="max_iterations_slider",
        )
        max_depth = st.slider(
            "Maximal number of reaction steps",
            min_value=3,
            max_value=9,
            value=6,
            key="max_depth_slider",
        )
        min_mol_size = st.slider(
            "Minimum size of a molecule to be precursor",
            min_value=0,
            max_value=7,
            value=0,
            key="min_mol_size_slider",
            help="Number of non-hydrogen atoms in molecule",
        )

    search_strategy_translator = {
        "Expansion first": "expansion_first",
        "Evaluation first": "evaluation_first",
    }
    search_strategy = search_strategy_translator[search_strategy_input]

    planning_params = {
        "search_strategy": search_strategy,
        "ucb_type": ucb_type,
        "c_ucb": c_ucb,
        "max_iterations": max_iterations,
        "max_depth": max_depth,
        "min_mol_size": min_mol_size,
    }

    if st.button("Start retrosynthetic planning", key="submit_planning_button"):
        # Reset downstream states if replanning
        st.session_state.planning_done = False
        st.session_state.clustering_done = False
        st.session_state.subclustering_done = False
        st.session_state.tree = None
        st.session_state.res = None
        st.session_state.clusters = None
        st.session_state.reactions_dict = None
        st.session_state.subclusters = None
        st.session_state.route_cgrs_dict = None
        st.session_state.r_route_cgrs_dict = None
        active_smile_code = st.session_state.get(
            "ketcher", DEFAULT_MOL
        )  # Get current SMILES
        st.session_state.target_smiles = (
            active_smile_code  # Store the SMILES used for this run
        )

        try:
            target_molecule = mol_from_smiles(active_smile_code)
            if target_molecule is None:
                st.error(f"Could not parse the input SMILES: {active_smile_code}")
            else:
                (
                    building_blocks_path,
                    ranking_policy_weights_path,
                    reaction_rules_path,
                ) = load_planning_resources_cached()
                with st.spinner("Running retrosynthetic planning..."):
                    with st.status("Loading resources...", expanded=False) as status:
                        st.write("Loading building blocks...")
                        building_blocks = load_building_blocks(
                            building_blocks_path, standardize=False
                        )
                        st.write("Loading reaction rules...")
                        reaction_rules = load_reaction_rules(reaction_rules_path)
                        st.write("Loading policy network...")
                        policy_config = PolicyNetworkConfig(
                            weights_path=ranking_policy_weights_path
                        )
                        policy_function = PolicyNetworkFunction(
                            policy_config=policy_config
                        )
                        status.update(label="Resources loaded!", state="complete")

                    tree_config = TreeConfig(
                        search_strategy=planning_params["search_strategy"],
                        evaluation_type="rollout",  # This was hardcoded, keeping it.
                        max_iterations=planning_params["max_iterations"],
                        max_depth=planning_params["max_depth"],
                        min_mol_size=planning_params["min_mol_size"],
                        init_node_value=0.5,  # This was hardcoded
                        ucb_type=planning_params["ucb_type"],
                        c_ucb=planning_params["c_ucb"],
                        silent=True,  # This was hardcoded
                    )

                    tree = Tree(
                        target=target_molecule,
                        config=tree_config,
                        reaction_rules=reaction_rules,
                        building_blocks=building_blocks,
                        expansion_function=policy_function,
                        evaluation_function=None,  # This was hardcoded
                    )

                    mcts_progress_text = "Running MCTS iterations..."
                    mcts_bar = st.progress(0, text=mcts_progress_text)
                    for step, (solved, node_id) in enumerate(tree):
                        progress_value = min(
                            1.0, (step + 1) / planning_params["max_iterations"]
                        )
                        mcts_bar.progress(
                            progress_value,
                            text=f"{mcts_progress_text} ({step+1}/{planning_params['max_iterations']})",
                        )

                    res = extract_tree_stats(tree, target_molecule)

                    st.session_state["tree"] = tree
                    st.session_state["res"] = res
                    st.session_state.planning_done = True
                    st.rerun()

        except Exception as e:
            st.error(f"An error occurred during planning: {e}")
            st.session_state.planning_done = False


def display_planning_results():
    """5. Planning Results Display: Handling the presentation of results."""
    if st.session_state.get("planning_done", False):
        res = st.session_state.res
        tree = st.session_state.tree

        if res is None or tree is None:
            st.error(
                "Planning results are missing from session state. Please re-run planning."
            )
            st.session_state.planning_done = False  # Reset state
            return  # Exit this function if no results

        if res.get("solved", False):  # Use .get for safety
            st.header("Planning Results")
            winning_nodes = (
                sorted(set(tree.winning_nodes))
                if hasattr(tree, "winning_nodes") and tree.winning_nodes
                else []
            )
            st.subheader(f"Number of unique routes found: {len(winning_nodes)}")

            st.subheader("Examples of found retrosynthetic routes")
            image_counter = 0
            visualised_node_ids = set()

            if not winning_nodes:
                st.warning(
                    "Planning solved, but no winning nodes found in the tree object."
                )
            else:
                for n, node_id in enumerate(winning_nodes):
                    if image_counter >= 3:
                        break
                    if node_id not in visualised_node_ids:
                        try:
                            visualised_node_ids.add(node_id)
                            num_steps = len(tree.synthesis_route(node_id))
                            route_score = round(tree.route_score(node_id), 3)
                            svg = get_route_svg(tree, node_id)
                            if svg:
                                st.image(
                                    svg,
                                    caption=f"Route {node_id}; {num_steps} steps; Route score: {route_score}",
                                )
                                image_counter += 1
                            else:
                                st.warning(
                                    f"Could not generate SVG for route {node_id}."
                                )
                        except Exception as e:
                            st.error(f"Error displaying route {node_id}: {e}")
        else:  # Not solved
            st.header("Planning Results")
            st.warning(
                "No reaction path found for the target molecule with the current settings."
            )
            st.write(
                "Consider adjusting planning options (e.g., increase iterations, adjust depth, check molecule validity)."
            )
            stat_col, _ = st.columns(2)
            with stat_col:
                st.subheader("Run Statistics (No Solution)")
                try:
                    if (
                        "target_smiles" not in res
                        and "target_smiles" in st.session_state
                    ):
                        res["target_smiles"] = st.session_state.target_smiles
                    cols_to_show = [
                        col
                        for col in [
                            "target_smiles",
                            "num_nodes",
                            "num_iter",
                            "search_time",
                        ]
                        if col in res
                    ]
                    if cols_to_show:
                        df = pd.DataFrame(res, index=[0])[cols_to_show]
                        st.dataframe(df)
                    else:
                        st.write("No statistics to display for the unsuccessful run.")
                except Exception as e:
                    st.error(f"Error displaying statistics: {e}")
                    st.write(res)


def download_planning_results():
    """6. Planning Results Download: Providing functionality to download."""
    if (
        st.session_state.get("planning_done", False)
        and st.session_state.res
        and st.session_state.res.get("solved", False)
    ):
        res = st.session_state.res
        tree = st.session_state.tree
        # This section is usually placed within a column in the original script
        # We'll assume it's called after display_planning_results and can use a new column or area.
        # For proper layout, this should be integrated with display_planning_results' columns.
        # For now, creating a placeholder or separate section for downloads:
        # st.subheader("Downloads") # This might be redundant if called within a layout context.

        # The original code places downloads in the second column of planning results.
        # To replicate, we'd need to pass the column object or call this within that context.
        # Simulating this by just creating the download links:
        try:
            html_body = generate_results_html(tree, html_path=None, extended=True)
            dl_html = download_button(
                html_body,
                f"results_synplanner_{st.session_state.target_smiles}.html",
                "Download results (HTML)",
            )
            if dl_html:
                st.markdown(dl_html, unsafe_allow_html=True)

            try:
                res_df = pd.DataFrame(res, index=[0])
                dl_csv = download_button(
                    res_df,
                    f"stats_synplanner_{st.session_state.target_smiles}.csv",
                    "Download statistics (CSV)",
                )
                if dl_csv:
                    st.markdown(dl_csv, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not prepare statistics CSV for download: {e}")

        except Exception as e:
            st.error(f"Error generating download links for planning results: {e}")


def setup_clustering():
    """7. Clustering: Encapsulating the logic related to the "clustering" functionality."""
    if (
        st.session_state.get("planning_done", False)
        and st.session_state.res
        and st.session_state.res.get("solved", False)
    ):
        st.divider()
        st.header("Clustering the retrosynthetic routes")

        # num_clusters_input = st.number_input( # This input was removed in the final user code, so omitting.
        #     "Desired Number of Clusters (approximate):",
        #     min_value=2, max_value=50, value=st.session_state.get("num_clusters_setting", 10),
        #     key="num_clusters_input_key"
        # )

        if st.button("Run Clustering", key="submit_clustering_button"):
            # st.session_state.num_clusters_setting = num_clusters_input
            st.session_state.clustering_done = False
            st.session_state.subclustering_done = False
            st.session_state.clusters = None
            st.session_state.reactions_dict = None
            st.session_state.subclusters = None
            st.session_state.route_cgrs_dict = None
            st.session_state.r_route_cgrs_dict = None

            with st.spinner("Performing clustering..."):
                try:
                    current_tree = st.session_state.tree
                    if not current_tree:
                        st.error("Tree object not found. Please re-run planning.")
                        return

                    st.write("Calculating RoutesCGRs...")
                    route_cgrs_dict = compose_all_route_cgrs(current_tree)
                    st.write("Processing ReducedRoutesCGRs...")
                    r_route_cgrs_dict = compose_all_reduced_route_cgrs(route_cgrs_dict)

                    results = cluster_routes(
                        r_route_cgrs_dict, use_strat=False
                    )  # num_clusters was removed from args
                    results = dict(sorted(results.items()))

                    st.session_state.clusters = results
                    st.session_state.route_cgrs_dict = route_cgrs_dict
                    st.session_state.r_route_cgrs_dict = r_route_cgrs_dict
                    st.write("Extracting reactions...")
                    st.session_state.reactions_dict = extract_reactions(current_tree)

                    if (
                        st.session_state.clusters is not None
                        and st.session_state.reactions_dict is not None
                    ):  # Check for None explicitly
                        st.session_state.clustering_done = True
                        st.success(
                            f"Clustering complete. Found {len(st.session_state.clusters)} clusters."
                        )
                    else:
                        st.error("Clustering failed or returned empty results.")
                        st.session_state.clustering_done = False

                    del results  # route_cgrs_dict, r_route_cgrs_dict are stored
                    gc.collect()
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during clustering: {e}")
                    st.session_state.clustering_done = False


def display_clustering_results():
    """8. Clustering Results Display: Handling the presentation of results."""
    if st.session_state.get("clustering_done", False):
        clusters = st.session_state.clusters
        # reactions_dict = st.session_state.reactions_dict # Needed for download, not directly for display here
        tree = st.session_state.tree
        MAX_DISPLAY_CLUSTERS_DATA = 10

        if (
            clusters is None or tree is None
        ):  # reactions_dict removed as not critical for display part
            st.error(
                "Clustering results (clusters or tree) are missing. Please re-run clustering."
            )
            st.session_state.clustering_done = False
            return

        st.subheader(f"Best routes from {len(clusters)} Found Clusters")
        clusters_items = list(clusters.items())
        first_items = clusters_items[:MAX_DISPLAY_CLUSTERS_DATA]
        remaining_items = clusters_items[MAX_DISPLAY_CLUSTERS_DATA:]

        for cluster_num, group_data in first_items:
            if (
                not group_data
                or "node_ids" not in group_data
                or not group_data["node_ids"]
            ):
                st.warning(f"Cluster {cluster_num} has no data or node_ids.")
                continue
            st.markdown(
                f"**Cluster {cluster_num}** (Size: {group_data.get('group_size', 'N/A')})"
            )
            node_id = group_data["node_ids"][0]
            try:
                num_steps = len(tree.synthesis_route(node_id))
                route_score = round(tree.route_score(node_id), 3)
                svg = get_route_svg(tree, node_id)
                r_route_cgr = group_data.get("r_route_cgr")  # Safely get r_route_cgr
                r_route_cgr_svg = None
                if r_route_cgr:
                    r_route_cgr.clean2d()
                    r_route_cgr_svg = cgr_display(r_route_cgr)

                if svg and r_route_cgr_svg:
                    col1, col2 = st.columns([0.2, 0.8])
                    with col1:
                        st.image(r_route_cgr_svg, caption="ReducedRouteCGR")
                    with col2:
                        st.image(
                            svg,
                            caption=f"Route {node_id}; {num_steps} steps; Route score: {route_score}",
                        )
                elif svg:  # Only route SVG available
                    st.image(
                        svg,
                        caption=f"Route {node_id}; {num_steps} steps; Route score: {route_score}",
                    )
                    st.warning(
                        f"ReducedRouteCGR could not be displayed for cluster {cluster_num}."
                    )
                else:
                    st.warning(
                        f"Could not generate SVG for route {node_id} or its ReducedRouteCGR."
                    )
            except Exception as e:
                st.error(
                    f"Error displaying route {node_id} for cluster {cluster_num}: {e}"
                )

        if remaining_items:
            with st.expander(f"... and {len(remaining_items)} more clusters"):
                for cluster_num, group_data in remaining_items:
                    if (
                        not group_data
                        or "node_ids" not in group_data
                        or not group_data["node_ids"]
                    ):
                        st.warning(
                            f"Cluster {cluster_num} in expansion has no data or node_ids."
                        )
                        continue
                    st.markdown(
                        f"**Cluster {cluster_num}** (Size: {group_data.get('group_size', 'N/A')})"
                    )
                    node_id = group_data["node_ids"][0]
                    try:
                        num_steps = len(tree.synthesis_route(node_id))
                        route_score = round(tree.route_score(node_id), 3)
                        svg = get_route_svg(tree, node_id)
                        r_route_cgr = group_data.get("r_route_cgr")
                        r_route_cgr_svg = None
                        if r_route_cgr:
                            r_route_cgr.clean2d()
                            r_route_cgr_svg = cgr_display(r_route_cgr)

                        if svg and r_route_cgr_svg:
                            col1, col2 = st.columns([0.2, 0.8])
                            with col1:
                                st.image(r_route_cgr_svg, caption="ReducedRouteCGR")
                            with col2:
                                st.image(
                                    svg,
                                    caption=f"Route {node_id}; {num_steps} steps; Route score: {route_score}",
                                )
                        elif svg:
                            st.image(
                                svg,
                                caption=f"Route {node_id}; {num_steps} steps; Route score: {route_score}",
                            )
                            st.warning(
                                f"ReducedRouteCGR could not be displayed for cluster {cluster_num}."
                            )
                        else:
                            st.warning(
                                f"Could not generate SVG for route {node_id} or its ReducedRouteCGR."
                            )
                    except Exception as e:
                        st.error(
                            f"Error displaying route {node_id} for cluster {cluster_num}: {e}"
                        )


def download_clustering_results():
    """9. Clustering Results Download: Providing functionality to download."""
    if st.session_state.get("clustering_done", False):
        tree_for_html = st.session_state.get("tree")
        clusters_for_html = st.session_state.get("clusters")
        r_route_cgrs_for_html = st.session_state.get(
            "r_route_cgrs_dict"
        )  # This was used instead of reactions_dict in the original for report

        if not tree_for_html:
            st.warning("MCTS Tree data not found. Cannot generate cluster reports.")
            return
        if not clusters_for_html:
            st.warning("Cluster data not found. Cannot generate cluster reports.")
            return
        # r_route_cgrs_for_html is optional for routes_clustering_report if not essential

        st.subheader("Cluster Reports")  # Changed subheader in original
        st.write("Generate downloadable HTML reports for each cluster:")

        MAX_DOWNLOAD_LINKS_DISPLAYED = 10
        num_clusters_total = len(clusters_for_html)
        clusters_items = list(clusters_for_html.items())

        for i, (cluster_idx, group_data) in enumerate(
            clusters_items
        ):  # group_data might not be needed here if report uses cluster_idx
            if i >= MAX_DOWNLOAD_LINKS_DISPLAYED:
                break
            try:
                html_content = routes_clustering_report(
                    tree_for_html,
                    clusters_for_html,  # Pass the whole dict
                    str(cluster_idx),  # Pass the key of the cluster
                    r_route_cgrs_for_html,  # Pass the r_route_cgrs dict
                    aam=False,
                )
                st.download_button(
                    label=f"Download report for cluster {cluster_idx}",
                    data=html_content,
                    file_name=f"cluster_{cluster_idx}_{st.session_state.target_smiles}.html",
                    mime="text/html",
                    key=f"download_cluster_{cluster_idx}",
                )
            except Exception as e:
                st.error(f"Error generating report for cluster {cluster_idx}: {e}")

        if num_clusters_total > MAX_DOWNLOAD_LINKS_DISPLAYED:
            remaining_items = clusters_items[MAX_DOWNLOAD_LINKS_DISPLAYED:]
            remaining_count = len(remaining_items)
            expander_label = f"Show remaining {remaining_count} cluster reports"
            with st.expander(expander_label):
                for (
                    group_index,
                    _,
                ) in remaining_items:  # group_data not needed here either
                    try:
                        html_content = routes_clustering_report(
                            tree_for_html,
                            clusters_for_html,
                            str(group_index),
                            r_route_cgrs_for_html,
                            aam=False,
                        )
                        st.download_button(
                            label=f"Download report for cluster {group_index}",
                            data=html_content,
                            file_name=f"cluster_{group_index}_{st.session_state.target_smiles}.html",
                            mime="text/html",
                            key=f"download_cluster_expanded_{group_index}",
                        )
                    except Exception as e:
                        st.error(
                            f"Error generating report for cluster {group_index} (expanded): {e}"
                        )

        try:
            buffer = io.BytesIO()
            with zipfile.ZipFile(
                buffer, mode="w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for idx, _ in clusters_items:  # group_data not needed
                    html_content_zip = routes_clustering_report(
                        tree_for_html,
                        clusters_for_html,
                        str(idx),
                        r_route_cgrs_for_html,
                        aam=False,
                    )
                    filename = f"cluster_{idx}_{st.session_state.target_smiles}.html"
                    zf.writestr(filename, html_content_zip)
            buffer.seek(0)

            st.download_button(
                label="📦 Download all cluster reports as ZIP",
                data=buffer,
                file_name=f"all_cluster_reports_{st.session_state.target_smiles}.zip",
                mime="application/zip",
                key="download_all_clusters_zip",
            )
        except Exception as e:
            st.error(f"Error generating ZIP file for cluster reports: {e}")


def setup_subclustering():
    """10. Subclustering: Encapsulating the logic related to the "subclustering" functionality."""
    if st.session_state.get(
        "clustering_done", False
    ):  # Subclustering depends on clustering being done
        st.divider()
        st.header("Sub-Clustering within a selected Cluster")

        if st.button("Run Subclustering Analysis", key="submit_subclustering_button"):
            st.session_state.subclustering_done = False
            st.session_state.subclusters = None
            with st.spinner("Performing subclustering analysis..."):
                try:
                    clusters_for_sub = st.session_state.get("clusters")
                    r_route_cgrs_dict_for_sub = st.session_state.get(
                        "r_route_cgrs_dict"
                    )  # Corrected name
                    route_cgrs_dict_for_sub = st.session_state.get(
                        "route_cgrs_dict"
                    )  # Corrected name

                    if (
                        clusters_for_sub
                        and r_route_cgrs_dict_for_sub
                        and route_cgrs_dict_for_sub
                    ):  # Ensure all are present
                        all_subgroups = subcluster_all_clusters(
                            clusters_for_sub, r_route_cgrs_dict_for_sub, route_cgrs_dict_for_sub
                        )
                        st.session_state.subclusters = all_subgroups
                        st.session_state.subclustering_done = True
                        st.success("Subclustering analysis complete.")
                        gc.collect()
                        st.rerun()
                    else:
                        missing = []
                        if not clusters_for_sub:
                            missing.append("clusters")
                        if not r_route_cgrs_dict_for_sub:
                            missing.append("ReducedRouteCGRs dictionary")
                        if not route_cgrs_dict_for_sub:
                            missing.append("RouteCGRs dictionary")
                        st.error(
                            f"Cannot run subclustering. Missing data: {', '.join(missing)}. Please ensure clustering ran successfully."
                        )
                        st.session_state.subclustering_done = False

                except Exception as e:
                    st.error(f"An error occurred during subclustering: {e}")
                    st.session_state.subclustering_done = False


def display_subclustering_results():
    """11. Subclustering Results Display: Handling the presentation of results."""
    if st.session_state.get("subclustering_done", False):
        sub = st.session_state.get("subclusters")
        tree = st.session_state.get("tree")
        # clusters_for_sub_display = st.session_state.get('clusters') # Not directly used in display logic from original code snippet

        if not sub or not tree:
            st.error(
                "Subclustering results (subclusters or tree) are missing. Please re-run subclustering."
            )
            st.session_state.subclustering_done = False
            return

        sub_input_col, sub_display_col = st.columns([0.25, 0.75])

        with sub_input_col:
            st.subheader("Select Cluster and Subcluster")
            available_cluster_nums = list(sub.keys())
            if not available_cluster_nums:
                st.warning("No clusters available in subclustering results.")
                return  # Exit if no clusters to select

            user_input_cluster_num_display = st.selectbox(
                "Select Cluster #:",
                options=sorted(available_cluster_nums),
                key="subcluster_num_select_key",
            )

            selected_subcluster_idx = 0

            if user_input_cluster_num_display in sub:
                sub_step_cluster = sub[user_input_cluster_num_display]
                allowed_subclusters_indices = sorted(list(sub_step_cluster.keys()))

                if not allowed_subclusters_indices:
                    st.warning(
                        f"No reaction steps (subclusters) found for Cluster {user_input_cluster_num_display}."
                    )
                else:
                    selected_subcluster_idx = st.selectbox(
                        "Select Subcluster Index:",
                        options=allowed_subclusters_indices,
                        key="subcluster_index_select_key",
                    )
                    if selected_subcluster_idx in sub[user_input_cluster_num_display]:
                        current_subcluster_data = sub[user_input_cluster_num_display][
                            selected_subcluster_idx
                        ]
                        if "r_route_cgr" in current_subcluster_data:
                            cluster_r_route_cgr_display = current_subcluster_data["r_route_cgr"]
                            cluster_r_route_cgr_display.clean2d()
                            st.image(
                                cluster_r_route_cgr_display.depict(),
                                caption=f"ReducedRouteCGR of parent Cluster {user_input_cluster_num_display}",
                            )
                        else:
                            st.warning("ReducedRouteCGR for this subcluster not found.")
            else:
                st.warning(
                    f"Selected cluster {user_input_cluster_num_display} not found in subclustering results."
                )
                return

        with sub_display_col:
            st.subheader("Subcluster Details")
            if (
                user_input_cluster_num_display in sub
                and selected_subcluster_idx in sub[user_input_cluster_num_display]
            ):

                subcluster_content = sub[user_input_cluster_num_display][
                    selected_subcluster_idx
                ]

                subcluster_to_display = post_process_subgroup(subcluster_content)

                if (
                    not subcluster_to_display
                    or "nodes_data" not in subcluster_to_display
                    or not subcluster_to_display["nodes_data"]
                ):
                    st.info("No routes or data found for this subcluster selection.")
                else:
                    MAX_ROUTES_PER_SUBCLUSTER = 5
                    all_route_ids_in_subcluster = list(
                        subcluster_to_display["nodes_data"].keys()
                    )
                    routes_to_display_direct = all_route_ids_in_subcluster[
                        :MAX_ROUTES_PER_SUBCLUSTER
                    ]
                    remaining_routes_sub = all_route_ids_in_subcluster[
                        MAX_ROUTES_PER_SUBCLUSTER:
                    ]

                    st.markdown(
                        f"--- \n**Subcluster {user_input_cluster_num_display}.{selected_subcluster_idx}** (Size: {len(all_route_ids_in_subcluster)})"
                    )

                    if "synthon_reaction" in subcluster_to_display:
                        synthon_reaction = subcluster_to_display["synthon_reaction"]
                        try:
                            st.image(
                                depict_custom_reaction(synthon_reaction),
                                caption=f"Markush-like pseudo reaction of subcluster",
                            )  # Assuming depict_custom_reaction
                        except Exception as e_depict:
                            st.warning(f"Could not depict synthon reaction: {e_depict}")
                    else:
                        st.info("No synthon reaction data for this subcluster.")

                    for route_id in routes_to_display_direct:
                        try:
                            route_score_sub = round(tree.route_score(route_id), 3)
                            svg_sub = get_route_svg(tree, route_id)
                            if svg_sub:
                                st.image(
                                    svg_sub,
                                    caption=f"Route {route_id}; Score: {route_score_sub}",
                                )
                            else:
                                st.warning(
                                    f"Could not generate SVG for route {route_id}."
                                )
                        except Exception as e:
                            st.error(
                                f"Error displaying route {route_id} in subcluster: {e}"
                            )

                    if remaining_routes_sub:
                        with st.expander(
                            f"... and {len(remaining_routes_sub)} more routes in this subcluster"
                        ):
                            for route_id in remaining_routes_sub:
                                try:
                                    route_score_sub = round(
                                        tree.route_score(route_id), 3
                                    )
                                    # num_steps_sub = len(tree.synthesis_route(route_id))
                                    svg_sub = get_route_svg(tree, route_id)
                                    if svg_sub:
                                        st.image(
                                            svg_sub,
                                            caption=f"Route {route_id}; Score: {route_score_sub}",
                                        )
                                    else:
                                        st.warning(
                                            f"Could not generate SVG for route {route_id}."
                                        )
                                except Exception as e:
                                    st.error(
                                        f"Error displaying route {route_id} in subcluster (expanded): {e}"
                                    )
            else:
                st.info("Select a valid cluster and subcluster index to see details.")


def download_subclustering_results():
    """12. Subclustering Results Download: Providing functionality to download."""
    if (
        st.session_state.get("subclustering_done", False)
        and "subcluster_num_select_key" in st.session_state
        and "subcluster_index_select_key" in st.session_state
    ):

        sub = st.session_state.get("subclusters")
        tree = st.session_state.get("tree")
        r_route_cgrs_for_report = st.session_state.get(
            "r_route_cgrs_dict"
        )  # Used by routes_subclustering_report

        user_input_cluster_num_display = st.session_state.subcluster_num_select_key
        selected_subcluster_idx = st.session_state.subcluster_index_select_key

        if not tree or not sub or not r_route_cgrs_for_report:
            st.warning(
                "Missing data for subclustering report generation (tree, subclusters, or ReducedRouteCGRs)."
            )
            return

        if (
            user_input_cluster_num_display in sub
            and selected_subcluster_idx in sub[user_input_cluster_num_display]
        ):

            subcluster_data_for_report = sub[user_input_cluster_num_display][
                selected_subcluster_idx
            ]
            # Apply the same post-processing as in display
            processed_subcluster_data = post_process_subgroup(
                subcluster_data_for_report
            )
            if "nodes_data" in subcluster_data_for_report and isinstance(
                subcluster_data_for_report["nodes_data"], dict
            ):
                processed_subcluster_data["group_lgs"] = group_by_identical_values(
                    subcluster_data_for_report["nodes_data"]
                )
            else:
                processed_subcluster_data["group_lgs"] = {}

            try:
                subcluster_html_content = routes_subclustering_report(
                    tree,
                    processed_subcluster_data,  # Pass the specific post-processed subcluster data
                    user_input_cluster_num_display,
                    selected_subcluster_idx,
                    r_route_cgrs_for_report,  # Pass the whole r_route_cgrs dict
                    if_lg_group=True,  # This parameter was in the original call
                )
                st.download_button(
                    label=f"Download report for subcluster {user_input_cluster_num_display}.{selected_subcluster_idx}",
                    data=subcluster_html_content,
                    file_name=f"subcluster_{user_input_cluster_num_display}.{selected_subcluster_idx}_{st.session_state.target_smiles}.html",
                    mime="text/html",
                    key=f"download_subcluster_{user_input_cluster_num_display}_{selected_subcluster_idx}",
                )
            except Exception as e:
                st.error(
                    f"Error generating download report for subcluster {user_input_cluster_num_display}.{selected_subcluster_idx}: {e}"
                )
        # else:
        # This case is handled by the display logic mostly, download button just won't appear or will be for previous valid selection.


def implement_restart():
    """13. Restart: Implementing the logic to reset or restart the application state."""
    st.divider()
    st.header("Restart Application State")
    if st.button("Clear All Results & Restart", key="restart_button"):
        keys_to_clear = [
            "planning_done",
            "tree",
            "res",
            "target_smiles",
            "clustering_done",
            "clusters",
            "reactions_dict",
            "num_clusters_setting",
            "route_cgrs_dict",
            "r_route_cgrs_dict",
            "subclustering_done",
            "subclusters",  # "sub" was renamed
            "clusters_downloaded",
            # Potentially ketcher related keys if they need manual reset beyond new input
            "ketcher_widget",
            "smiles_text_input_key",  # Keys for widgets
            "subcluster_num_select_key",
            "subcluster_index_select_key",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Reset ketcher input to default by resetting its session state variable
        st.session_state.ketcher = DEFAULT_MOL
        # Also explicitly set target_smiles to empty or default to avoid stale data
        st.session_state.target_smiles = ""

        # It's generally better to let Streamlit manage widget state if possible,
        # but for a full reset, clearing their explicit session state keys might be needed.
        st.rerun()


# --- Main Application Flow ---
def main():
    initialize_app()
    setup_sidebar()
    current_smile_code = handle_molecule_input()
    # Update session_state.ketcher if current_smile_code has changed from ketcher output
    if st.session_state.get("ketcher") != current_smile_code:
        st.session_state.ketcher = current_smile_code
        # No rerun here, let the flow continue. handle_molecule_input already warns.

    setup_planning_options()  # This function now also handles the button press and logic for planning

    # Display planning results and download options together
    if st.session_state.get("planning_done", False):
        display_planning_results()  # Displays stats and routes
        if st.session_state.res and st.session_state.res.get("solved", False):
            stat_col, download_col = st.columns(
                2, gap="medium"
            )  # Placeholder for download column
            with stat_col:
                st.subheader("Statistics")
                try:
                    res = st.session_state.res
                    if (
                        "target_smiles" not in res
                        and "target_smiles" in st.session_state
                    ):
                        res["target_smiles"] = st.session_state.target_smiles
                    cols_to_show = [
                        col
                        for col in [
                            "target_smiles",
                            "num_routes",
                            "num_nodes",
                            "num_iter",
                            "search_time",
                        ]
                        if col in res
                    ]
                    if cols_to_show:  # Ensure there are columns to show
                        df = pd.DataFrame(res, index=[0])[cols_to_show]
                        st.dataframe(df)
                    else:
                        st.write("No statistics to display from planning results.")
                except Exception as e:
                    st.error(f"Error displaying statistics: {e}")
                    st.write(res)  # Show raw dict if DataFrame fails
            with download_col:
                st.subheader("Planning Downloads")  # Adding a subheader for clarity
                download_planning_results()

    # Clustering section (setup button, display, download)
    if (
        st.session_state.get("planning_done", False)
        and st.session_state.res
        and st.session_state.res.get("solved", False)
    ):
        setup_clustering()  # Contains the "Run Clustering" button and logic
        if st.session_state.get("clustering_done", False):
            display_clustering_results()  # Displays cluster routes and stats
            cluster_stat_col, cluster_download_col = st.columns(2, gap="medium")

            with cluster_stat_col:
                clusters = st.session_state.clusters
                cluster_sizes = [
                    cluster.get("group_size", 0)
                    for cluster in clusters.values()
                    if cluster
                ]  # Safe get
                st.subheader("Cluster Statistics")
                if cluster_sizes:
                    cluster_df = pd.DataFrame(
                        {
                            "Cluster": [
                                k for k, v in clusters.items() if v
                            ],  # Filter out empty clusters
                            "Number of Routes": [
                                v["group_size"] for v in clusters.values() if v
                            ],
                        }
                    )
                    if not cluster_df.empty:
                        cluster_df.index += 1
                        st.dataframe(cluster_df)
                    else:
                        st.write("No valid cluster data to display statistics for.")

                else:
                    st.write("No cluster data to display statistics for.")
            with cluster_download_col:
                download_clustering_results()

    # Subclustering section (setup button, display, download)
    if st.session_state.get("clustering_done", False):  # Depends on clustering
        setup_subclustering()  # Contains "Run Subclustering" button
        if st.session_state.get("subclustering_done", False):
            display_subclustering_results()  # Displays subcluster details and routes
            download_subclustering_results()  # This needs to be called after selections are made in display.

    implement_restart()


if __name__ == "__main__":
    main()
