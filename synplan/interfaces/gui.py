import base64
import gc
import io
import pickle
import re
import uuid
import zipfile

from CGRtools.files import SMILESRead
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher

from synplan.chem.reaction_routes.clustering import *
from synplan.chem.reaction_routes.route_cgr import *
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.search import extract_tree_stats
from synplan.mcts.tree import Tree
from synplan.utils.config import TreeConfig
from synplan.utils.loading import (
    load_building_blocks,
    load_policy_function,
    load_reaction_rules,
)
from synplan.utils.visualisation import (
    generate_results_html,
    get_route_svg_from_json,
    html_top_routes_cluster,
    routes_clustering_report,
    routes_subclustering_report,
)

disable_progress_bars("huggingface_hub")

smiles_parser = SMILESRead.create_parser(ignore=True)
DEFAULT_MOL = "c1cc(ccc1Cl)C(CCO)NC(C2(CCN(CC2)c3c4cc[nH]c4ncn3)N)=O"


# --- Helper Functions ---
def download_button(
    object_to_download, download_filename, button_text, pickle_it=False
):
    """
    Generates a link to download the given object_to_download.
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
    button_id = re.sub(r"\d+", "", button_uuid)

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
def load_planning_resources_cached():
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
    st.set_page_config(
        page_title="SynPlanner GUI",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root {
            color-scheme: light !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state variables if they don't exist.
    if "planning_done" not in st.session_state:
        st.session_state.planning_done = False
    if "tree" not in st.session_state:
        st.session_state.tree = None
    if "res" not in st.session_state:
        st.session_state.res = None
    if "target_smiles" not in st.session_state:
        st.session_state.target_smiles = ""

    # Clustering state
    if "clustering_done" not in st.session_state:
        st.session_state.clustering_done = False
    if "clusters" not in st.session_state:
        st.session_state.clusters = None
    if "reactions_dict" not in st.session_state:
        st.session_state.reactions_dict = None
    if "num_clusters_setting" not in st.session_state:
        st.session_state.num_clusters_setting = 10
    if "route_cgrs_dict" not in st.session_state:
        st.session_state.route_cgrs_dict = None
    if "sb_cgrs_dict" not in st.session_state:
        st.session_state.sb_cgrs_dict = None
    if "route_json" not in st.session_state:
        st.session_state.route_json = None

    # Subclustering state
    if "subclustering_done" not in st.session_state:
        st.session_state.subclustering_done = False
    if "subclusters" not in st.session_state:
        st.session_state.subclusters = None

    # Download state
    if "clusters_downloaded" not in st.session_state:
        st.session_state.clusters_downloaded = False

    if "ketcher" not in st.session_state:
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
    st.sidebar.title("Docs")
    st.sidebar.markdown("https://synplanner.readthedocs.io/en/latest/")

    st.sidebar.title("Tutorials")
    st.sidebar.markdown(
        "[Link](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials)"
    )

    st.sidebar.title("Preprint")
    st.sidebar.markdown("[Link](https://doi.org/10.26434/chemrxiv-2024-bzpnd)")

    st.sidebar.title("Paper")
    st.sidebar.markdown("[Link](https://doi.org/10.1021/acs.jcim.4c02004)")

    st.sidebar.title("Issues")
    st.sidebar.markdown(
        "[Report a bug 🐞](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=%5BBUG%5D)"
    )


def handle_molecule_input():
    st.header("Molecule input")
    st.markdown(
        """
        You can provide a molecular structure by either providing:
        * SMILES string + Enter
        * Draw it + Apply
        """
    )

    if "shared_smiles" not in st.session_state:
        st.session_state.shared_smiles = st.session_state.get("ketcher", DEFAULT_MOL)

    if "ketcher_render_count" not in st.session_state:
        st.session_state.ketcher_render_count = 0

    def text_input_changed_callback():
        new_text_value = st.session_state.smiles_text_input_key_for_sync
        if new_text_value != st.session_state.shared_smiles:
            st.session_state.shared_smiles = new_text_value
            st.session_state.ketcher = new_text_value
            st.session_state.ketcher_render_count += 1

    st.text_input(
        "SMILES:",
        value=st.session_state.shared_smiles,
        key="smiles_text_input_key_for_sync",
        on_change=text_input_changed_callback,
        help="Enter SMILES string and press Enter. The drawing will update, and vice-versa.",
    )

    ketcher_key = f"ketcher_widget_for_sync_{st.session_state.ketcher_render_count}"
    smile_code_output_from_ketcher = st_ketcher(
        st.session_state.shared_smiles, key=ketcher_key
    )

    if smile_code_output_from_ketcher != st.session_state.shared_smiles:
        st.session_state.shared_smiles = smile_code_output_from_ketcher
        st.session_state.ketcher = smile_code_output_from_ketcher
        st.rerun()

    current_smiles_for_planning = st.session_state.shared_smiles

    last_planned_smiles = st.session_state.get("target_smiles")
    if (
        last_planned_smiles
        and current_smiles_for_planning != last_planned_smiles
        and st.session_state.get("planning_done", False)
    ):
        st.warning(
            "Molecule structure has changed since the last successful planning run. "
            "Results shown below (if any) are for the previous molecule. "
            "Please re-run planning for the current structure."
        )

    if st.session_state.get("ketcher") != current_smiles_for_planning:
        st.session_state.ketcher = current_smiles_for_planning

    return current_smiles_for_planning


def setup_planning_options():
    st.header("Launch calculation")
    st.markdown(
        """If you modified the structure, please ensure you clicked on `Apply` (bottom right of the molecular editor)."""
    )

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
        )
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
        st.session_state.sb_cgrs_dict = None
        st.session_state.route_json = None

        active_smile_code = st.session_state.get("ketcher", DEFAULT_MOL)
        st.session_state.target_smiles = active_smile_code

        try:
            target_molecule = mol_from_smiles(active_smile_code, clean_stereo=True)
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
                        policy_function = load_policy_function(
                            weights_path=ranking_policy_weights_path
                        )
                        status.update(label="Resources loaded!", state="complete")

                    from synplan.utils.config import RolloutEvaluationConfig
                    from synplan.utils.loading import load_evaluation_function

                    tree_config = TreeConfig(
                        search_strategy=planning_params["search_strategy"],
                        max_iterations=planning_params["max_iterations"],
                        max_depth=planning_params["max_depth"],
                        min_mol_size=planning_params["min_mol_size"],
                        init_node_value=0.5,
                        ucb_type=planning_params["ucb_type"],
                        c_ucb=planning_params["c_ucb"],
                        silent=True,
                    )

                    eval_config = RolloutEvaluationConfig(
                        policy_network=policy_function,
                        reaction_rules=reaction_rules,
                        building_blocks=building_blocks,
                        min_mol_size=tree_config.min_mol_size,
                        max_depth=tree_config.max_depth,
                    )
                    evaluator = load_evaluation_function(eval_config)

                    tree = Tree(
                        target=target_molecule,
                        config=tree_config,
                        reaction_rules=reaction_rules,
                        building_blocks=building_blocks,
                        expansion_function=policy_function,
                        evaluation_function=evaluator,
                    )

                    mcts_progress_text = "Running MCTS iterations..."
                    mcts_bar = st.progress(0, text=mcts_progress_text)
                    for step, (solved, route_id) in enumerate(tree):
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
        except Exception as e:
            st.error(f"An error occurred during planning: {e}")
            st.session_state.planning_done = False


def display_planning_results():
    """
    Planning results for the NOT-SOLVED case only.
    For solved runs, we use the unified planning+clustering section.
    """
    if not st.session_state.get("planning_done", False):
        return

    res = st.session_state.res
    tree = st.session_state.tree

    if res is None or tree is None:
        st.error(
            "Planning results are missing from session state. Please re-run planning."
        )
        st.session_state.planning_done = False
        return

    if res.get("solved", False):
        # Solved case handled in unified section.
        return

    st.header("Planning results")
    st.warning(
        "No reaction path found for the target molecule with the current settings."
    )
    st.write(
        "Consider adjusting planning options (e.g., increase iterations, adjust depth, check molecule validity)."
    )
    stat_col, _ = st.columns(2)
    with stat_col:
        st.subheader("Run statistics (no solution)")
        try:
            if "target_smiles" not in res and "target_smiles" in st.session_state:
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
    """
    Planning results download (full HTML report).
    Only uses internal state; no headers here to keep UX unified.
    """
    if (
        st.session_state.get("planning_done", False)
        and st.session_state.res
        and st.session_state.res.get("solved", False)
    ):
        try:
            if st.button("Generate full HTML report", key="gen_plan_html"):
                with st.spinner("Generating HTML report..."):
                    st.session_state.planning_report_html = generate_results_html(
                        st.session_state.tree, html_path=None, extended=True
                    )

            if st.session_state.get("planning_report_html"):
                st.download_button(
                    label="Download full planning report (HTML)",
                    data=st.session_state.planning_report_html,
                    file_name=f"full_report_{st.session_state.target_smiles}.html",
                    mime="text/html",
                    key="download_full_planning_html",
                )
        except Exception as e:
            st.error(f"Error generating download links for planning results: {e}")


def filter_routes_for_clustering(
    clusters: dict,
    sb_cgrs_dict: dict,
    route_cgrs_dict: dict,
    skip_ids: set[int],
):
    """
    Remove routes in skip_ids from:
      - clusters (route_ids and group_size)
      - SB-CGR dictionary
      - RouteCGR dictionary

    Clusters that become empty are dropped.
    """
    if not skip_ids:
        return clusters, sb_cgrs_dict, route_cgrs_dict

    # Filter SB-CGRs and RouteCGRs
    sb_cgrs_filtered = {
        rid: cgr for rid, cgr in sb_cgrs_dict.items() if rid not in skip_ids
    }
    route_cgrs_filtered = {
        rid: cgr for rid, cgr in route_cgrs_dict.items() if rid not in skip_ids
    }

    # Filter clusters and recompute group_size
    filtered_clusters = {}
    for cid, data in clusters.items():
        route_ids = [rid for rid in data.get("route_ids", []) if rid not in skip_ids]
        if not route_ids:
            # whole cluster becomes empty -> drop it
            continue

        new_data = dict(data)
        new_data["route_ids"] = route_ids
        new_data["group_size"] = len(route_ids)
        filtered_clusters[cid] = new_data

    return filtered_clusters, sb_cgrs_filtered, route_cgrs_filtered


def run_clustering_core():
    """Core clustering logic (no explicit UI button)."""
    st.session_state.clustering_done = False
    st.session_state.subclustering_done = False
    st.session_state.clusters = None
    st.session_state.reactions_dict = None
    st.session_state.subclusters = None
    st.session_state.route_cgrs_dict = None
    st.session_state.sb_cgrs_dict = None
    st.session_state.route_json = None

    with st.spinner("Performing clustering..."):
        try:
            current_tree = st.session_state.tree
            if not current_tree:
                st.error("Tree object not found. Please re-run planning.")
                return

            st.write("Calculating RouteCGRs...")
            route_cgrs_dict = compose_all_route_cgrs(current_tree)
            st.write("Processing SB-CGRs...")
            sb_cgrs_dict = compose_all_sb_cgrs(route_cgrs_dict)

            results = cluster_routes(sb_cgrs_dict, use_strat=False)
            results = dict(sorted(results.items(), key=lambda x: float(x[0])))

            st.session_state.clusters = results
            st.session_state.route_cgrs_dict = route_cgrs_dict
            st.session_state.sb_cgrs_dict = sb_cgrs_dict
            st.write("Extracting reactions...")
            st.session_state.reactions_dict = extract_reactions(current_tree)
            st.session_state.route_json = make_json(st.session_state.reactions_dict)

            if (
                st.session_state.clusters is not None
                and st.session_state.reactions_dict is not None
            ):
                st.session_state.clustering_done = True
                st.success(
                    f"Clustering complete. Found {len(st.session_state.clusters)} clusters."
                )
            else:
                st.error("Clustering failed or returned empty results.")
                st.session_state.clustering_done = False

            gc.collect()
        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
            st.session_state.clustering_done = False


def download_clustering_results():
    """Clustering results download: per-cluster reports + ZIP."""
    if st.session_state.get("clustering_done", False):
        tree_for_html = st.session_state.get("tree")
        clusters_for_html = st.session_state.get("clusters")
        sb_cgrs_for_html = st.session_state.get("sb_cgrs_dict")

        if not tree_for_html:
            st.warning("MCTS tree data not found. Cannot generate cluster reports.")
            return
        if not clusters_for_html:
            st.warning("Cluster data not found. Cannot generate cluster reports.")
            return

        st.caption("Generate downloadable HTML reports for each cluster.")

        MAX_DOWNLOAD_LINKS_DISPLAYED = 10
        num_clusters_total = len(clusters_for_html)
        clusters_items = list(clusters_for_html.items())

        for i, (cluster_idx, group_data) in enumerate(clusters_items):
            if i >= MAX_DOWNLOAD_LINKS_DISPLAYED:
                break
            try:
                html_content = routes_clustering_report(
                    tree_for_html,
                    clusters_for_html,
                    str(cluster_idx),
                    sb_cgrs_for_html,
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
                for group_index, _ in remaining_items:
                    try:
                        html_content = routes_clustering_report(
                            tree_for_html,
                            clusters_for_html,
                            str(group_index),
                            sb_cgrs_for_html,
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

        # ZIP of all clusters
        try:
            buffer = io.BytesIO()
            with zipfile.ZipFile(
                buffer, mode="w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for idx, _ in clusters_items:
                    html_content_zip = routes_clustering_report(
                        tree_for_html,
                        clusters_for_html,
                        str(idx),
                        sb_cgrs_for_html,
                        aam=False,
                    )
                    filename = f"cluster_{idx}_{st.session_state.target_smiles}.html"
                    zf.writestr(filename, html_content_zip)
            buffer.seek(0)

            st.download_button(
                label="Download all cluster reports as ZIP",
                data=buffer,
                file_name=f"all_cluster_reports_{st.session_state.target_smiles}.zip",
                mime="application/zip",
                key="download_all_clusters_zip",
            )
        except Exception as e:
            st.error(f"Error generating ZIP file for cluster reports: {e}")


def display_planning_and_clustering_results_unified():
    """
    Overview tab: planning summary, cluster summary, and best routes from clusters.
    """
    res = st.session_state.res
    tree = st.session_state.tree
    clusters = st.session_state.clusters
    route_json = st.session_state.route_json

    if not (res and tree and clusters and route_json):
        st.error(
            "Missing data for unified planning+clustering display. Please re-run planning."
        )
        return

    # --- Compact planning + cluster summaries instead of big tables ---
    stat_col, cluster_stat_col = st.columns(2, gap="medium")

    with stat_col:
        st.subheader("Planning summary")
        smi = st.session_state.get("target_smiles", "")
        num_routes = res.get("num_routes", "—")
        num_nodes = res.get("num_nodes", "—")
        num_iter = res.get("num_iter", "—")
        search_time = res.get("search_time", "—")

        st.markdown(
            f"""
- **Target SMILES**: `{smi}`
- **Routes explored**: **{num_routes}**
- **Tree nodes**: {num_nodes}
- **MCTS iterations**: {num_iter}
- **Search time**: {search_time} s
"""
        )

    with cluster_stat_col:
        st.subheader("Cluster summary")
        clusters_dict = clusters or {}
        non_empty_clusters = [v for v in clusters_dict.values() if v]
        n_clusters = len(non_empty_clusters)
        route_counts = [v.get("group_size", 0) for v in non_empty_clusters]
        if route_counts:
            min_routes = min(route_counts)
            max_routes = max(route_counts)
            avg_routes = sum(route_counts) / len(route_counts)
        else:
            min_routes = max_routes = avg_routes = 0

        st.markdown(
            f"""
- **Number of clusters**: **{n_clusters}**
- **Routes per cluster**: min {min_routes}, max {max_routes}, avg {avg_routes:.1f}
"""
        )

        if clusters_dict:
            best_route_html = html_top_routes_cluster(
                clusters_dict,
                st.session_state.tree,
                st.session_state.target_smiles,
            )
            st.download_button(
                label="Download best route from each cluster (HTML)",
                data=best_route_html,
                file_name=f"cluster_best_{st.session_state.target_smiles}.html",
                mime="text/html",
                key="download_cluster_best_unified",
            )

    st.markdown("---")
    st.subheader(f"Best routes from {len(clusters)} found clusters")

    # --- Best routes from clusters (always shown, not hidden in expanders) ---
    MAX_DISPLAY_CLUSTERS_DATA = 10
    clusters_items = list(clusters.items())
    first_items = clusters_items[:MAX_DISPLAY_CLUSTERS_DATA]
    remaining_items = clusters_items[MAX_DISPLAY_CLUSTERS_DATA:]

    for cluster_num, group_data in first_items:
        if (
            not group_data
            or "route_ids" not in group_data
            or not group_data["route_ids"]
        ):
            st.warning(f"Cluster {cluster_num} has no data or route_ids.")
            continue

        st.markdown(
            f"**Cluster {cluster_num}** (size: {group_data.get('group_size', 'N/A')})"
        )
        route_id = group_data["route_ids"][0]
        try:
            num_steps = len(tree.synthesis_route(route_id))
            route_score = round(tree.route_score(route_id), 3)
            svg = get_route_svg_from_json(route_json, route_id)

            sb_cgr = group_data.get("sb_cgr")
            sb_cgr_svg = None
            if sb_cgr:
                sb_cgr.clean2d()
                sb_cgr_svg = cgr_display(sb_cgr)

            if svg and sb_cgr_svg:
                col1, col2 = st.columns([0.2, 0.8])
                with col1:
                    st.image(sb_cgr_svg, caption="SB-CGR")
                with col2:
                    st.image(
                        svg,
                        caption=f"Route {route_id}; {num_steps} steps; route score: {route_score}",
                    )
            elif svg:
                st.image(
                    svg,
                    caption=f"Route {route_id}; {num_steps} steps; route score: {route_score}",
                )
                st.warning(f"SB-CGR could not be displayed for cluster {cluster_num}.")
            else:
                st.warning(
                    f"Could not generate SVG for route {route_id} or its SB-CGR."
                )
        except Exception as e:
            st.error(
                f"Error displaying route {route_id} for cluster {cluster_num}: {e}"
            )

    if remaining_items:
        with st.expander(f"... and {len(remaining_items)} more clusters"):
            for cluster_num, group_data in remaining_items:
                if (
                    not group_data
                    or "route_ids" not in group_data
                    or not group_data["route_ids"]
                ):
                    st.warning(
                        f"Cluster {cluster_num} in expansion has no data or route_ids."
                    )
                    continue
                st.markdown(
                    f"**Cluster {cluster_num}** (size: {group_data.get('group_size', 'N/A')})"
                )
                route_id = group_data["route_ids"][0]
                try:
                    num_steps = len(tree.synthesis_route(route_id))
                    route_score = round(tree.route_score(route_id), 3)
                    svg = get_route_svg_from_json(route_json, route_id)

                    sb_cgr = group_data.get("sb_cgr")
                    sb_cgr_svg = None
                    if sb_cgr:
                        sb_cgr.clean2d()
                        sb_cgr_svg = cgr_display(sb_cgr)

                    if svg and sb_cgr_svg:
                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.image(sb_cgr_svg, caption="SB-CGR")
                        with col2:
                            st.image(
                                svg,
                                caption=f"Route {route_id}; {num_steps} steps; route score: {route_score}",
                            )
                    elif svg:
                        st.image(
                            svg,
                            caption=f"Route {route_id}; {num_steps} steps; route score: {route_score}",
                        )
                        st.warning(
                            f"SB-CGR could not be displayed for cluster {cluster_num}."
                        )
                    else:
                        st.warning(
                            f"Could not generate SVG for route {route_id} or its SB-CGR."
                        )
                except Exception as e:
                    st.error(
                        f"Error displaying route {route_id} for cluster {cluster_num}: {e}"
                    )


# --- Subclustering-related cached helpers ---


@st.cache_data
def generate_sb_cgr_image(_cgr):
    _cgr.clean2d()
    return _cgr.depict()


@st.cache_data
def generate_synthon_reaction_image(_synthon_reaction):
    _synthon_reaction.clean2d()
    return depict_custom_reaction(_synthon_reaction)


@st.cache_data
def get_cached_route_svg(_route_json, route_id):
    return get_route_svg_from_json(_route_json, route_id)


@st.cache_data
def get_route_details(_tree, route_id):
    score = round(_tree.route_score(route_id), 3)
    length = len(_tree.synthesis_route(route_id))
    return {"score": score, "length": length}


def display_single_route(route_id, route_json, details):
    try:
        svg_sub = get_cached_route_svg(route_json, route_id)
        if svg_sub:
            st.image(
                svg_sub,
                caption=f"Route {route_id}; score: {details['score']}; steps: {details['length']}",
            )
        else:
            st.warning(f"Could not generate SVG for route {route_id}.")
    except Exception as e:
        st.error(f"Error displaying route {route_id}: {e}")


# --- Subclustering: core + UI + downloads ---


def run_subclustering_core():
    """
    Core subclustering logic, with automatic skipping of problematic routes.

    Strategy:
    - Try subcluster_all_clusters.
    - If it fails and the error message contains 'route <id>',
      remove that route from all clustering dictionaries and retry.
    - Repeat a few times; if still failing, give up with an error.
    """
    st.session_state.subclustering_done = False
    st.session_state.subclusters = None

    clusters = st.session_state.get("clusters")
    sb_cgrs_dict = st.session_state.get("sb_cgrs_dict")
    route_cgrs_dict = st.session_state.get("route_cgrs_dict")

    if not (clusters and sb_cgrs_dict and route_cgrs_dict):
        st.error(
            "Cannot run subclustering. Missing clusters / SB-CGRs / RouteCGRs. "
            "Please ensure clustering ran successfully."
        )
        return

    skipped_routes = set()

    # Arbitrary cap to avoid infinite loops if something else is wrong
    MAX_ATTEMPTS = 5

    for attempt in range(1, MAX_ATTEMPTS + 1):
        with st.spinner(f"Performing subclustering analysis (attempt {attempt})..."):
            try:
                all_subgroups = subcluster_all_clusters(
                    clusters,
                    sb_cgrs_dict,
                    route_cgrs_dict,
                )

                # Success
                st.session_state.clusters = clusters
                st.session_state.sb_cgrs_dict = sb_cgrs_dict
                st.session_state.route_cgrs_dict = route_cgrs_dict
                st.session_state.subclusters = all_subgroups
                st.session_state.subclustering_done = True

                if skipped_routes:
                    st.info(
                        "Subclustering finished after automatically skipping "
                        f"{len(skipped_routes)} problematic route(s): "
                        + ", ".join(str(r) for r in sorted(skipped_routes))
                    )

                gc.collect()
                return

            except Exception as e:
                msg = str(e)
                # Look for "... route 1267" in the error text
                m = re.search(r"route\s+(\d+)", msg)
                if not m:
                    # No route id -> we cannot automatically fix this
                    st.error(f"An error occurred during subclustering: {e}")
                    st.session_state.subclustering_done = False
                    return

                bad_id = int(m.group(1))
                if bad_id in skipped_routes:
                    # We already skipped this one but it still fails -> abort
                    st.error(
                        "Subclustering still failing even after skipping route "
                        f"{bad_id}. Last error: {e}"
                    )
                    st.session_state.subclustering_done = False
                    return

                skipped_routes.add(bad_id)
                st.warning(
                    f"Subclustering failed for route {bad_id}; "
                    "it will be ignored in clusters, Overview and subclustering."
                )

                # Filter this route out and retry
                clusters, sb_cgrs_dict, route_cgrs_dict = filter_routes_for_clustering(
                    clusters,
                    sb_cgrs_dict,
                    route_cgrs_dict,
                    {bad_id},
                )

    st.error(
        "Subclustering failed after multiple attempts even after skipping "
        "problematic routes."
    )
    st.session_state.subclustering_done = False


def setup_subclustering():
    """Subclustering tab header + optional re-run button."""
    if not st.session_state.get("clustering_done", False):
        return

    st.header("Subclustering within a selected cluster")

    with st.expander("What is subclustering?"):
        st.markdown(
            "The first two numbers define the cluster of interest (e.g., 2.1), "
            "while the final designation (such as 3_1) indicates that the selected "
            "subcluster contains three leaving groups in the Markush-like representation "
            "of the abstracted RouteCGR, with “1” specifying a particular set of leaving groups."
        )

    st.caption(
        "Subclustering is pre-computed after clustering so you can switch tabs without extra waiting."
    )

    if st.button("Re-run subclustering analysis", key="rerun_subclustering_button"):
        run_subclustering_core()
        st.rerun()


def display_subclustering_results():
    """Subclustering results display."""
    if not st.session_state.get("subclustering_done", False):
        st.info("Subclustering is not available. Please check clustering results.")
        return

    sub = st.session_state.get("subclusters")
    tree = st.session_state.get("tree")
    route_json = st.session_state.get("route_json")

    if not all([sub, tree, route_json]):
        st.error("Subclustering results are missing. Please re-run subclustering.")
        st.session_state.subclustering_done = False
        return

    sub_input_col, sub_display_col = st.columns([0.15, 0.85])

    with sub_input_col:
        st.subheader("Select cluster")
        available_cluster_nums = sorted(list(sub.keys()))
        if not available_cluster_nums:
            st.warning("No clusters available in subclustering results.")
            return

        sel_cluster_num = st.selectbox(
            "Cluster #",
            options=available_cluster_nums,
            key="subcluster_num_select_key",
        )

        sub_step_cluster = sub.get(sel_cluster_num, {})
        allowed_subclusters = sorted(list(sub_step_cluster.keys()))

        if not allowed_subclusters:
            st.warning(f"No subclusters found for cluster {sel_cluster_num}.")
            return

        sel_subcluster_idx = st.selectbox(
            "Subcluster index",
            options=allowed_subclusters,
            key="subcluster_index_select_key",
        )

        current_subcluster_data = sub_step_cluster.get(sel_subcluster_idx)

        if not current_subcluster_data:
            st.warning("Selected subcluster not found.")
            return

        if "sb_cgr" in current_subcluster_data:
            st.image(
                generate_sb_cgr_image(current_subcluster_data["sb_cgr"]),
                caption=f"SB-CGR of parent cluster {sel_cluster_num}",
            )

        all_routes_in_subcluster = current_subcluster_data.get("routes_data", {}).keys()
        route_details_list = [
            get_route_details(tree, rid) for rid in all_routes_in_subcluster
        ]

        if not route_details_list:
            min_steps, max_steps = 1, 2
        else:
            all_steps = [details["length"] for details in route_details_list]
            min_steps, max_steps = min(all_steps), max(all_steps)

        if min_steps < max_steps:
            min_max_step = st.slider(
                "Filter by number of steps",
                min_value=min_steps,
                max_value=max_steps,
                value=(min_steps, max_steps),
            )
        else:
            st.write(f"Routes with only one possible number of steps: **{min_steps}**")
            min_max_step = (min_steps, max_steps)

    with sub_display_col:
        st.subheader(
            f"Details for subcluster {sel_cluster_num}.{sel_subcluster_idx}: "
            f"total {len(all_routes_in_subcluster)} routes"
        )

        filtered_routes = [
            (rid, details)
            for rid, details in zip(all_routes_in_subcluster, route_details_list)
            if min_max_step[0] <= details["length"] <= min_max_step[1]
        ]

        if not filtered_routes:
            st.info("No routes match the current filter settings.")
            return

        st.markdown(
            f"--- \n**Displaying {len(filtered_routes)} routes "
            f"(from {min_max_step[0]} to {min_max_step[1]} reaction steps)**"
        )

        if "synthon_reaction" in current_subcluster_data:
            try:
                st.image(
                    generate_synthon_reaction_image(
                        current_subcluster_data["synthon_reaction"]
                    ),
                    caption="Markush-like pseudo reaction of subcluster",
                )
            except Exception as e_depict:
                st.warning(f"Could not depict synthon reaction: {e_depict}")

        MAX_ROUTES_PER_SUBCLUSTER = 5
        routes_to_display_direct = filtered_routes[:MAX_ROUTES_PER_SUBCLUSTER]
        remaining_routes = filtered_routes[MAX_ROUTES_PER_SUBCLUSTER:]

        with st.container(height=500):
            for route_id, details in routes_to_display_direct:
                display_single_route(route_id, route_json, details)

            if remaining_routes:
                with st.expander(f"... and {len(remaining_routes)} more routes"):
                    for route_id, details in remaining_routes:
                        display_single_route(route_id, route_json, details)


def download_subclustering_results():
    """Subclustering results download for the currently selected subcluster."""
    if (
        st.session_state.get("subclustering_done", False)
        and "subcluster_num_select_key" in st.session_state
        and "subcluster_index_select_key" in st.session_state
    ):

        sub = st.session_state.get("subclusters")
        tree = st.session_state.get("tree")
        sb_cgrs_for_report = st.session_state.get("sb_cgrs_dict")

        user_input_cluster_num_display = st.session_state.subcluster_num_select_key
        selected_subcluster_idx = st.session_state.subcluster_index_select_key

        if not tree or not sub or not sb_cgrs_for_report:
            st.warning(
                "Missing data for subclustering report generation (tree, subclusters, or SB-CGRs)."
            )
            return

        if (
            user_input_cluster_num_display in sub
            and selected_subcluster_idx in sub[user_input_cluster_num_display]
        ):

            subcluster_data_for_report = sub[user_input_cluster_num_display][
                selected_subcluster_idx
            ]
            processed_subcluster_data = post_process_subgroup(
                subcluster_data_for_report
            )
            if "routes_data" in subcluster_data_for_report and isinstance(
                subcluster_data_for_report["routes_data"], dict
            ):
                processed_subcluster_data["group_lgs"] = group_by_identical_values(
                    subcluster_data_for_report["routes_data"]
                )
            else:
                processed_subcluster_data["group_lgs"] = {}

            try:
                subcluster_html_content = routes_subclustering_report(
                    tree,
                    processed_subcluster_data,
                    user_input_cluster_num_display,
                    selected_subcluster_idx,
                    sb_cgrs_for_report,
                    if_lg_group=True,
                )
                st.download_button(
                    label=(
                        f"Download report for subcluster "
                        f"{user_input_cluster_num_display}.{selected_subcluster_idx}"
                    ),
                    data=subcluster_html_content,
                    file_name=(
                        f"subcluster_{user_input_cluster_num_display}."
                        f"{selected_subcluster_idx}_{st.session_state.target_smiles}.html"
                    ),
                    mime="text/html",
                    key=(
                        f"download_subcluster_{user_input_cluster_num_display}_"
                        f"{selected_subcluster_idx}"
                    ),
                )
            except Exception as e:
                st.error(
                    f"Error generating download report for subcluster "
                    f"{user_input_cluster_num_display}.{selected_subcluster_idx}: {e}"
                )


def display_downloads_tab():
    """All download actions grouped in one place."""
    st.subheader("Planning reports")
    download_planning_results()

    st.markdown("---")
    st.subheader("Cluster reports")
    download_clustering_results()

    st.markdown("---")
    st.subheader("Subclustering reports")
    st.caption(
        "Select a cluster and subcluster in the Subclustering tab first, "
        "then return here to export the corresponding report."
    )
    download_subclustering_results()


def implement_restart():
    """Restart: reset application state."""
    st.divider()
    st.header("Restart application state")
    if st.button("Clear all results & restart", key="restart_button"):
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
            "sb_cgrs_dict",
            "route_json",
            "subclustering_done",
            "subclusters",
            "clusters_downloaded",
            "ketcher_widget",
            "smiles_text_input_key",
            "subcluster_num_select_key",
            "subcluster_index_select_key",
            "planning_report_html",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.ketcher = DEFAULT_MOL
        st.session_state.target_smiles = ""
        st.rerun()


# --- Main Application Flow with tabs ---


def main():
    initialize_app()
    setup_sidebar()

    # --- Top section: Input & Planning ---
    current_smile_code = handle_molecule_input()
    if st.session_state.get("ketcher") != current_smile_code:
        st.session_state.ketcher = current_smile_code

    setup_planning_options()

    # If nothing has been run yet, stop here
    if not st.session_state.get("planning_done", False):
        implement_restart()
        return

    res = st.session_state.res

    # Not solved: show simple planning section, no tabs needed
    if not (res and res.get("solved", False)):
        st.markdown("---")
        st.header("Results")
        display_planning_results()
        implement_restart()
        return

    # Solved: run clustering (once)
    if not st.session_state.get("clustering_done", False):
        run_clustering_core()

    if not st.session_state.get("clustering_done", False):
        # Planning solved but clustering failed
        st.markdown("---")
        st.header("Results")
        st.success("Planning succeeded.")
        st.error(
            "Clustering did not complete successfully. Please check logs or adjust settings."
        )
        st.subheader("Planning downloads")
        download_planning_results()
        implement_restart()
        return

    # Clustering done; pre-compute subclustering so navigation is instant
    if not st.session_state.get("subclustering_done", False):
        run_subclustering_core()

    # From this point we have: planning_done, solved, clustering_done (+ usually subclustering_done)
    # Clear separation between planning controls and results
    st.markdown("---")
    st.header("Results")

    # Small status row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Solution found", "Yes")
    with col2:
        st.metric("Routes", st.session_state.res.get("num_routes", "—"))
    with col3:
        st.metric("Clusters", len(st.session_state.clusters))

    # Results tabs
    tab_overview, tab_subclustering, tab_downloads = st.tabs(
        ["Overview", "Subclustering", "Downloads"]
    )

    with tab_overview:
        display_planning_and_clustering_results_unified()

    with tab_subclustering:
        setup_subclustering()
        if st.session_state.get("subclustering_done", False):
            st.caption(
                "Select a cluster and subcluster, and optionally filter routes by number of steps."
            )
            display_subclustering_results()

    with tab_downloads:
        display_downloads_tab()

    implement_restart()


if __name__ == "__main__":
    main()
