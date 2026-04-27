import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from huggingface_hub.utils import disable_progress_bars
from streamlit_ketcher import st_ketcher

disable_progress_bars("huggingface_hub")

DEFAULT_MOL = "N#CC1(c2ccc(NC(=O)c3cccnc3NCc3ccncc3)cc2)CCCC1"
_BASE_GUI = None


def get_base_gui():
    global _BASE_GUI
    if _BASE_GUI is None:
        import synplan.interfaces.gui as base_gui

        _BASE_GUI = base_gui
    return _BASE_GUI


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

    if "planning_done" not in st.session_state:
        st.session_state.planning_done = False
    if "tree" not in st.session_state:
        st.session_state.tree = None
    if "tree_pickle_bytes" not in st.session_state:
        st.session_state.tree_pickle_bytes = None
    if "res" not in st.session_state:
        st.session_state.res = None
    if "target_smiles" not in st.session_state:
        st.session_state.target_smiles = ""
    if "planning_report_html" not in st.session_state:
        st.session_state.planning_report_html = None
    if "expansion_visualization_html" not in st.session_state:
        st.session_state.expansion_visualization_html = None
    if "tree_visualization_html" not in st.session_state:
        st.session_state.tree_visualization_html = None
    if "ready_for_clustering" not in st.session_state:
        st.session_state.ready_for_clustering = False
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
    if "subclustering_done" not in st.session_state:
        st.session_state.subclustering_done = False
    if "subclusters" not in st.session_state:
        st.session_state.subclusters = None
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
        with st.spinner("Loading planning engine..."):
            base_gui = get_base_gui()
        st.session_state.planning_done = False
        st.session_state.ready_for_clustering = False
        st.session_state.clustering_done = False
        st.session_state.subclustering_done = False
        st.session_state.tree = None
        st.session_state.tree_pickle_bytes = None
        st.session_state.res = None
        st.session_state.clusters = None
        st.session_state.reactions_dict = None
        st.session_state.subclusters = None
        st.session_state.route_cgrs_dict = None
        st.session_state.sb_cgrs_dict = None
        st.session_state.route_json = None
        st.session_state.planning_report_html = None
        st.session_state.expansion_visualization_html = None
        st.session_state.tree_visualization_html = None

        active_smile_code = st.session_state.get("ketcher", DEFAULT_MOL)
        st.session_state.target_smiles = active_smile_code

        try:
            target_molecule = base_gui.mol_from_smiles(
                active_smile_code, clean_stereo=True
            )
            if target_molecule is None:
                st.error(f"Could not parse the input SMILES: {active_smile_code}")
            else:
                (
                    building_blocks_path,
                    ranking_policy_weights_path,
                    reaction_rules_path,
                ) = base_gui.load_planning_resources_cached()
                with st.spinner("Running retrosynthetic planning..."):
                    with st.status("Loading resources...", expanded=False) as status:
                        st.write("Loading building blocks...")
                        building_blocks = base_gui.load_building_blocks(
                            building_blocks_path, standardize=False
                        )
                        st.write("Loading reaction rules...")
                        reaction_rules = base_gui.load_reaction_rules(
                            reaction_rules_path
                        )
                        st.write("Loading policy network...")
                        policy_function = base_gui.load_policy_function(
                            weights_path=ranking_policy_weights_path
                        )
                        status.update(label="Resources loaded!", state="complete")

                    from synplan.utils.config import RolloutEvaluationConfig
                    from synplan.utils.loading import load_evaluation_function

                    tree_config = base_gui.TreeConfig(
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

                    route_scorer = base_gui.ProtectionRouteScorer.from_config()

                    tree = base_gui.Tree(
                        target=target_molecule,
                        config=tree_config,
                        reaction_rules=reaction_rules,
                        building_blocks=building_blocks,
                        expansion_function=policy_function,
                        evaluation_function=evaluator,
                        route_scorer=route_scorer,
                    )

                    mcts_progress_text = "Running MCTS iterations..."
                    mcts_bar = st.progress(0, text=mcts_progress_text)
                    for step, (_solved, _route_id) in enumerate(tree):
                        progress_value = min(
                            1.0, (step + 1) / planning_params["max_iterations"]
                        )
                        mcts_bar.progress(
                            progress_value,
                            text=f"{mcts_progress_text} ({step+1}/{planning_params['max_iterations']})",
                        )

                    res = base_gui.extract_tree_stats(tree, target_molecule)

                    st.session_state.tree = tree
                    try:
                        st.session_state.tree_pickle_bytes = pickle.dumps(
                            tree, protocol=pickle.HIGHEST_PROTOCOL
                        )
                    except Exception:
                        st.session_state.tree_pickle_bytes = None
                    st.session_state.res = res
                    st.session_state.planning_done = True
        except Exception as e:
            st.error(f"An error occurred during planning: {e}")
            st.session_state.planning_done = False


def download_planning_results():
    get_base_gui().download_planning_results()

    if st.session_state.get("expansion_visualization_html"):
        st.download_button(
            label="Download expansion timeline (HTML)",
            data=st.session_state.expansion_visualization_html,
            file_name=f"expansion_tree_{st.session_state.target_smiles}.html",
            mime="text/html",
            key="download_expansion_planning_html",
        )


def _generate_expansion_visualization_in_process(
    html_output_path: Path,
    tree,
    clusters,
    env: dict[str, str],
) -> str:
    old_env = {
        key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")
    }

    try:
        os.environ["MPLCONFIGDIR"] = env["MPLCONFIGDIR"]
        os.environ["XDG_CACHE_HOME"] = env["XDG_CACHE_HOME"]
        from synplan.utils.visualization.visualize_tree_main import (
            generate_expansion_html,
        )

        generate_expansion_html(
            tree,
            html_output_path,
            max_nodes=None,
            sample_step=1,
            with_svg=True,
            radius_step=1.0,
            render_scale=220.0,
            node_radius=None,
            bubble_scale=10.0,
            clusters_data=clusters,
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return html_output_path.read_text(encoding="utf-8")


def generate_expansion_visualization_html(
    tree, clusters=None, tree_pickle_bytes=None
) -> str:
    repo_root = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory(prefix="synplan_expansion_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        tree_pickle_path = temp_dir_path / "tree.pkl"
        clusters_pickle_path = temp_dir_path / "clusters.pkl"
        html_output_path = temp_dir_path / "expansion_tree.html"
        cache_root = temp_dir_path / "cache"
        mpl_cache_dir = cache_root / "matplotlib"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(mpl_cache_dir)
        env["XDG_CACHE_HOME"] = str(cache_root)

        try:
            if tree_pickle_bytes:
                tree_pickle_path.write_bytes(tree_pickle_bytes)
            else:
                with tree_pickle_path.open("wb") as handle:
                    pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return _generate_expansion_visualization_in_process(
                html_output_path,
                tree,
                clusters,
                env,
            )

        command = [
            sys.executable,
            "-m",
            "synplan.utils.visualization.visualize_tree_main",
            "--tree",
            str(tree_pickle_path),
            "--output",
            str(html_output_path),
        ]
        if clusters is not None:
            try:
                with clusters_pickle_path.open("wb") as handle:
                    pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)
                command.extend(["--clusters", str(clusters_pickle_path)])
            except Exception:
                return _generate_expansion_visualization_in_process(
                    html_output_path,
                    tree,
                    clusters,
                    env,
                )

        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "Expansion visualization generation failed."
                + (f" Details: {details}" if details else "")
            )

        return html_output_path.read_text(encoding="utf-8")


def process_tree_visualization_section():
    st.markdown("---")
    st.subheader("Tree visualization")

    process_col, download_col = st.columns(2, gap="medium")
    with process_col:
        if st.button(
            "Process the Tree Visualization",
            key="process_tree_visualization_button",
        ):
            tree = st.session_state.get("tree")
            clusters = st.session_state.get("clusters")
            if not tree:
                st.session_state.tree_visualization_html = None
                st.error("Tree object not found. Please re-run planning.")
            elif not clusters:
                st.session_state.tree_visualization_html = None
                st.error("Clusters not found. Please run clustering first.")
            else:
                try:
                    st.session_state.tree_visualization_html = None
                    with st.spinner(
                        "Generating tree visualization with Strategic Bond Search Mode..."
                    ):
                        st.session_state.tree_visualization_html = (
                            generate_expansion_visualization_html(
                                tree,
                                clusters=clusters,
                                tree_pickle_bytes=st.session_state.get(
                                    "tree_pickle_bytes"
                                ),
                            )
                        )
                    st.success("Tree visualization is ready.")
                except Exception as e:
                    st.error(f"Could not generate tree visualization: {e}")

    with download_col:
        st.download_button(
            label="Download Tree Visualization",
            data=st.session_state.get("tree_visualization_html") or "",
            file_name=f"tree_visualization_{st.session_state.target_smiles}.html",
            mime="text/html",
            key="download_tree_visualization_html",
            disabled=not bool(st.session_state.get("tree_visualization_html")),
        )


def setup_pre_clustering_actions():
    st.markdown("---")
    st.header("Post-search actions")
    st.success("Planning succeeded.")
    st.caption(
        "Generate the expansion timeline from the current search tree before "
        "starting clustering."
    )

    visualize_col, download_col, continue_col = st.columns(3, gap="medium")

    with visualize_col:
        if st.button(
            "Visualize expansion",
            key="generate_expansion_visualization_button",
        ):
            if not st.session_state.get("tree"):
                st.session_state.expansion_visualization_html = None
                st.error("Tree object not found. Please re-run planning.")
            else:
                try:
                    st.session_state.expansion_visualization_html = None
                    with st.spinner(
                        "Pickling the tree and generating expansion timeline HTML..."
                    ):
                        st.session_state.expansion_visualization_html = (
                            generate_expansion_visualization_html(
                                st.session_state.tree,
                                tree_pickle_bytes=st.session_state.get(
                                    "tree_pickle_bytes"
                                ),
                            )
                        )
                    st.success("Expansion visualization is ready.")
                except Exception as e:
                    st.error(f"Could not generate expansion visualization: {e}")

    with download_col:
        st.download_button(
            label="Download expansion HTML",
            data=st.session_state.get("expansion_visualization_html") or "",
            file_name=f"expansion_tree_{st.session_state.target_smiles}.html",
            mime="text/html",
            key="download_expansion_pre_clustering_html",
            disabled=not bool(st.session_state.get("expansion_visualization_html")),
        )

    with continue_col:
        if st.button("Continue to clustering", key="continue_to_clustering_button"):
            st.session_state.ready_for_clustering = True
            st.rerun()


def display_downloads_tab():
    st.subheader("Planning reports")
    download_planning_results()

    st.markdown("---")
    st.subheader("Cluster reports")
    get_base_gui().download_clustering_results()

    process_tree_visualization_section()

    st.markdown("---")
    st.subheader("Subclustering reports")
    st.caption(
        "Select a cluster and subcluster in the Subclustering tab first, "
        "then return here to export the corresponding report."
    )
    get_base_gui().download_subclustering_results()


def implement_restart():
    st.divider()
    st.header("Restart application state")
    if st.button("Clear all results & restart", key="restart_button"):
        keys_to_clear = [
            "planning_done",
            "tree",
            "tree_pickle_bytes",
            "res",
            "target_smiles",
            "ready_for_clustering",
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
            "expansion_visualization_html",
            "tree_visualization_html",
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


def main():
    initialize_app()
    setup_sidebar()

    current_smile_code = handle_molecule_input()
    if st.session_state.get("ketcher") != current_smile_code:
        st.session_state.ketcher = current_smile_code

    setup_planning_options()

    if not st.session_state.get("planning_done", False):
        implement_restart()
        return

    res = st.session_state.res

    if not (res and res.get("solved", False)):
        st.markdown("---")
        st.header("Results")
        get_base_gui().display_planning_results()
        implement_restart()
        return

    if not st.session_state.get("ready_for_clustering", False):
        setup_pre_clustering_actions()
        implement_restart()
        return

    if not st.session_state.get("clustering_done", False):
        get_base_gui().run_clustering_core()

    if not st.session_state.get("clustering_done", False):
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

    if not st.session_state.get("subclustering_done", False):
        get_base_gui().run_subclustering_core()

    st.markdown("---")
    st.header("Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Solution found", "Yes")
    with col2:
        st.metric("Routes", st.session_state.res.get("num_routes", "—"))
    with col3:
        st.metric("Clusters", len(st.session_state.clusters))

    tab_overview, tab_subclustering, tab_downloads = st.tabs(
        ["Overview", "Subclustering", "Downloads"]
    )

    with tab_overview:
        get_base_gui().display_planning_and_clustering_results_unified()

    with tab_subclustering:
        get_base_gui().setup_subclustering()
        if st.session_state.get("subclustering_done", False):
            st.caption(
                "Select a cluster and subcluster, and optionally filter routes by number of steps."
            )
            get_base_gui().display_subclustering_results()

    with tab_downloads:
        display_downloads_tab()

    implement_restart()


if __name__ == "__main__":
    main()
