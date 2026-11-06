"""Streamlit UI for LAN network inspector workflows."""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ssms
import streamlit as st

if __package__ in (None, ""):
    # Support direct execution with:
    # streamlit run src/lanfactory/network_inspectors/streamlit_app.py
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from lanfactory.network_inspectors import ui_config as ui
from lanfactory.network_inspectors.api import (
    compute_kde_vs_lan_likelihoods,
    compute_lan_manifold,
)
from lanfactory.network_inspectors.config import (
    GridSpec,
    ModelSpec,
    PlotConfig,
)
from lanfactory.network_inspectors.discovery import (
    model_directories,
    resolve_model_paths,
)
from lanfactory.network_inspectors.loaders import get_torch_mlp
from lanfactory.network_inspectors.parameters import (
    make_parameter_df,
    sweep_bounds,
)
from lanfactory.network_inspectors.plotting import (
    build_kde_vs_lan_figure,
    build_manifold_figure,
)


def _load_stylesheet() -> str:
    """Load Streamlit CSS from package data with local-file fallback."""
    try:
        return (
            files("lanfactory.network_inspectors")
            .joinpath("styles.css")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return (
            Path(__file__).resolve().with_name("styles.css").read_text(encoding="utf-8")
        )


def _load_predictor(
    base_dir: str, model: str, directories: dict[str, Path] | None = None
):
    """Load the predictor for the newest run, re-resolving artifacts every call.

    Resolution happens outside the resource cache and the cache key includes
    the artifact paths and modification times, so a newly finished training run
    (or a replaced checkpoint) is picked up instead of serving stale weights.
    ``directories`` reuses this rerun's discovery scan.
    """
    state_dict_path, network_config_path = resolve_model_paths(
        base_dir, model, directories
    )
    return _load_predictor_cached(
        model=model,
        state_dict_path=str(state_dict_path),
        network_config_path=str(network_config_path),
        artifact_mtimes=(
            state_dict_path.stat().st_mtime_ns,
            network_config_path.stat().st_mtime_ns,
        ),
    )


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_predictor_cached(
    model: str,
    state_dict_path: str,
    network_config_path: str,
    artifact_mtimes: tuple[int, int],
):
    del artifact_mtimes  # only part of the cache key
    spec = ModelSpec.from_model(model)
    input_dim = len(spec.params) + 2
    return get_torch_mlp(
        model_file_path=state_dict_path,
        network_config=network_config_path,
        input_dim=input_dim,
    )


def _default_base_dir() -> str:
    candidates = [Path("data/torch_models"), Path("../data/torch_models")]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "data/torch_models"


def _kde_tab(model: str, predictor) -> None:
    st.subheader("KDE vs LAN Likelihoods")
    seed = st.number_input(**ui.SEED.kwargs())
    parameter_col, samples_col, reps_col = st.columns(3)
    n_parameter_sets = parameter_col.slider(**ui.PARAMETER_SETS.kwargs())
    n_samples = samples_col.slider(**ui.SIMULATOR_SAMPLES.kwargs())
    n_reps = reps_col.slider(**ui.KDE_REPETITIONS.kwargs())

    grid_col, plot_col = st.columns(2)
    with grid_col, grid_col.expander("KDE Grid", expanded=True):
        n_points_2c = st.slider(**ui.KDE_GRID_POINTS.kwargs())
        rt_step_2c = st.number_input(**ui.KDE_GRID_STEP.kwargs())

    with plot_col, plot_col.expander("Plot", expanded=True):
        cols = st.slider(**ui.KDE_PLOT_COLUMNS.kwargs())
        alpha = st.slider(**ui.KDE_ALPHA.kwargs())
        font_scale = st.slider(**ui.KDE_FONT_SCALE.kwargs())

    grid_spec = GridSpec(
        n_points_2c=n_points_2c,
        rt_step_2c=float(rt_step_2c),
    )
    n_rows = -(-n_parameter_sets // cols)  # ceil division
    plot_cfg = PlotConfig(
        show=False,
        save=False,
        cols=cols,
        alpha=alpha,
        font_scale=font_scale,
        figsize=(
            ui.KDE_FIGURE_WIDTH_IN,
            n_rows * ui.KDE_PANEL_ROW_HEIGHT_IN + ui.KDE_TITLE_HEIGHT_IN,
        ),
    )

    parameter_df = make_parameter_df(model=model, n_rows=n_parameter_sets, seed=seed)
    st.caption("Generated parameter vectors")
    st.dataframe(parameter_df, width="content")

    if st.button("Run KDE vs LAN", type="primary", width="stretch"):
        with st.spinner("Computing likelihoods..."):
            comparison = compute_kde_vs_lan_likelihoods(
                parameter_df=parameter_df,
                model=model,
                torch_mlp_predict=predictor,
                n_samples=n_samples,
                n_reps=n_reps,
                grid=grid_spec,
            )
            fig = build_kde_vs_lan_figure(comparison, plot_cfg)

        st.caption(
            "KDE vs LAN chart. X-axis is signed reaction time and y-axis is "
            "likelihood. Gray curves are KDE estimates and indigo curves are LAN outputs."
        )
        try:
            st.pyplot(fig, clear_figure=True, width=ui.KDE_FIGURE_DISPLAY_WIDTH_PX)
        finally:
            plt.close(fig)


def _manifold_tab(model: str, predictor) -> None:
    st.subheader("LAN Manifold")
    params = ssms.config.model_config[model]["params"]
    defaults = ssms.config.model_config[model]["default_params"]

    with st.expander("Manifold Grid", expanded=True):
        n_rt_steps = st.slider(**ui.MANIFOLD_RT_STEPS.kwargs())
        max_rt = st.slider(**ui.MANIFOLD_MAX_RT.kwargs())

    # Base vector and swept parameter come first so the sweep range below can
    # resolve dependent bounds (full_ddm_rv: st <= t) against the edited values.
    st.caption("Base parameter vector")
    base_parameter_df = pd.DataFrame([defaults], columns=params)
    edited_df = st.data_editor(
        base_parameter_df,
        num_rows="fixed",
        width="content",
    )
    vary_param = st.selectbox(
        **ui.SWEEP_PARAMETER.kwargs(options=params, key="manifold_vary_param")
    )
    base_values = {param: float(edited_df[param].iloc[0]) for param in params}
    p_min, p_max = sweep_bounds(model, vary_param, base_values)

    col_a, col_b, col_c = st.columns(3)
    sweep_min = col_a.number_input(**ui.SWEEP_MIN.kwargs(value=p_min))
    sweep_max = col_b.number_input(**ui.SWEEP_MAX.kwargs(value=p_max))
    sweep_steps = col_c.slider(**ui.SWEEP_STEPS.kwargs())

    run_manifold = st.button("Run Manifold", type="primary", width="stretch")

    grid_spec = GridSpec(n_rt_steps=n_rt_steps, max_rt=max_rt)
    plot_cfg = PlotConfig(show=False, save=False)

    if run_manifold:
        if sweep_max <= sweep_min:
            st.error("Sweep max must be larger than sweep min.")
            return

        vary_values = np.linspace(sweep_min, sweep_max, sweep_steps)
        try:
            with st.spinner("Computing manifold..."):
                computation = compute_lan_manifold(
                    parameter_df=edited_df,
                    vary_dict={vary_param: vary_values},
                    model=model,
                    torch_mlp_predict=predictor,
                    grid=grid_spec,
                )
                fig = build_manifold_figure(computation, plot_cfg)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.caption(
            "3D manifold chart. X-axis is signed reaction time, y-axis is the swept "
            "parameter value, and z-axis is likelihood."
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Computed manifold table")
        st.dataframe(computation.manifold, width="content")


def run() -> None:
    """Render the Streamlit app."""
    st.set_page_config(
        page_title="LANfactory Network Inspectors",
        page_icon="LAN",
        layout="wide",
    )
    st.title("LANfactory Network Inspectors")
    st.write(
        "Inspect trained LAN likelihood behavior with KDE comparisons and manifold plots."
    )
    st.caption(
        "Network Inspectors currently support Torch models only. "
        "Each model needs a state dict (.pt) and network config (.pickle)."
    )
    # st.caption(
    #     "Accessibility: all controls include visible labels, keyboard focus outlines, "
    #     "and descriptive helper text."
    # )
    css = _load_stylesheet()
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)

    all_models = sorted(ssms.config.model_config.keys())

    with st.sidebar, st.expander("Model Selection", expanded=True):
        if "torch_models_base_dir" not in st.session_state:
            st.session_state["torch_models_base_dir"] = _default_base_dir()

        base_dir = st.text_input(**ui.BASE_DIR.kwargs(key="torch_models_base_dir"))
        # One scan per rerun, shared by model selection and loading; the next
        # rerun scans again so new checkpoints are still picked up.
        directories = model_directories(base_dir)
        detected_models = sorted(directories)

        if detected_models:
            default_model = "ddm" if "ddm" in detected_models else detected_models[0]
            model = st.selectbox(
                **ui.MODEL_DETECTED.kwargs(
                    options=detected_models,
                    index=detected_models.index(default_model),
                )
            )
            st.caption("Models detected on disk: " + ", ".join(detected_models))
        else:
            default_model = "ddm" if "ddm" in all_models else all_models[0]
            model = st.selectbox(
                **ui.MODEL_ANY.kwargs(
                    options=all_models,
                    index=all_models.index(default_model),
                )
            )
            st.warning(
                "No valid model folders found in the selected base directory. "
                "Set a folder containing per-model subdirectories with both "
                "*state_dict* and *network_config* files."
            )

    try:
        predictor = _load_predictor(base_dir, model, directories)
    except Exception as exc:  # pragma: no cover - Streamlit interactive path
        st.error(str(exc))
        st.stop()

    kde_view, manifold_view = st.tabs(["KDE vs LAN", "3D Manifold"])

    with kde_view:
        _kde_tab(model, predictor)

    with manifold_view:
        _manifold_tab(model, predictor)


if __name__ == "__main__":
    run()
