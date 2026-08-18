"""Streamlit UI for LAN network inspector workflows."""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path

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

from lanfactory.network_inspectors.api import (
    compute_kde_vs_lan_likelihoods,
    compute_lan_manifold,
)
from lanfactory.network_inspectors.config import (
    GridSpec,
    ModelSpec,
    PlotConfig,
)
from lanfactory.network_inspectors.loaders import get_torch_mlp
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


def _available_models(base_dir: str) -> list[str]:
    return sorted(_model_directories(base_dir))


def _model_directories(base_dir: str) -> dict[str, Path]:
    model_root = Path(base_dir).expanduser()
    if not model_root.exists() or not model_root.is_dir():
        return {}

    valid_ssms_models = set(ssms.config.model_config.keys())
    model_directories: dict[str, Path] = {}
    for model_dir in sorted(path for path in model_root.rglob("*") if path.is_dir()):
        if model_dir.name not in valid_ssms_models:
            continue
        has_state = any(model_dir.glob("*state_dict*"))
        has_cfg = any(model_dir.glob("*network_config*"))
        if has_state and has_cfg:
            model_directories.setdefault(model_dir.name, model_dir)

    return model_directories


def _resolve_model_paths(base_dir: str, model: str) -> tuple[Path, Path]:
    model_dir = _model_directories(base_dir).get(model)
    if model_dir is None:
        available = _available_models(base_dir)
        available_str = ", ".join(available) if available else "none detected"
        raise FileNotFoundError(
            f"Model directory not found for '{model}' under {base_dir}. "
            "Set the base directory to your torch_models folder. "
            f"Detected models in this base dir: {available_str}."
        )

    state_candidates = sorted(model_dir.glob("*state_dict*"))
    config_candidates = sorted(model_dir.glob("*network_config*"))

    if not state_candidates:
        raise FileNotFoundError(f"No state dict file found in {model_dir}.")
    if not config_candidates:
        raise FileNotFoundError(f"No network config file found in {model_dir}.")

    return state_candidates[0], config_candidates[0]


@st.cache_resource(show_spinner=False)
def _load_predictor(base_dir: str, model: str):
    spec = ModelSpec.from_model(model)
    state_dict_path, network_config_path = _resolve_model_paths(base_dir, model)
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


def _make_parameter_df(model: str, n_rows: int, seed: int) -> pd.DataFrame:
    params = ssms.config.model_config[model]["params"]
    lb, ub = ssms.config.model_config[model]["param_bounds"]
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(lb, ub, size=(n_rows, len(params))), columns=params)


def _kde_tab(model: str, predictor) -> None:
    st.subheader("KDE vs LAN Likelihoods")
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2_000_000,
        value=123,
        width=160,
        help="Controls reproducible generation of parameter vectors.",
    )
    parameter_col, samples_col, reps_col = st.columns(3)
    n_parameter_sets = parameter_col.slider(
        "Parameter sets",
        min_value=1,
        max_value=20,
        value=6,
        help="Number of parameter vectors sampled from the model bounds.",
    )
    n_samples = samples_col.slider(
        "Simulator samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=200,
        help="Number of samples used per simulated dataset.",
    )
    n_reps = reps_col.slider(
        "KDE repetitions",
        min_value=1,
        max_value=30,
        value=8,
        help="How many KDE estimates are overlaid for each parameter set.",
    )

    grid_col, plot_col = st.columns(2)
    with grid_col, grid_col.expander("KDE Grid", expanded=True):
        n_points_2c = st.slider(
            "KDE grid points (2-choice)",
            min_value=200,
            max_value=5000,
            value=2000,
            help="Number of reaction-time grid points used for KDE/LAN comparison.",
        )
        rt_step_2c = st.number_input(
            "KDE grid step (2-choice)",
            value=0.0025,
            help="Reaction-time spacing in the KDE/LAN evaluation grid.",
        )

    with plot_col, plot_col.expander("Plot", expanded=True):
        cols = st.slider(
            "KDE plot columns",
            min_value=1,
            max_value=6,
            value=3,
            help="Number of subplot columns for KDE/LAN comparison charts.",
        )
        alpha = st.slider(
            "KDE alpha",
            min_value=0.01,
            max_value=0.8,
            value=0.1,
            help="Opacity of KDE overlay lines.",
        )
        font_scale = st.slider(
            "KDE font scale",
            min_value=0.8,
            max_value=2.5,
            value=1.3,
            help="Scale factor for Seaborn text in the KDE chart.",
        )

    grid_spec = GridSpec(
        n_points_2c=n_points_2c,
        rt_step_2c=float(rt_step_2c),
    )
    plot_cfg = PlotConfig(
        show=False, save=False, cols=cols, alpha=alpha, font_scale=font_scale
    )

    parameter_df = _make_parameter_df(model=model, n_rows=n_parameter_sets, seed=seed)
    st.caption("Generated parameter vectors")
    st.dataframe(parameter_df, width="content")

    if st.button("Run KDE vs LAN", use_container_width=True):
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
            "likelihood. Black curves are KDE estimates and green curves are LAN outputs."
        )
        st.pyplot(fig, clear_figure=True, use_container_width=True)


def _manifold_tab(model: str, predictor) -> None:
    st.subheader("LAN Manifold")
    params = ssms.config.model_config[model]["params"]
    defaults = ssms.config.model_config[model]["default_params"]
    lb, ub = ssms.config.model_config[model]["param_bounds"]

    selected_vary_param = st.session_state.get("manifold_vary_param", params[0])
    if selected_vary_param not in params:
        selected_vary_param = params[0]
    vary_idx = params.index(selected_vary_param)
    p_min = float(lb[vary_idx])
    p_max = float(ub[vary_idx])

    with st.expander("Manifold Grid", expanded=True):
        n_rt_steps = st.slider(
            "Manifold RT steps",
            min_value=50,
            max_value=800,
            value=300,
            help="Number of reaction-time points for manifold evaluation.",
        )
        max_rt = st.slider(
            "Manifold max RT",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            help="Maximum reaction time represented in manifold plots.",
        )

    col_a, col_b, col_c = st.columns(3)
    sweep_min = col_a.number_input(
        "Sweep min",
        value=p_min,
        help="Minimum value in the parameter sweep range.",
    )
    sweep_max = col_b.number_input(
        "Sweep max",
        value=p_max,
        help="Maximum value in the parameter sweep range.",
    )
    sweep_steps = col_c.slider(
        "Sweep steps",
        min_value=5,
        max_value=80,
        value=20,
        help="Number of points between sweep min and sweep max.",
    )

    run_manifold = st.button("Run Manifold", use_container_width=True)

    st.caption("Base parameter vector")
    base_parameter_df = pd.DataFrame([defaults], columns=params)
    edited_df = st.data_editor(
        base_parameter_df,
        num_rows="fixed",
        width="content",
    )
    vary_param = st.selectbox(
        "Parameter to sweep",
        options=params,
        index=params.index(selected_vary_param),
        key="manifold_vary_param",
        help="Choose one parameter to vary while others remain fixed.",
    )

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

        base_dir = st.text_input(
            "Torch models base directory",
            key="torch_models_base_dir",
            help="Folder containing model subfolders, each with state_dict and network_config files.",
        )
        available_models = _available_models(base_dir)

        if available_models:
            default_model = "ddm" if "ddm" in available_models else available_models[0]
            model = st.selectbox(
                "Model",
                options=available_models,
                index=available_models.index(default_model),
                help="Choose a model that exists in the selected torch models directory.",
            )
            st.caption("Models detected on disk: " + ", ".join(available_models))
        else:
            default_model = "ddm" if "ddm" in all_models else all_models[0]
            model = st.selectbox(
                "Model",
                options=all_models,
                index=all_models.index(default_model),
                help="Choose a model name. You still need matching files on disk.",
            )
            st.warning(
                "No valid model folders found in the selected base directory. "
                "Set a folder containing per-model subdirectories with both "
                "*state_dict* and *network_config* files."
            )

    try:
        predictor = _load_predictor(base_dir, model)
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
