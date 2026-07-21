"""Thin entry points wiring config, computation, and plotting together."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .compute import (
    build_manifold,
    evaluate_kde,
    evaluate_network,
    make_manifold_grid,
    make_rt_choice_grid,
    simulate_ground_truth,
)
from .config import GridSpec, ModelSpec, PlotConfig
from .plotting import LikelihoodResult, plot_kde_vs_lan, plot_manifold

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import plotly.graph_objects as go


def kde_vs_lan_likelihoods(
    parameter_df: pd.DataFrame,
    model: str,
    torch_mlp_predict: Callable[[NDArray[np.float32]], Any],
    n_samples: int = 10,
    n_reps: int = 10,
    grid: GridSpec | None = None,
    plot: PlotConfig | None = None,
) -> None:
    """Compare kernel density estimates from simulation data with LAN output.

    parameter_df: one model-compatible parameter vector per row.
    model: model name. torch_mlp_predict: predict_on_batch from get_torch_mlp.
    n_samples/n_reps: samples per KDE / KDEs per subplot.
    grid: optional GridSpec. plot: optional PlotConfig.
    """
    if parameter_df is None or model is None or torch_mlp_predict is None:
        raise ValueError(
            "parameter_df, model, and torch_mlp_predict are required; build the"
            " predictor with get_torch_mlp()."
        )
    if not isinstance(parameter_df, pd.DataFrame):
        raise TypeError("parameter_df must be a pandas.DataFrame.")
    if parameter_df.empty:
        raise ValueError("parameter_df must contain at least one parameter vector.")

    spec = ModelSpec.from_model(model, predictor=torch_mlp_predict)
    missing_params = [param for param in spec.params if param not in parameter_df]
    if missing_params:
        raise ValueError(
            "parameter_df is missing model parameter columns: "
            + ", ".join(missing_params)
        )
    cfg = plot or PlotConfig()
    grid_arr = make_rt_choice_grid(spec, grid)

    results: list[LikelihoodResult] = []
    for i in range(parameter_df.shape[0]):
        params = parameter_df.iloc[i, :][spec.params].values.astype(np.float32)
        lan_like = np.exp(evaluate_network(spec, params, grid_arr))
        kdes = [
            np.exp(
                evaluate_kde(simulate_ground_truth(spec, params, n_samples), grid_arr)
            )
            for _ in range(n_reps)
        ]
        results.append({"lan": lan_like, "kdes": kdes})

    return plot_kde_vs_lan(grid_arr, results, spec, cfg)


def lan_manifold(
    parameter_df: pd.DataFrame | np.ndarray | None = None,
    vary_dict: dict[str, Any] | None = None,
    model: str = "ddm",
    torch_mlp_predict: Callable[[NDArray[np.float32]], Any] | None = None,
    grid: GridSpec | None = None,
    plot: PlotConfig | None = None,
) -> go.Figure:
    """Plot LAN likelihoods as a 3D manifold while sweeping one parameter.

    parameter_df: parameter vector (first row used). vary_dict: {param: values}.
    model: model name. torch_mlp_predict: predict_on_batch from get_torch_mlp.
    grid: optional GridSpec. plot: optional PlotConfig. Returns a Plotly Figure.
    """
    if parameter_df is None or torch_mlp_predict is None:
        raise ValueError(
            "parameter_df and torch_mlp_predict are required; build the predictor"
            " with get_torch_mlp()."
        )
    if vary_dict is None:
        vary_dict = {"v": [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]}
    if len(vary_dict) != 1:
        raise ValueError("vary_dict must contain exactly one parameter.")

    vary_name, raw_values = next(iter(vary_dict.items()))
    vary_values = np.asarray(raw_values)
    if vary_values.size == 0:
        raise ValueError("The parameter sweep must contain at least one value.")

    spec = ModelSpec.from_model(model, predictor=torch_mlp_predict)

    if spec.n_choices != 2:
        raise ValueError(
            "lan_manifold currently supports only 2-choice models; "
            f"got {spec.n_choices} choices."
        )

    if isinstance(parameter_df, pd.DataFrame):
        if parameter_df.empty:
            raise ValueError("parameter_df must contain at least one parameter vector.")
        missing_params = [param for param in spec.params if param not in parameter_df]
        if missing_params:
            raise ValueError(
                "parameter_df is missing model parameter columns: "
                + ", ".join(missing_params)
            )
        if parameter_df.shape[0] > 1:
            logger.info("Using only the first row of the supplied parameter array.")
        parameters = parameter_df.iloc[0, :][spec.params].values.astype(np.float32)
    else:
        parameters = np.asarray(parameter_df, dtype=np.float32)
        if parameters.ndim == 2:
            if parameters.shape[0] == 0:
                raise ValueError("parameter_df must contain at least one row.")
            parameters = parameters[0]
        if parameters.ndim != 1 or parameters.size != spec.n_params:
            raise ValueError(
                f"Expected one parameter vector with {spec.n_params} values."
            )

    grid_arr = make_manifold_grid(grid or GridSpec())
    manifold = build_manifold(spec, parameters, vary_name, vary_values, grid_arr)

    return plot_manifold(manifold, spec, vary_name, plot or PlotConfig())
