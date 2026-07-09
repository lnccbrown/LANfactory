"""Thin entry points wiring config, computation, and plotting together."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .compute import (
    build_manifold,
    evaluate_kde,
    evaluate_network,
    make_manifold_grid,
    make_rt_choice_grid,
    simulate_ground_truth,
)
from .config import GridSpec, ModelSpec, PlotConfig
from .plotting import plot_kde_vs_lan, plot_manifold


def kde_vs_lan_likelihoods(
    parameter_df,
    model,
    torch_mlp_predict,
    n_samples=10,
    n_reps=10,
    grid=None,
    plot=None,
):
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

    spec = ModelSpec.from_model(model, predictor=torch_mlp_predict)
    cfg = plot or PlotConfig()
    grid_arr = make_rt_choice_grid(spec, grid)

    results = []
    for i in range(parameter_df.shape[0]):
        params = parameter_df.iloc[i, :].values
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
    parameter_df=None,
    vary_dict={"v": [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]},
    model="ddm",
    torch_mlp_predict=None,
    grid=None,
    plot=None,
):
    """Plot LAN likelihoods as a 3D manifold while sweeping one parameter.

    parameter_df: parameter vector (first row used). vary_dict: {param: values}.
    model: model name. torch_mlp_predict: predict_on_batch from get_torch_mlp.
    grid: optional GridSpec. plot: optional PlotConfig.
    """
    if parameter_df is None or torch_mlp_predict is None:
        raise ValueError(
            "parameter_df and torch_mlp_predict are required; build the predictor"
            " with get_torch_mlp()."
        )

    spec = ModelSpec.from_model(model, predictor=torch_mlp_predict)

    assert spec.n_choices == 2, (
        "This plot works only for 2-choice models at the moment. Improvements coming!"
    )

    if parameter_df.shape[0] > 0:
        parameters = parameter_df.iloc[0, :]
        print("Using only the first row of the supplied parameter array !")

    if isinstance(parameter_df, pd.DataFrame):
        parameters = np.squeeze(parameters[spec.params].values.astype(np.float32))
    else:
        parameters = parameter_df

    vary_name = list(vary_dict.keys())[0]
    vary_values = np.asarray(vary_dict[vary_name])

    grid_arr = make_manifold_grid(grid or GridSpec())
    manifold = build_manifold(spec, parameters, vary_name, vary_values, grid_arr)

    return plot_manifold(manifold, spec, vary_name, plot or PlotConfig())
