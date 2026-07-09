"""Pure computation: RT grids, LAN/KDE likelihoods, and manifold data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ssms.basic_simulators.simulator import simulator
from ssms.support_utils.kde_class import LogKDE

from .config import GridSpec


def _symmetric_2choice_grid(n, step):
    """(rt, choice) grid: n points per side at rt = i*step, choices -1 and +1."""
    plot_data = np.zeros((2 * n, 2))
    plot_data[:, 0] = np.concatenate(
        (
            [i * step for i in range(n, 0, -1)],
            [i * step for i in range(1, n + 1, 1)],
        )
    )
    plot_data[:, 1] = np.concatenate((np.repeat(-1, n), np.repeat(1, n)))
    return plot_data


def make_rt_choice_grid(spec, grid_spec=None):
    """Build the (rt, choice) evaluation grid for kde_vs_lan_likelihoods."""
    grid_spec = grid_spec or GridSpec()
    if spec.n_choices == 2:
        return _symmetric_2choice_grid(grid_spec.n_points_2c, grid_spec.rt_step_2c)

    n, step = grid_spec.n_points_mc, grid_spec.rt_step_mc
    plot_data = np.zeros((spec.n_choices * n, 2))
    plot_data[:, 0] = np.concatenate(
        [[i * step for i in range(1, n + 1, 1)] for _ in range(spec.n_choices)]
    )
    plot_data[:, 1] = np.concatenate([np.repeat(i, n) for i in range(spec.n_choices)])
    return plot_data


def make_manifold_grid(grid_spec):
    """Build the (rt, choice) grid for the 2-choice manifold plot."""
    return _symmetric_2choice_grid(
        grid_spec.n_rt_steps, grid_spec.max_rt / grid_spec.n_rt_steps
    )


def evaluate_network(spec, params, grid):
    """LAN log-likelihood over ``grid`` for a single parameter vector.

    Builds the torch input batch (parameter vector trailed by rt and choice)
    and returns the per-row log-likelihood from ``predict_on_batch``.
    """
    params = np.asarray(params, dtype=np.float32)
    input_batch = np.zeros((grid.shape[0], spec.n_params + 2))
    input_batch[:, : spec.n_params] = params
    input_batch[:, spec.n_params :] = grid
    ll_out = spec.predictor(input_batch.astype(np.float32))
    return np.asarray(ll_out)[:, 0]


def simulate_ground_truth(spec, params, n_samples, max_t=20, delta_t=0.001):
    """Simulate data via ssms; the returned dict is what LogKDE expects."""
    return simulator(
        theta=np.asarray(params),
        model=spec.name,
        n_samples=n_samples,
        max_t=max_t,
        delta_t=delta_t,
    )


def evaluate_kde(sim_out, grid):
    """KDE log-likelihood over ``grid`` from one simulation output."""
    mykde = LogKDE(sim_out)
    return mykde.kde_eval({"rts": grid[:, 0], "choices": grid[:, 1]})


def build_manifold(spec, base_params, vary_name, vary_values, grid):
    """LAN likelihoods over a grid while sweeping one param (tidy DataFrame)."""
    parameters = np.asarray(base_params, dtype=np.float32).copy()
    idx = spec.params.index(vary_name)

    blocks = []
    for par_tmp in np.asarray(vary_values):
        parameters[idx] = par_tmp
        like = np.exp(evaluate_network(spec, parameters, grid))
        blocks.append(
            pd.DataFrame(
                {
                    "rt": grid[:, 0],
                    "choice": grid[:, 1],
                    "vary": par_tmp,
                    "likelihood": like,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)
