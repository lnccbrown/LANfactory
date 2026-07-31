"""Tests for network inspector plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lanfactory.network_inspectors.config import ModelSpec, PlotConfig
from lanfactory.network_inspectors.contracts import (
    LikelihoodComparison,
    LikelihoodRow,
    ManifoldComputation,
)
from lanfactory.network_inspectors.plotting import (
    build_kde_vs_lan_figure,
    build_manifold_figure,
    plot_manifold,
)
from matplotlib.figure import Figure


def test_build_kde_vs_lan_figure_returns_matplotlib_figure():
    grid = np.array([[-1.0, -1.0], [1.0, 1.0]])
    rows = [
        LikelihoodRow(
            params=np.array([0.2, 1.5, 0.5, 0.3], dtype=np.float32),
            lan=np.array([0.4, 0.6]),
            kdes=[np.array([0.3, 0.7])],
        )
    ]
    comparison = LikelihoodComparison(
        spec=ModelSpec(name="ddm", params=["v", "a", "z", "t"], choices=[-1, 1]),
        grid=grid,
        rows=rows,
    )

    fig = build_kde_vs_lan_figure(comparison, PlotConfig(show=False, save=False))

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_manifold_returns_interactive_plotly_figure(tmp_path):
    """The manifold renderer should produce an interactive Plotly surface."""
    manifold = pd.DataFrame(
        {
            "rt": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "choice": [-1, -1, 1, 1, -1, -1, 1, 1],
            "vary": [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
            "likelihood": [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5],
        }
    )
    spec = ModelSpec(name="ddm", params=["v"], choices=[-1, 1])
    cfg = PlotConfig(show=False, save=True, save_dir=str(tmp_path))
    computation = ManifoldComputation(
        spec=spec,
        vary_name="v",
        vary_values=np.array([0.1, 0.2]),
        grid=np.array([[1.0, -1.0], [2.0, -1.0], [1.0, 1.0], [2.0, 1.0]]),
        manifold=manifold,
    )

    fig = plot_manifold(computation, cfg)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "surface"
    np.testing.assert_array_equal(fig.data[0].x, [-2.0, -1.0, 1.0, 2.0])
    np.testing.assert_array_equal(fig.data[0].y, [0.1, 0.2])
    np.testing.assert_array_equal(
        fig.data[0].z,
        [[0.2, 0.1, 0.3, 0.4], [0.3, 0.2, 0.4, 0.5]],
    )
    assert (tmp_path / "mlp_manifold_ddm.html").exists()


def test_build_manifold_figure_returns_interactive_plotly_figure():
    manifold = pd.DataFrame(
        {
            "rt": [1.0, 2.0, 1.0, 2.0],
            "choice": [-1, -1, 1, 1],
            "vary": [0.1, 0.1, 0.1, 0.1],
            "likelihood": [0.1, 0.2, 0.3, 0.4],
        }
    )
    computation = ManifoldComputation(
        spec=ModelSpec(name="ddm", params=["v"], choices=[-1, 1]),
        vary_name="v",
        vary_values=np.array([0.1]),
        grid=np.array([[1.0, -1.0], [2.0, -1.0], [1.0, 1.0], [2.0, 1.0]]),
        manifold=manifold,
    )

    fig = build_manifold_figure(computation, PlotConfig(show=False, save=False))

    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "surface"
