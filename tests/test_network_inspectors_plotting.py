"""Tests for network inspector plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lanfactory.network_inspectors.config import ModelSpec, PlotConfig
from lanfactory.network_inspectors.plotting import plot_manifold


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

    fig = plot_manifold(manifold, spec, "v", cfg)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "surface"
    np.testing.assert_array_equal(fig.data[0].x, [-2.0, -1.0, 1.0, 2.0])
    np.testing.assert_array_equal(fig.data[0].y, [0.1, 0.2])
    np.testing.assert_array_equal(
        fig.data[0].z,
        [[0.2, 0.1, 0.3, 0.4], [0.3, 0.2, 0.4, 0.5]],
    )
    assert (tmp_path / "mlp_manifold_ddm.html").exists()
