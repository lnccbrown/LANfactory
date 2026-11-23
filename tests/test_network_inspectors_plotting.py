"""Tests for network inspector plotting helpers."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import seaborn as sns
from matplotlib.figure import Figure

from lanfactory.network_inspectors.config import ModelSpec, PlotConfig
from lanfactory.network_inspectors.plotting import plot_kde_vs_lan, plot_manifold


@pytest.fixture(autouse=True)
def _headless_figures():
    """Draw without a display and leave no figure open for the next test."""
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def _kde_vs_lan_inputs(choices=(-1, 1), n_panels=2):
    """Two flat likelihood curves per panel on a five-point rt grid per choice."""
    spec = ModelSpec(name="ddm", params=["v", "a"], choices=list(choices))
    grid = np.column_stack(
        [np.tile(np.linspace(0.1, 1.0, 5), len(choices)), np.repeat(choices, 5)]
    )
    results = [
        {
            "lan": np.full(len(grid), 0.5),
            "kdes": [np.full(len(grid), 0.4), np.full(len(grid), 0.6)],
            "params": np.array([0.5 + i, 1.25], dtype=np.float32),
        }
        for i in range(n_panels)
    ]
    return grid, results, spec


def _panel_titles(fig):
    return [ax.get_title() for ax in fig.axes if ax.axison]


@pytest.mark.parametrize("choices", [(-1, 1), (0, 1, 2)])
def test_kde_vs_lan_defaults_draw_the_original_figure(tmp_path, choices):
    """With an untouched PlotConfig the text, sizes and file name do not change."""
    grid, results, spec = _kde_vs_lan_inputs(choices)
    cfg = PlotConfig(show=False, save=True, save_dir=str(tmp_path))

    fig = plot_kde_vs_lan(grid, results, spec, cfg)

    assert isinstance(fig, Figure)
    assert fig.get_suptitle() == "Likelihoods KDE vs. LAN: DDM"
    assert fig.texts[0].get_fontsize() == 30
    assert _panel_titles(fig) == ["0", "1"]
    first = fig.axes[0]
    assert first.title.get_fontsize() == 20
    assert first.xaxis.label.get_fontsize() == 24
    assert first.yaxis.label.get_fontsize() == 20
    legend_texts = first.get_legend().get_texts()
    assert [text.get_text() for text in legend_texts] == ["KDE", "MLP"]
    assert legend_texts[0].get_fontsize() == 12
    assert (tmp_path / "kde_vs_mlp_plot.png").exists()


def test_kde_vs_lan_takes_titles_and_font_sizes_from_the_config():
    grid, results, spec = _kde_vs_lan_inputs()
    cfg = PlotConfig(
        show=False,
        title="Network check",
        panel_titles=["decay 0", "decay 2"],
        title_fontsize=14,
        panel_title_fontsize=9,
        xlabel_fontsize=8,
        ylabel_fontsize=7,
        legend_fontsize=6,
    )

    fig = plot_kde_vs_lan(grid, results, spec, cfg)

    assert fig.get_suptitle() == "Network check"
    assert fig.texts[0].get_fontsize() == 14
    assert _panel_titles(fig) == ["decay 0", "decay 2"]
    first = fig.axes[0]
    assert first.title.get_fontsize() == 9
    assert first.xaxis.label.get_fontsize() == 8
    assert first.yaxis.label.get_fontsize() == 7
    assert first.get_legend().get_texts()[0].get_fontsize() == 6


def test_kde_vs_lan_title_none_draws_no_figure_title():
    grid, results, spec = _kde_vs_lan_inputs()

    fig = plot_kde_vs_lan(grid, results, spec, PlotConfig(show=False, title=None))

    assert fig.get_suptitle() == ""
    assert fig.texts == []


def test_kde_vs_lan_panel_titles_from_parameter_values():
    grid, results, spec = _kde_vs_lan_inputs()
    cfg = PlotConfig(show=False, panel_titles="params")

    fig = plot_kde_vs_lan(grid, results, spec, cfg)

    assert _panel_titles(fig) == ["v=0.5, a=1.25", "v=1.5, a=1.25"]


def test_plot_config_rejects_unknown_panel_title_mode():
    with pytest.raises(ValueError, match="'index', 'params' or a sequence"):
        PlotConfig(panel_titles="values")


def test_kde_vs_lan_saves_under_the_configured_name_and_dpi(tmp_path):
    grid, results, spec = _kde_vs_lan_inputs()
    cfg = PlotConfig(
        show=False,
        save=True,
        save_dir=str(tmp_path),
        figsize=(4, 2),
        filename="network_check.png",
        dpi=50,
    )

    plot_kde_vs_lan(grid, results, spec, cfg)

    # 4 x 2 inches at 50 dpi
    assert plt.imread(tmp_path / "network_check.png").shape[:2] == (100, 200)
    assert not (tmp_path / "kde_vs_mlp_plot.png").exists()


def test_kde_vs_lan_file_extension_picks_the_format(tmp_path):
    grid, results, spec = _kde_vs_lan_inputs()
    cfg = PlotConfig(
        show=False, save=True, save_dir=str(tmp_path), filename="network_check.pdf"
    )

    plot_kde_vs_lan(grid, results, spec, cfg)

    assert (tmp_path / "network_check.pdf").read_bytes().startswith(b"%PDF")


def test_kde_vs_lan_leaves_the_global_style_alone():
    """sns.set() used to restyle every later figure and remap the 'g' colour."""
    grid, results, spec = _kde_vs_lan_inputs()
    # Start from a style sns.set() would overwrite, so the check does not
    # depend on what earlier tests left behind.
    sns.set_color_codes("reset")
    callers_style = {"axes.facecolor": "red", "font.size": 7.0}

    with plt.rc_context(callers_style):
        rc_before = dict(mpl.rcParams)
        plot_kde_vs_lan(grid, results, spec, PlotConfig(show=False))
        assert dict(mpl.rcParams) == rc_before

    assert mpl.colors.to_rgb("g") == (0.0, 0.5, 0.0)


def test_kde_vs_lan_keeps_the_figure_open_unless_it_showed_it(monkeypatch):
    grid, results, spec = _kde_vs_lan_inputs()
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    kept = plot_kde_vs_lan(grid, results, spec, PlotConfig(show=False))
    assert shown == []
    assert plt.fignum_exists(kept.number)

    displayed = plot_kde_vs_lan(grid, results, spec, PlotConfig(show=True))
    assert shown == [True]
    assert not plt.fignum_exists(displayed.number)
    assert plt.fignum_exists(kept.number)


def test_kde_vs_lan_layout_engine_keeps_text_inside_a_small_figure():
    """Fixed margins clip the x label of a short figure; a layout engine fits it."""
    grid, results, spec = _kde_vs_lan_inputs()

    def text_inside_figure(layout):
        cfg = PlotConfig(show=False, figsize=(6, 2), layout=layout)
        fig = plot_kde_vs_lan(grid, results, spec, cfg)
        fig.canvas.draw()
        labels = [ax.xaxis.label for ax in fig.axes] + fig.texts
        return all(
            fig.bbox.contains(*corner)
            for label in labels
            for corner in label.get_window_extent().get_points()
        )

    assert not text_inside_figure(None)
    assert text_inside_figure("constrained")


@pytest.mark.parametrize("layout", [None, "constrained"])
def test_kde_vs_lan_returned_figure_renders_like_the_saved_file(tmp_path, layout):
    """The caller's own style must not change the returned figure.

    Matplotlib reads the font and the tick-label size when it draws, so a
    figure built under a temporary style can come out different later.
    """
    grid, results, spec = _kde_vs_lan_inputs()
    cfg = PlotConfig(show=False, save=True, save_dir=str(tmp_path), layout=layout)

    fig = plot_kde_vs_lan(grid, results, spec, cfg)
    other_style = {"font.family": "serif", "xtick.labelsize": 4, "ytick.labelsize": 4}
    with plt.rc_context(other_style):
        fig.savefig(tmp_path / "saved_by_caller.png")

    np.testing.assert_array_equal(
        plt.imread(tmp_path / "saved_by_caller.png"),
        plt.imread(tmp_path / "kde_vs_mlp_plot.png"),
    )


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
