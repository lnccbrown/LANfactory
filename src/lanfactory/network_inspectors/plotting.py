"""Rendering: KDE-vs-LAN comparison and 3D LAN likelihood manifold."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from numpy.typing import NDArray

from .config import ModelSpec, PlotConfig

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class LikelihoodResult(TypedDict):
    """Likelihood arrays for one parameter vector."""

    lan: NDArray[np.float64]
    kdes: list[NDArray[np.float64]]
    params: NDArray[np.float32]


def _panel_title(
    i: int, res: LikelihoodResult, spec: ModelSpec, cfg: PlotConfig
) -> str:
    if cfg.panel_titles == "index":
        return str(i)
    if cfg.panel_titles == "params":
        return ", ".join(
            f"{name}={value:.3g}" for name, value in zip(spec.params, res["params"])
        )
    return cfg.panel_titles[i]


def plot_kde_vs_lan(
    grid: NDArray[np.float64],
    results: list[LikelihoodResult],
    spec: ModelSpec,
    cfg: PlotConfig,
) -> Figure:
    """Render the KDE-vs-LAN comparison from precomputed likelihoods.

    results: list of {"lan": array, "kdes": [arrays], "params": array}, one per
    parameter vector. Returns the figure, closed if cfg.show displayed it.
    """
    rows = int(np.ceil(len(results) / cfg.cols))
    per_choice = grid.shape[0] // spec.n_choices
    title = cfg.title
    if title == "auto":
        title = "Likelihoods KDE vs. LAN: " + spec.name.upper().replace("_", "-")

    # Style contexts, not sns.set(): the caller's later figures keep their style.
    with (
        sns.axes_style("white"),
        sns.plotting_context("notebook", font_scale=cfg.font_scale),
    ):
        fig, ax = plt.subplots(
            rows,
            cfg.cols,
            figsize=cfg.figsize,
            sharex=True,
            sharey=False,
            squeeze=False,
            layout=cfg.layout,
        )

        if title is not None:
            fig.suptitle(title, fontsize=cfg.title_fontsize)
        sns.despine(right=True)

        for i, res in enumerate(results):
            logger.info("%d of %d", i + 1, len(results))

            row_tmp = i // cfg.cols
            col_tmp = i - (cfg.cols * row_tmp)

            for j, kde_like in enumerate(res["kdes"]):
                if j == 0:
                    label = "KDE"
                else:
                    label = None

                if spec.n_choices == 2:
                    sns.lineplot(
                        x=grid[:, 0] * grid[:, 1],
                        y=kde_like,
                        color="black",
                        alpha=cfg.alpha,
                        label=label,
                        ax=ax[row_tmp, col_tmp],
                    )
                else:
                    for k in range(spec.n_choices):
                        if k > 0:
                            label = None
                        sns.lineplot(
                            x=grid[per_choice * k : per_choice * (k + 1), 0],
                            y=kde_like[per_choice * k : per_choice * (k + 1)],
                            color="black",
                            alpha=cfg.alpha,
                            label=label,
                            ax=ax[row_tmp, col_tmp],
                        )

            lan_like = res["lan"]
            if spec.n_choices == 2:
                sns.lineplot(
                    x=grid[:, 0] * grid[:, 1],
                    y=lan_like,
                    color="green",
                    label="MLP",
                    alpha=1,
                    ax=ax[row_tmp, col_tmp],
                )
            else:
                for k in range(spec.n_choices):
                    if k == 0:
                        label = "MLP"
                    else:
                        label = None

                    sns.lineplot(
                        x=grid[per_choice * k : per_choice * (k + 1), 0],
                        y=lan_like[per_choice * k : per_choice * (k + 1)],
                        color="green",
                        label=label,
                        alpha=1,
                        ax=ax[row_tmp, col_tmp],
                    )

            if row_tmp == 0 and col_tmp == 0:
                ax[row_tmp, col_tmp].legend(
                    loc="upper left",
                    fancybox=True,
                    shadow=True,
                    fontsize=cfg.legend_fontsize,
                )
            else:
                ax[row_tmp, col_tmp].legend().set_visible(False)

            if row_tmp == rows - 1:
                ax[row_tmp, col_tmp].set_xlabel("rt", fontsize=cfg.xlabel_fontsize)
            else:
                ax[row_tmp, col_tmp].tick_params(color="white")

            if col_tmp == 0:
                ax[row_tmp, col_tmp].set_ylabel(
                    "likelihood", fontsize=cfg.ylabel_fontsize
                )

            ax[row_tmp, col_tmp].set_title(
                _panel_title(i, res, spec, cfg), fontsize=cfg.panel_title_fontsize
            )
            ax[row_tmp, col_tmp].tick_params(axis="y", size=14)
            ax[row_tmp, col_tmp].tick_params(axis="x", size=14)

        for i in range(len(results), rows * cfg.cols, 1):
            row_tmp = i // cfg.cols
            col_tmp = i - (cfg.cols * row_tmp)
            ax[row_tmp, col_tmp].axis("off")

        if cfg.layout is None:
            fig.subplots_adjust(top=0.9)
            fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # Matplotlib reads the font and the tick-label size when it draws. Pin
        # what this style resolves to, so the returned figure renders the same
        # after the context ends.
        font = FontProperties().get_name()
        for text in fig.findobj(Text):
            text.set_fontfamily(font)
        for axes in fig.axes:
            axes.tick_params(labelsize=plt.rcParams["xtick.labelsize"])

        if cfg.save:
            os.makedirs(cfg.save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(cfg.save_dir, cfg.filename),
                dpi=cfg.dpi,
                transparent=False,
            )

        if cfg.show:
            plt.show()
            plt.close(fig)

    return fig


def plot_manifold(
    manifold: pd.DataFrame, spec: ModelSpec, vary_name: str, cfg: PlotConfig
) -> go.Figure:
    """Render an interactive 3D LAN likelihood manifold."""
    plot_data = manifold.assign(signed_rt=manifold["rt"] * manifold["choice"])
    surface = (
        plot_data.pivot(index="vary", columns="signed_rt", values="likelihood")
        .sort_index()
        .sort_index(axis=1)
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=surface.columns.to_numpy(dtype=float),
                y=surface.index.to_numpy(dtype=float),
                z=surface.to_numpy(dtype=float),
                colorscale="RdBu",
                colorbar={"title": "Likelihood"},
            )
        ]
    )
    fig.update_layout(
        title=spec.name.upper().replace("_", "-") + " - MLP: Manifold",
        width=int(900 * cfg.fig_scale),
        height=int(620 * cfg.fig_scale),
        scene={
            "xaxis_title": "Signed RT",
            "yaxis_title": vary_name.upper().replace("_", "-"),
            "zaxis_title": "Likelihood",
        },
    )

    if cfg.save:
        os.makedirs(cfg.save_dir, exist_ok=True)
        fig.write_html(
            os.path.join(cfg.save_dir, "mlp_manifold_" + spec.name + ".html"),
            include_plotlyjs=True,
        )

    if cfg.show:
        fig.show()

    return fig
