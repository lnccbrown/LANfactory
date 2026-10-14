"""Rendering: KDE-vs-LAN comparison and 3D LAN likelihood manifold."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from numpy.typing import NDArray

from .config import ModelSpec, PlotConfig

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class LikelihoodResult(TypedDict):
    """Likelihood arrays for one parameter vector."""

    lan: NDArray[np.float64]
    kdes: list[NDArray[np.float64]]


def _save_figure(filename: str, cfg: PlotConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    plt.savefig(os.path.join(cfg.save_dir, filename), format="png", transparent=False)


def plot_kde_vs_lan(
    grid: NDArray[np.float64],
    results: list[LikelihoodResult],
    spec: ModelSpec,
    cfg: PlotConfig,
) -> None:
    """Render the KDE-vs-LAN comparison from precomputed likelihoods.

    results: list of {"lan": array, "kdes": [arrays]}, one per parameter vector.
    """
    rows = int(np.ceil(len(results) / cfg.cols))
    per_choice = grid.shape[0] // spec.n_choices
    sns.set(style="white", palette="muted", color_codes=True, font_scale=cfg.font_scale)

    fig, ax = plt.subplots(
        rows, cfg.cols, figsize=cfg.figsize, sharex=True, sharey=False, squeeze=False
    )

    fig.suptitle(
        "Likelihoods KDE vs. LAN" + ": " + spec.name.upper().replace("_", "-"),
        fontsize=30,
    )
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
                loc="upper left", fancybox=True, shadow=True, fontsize=12
            )
        else:
            ax[row_tmp, col_tmp].legend().set_visible(False)

        if row_tmp == rows - 1:
            ax[row_tmp, col_tmp].set_xlabel("rt", fontsize=24)
        else:
            ax[row_tmp, col_tmp].tick_params(color="white")

        if col_tmp == 0:
            ax[row_tmp, col_tmp].set_ylabel("likelihood", fontsize=20)

        ax[row_tmp, col_tmp].set_title(str(i), fontsize=20)
        ax[row_tmp, col_tmp].tick_params(axis="y", size=14)
        ax[row_tmp, col_tmp].tick_params(axis="x", size=14)

    for i in range(len(results), rows * cfg.cols, 1):
        row_tmp = i // cfg.cols
        col_tmp = i - (cfg.cols * row_tmp)
        ax[row_tmp, col_tmp].axis("off")

    plt.subplots_adjust(top=0.9)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if cfg.save:
        _save_figure("kde_vs_mlp_plot.png", cfg)

    if cfg.show:
        plt.show()

    plt.close()


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
