"""Rendering: KDE-vs-LAN comparison and 3D LAN likelihood manifold."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
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
) -> None:
    """Render a 3D LAN likelihood manifold from a build_manifold frame."""
    signed_rt = (manifold["rt"] * manifold["choice"]).values
    vary = manifold["vary"].values
    like = manifold["likelihood"].values

    fig = plt.figure(figsize=(8 * cfg.fig_scale, 5.5 * cfg.fig_scale))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        signed_rt,
        vary,
        like,
        linewidth=0.5,
        alpha=1.0,
        cmap=cm.coolwarm,
    )

    ax.set_ylabel(vary_name.upper().replace("_", "-"), fontsize=16, labelpad=20)

    ax.set_xlabel("RT", fontsize=16, labelpad=20)

    ax.set_zlabel("Likelihood", fontsize=16, labelpad=20)

    ax.set_zticks(np.round(np.linspace(min(like), max(like), 5), 1))

    ax.set_yticks(np.round(np.linspace(min(vary), max(vary), 5), 1))

    ax.set_xticks(np.round(np.linspace(min(signed_rt), max(signed_rt), 5), 1))

    ax.tick_params(labelsize=16)
    ax.set_title(
        spec.name.upper().replace("_", "-") + " - MLP: Manifold", fontsize=20, pad=20
    )

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if cfg.save:
        _save_figure("mlp_manifold_" + spec.name + ".png", cfg)

    if cfg.show:
        plt.show()

    plt.close()
