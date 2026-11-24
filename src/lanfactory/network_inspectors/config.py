"""Config objects: model metadata and plotting/grid defaults."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ssms.config import ModelConfigBuilder


@dataclass
class ModelSpec:
    """Model metadata (name, params, choices) plus a supplied LAN predictor."""

    name: str
    params: list[str]
    choices: list[int]
    predictor: Callable[[NDArray[np.float32]], Any] | None = None

    @classmethod
    def from_model(
        cls,
        model: str,
        predictor: Callable[[NDArray[np.float32]], Any] | None = None,
    ) -> ModelSpec:
        """Build a ModelSpec from an ssms model name and optional predictor."""
        cfg = ModelConfigBuilder.from_model(model)
        return cls(
            name=model,
            params=list(cfg["params"]),
            choices=list(cfg["choices"]),
            predictor=predictor,
        )

    @property
    def n_params(self) -> int:
        """Number of model parameters."""
        return len(self.params)

    @property
    def n_choices(self) -> int:
        """Number of possible choices."""
        return len(self.choices)


@dataclass
class PlotConfig:
    """Plotting defaults shared by the entry points.

    ``save``, ``show`` and ``save_dir`` apply to both plots and ``fig_scale``
    only to the manifold. Every other field styles the KDE-vs-LAN figure.

    Parameters
    ----------
    font_scale
        Seaborn font scale; sets the tick-label size.
    figsize
        Figure size in inches.
    cols
        Number of panel columns.
    alpha
        Opacity of the KDE lines.
    save
        Write the figure to ``save_dir``.
    show
        Display the figure.
    save_dir
        Directory for saved figures; created if missing.
    fig_scale
        Size multiplier for the manifold figure.
    title
        Figure title. ``"auto"`` gives "Likelihoods KDE vs. LAN: <MODEL>",
        ``None`` draws no title, any other string is used as given.
    panel_titles
        ``"index"`` titles each panel with the row number of its parameter
        vector, ``"params"`` with that vector's values (``v=1, a=1.5, ...``).
        A sequence of strings gives one title per parameter vector.
    title_fontsize
        Font size of the figure title, in points.
    panel_title_fontsize
        Font size of the panel titles.
    xlabel_fontsize
        Font size of the x-axis label.
    ylabel_fontsize
        Font size of the y-axis label.
    legend_fontsize
        Font size of the legend.
    layout
        Matplotlib layout engine, ``"constrained"`` or ``"tight"``, which sizes
        the margins to the titles and labels. ``None`` keeps the fixed margins,
        which suit the default figure size and clip smaller figures. No engine
        can fit a title that is wider than the figure; lower its font size.
    filename
        File name inside ``save_dir``; its extension picks the format.
    dpi
        Resolution of the saved file; ``None`` uses Matplotlib's ``savefig.dpi``.
    """

    font_scale: float = 1.5
    figsize: tuple[float, float] = (10, 10)
    cols: int = 3
    alpha: float = 0.1
    save: bool = False
    show: bool = True
    save_dir: str = "figures/"
    fig_scale: float = 1.0
    title: str | None = "auto"
    panel_titles: str | Sequence[str] = "index"
    title_fontsize: float = 30
    panel_title_fontsize: float = 20
    xlabel_fontsize: float = 24
    ylabel_fontsize: float = 20
    legend_fontsize: float = 12
    layout: str | None = None
    filename: str = "kde_vs_mlp_plot.png"
    dpi: float | None = None

    def __post_init__(self) -> None:
        """Reject a panel_titles string that is not one of the two modes."""
        if isinstance(self.panel_titles, str) and self.panel_titles not in (
            "index",
            "params",
        ):
            raise ValueError(
                "panel_titles must be 'index', 'params' or a sequence of titles;"
                f" got {self.panel_titles!r}."
            )


@dataclass
class GridSpec:
    """Reaction-time grid resolution for the inspection plots."""

    n_rt_steps: int = 200
    max_rt: float = 5.0
    n_points_2c: int = 2000
    rt_step_2c: float = 0.0025
    n_points_mc: int = 1000
    rt_step_mc: float = 0.01
