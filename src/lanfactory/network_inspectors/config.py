"""Config objects: model metadata and plotting/grid defaults."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
            # Some ssms configs give only ``nchoices`` (e.g. lca_3); their
            # simulators emit choices 0..n-1.
            choices=list(cfg.get("choices") or range(cfg["nchoices"])),
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


# HSSM ecosystem plot colors, from the design tokens at
# https://github.com/lnccbrown/hssm_website/blob/main/design-system/tokens.json
# Plot lines use the same colors as the UI controls, so the figure matches the
# page: the LAN output in the theme's primary indigo (buttons, sliders) and the
# KDE estimates in ink, the neutral text color, as the layer underneath.
HSSM_PRIMARY = "#3538a1"  # primary (indigo)
HSSM_PAPER_INK = "#20303a"  # paper-ink: text, axes, and KDE curves
# The logo's 16-step "wave" ramp, dark to light. OKLCH lightness rises
# monotonically, so it is a valid sequential scale.
HSSM_WAVE = (
    "#480080", "#402392", "#3538a1", "#264bae", "#115bb9", "#006bc1",
    "#007bc8", "#008acd", "#1799d1", "#00aad8", "#00b9d8", "#00c8d5",
    "#00d8cf", "#00e6c3", "#3bf3b4", "#72ffa4",
)  # fmt: skip


@dataclass
class PlotConfig:
    """Plotting defaults shared by the entry points."""

    font_scale: float = 1.5
    figsize: tuple[float, float] = (10, 10)
    cols: int = 3
    alpha: float = 0.1
    save: bool = False
    show: bool = True
    save_dir: str = "figures/"
    fig_scale: float = 1.0
    kde_color: str = HSSM_PAPER_INK
    lan_color: str = HSSM_PRIMARY
    ink_color: str = HSSM_PAPER_INK
    manifold_colorscale: list[str] = field(default_factory=lambda: list(HSSM_WAVE))


@dataclass
class GridSpec:
    """Reaction-time grid resolution for the inspection plots."""

    n_rt_steps: int = 200
    max_rt: float = 5.0
    n_points_2c: int = 2000
    rt_step_2c: float = 0.0025
    n_points_mc: int = 1000
    rt_step_mc: float = 0.01
