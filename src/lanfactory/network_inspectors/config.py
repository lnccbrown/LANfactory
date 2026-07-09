"""Config objects: model metadata and plotting/grid defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ssms.config import ModelConfigBuilder


@dataclass
class ModelSpec:
    """Model metadata (name, params, choices) plus a supplied LAN predictor."""

    name: str
    params: list[str]
    choices: list[int]
    predictor: Callable | None = None

    @classmethod
    def from_model(cls, model, predictor=None):
        """Build a ModelSpec from an ssms model name and optional predictor."""
        cfg = ModelConfigBuilder.from_model(model)
        return cls(
            name=model,
            params=list(cfg["params"]),
            choices=list(cfg["choices"]),
            predictor=predictor,
        )

    @property
    def n_params(self):
        """Number of model parameters."""
        return len(self.params)

    @property
    def n_choices(self):
        """Number of possible choices."""
        return len(self.choices)


@dataclass
class PlotConfig:
    """Plotting defaults shared by the entry points."""

    font_scale: float = 1.5
    figsize: tuple[int, int] = (10, 10)
    cols: int = 3
    alpha: float = 0.1
    save: bool = False
    show: bool = True
    save_dir: str = "figures/"
    fig_scale: float = 1.0


@dataclass
class GridSpec:
    """Reaction-time grid resolution for the inspection plots."""

    n_rt_steps: int = 200
    max_rt: float = 5.0
    n_points_2c: int = 2000
    rt_step_2c: float = 0.0025
    n_points_mc: int = 1000
    rt_step_mc: float = 0.01
