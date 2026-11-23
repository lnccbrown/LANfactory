"""Shared result contracts for network inspector compute and plotting layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import ModelSpec


@dataclass
class LikelihoodRow:
    """Likelihood arrays for one parameter vector."""

    params: NDArray[np.float32]
    lan: NDArray[np.float64]
    kdes: list[NDArray[np.float64]]


@dataclass
class LikelihoodComparison:
    """Computed LAN/KDE likelihoods over a shared reaction-time grid."""

    spec: ModelSpec
    grid: NDArray[np.float64]
    rows: list[LikelihoodRow]


@dataclass
class ManifoldComputation:
    """Computed LAN manifold payload and metadata for plotting or UI display."""

    spec: ModelSpec
    vary_name: str
    vary_values: NDArray[np.float64]
    grid: NDArray[np.float64]
    manifold: pd.DataFrame
