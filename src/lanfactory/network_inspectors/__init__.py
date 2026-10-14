"""Compare LAN vs KDE likelihoods and plot LAN manifolds (LANfactory backend).

Modular layout:
    loaders   -- build LAN predictors from the LANfactory torch backend
    config    -- ModelSpec / PlotConfig / GridSpec config objects
    compute   -- headless RT-grid and likelihood computation
    plotting  -- rendering of the comparison and manifold figures
    api       -- thin entry points wiring the above together
"""

from __future__ import annotations

from .api import kde_vs_lan_likelihoods, lan_manifold
from .config import GridSpec, ModelSpec, PlotConfig
from .loaders import get_torch_mlp

__all__ = [
    "get_torch_mlp",
    "kde_vs_lan_likelihoods",
    "lan_manifold",
    "ModelSpec",
    "PlotConfig",
    "GridSpec",
]
