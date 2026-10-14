"""Network loaders: build LAN predictors from the LANfactory torch backend."""

from __future__ import annotations

from collections.abc import Callable
import os
from os import PathLike
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray

LoadTorchMLPInfer: Any
try:
    from lanfactory.trainers import LoadTorchMLPInfer
except ImportError:
    LoadTorchMLPInfer = None
    warnings.warn(
        "PyTorch (via lanfactory) is not installed. Network-loading functions"
        " in the network_inspectors package will not be available.",
        stacklevel=2,
    )


def get_torch_mlp(
    model_file_path: str | PathLike[str],
    network_config: str | PathLike[str] | dict[str, Any],
    input_dim: int,
) -> Callable[[NDArray[np.float32]], Any]:
    """Return a ``predict_on_batch`` callable for the TORCH_MLP likelihood.

    The returned function expects a 2d float32 array whose rows are a parameter
    vector trailed by a reaction time and a choice, and returns the per-row LAN
    log-likelihood.
    """
    if LoadTorchMLPInfer is None:
        raise ImportError(
            "get_torch_mlp requires PyTorch (via lanfactory), which is not"
            " installed. Install lanfactory with PyTorch to load networks."
        )
    network = LoadTorchMLPInfer(
        model_file_path=os.fspath(model_file_path),
        network_config=(
            os.fspath(network_config)
            if isinstance(network_config, os.PathLike)
            else network_config
        ),
        input_dim=input_dim,
    )
    return network.predict_on_batch
