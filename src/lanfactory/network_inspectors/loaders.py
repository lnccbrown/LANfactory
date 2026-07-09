"""Network loaders: build LAN predictors from the LANfactory torch backend."""

from __future__ import annotations

import warnings

try:
    from lanfactory.trainers import LoadTorchMLPInfer
except ImportError:
    LoadTorchMLPInfer = None  # type: ignore[misc]
    warnings.warn(
        "PyTorch (via lanfactory) is not installed. Network-loading functions"
        " in the network_inspectors package will not be available.",
        stacklevel=2,
    )


def get_torch_mlp(model_file_path, network_config, input_dim):
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
        model_file_path=model_file_path,
        network_config=network_config,
        input_dim=input_dim,
    )
    return network.predict_on_batch
