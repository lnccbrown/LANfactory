import warnings

from .torch_mlp import (
    DatasetTorch,
    TorchMLP,
    TorchMLPFactory,
    ModelTrainerTorchMLP,
    LoadTorchMLP,
    LoadTorchMLPInfer,
    make_dataloader,
    make_train_valid_dataloaders,
)
from .jax_mlp import JaxMLPFactory, JaxMLP, ModelTrainerJaxMLP

__all__ = [
    # Dataset and DataLoader helpers
    "DatasetTorch",
    "make_dataloader",
    "make_train_valid_dataloaders",
    # Torch MLP
    "TorchMLP",
    "TorchMLPFactory",
    "ModelTrainerTorchMLP",
    "LoadTorchMLP",
    "LoadTorchMLPInfer",
    # Jax MLP
    "JaxMLPFactory",
    "JaxMLP",
    "ModelTrainerJaxMLP",
]

_DEPRECATED_ALIASES = {
    "MLPJax": "JaxMLP",
    "MLPJaxFactory": "JaxMLPFactory",
}


def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
