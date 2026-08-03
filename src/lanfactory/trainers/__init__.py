import warnings

from .jax_mlp import JaxMLP, JaxMLPFactory, ModelTrainerJaxMLP
from .torch_mlp import (
    DatasetTorch,
    LoadTorchMLP,
    LoadTorchMLPInfer,
    ModelTrainerTorchMLP,
    TorchMLP,
    TorchMLPFactory,
    make_dataloader,
    make_train_valid_dataloaders,
)

__all__ = [
    # Dataset and DataLoader helpers
    "DatasetTorch",
    "JaxMLP",
    # Jax MLP
    "JaxMLPFactory",
    "LoadTorchMLP",
    "LoadTorchMLPInfer",
    "ModelTrainerJaxMLP",
    "ModelTrainerTorchMLP",
    # Torch MLP
    "TorchMLP",
    "TorchMLPFactory",
    "make_dataloader",
    "make_train_valid_dataloaders",
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
