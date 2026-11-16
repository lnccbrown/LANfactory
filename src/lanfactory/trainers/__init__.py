import importlib
import warnings
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from .jax_mlp import JaxMLP, JaxMLPFactory, ModelTrainerJaxMLP

# The jax trainer is imported on first access, not here: `import lanfactory`
# (and so torch training and every CLI's --help) would otherwise load jax,
# flax and optax.
_LAZY_JAX_EXPORTS = frozenset({"JaxMLP", "JaxMLPFactory", "ModelTrainerJaxMLP"})

__all__ = [  # noqa: RUF022 - grouped by backend, not alphabetically
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
    if name in _LAZY_JAX_EXPORTS:
        value = getattr(importlib.import_module(f"{__name__}.jax_mlp"), name)
        globals()[name] = value
        return value
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return __getattr__(new_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_JAX_EXPORTS})
