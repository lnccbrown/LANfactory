__version__ = "0.8.0"

from . import config
from . import trainers
from . import utils
from . import onnx

__all__ = ["config", "trainers", "utils", "onnx", "network_inspectors"]


def __getattr__(name):
    # lazy import so `import lanfactory` doesn't require sklearn etc.
    # (importlib here, since `from . import` would recurse back into __getattr__)
    if name == "network_inspectors":
        import importlib

        module = importlib.import_module(f"{__name__}.network_inspectors")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
