__version__ = "0.5.3"

from . import config
from . import trainers
from . import utils
from . import onnx

__all__ = ["config", "trainers", "utils", "onnx", "network_inspectors"]


def __getattr__(name):
    # Lazily import network_inspectors so that `import lanfactory` does not
    # eagerly pull in its heavy / optional dependencies (e.g. scikit-learn).
    if name == "network_inspectors":
        import importlib

        module = importlib.import_module(".network_inspectors", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
