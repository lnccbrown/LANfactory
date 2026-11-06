"""Widget settings for the network inspector Streamlit UI.

Each spec holds a widget's label, range, default, and help text, so the layout
code in ``streamlit_app`` only places widgets. Specs are plain data; nothing
here imports streamlit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WidgetSpec:
    """Keyword arguments for one Streamlit input widget.

    Fields left as ``None`` are not passed, so Streamlit's defaults apply and
    callers can supply them at render time (e.g. a data-dependent ``value``).
    """

    label: str
    help: str
    value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    format: str | None = None
    width: int | None = None

    def kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return the widget keyword arguments, with ``overrides`` applied."""
        spec = {key: val for key, val in asdict(self).items() if val is not None}
        return spec | overrides


# KDE vs LAN tab --------------------------------------------------------------

SEED = WidgetSpec(
    label="Random seed",
    help="Controls reproducible generation of parameter vectors.",
    value=123,
    min_value=0,
    max_value=2_000_000,
    width=160,
)
PARAMETER_SETS = WidgetSpec(
    label="Parameter sets",
    help="Number of parameter vectors sampled from the model bounds.",
    value=6,
    min_value=1,
    max_value=20,
)
SIMULATOR_SAMPLES = WidgetSpec(
    label="Simulator samples",
    help="Number of samples used per simulated dataset.",
    value=1000,
    min_value=100,
    max_value=5000,
    step=200,
)
KDE_REPETITIONS = WidgetSpec(
    label="KDE repetitions",
    help="How many KDE estimates are overlaid for each parameter set.",
    value=8,
    min_value=1,
    max_value=30,
)
KDE_GRID_POINTS = WidgetSpec(
    label="KDE grid points (2-choice)",
    help="Number of reaction-time grid points used for KDE/LAN comparison.",
    value=2000,
    min_value=200,
    max_value=5000,
)
KDE_GRID_STEP = WidgetSpec(
    label="KDE grid step (2-choice)",
    help="Reaction-time spacing in the KDE/LAN evaluation grid.",
    value=0.0025,
    min_value=1e-6,
    step=0.0025,
    format="%g",
)
KDE_PLOT_COLUMNS = WidgetSpec(
    label="KDE plot columns",
    help="Number of subplot columns for KDE/LAN comparison charts.",
    value=3,
    min_value=1,
    max_value=6,
)
KDE_ALPHA = WidgetSpec(
    label="KDE alpha",
    help="Opacity of KDE overlay lines.",
    value=0.1,
    min_value=0.01,
    max_value=0.8,
)
KDE_FONT_SCALE = WidgetSpec(
    label="KDE font scale",
    help="Scale factor for all text in the KDE chart (title, labels, ticks).",
    value=1.0,
    min_value=0.6,
    max_value=2.0,
)

# KDE figure size. The width stays at the 10in the figure's font sizes are tuned
# for; the height follows the number of panel rows instead of a fixed square, and
# the figure is shown at a fixed pixel width rather than stretched to the page.
KDE_FIGURE_WIDTH_IN = 10.0
KDE_PANEL_ROW_HEIGHT_IN = 3.0
KDE_TITLE_HEIGHT_IN = 1.0
KDE_FIGURE_DISPLAY_WIDTH_PX = 900

# 3D manifold tab --------------------------------------------------------------

MANIFOLD_RT_STEPS = WidgetSpec(
    label="Manifold RT steps",
    help="Number of reaction-time points for manifold evaluation.",
    value=300,
    min_value=50,
    max_value=800,
)
MANIFOLD_MAX_RT = WidgetSpec(
    label="Manifold max RT",
    help="Maximum reaction time represented in manifold plots.",
    value=5.0,
    min_value=1.0,
    max_value=10.0,
)
# The sweep range default depends on the selected parameter; it is passed as
# ``value`` when rendering.
SWEEP_MIN = WidgetSpec(
    label="Sweep min",
    format="%g",
    help="Minimum value in the parameter sweep range.",
)
SWEEP_MAX = WidgetSpec(
    label="Sweep max",
    format="%g",
    help="Maximum value in the parameter sweep range.",
)
SWEEP_STEPS = WidgetSpec(
    label="Sweep steps",
    help="Number of points between sweep min and sweep max.",
    value=20,
    min_value=5,
    max_value=80,
)
SWEEP_PARAMETER = WidgetSpec(
    label="Parameter to sweep",
    help="Choose one parameter to vary while others remain fixed.",
)

# Sidebar ----------------------------------------------------------------------

BASE_DIR = WidgetSpec(
    label="Torch models base directory",
    help=(
        "Folder containing model subfolders, each with state_dict and "
        "network_config files."
    ),
)
MODEL_DETECTED = WidgetSpec(
    label="Model",
    help="Choose a model that exists in the selected torch models directory.",
)
MODEL_ANY = WidgetSpec(
    label="Model",
    help="Choose a model name. You still need matching files on disk.",
)
