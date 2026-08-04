"""Launcher for the network inspectors Streamlit app."""

from __future__ import annotations

from pathlib import Path
import sys


def app() -> None:
    """Start the LANfactory network inspectors UI via Streamlit."""
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Streamlit is required for the network-inspectors-ui command. "
            "Install optional dependencies with: uv sync --extra ui"
        ) from exc

    app_path = (
        Path(__file__).resolve().parents[1] / "network_inspectors" / "streamlit_app.py"
    )
    sys.argv = ["streamlit", "run", str(app_path)]
    raise SystemExit(stcli.main())
