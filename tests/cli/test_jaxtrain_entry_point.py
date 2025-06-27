import subprocess
import sys
from pathlib import Path

import pytest

from lanfactory.cli import jax_train

non_negative_int = jax_train.non_negative_int


def test_non_negative_int_valid():
    assert non_negative_int("5") == 5
    assert non_negative_int("0") == 0


def test_non_negative_int_negative():
    with pytest.raises(Exception):
        non_negative_int("-1")


def test_non_negative_int_non_integer():
    with pytest.raises(Exception):
        non_negative_int("abc")


def test_jax_train_cli_smoke(tmp_path):
    """Smoke test: actually runs the CLI with real data."""
    # Path to the training data directory
    __dir__ = Path(__file__).parent
    training_data_folder = str(
        __dir__ / "test_training_data/training_data/lan/training_data_n_samples_2000_dt_0.001/ddm"
    )

    config_path = str(__dir__ / "config_network_training_lan.yaml")

    networks_path = tmp_path / "networks"
    networks_path.mkdir()

    cmd = [
        "jaxtrain",
        "--config-path",
        config_path,
        "--training-data-folder",
        training_data_folder,
        "--networks-path-base",
        str(networks_path),
        "--log-level",
        "WARNING",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"jax_train.py failed: {result.stderr}"
