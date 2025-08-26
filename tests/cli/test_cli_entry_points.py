import subprocess
from pathlib import Path


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
        "--training-data-folder",
        training_data_folder,
        "--networks-path-base",
        str(networks_path),
        "--log-level",
        "WARNING",
        "--config-path",
        config_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"jax_train.py failed: {result.stderr}"

    # No config path provided, should use default config
    result = subprocess.run(cmd[:-2], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"jax_train.py failed: {result.stderr}"


def test_torch_train_cli_smoke(tmp_path):
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
        "torchtrain",
        "--config-path",
        config_path,
        "--training-data-folder",
        training_data_folder,
        "--networks-path-base",
        str(networks_path),
        "--log-level",
        "WARNING",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"torch_train.py failed: {result.stderr}"
