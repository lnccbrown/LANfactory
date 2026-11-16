import subprocess
from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from typer.testing import CliRunner

from lanfactory.cli.train import app, run_training

HERE = Path(__file__).parent
TRAINING_DATA_FOLDER = (
    HERE
    / "test_training_data/training_data/lan/training_data_n_samples_2000_dt_0.001/ddm"
)
# In principle callable without config, in which case the bundled default
# config is used, but the default config is too slow for testing.
CONFIG_PATH = HERE / "config_network_training_lan.yaml"

runner = CliRunner()


@pytest.fixture
def inprocess_dataloader(monkeypatch):
    """Load data in the test process: forked DataLoader workers are slow to start.

    The CLI cannot ask for it (``--dl-workers <= 0`` means "auto", never 0).
    """
    original_init = DataLoader.__init__

    def init_in_process(self, *args, **kwargs):
        original_init(self, *args, **{**kwargs, "num_workers": 0})

    monkeypatch.setattr(DataLoader, "__init__", init_in_process)


def _args(networks_path: Path, *extra: str) -> list[str]:
    return [
        "--config-path",
        str(CONFIG_PATH),
        "--training-data-folder",
        str(TRAINING_DATA_FOLDER),
        "--networks-path-base",
        str(networks_path),
        "--log-level",
        "WARNING",
        *extra,
    ]


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_train_cli_smoke(tmp_path, backend):
    """Smoke test: the installed `train` console script trains on real data."""
    networks_path = tmp_path / "networks"
    networks_path.mkdir()

    cmd = ["train", "--backend", backend, *_args(networks_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"train --backend {backend} failed: {result.stderr}"

    # Both backends write the HSSM-consumable ONNX by default.
    assert list((networks_path / "lan" / "ddm").glob("*_model.onnx"))


# In-process runs call run_training rather than CliRunner: pytest's live
# logging (log_cli) swaps sys.stdout on each record, closing CliRunner's
# captured stream mid-run. CliRunner is used only where parsing fails first.


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_train_no_export_onnx(tmp_path, backend, inprocess_dataloader):
    """`export_onnx=False` skips the ONNX artifact on either backend."""
    run_training(
        backend=backend,
        config_path=CONFIG_PATH,
        training_data_folder=TRAINING_DATA_FOLDER,
        networks_path_base=tmp_path,
        export_onnx=False,
    )
    model_dir = tmp_path / "lan" / "ddm"
    assert list(model_dir.glob("*network_config.pickle"))
    assert not list(model_dir.glob("*.onnx"))


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_train_dry_run(tmp_path, backend, inprocess_dataloader):
    """Dry run validates the pipeline and writes nothing."""
    run_training(
        backend=backend,
        config_path=CONFIG_PATH,
        training_data_folder=TRAINING_DATA_FOLDER,
        networks_path_base=tmp_path,
        dry_run=True,
    )
    assert not any(tmp_path.iterdir())


def test_train_requires_backend(tmp_path):
    result = runner.invoke(app, _args(tmp_path))
    assert result.exit_code != 0
    assert "--backend" in result.output


def test_train_rejects_unknown_backend(tmp_path):
    result = runner.invoke(app, ["--backend", "tensorflow", *_args(tmp_path)])
    assert result.exit_code != 0
    assert "tensorflow" in result.output


def test_train_backend_is_case_insensitive():
    """`JAX` parses; the run then stops at the next missing required option."""
    result = runner.invoke(app, ["--backend", "JAX"])
    assert result.exit_code != 0
    assert "--networks-path-base" in result.output
    assert "JAX" not in result.output
