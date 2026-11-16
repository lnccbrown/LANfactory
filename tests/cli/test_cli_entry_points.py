import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader
from typer.testing import CliRunner

from lanfactory.cli import train as train_module
from lanfactory.cli.train import (
    Backend,
    _folder_from_runs,
    _split_files,
    _start_mlflow_run,
    app,
    run_training,
)

HERE = Path(__file__).parent
TRAINING_DATA_FOLDER = (
    HERE
    / "test_training_data/training_data/lan/training_data_n_samples_2000_dt_0.001/ddm"
)
# In principle callable without config, in which case the bundled default
# config is used, but the default config is too slow for testing.
CONFIG_PATH = HERE / "config_network_training_lan.yaml"

runner = CliRunner()

# typer renders errors through rich, which colors them when it detects CI
# (e.g. GITHUB_ACTIONS): the escape codes split `--backend` into `-` + `-backend`.
_ANSI_RE = re.compile(r"\x1b\[[\d;]*m")


def _plain_output(result) -> str:
    return _ANSI_RE.sub("", result.output)


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
    assert "--backend" in _plain_output(result)


def test_train_rejects_unknown_backend(tmp_path):
    result = runner.invoke(app, ["--backend", "tensorflow", *_args(tmp_path)])
    assert result.exit_code != 0
    assert "tensorflow" in _plain_output(result)


def test_train_backend_is_case_insensitive():
    """`JAX` parses; the run then stops at the next missing required option."""
    result = runner.invoke(app, ["--backend", "JAX"])
    assert result.exit_code != 0
    output = _plain_output(result)
    assert "--networks-path-base" in output
    assert "JAX" not in output


def test_cli_import_does_not_load_jax():
    """Importing the CLI (torch runs, --help) must not pull in jax/flax/optax."""
    code = (
        "import sys, lanfactory.cli.train; "
        "print(*[m for m in ('jax', 'flax', 'optax') if m in sys.modules])"
    )
    # A fresh interpreter: this process has long since imported jax.
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"loaded on import: {result.stdout.strip()}"


def test_jax_trainer_exports_resolve_lazily():
    from lanfactory import trainers
    from lanfactory.trainers.jax_mlp import JaxMLP, JaxMLPFactory, ModelTrainerJaxMLP

    assert trainers.JaxMLP is JaxMLP
    assert trainers.JaxMLPFactory is JaxMLPFactory
    assert trainers.ModelTrainerJaxMLP is ModelTrainerJaxMLP
    assert {"JaxMLP", "JaxMLPFactory", "ModelTrainerJaxMLP"} <= set(dir(trainers))


@pytest.mark.parametrize(
    ("n_files", "cap", "split", "expected"),
    [
        (10, 4, 0.75, (3, 1)),  # the cap bounds validation too, not just training
        (2, 10_000, 0.98, (1, 1)),  # fewer files than the cap
        (10, 10, 0.8, (8, 2)),
    ],
)
def test_split_files_respects_the_cap(n_files, cap, split, expected):
    files = [Path(f"f{i}.pickle") for i in range(n_files)]
    train, val = _split_files(files, cap, split)
    assert (len(train), len(val)) == expected
    assert train + val == files[: sum(expected)]


@pytest.mark.parametrize(
    ("n_files", "cap", "split", "empty"),
    [
        (10, 5, 1.0, "validation"),  # a cap below the file count leaves none over
        (10, 1, 0.98, "training"),
    ],
)
def test_split_files_rejects_an_empty_split(n_files, cap, split, empty):
    files = [Path(f"f{i}.pickle") for i in range(n_files)]
    with pytest.raises(ValueError, match=f"No {empty} files"):
        _split_files(files, cap, split)


def test_folder_from_runs():
    assert _folder_from_runs([]) is None
    assert _folder_from_runs([{"data_output_folder": None}]) is None
    same = [{"data_output_folder": "/data/ddm"}] * 3
    assert _folder_from_runs(same) == Path("/data/ddm")
    disagree = [{"data_output_folder": "/data/a"}, {"data_output_folder": "/data/b"}]
    with pytest.raises(ValueError, match="2 different folders"):
        _folder_from_runs(disagree)


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    """An isolated MLflow tracking store; restores global MLflow state after."""
    mlflow = pytest.importorskip("mlflow")
    for var in ("MLFLOW_EXPERIMENT_NAME", "MLFLOW_ARTIFACT_LOCATION"):
        monkeypatch.delenv(var, raising=False)
    # MLflow's default artifact root is ./mlruns: keep any of it out of the repo.
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').absolute()}"
    artifacts = str((tmp_path / "artifacts").absolute())
    original_uri = mlflow.get_tracking_uri()
    yield SimpleNamespace(
        mlflow=mlflow,
        client=mlflow.MlflowClient(tracking_uri=tracking_uri),
        tracking_uri=tracking_uri,
        artifacts=artifacts,
    )
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(original_uri)


def _generation_experiment(env, filenames, folder=TRAINING_DATA_FOLDER) -> str:
    """A data-generation experiment with one run, as ssm-simulators records it."""
    experiment_id = env.client.create_experiment(
        "generation", artifact_location=env.artifacts
    )
    run_id = env.client.create_run(experiment_id).info.run_id
    if folder is not None:
        env.client.log_param(run_id, "data_output_folder", str(folder.absolute()))
    env.client.log_dict(
        run_id,
        {
            "num_files": len(filenames),
            "total_size_mb": 0.0,
            "files": [{"filename": name} for name in filenames],
        },
        "generated_files_inventory.json",
    )
    env.client.set_terminated(run_id)
    return experiment_id


TRAINING_FILES = sorted(f.name for f in TRAINING_DATA_FOLDER.glob("*.pickle"))


def test_train_from_data_generation_experiment_alone(mlflow_env, inprocess_dataloader):
    """MLflow-first: the data folder comes from the generation run's parameters."""
    env = mlflow_env
    generation = _generation_experiment(env, TRAINING_FILES)

    run_training(
        backend="torch",
        config_path=CONFIG_PATH,
        data_generation_experiment_id=generation,
        networks_path_base=Path("networks"),
        dry_run=True,
        mlflow_experiment_name="training",
        mlflow_tracking_uri=env.tracking_uri,
        mlflow_artifact_location=env.artifacts,
    )

    # The training run records where its data came from, and the folder it
    # resolved is the one the generation run logged.
    training = env.client.get_experiment_by_name("training")
    (train_run,) = env.client.search_runs([training.experiment_id])
    assert train_run.data.tags["data_generation_experiment_id"] == generation
    assert train_run.data.params["data_generation_total_files"] == str(
        len(TRAINING_FILES)
    )
    lineage = env.mlflow.artifacts.load_dict(
        f"{train_run.info.artifact_uri}/training_data_lineage.json"
    )
    assert lineage["training_data_folder"] == str(TRAINING_DATA_FOLDER.absolute())
    assert sorted(lineage["actual_files_used"]) == TRAINING_FILES


def test_experiment_without_a_recorded_folder_needs_one(mlflow_env):
    env = mlflow_env
    generation = _generation_experiment(env, TRAINING_FILES, folder=None)
    with pytest.raises(ValueError, match="no --training-data-folder provided"):
        run_training(
            backend="torch",
            config_path=CONFIG_PATH,
            data_generation_experiment_id=generation,
            networks_path_base=Path("networks"),
            dry_run=True,
            mlflow_tracking_uri=env.tracking_uri,
            mlflow_experiment_name="training",
            mlflow_artifact_location=env.artifacts,
        )


def test_validation_mode_rejects_missing_files(mlflow_env):
    """Both sources given: every file MLflow recorded must be in the folder."""
    env = mlflow_env
    generation = _generation_experiment(env, [*TRAINING_FILES, "absent.pickle"])
    with pytest.raises(FileNotFoundError, match="1 are missing"):
        run_training(
            backend="torch",
            config_path=CONFIG_PATH,
            training_data_folder=TRAINING_DATA_FOLDER,
            data_generation_experiment_id=generation,
            networks_path_base=Path("networks"),
            dry_run=True,
            mlflow_tracking_uri=env.tracking_uri,
            mlflow_experiment_name="training",
            mlflow_artifact_location=env.artifacts,
        )


def test_validation_mode_tolerates_extra_files(
    mlflow_env, inprocess_dataloader, caplog
):
    """Files in the folder that MLflow did not record are reported, not fatal."""
    env = mlflow_env
    generation = _generation_experiment(env, TRAINING_FILES[:1])
    run_training(
        backend="torch",
        config_path=CONFIG_PATH,
        training_data_folder=TRAINING_DATA_FOLDER,
        data_generation_experiment_id=generation,
        networks_path_base=Path("networks"),
        dry_run=True,
        mlflow_tracking_uri=env.tracking_uri,
        mlflow_experiment_name="training",
        mlflow_artifact_location=env.artifacts,
    )
    assert "1 extra files in training folder" in caplog.text


def test_start_mlflow_run_branches(mlflow_env):
    env = mlflow_env
    mlflow = env.mlflow
    existing = env.client.create_experiment("existing", artifact_location=env.artifacts)

    def start(**overrides):
        kwargs = {
            "tracking_uri": env.tracking_uri,
            "mlflow_experiment_name": "existing",
            "mlflow_artifact_location": env.artifacts,
            "mlflow_run_name": "run",
            "mlflow_run_id": None,
            **overrides,
        }
        _start_mlflow_run(**kwargs)
        run = mlflow.active_run()
        mlflow.end_run()
        return run

    # An existing experiment is reused whether or not an artifact location
    # is given (and is not re-created).
    assert start().info.experiment_id == existing
    resumable = start(mlflow_artifact_location=None)
    assert resumable.info.experiment_id == existing
    # A run id resumes that run.
    assert start(mlflow_run_id=resumable.info.run_id).info.run_id == (
        resumable.info.run_id
    )
    # No experiment name anywhere: the run starts in whatever experiment MLflow
    # already has active, and no experiment is created.
    n_experiments = len(env.client.search_experiments())
    assert start(mlflow_experiment_name=None) is not None
    assert len(env.client.search_experiments()) == n_experiments


def test_run_training_requires_a_data_source(tmp_path):
    with pytest.raises(ValueError, match="Must provide either"):
        run_training(
            backend="torch", config_path=CONFIG_PATH, networks_path_base=tmp_path
        )


def test_run_training_rejects_a_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_training(
            backend="torch",
            config_path=CONFIG_PATH,
            training_data_folder=tmp_path / "nowhere",
            networks_path_base=tmp_path,
        )


def test_default_config_and_auto_workers(tmp_path, inprocess_dataloader, monkeypatch):
    """No --config-path: the bundled config is used; --dl-workers 0 sizes itself."""
    # The bundled config's GPU batch size does not divide the test files' rows;
    # pin the CPU one so the result does not depend on the machine.
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    run_training(
        backend="torch",
        training_data_folder=TRAINING_DATA_FOLDER,
        networks_path_base=tmp_path,
        dl_workers=0,
        dry_run=True,
    )
    assert not any(tmp_path.iterdir())


def test_main_forwards_every_option(monkeypatch, tmp_path):
    """The typer command passes each CLI option through to run_training."""
    calls = []
    monkeypatch.setattr(train_module, "run_training", lambda **kw: calls.append(kw))
    result = runner.invoke(
        app,
        [
            "--backend", "jax",
            "--config-path", str(CONFIG_PATH),
            "--training-data-folder", str(TRAINING_DATA_FOLDER),
            "--networks-path-base", str(tmp_path),
            "--network-id", "2",
            "--dl-workers", "3",
            "--dry-run",
            "--no-export-onnx",
            "--mlflow-run-name", "name",
            "--mlflow-experiment-name", "exp",
            "--mlflow-run-id", "rid",
            "--data-generation-experiment-id", "7",
            "--mlflow-tracking-uri", "sqlite:///x.db",
            "--mlflow-artifact-location", "art",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "backend": Backend.jax,
            "config_path": CONFIG_PATH,
            "training_data_folder": TRAINING_DATA_FOLDER,
            "networks_path_base": tmp_path,
            "network_id": 2,
            "dl_workers": 3,
            "dry_run": True,
            "export_onnx": False,
            "mlflow_run_name": "name",
            "mlflow_experiment_name": "exp",
            "mlflow_run_id": "rid",
            "data_generation_experiment_id": "7",
            "mlflow_tracking_uri": "sqlite:///x.db",
            "mlflow_artifact_location": "art",
        }
    ]
