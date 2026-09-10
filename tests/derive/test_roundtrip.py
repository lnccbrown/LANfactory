"""derive → ``torchtrain`` → ONNX, on the production ddm LAN.

The corpus tests check that a derived folder is what the trainer reads; this
test drives the trainer's own CLI through it and checks what comes out: an
artifact that satisfies the single-trial ONNX contract and evaluates to a
log-probability, and an MLflow run that names the LAN the corpus came from.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from lanfactory.cli.torch_train import main as torchtrain
from lanfactory.derive import AUX_CATEGORY, DERIVATION_METHOD, derive_aux_corpus
from lanfactory.onnx.contract import assert_single_trial_contract
from lanfactory.trainers.torch_mlp import ModelTrainerTorchMLP
from tests.derive.conftest import DDM_ONNX

mlflow = pytest.importorskip("mlflow")

N_PARAMS = 4  # ddm: v, a, z, t
N_THETA = 64
# The trainer requires the batch size to divide the rows per file: a derived
# cpn corpus has one row per choice (two for ddm), opn / gonogo one per theta.
ROWS_PER_FILE = {"cpn": 2 * N_THETA, "opn": N_THETA, "gonogo": N_THETA}


def _history_write_is_broken() -> bool:
    """pandas >= 3 makes ``DataFrame.values`` read-only, and the trainers write
    their per-epoch history into it (``training_history.values[epoch, :] =``);
    fixed upstream in lnccbrown/LANfactory#145. The skip lifts itself once that
    fix is on the branch, and is never taken under pandas 2."""
    source = inspect.getsource(ModelTrainerTorchMLP.train_and_evaluate)
    return int(pd.__version__.split(".")[0]) >= 3 and "history.values[" in source


pytestmark = pytest.mark.skipif(
    _history_write_is_broken(),
    reason=(
        f"pandas {pd.__version__}: the trainer's per-epoch history write fails "
        "('assignment destination is read-only') until lnccbrown/LANfactory#145 "
        "is merged; run with `uv run --with 'pandas<3' pytest ...`"
    ),
)


@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Reset MLflow state after each run (``torchtrain`` sets the tracking URI)."""
    original_uri = mlflow.get_tracking_uri()
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlruns = Path.cwd() / "mlruns"
    if mlruns.exists():
        shutil.rmtree(mlruns)
    with contextlib.suppress(Exception):
        mlflow.set_tracking_uri(original_uri)


def _training_yaml(path: Path, network_type: str, corpus: Path) -> Path:
    config = {
        "NETWORK_TYPE": network_type,
        "CPU_BATCH_SIZE": ROWS_PER_FILE[network_type],
        "GPU_BATCH_SIZE": ROWS_PER_FILE[network_type],
        "GENERATOR_APPROACH": "derived",
        "OPTIMIZER_": "adam",
        "N_EPOCHS": 2,
        "TRAINING_DATA_FOLDER": str(corpus),
        "MODEL": "ddm",
        "SHUFFLE": True,
        "LAYER_SIZES": [[16, 16, 1]],
        "ACTIVATIONS": [["tanh", "tanh"]],
        "WEIGHT_DECAY": 0.0,
        "TRAIN_VAL_SPLIT": 0.5,  # one training file, one validation file
        "N_TRAINING_FILES": 10000,
        "LABELS_LOWER_BOUND": "None",
        "LEARNING_RATE": 0.001,
        "LR_SCHEDULER": "reduce_on_plateau",
        "LR_SCHEDULER_PARAMS": {"factor": 0.1, "patience": 2},
    }
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("network_type", ["cpn", "opn", "gonogo"])
def test_derive_train_export_round_trip(tmp_path, network_type):
    corpus = tmp_path / "corpus"
    files = derive_aux_corpus(
        DDM_ONNX, "ddm", network_type, corpus, n_files=2, n_theta_per_file=N_THETA
    )
    config_path = _training_yaml(
        tmp_path / f"{network_type}.yaml", network_type, corpus
    )
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').absolute()}"
    networks = tmp_path / "networks"

    # The command function itself, in-process, with every option spelled out
    # (their defaults are typer OptionInfo objects). CliRunner would do, but
    # its stdout buffer is closed under it by the trainer's DataLoader worker
    # teardown, and the run is then unreadable even though it succeeded.
    torchtrain(
        config_path=config_path,
        training_data_folder=corpus,
        networks_path_base=networks,
        network_id=0,
        dl_workers=1,
        dry_run=False,
        mlflow_run_name=f"roundtrip-{network_type}",
        mlflow_experiment_name="derive-roundtrip",
        mlflow_run_id=None,
        data_generation_experiment_id=None,
        mlflow_tracking_uri=tracking_uri,
        mlflow_artifact_location=str(tmp_path / "artifacts"),
        log_level="WARNING",
    )

    # The artifact: single-trial contract, n_params + 1 wide, log-probability.
    (onnx_path,) = list((networks / network_type / "ddm").glob("*_model.onnx"))
    assert_single_trial_contract(onnx_path, expected_input_width=N_PARAMS + 1)
    import onnxruntime as ort

    with open(files[0], "rb") as f:
        row = pickle.load(f)[f"{network_type}_data"][:1].astype(np.float32)
    session = ort.InferenceSession(str(onnx_path))
    (out,) = session.run(None, {session.get_inputs()[0].name: row})
    log_prob = float(np.asarray(out).reshape(-1)[0])
    assert np.isfinite(log_prob) and log_prob <= 0.0, log_prob

    # The run: the LAN the corpus came from, by name, plus the origin tag.
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("derive-roundtrip")
    (run,) = client.search_runs([experiment.experiment_id])
    params, tags = run.data.params, run.data.tags
    assert params["network_type"] == network_type
    assert params["derivation_method"] == DERIVATION_METHOD
    assert params["aux_category"] == AUX_CATEGORY[network_type]
    assert (
        params["source_lan_sha256"] == hashlib.sha256(DDM_ONNX.read_bytes()).hexdigest()
    )
    assert params["source_lan_run_uuid"] == ""  # a bare ddm.onnx has no run uuid
    assert tags["data_origin"] == "derived"
    assert tags["run_uuid"] in onnx_path.name  # the MLflow <-> disk join key
