"""derive → ``torchtrain`` → ONNX, on the production ddm LAN.

The corpus tests check that a derived folder is what the trainer reads; this
test drives the trainer's own CLI through it and checks what comes out: an
artifact that satisfies the single-trial ONNX contract and evaluates to a
log-probability, and an MLflow run that names the LAN the corpus came from —
every provenance key exactly as the corpus wrote it.
"""

from __future__ import annotations

import contextlib
import hashlib
import pickle
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import yaml
from torch.utils.data import DataLoader

from lanfactory.cli.torch_train import main as torchtrain
from lanfactory.derive import (
    AUX_CATEGORY,
    DERIVATION_METHOD,
    IntegrationGrid,
    derive_aux_corpus,
)
from lanfactory.onnx.contract import assert_single_trial_contract
from lanfactory.trainers.torch_mlp import ModelTrainerTorchMLP
from tests.derive.conftest import DDM_ONNX
from tests.utils import HISTORY_WRITE_SKIP_REASON, history_write_is_broken

mlflow = pytest.importorskip("mlflow")

N_PARAMS = 4  # ddm: v, a, z, t
N_THETA = 64
# The trainer requires the batch size to divide the rows per file: a derived
# cpn corpus has one row per choice (two for ddm), opn / gonogo one per theta.
ROWS_PER_FILE = {"cpn": 2 * N_THETA, "opn": N_THETA, "gonogo": N_THETA}

pytestmark = pytest.mark.skipif(
    history_write_is_broken(ModelTrainerTorchMLP.train_and_evaluate),
    reason=HISTORY_WRITE_SKIP_REASON,
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


@pytest.fixture(autouse=True)
def inprocess_dataloader(monkeypatch):
    """Load the two tiny files in the test process.

    The CLI cannot ask for it (``dl_workers <= 0`` means "auto", never 0),
    and spawning a worker per DataLoader costs ~25 s per case here against
    ~0.2 s in-process — the whole round trip is otherwise worker start-up.
    """
    original_init = DataLoader.__init__

    def init_in_process(self, *args, **kwargs):
        original_init(self, *args, **{**kwargs, "num_workers": 0})

    monkeypatch.setattr(DataLoader, "__init__", init_in_process)


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

    # The command function itself, in-process (so the DataLoader patch above
    # applies), with every option spelled out: their defaults are typer
    # OptionInfo objects.
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

    # The artifact: single-trial contract, n_params + 1 wide, and the
    # log-sigmoid head (Exp + Log) that the eval-mode logits network exports —
    # a raw-logit export has only Gemm/Tanh. Its output on a corpus row is
    # then a finite log-probability.
    (onnx_path,) = list((networks / network_type / "ddm").glob("*_model.onnx"))
    info = assert_single_trial_contract(onnx_path, expected_input_width=N_PARAMS + 1)
    assert {"Exp", "Log"} <= set(info["ops"]), info["ops"]
    with open(files[0], "rb") as f:
        pickled = pickle.load(f)
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    for row in pickled[f"{network_type}_data"][:4].astype(np.float32):
        (out,) = session.run(None, {input_name: row[None, :]})
        log_prob = float(np.asarray(out).reshape(-1)[0])
        assert np.isfinite(log_prob) and log_prob <= 0.0, log_prob

    # The run: what the corpus wrote, by name, plus the origin tag.
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("derive-roundtrip")
    (run,) = client.search_runs([experiment.experiment_id])
    params, tags = run.data.params, run.data.tags
    assert params["network_type"] == network_type
    assert tags["data_origin"] == "derived"
    assert tags["run_uuid"] in onnx_path.name  # the MLflow <-> disk join key

    source = pickled["generator_config"]["source"]
    assert params["derivation_method"] == DERIVATION_METHOD
    assert params["aux_category"] == AUX_CATEGORY[network_type]
    assert (
        params["source_lan_sha256"] == hashlib.sha256(DDM_ONNX.read_bytes()).hexdigest()
    )
    # A bare ddm.onnx has no run uuid and no Hub commit: logged as "", and
    # its MLflow run id is unknown: not logged at all.
    assert params["source_lan_run_uuid"] == ""
    assert params["source_lan_hf_commit"] == ""
    assert "source_lan_run_id" not in params
    assert params["integration_grid"] == str(IntegrationGrid().n_points)
    assert params["integration_max_t"] == str(IntegrationGrid().max_t)
    # Every other key the corpus wrote, verbatim.
    for key, value in source.items():
        if value is not None:
            assert params[key] == str(value), key

    # The mass statistics are per file and describe the first file the
    # dataset read, which file shuffling makes either one: the tags must be
    # exactly one corpus file's derive_stats, stringified.
    per_file_stats = []
    for file in files:
        with open(file, "rb") as f:
            stats = pickle.load(f)["generator_config"]["derive_stats"]
        per_file_stats.append({key: str(value) for key, value in stats.items()})
    logged_stats = {key: tags[key] for key in per_file_stats[0]}
    assert logged_stats in per_file_stats, (logged_stats, per_file_stats)
    assert 0.9 < float(tags["derive_total_mass_mean"]) < 1.1
