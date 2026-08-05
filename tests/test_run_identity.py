"""Tests for self-describing training runs (feat/mlflow-self-describing-training).

Covers:
- ``log_training_run_identity``: the identity params/tags every training run
  must carry (model, network_type, backend, run_uuid, bounds, ...)
- ``DatasetTorch`` retaining ``model_config`` from training data files
- ``_save_data_details`` carrying model_config into data_details.pickle
- the jax trainer's ``network_type`` passthrough (OPN was mislabeled as cpn)
"""

import contextlib
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")

import lanfactory
from lanfactory.cli.utils import log_training_run_identity


@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow():
    """Reset MLflow state after each test (mirrors test_mlflow_integration)."""
    original_uri = mlflow.get_tracking_uri()
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlruns_path = Path.cwd() / "mlruns"
    if mlruns_path.exists():
        shutil.rmtree(mlruns_path)
    with contextlib.suppress(Exception):
        mlflow.set_tracking_uri(original_uri)


@pytest.fixture
def tmp_tracking(tmp_path):
    """Isolated sqlite tracking backend."""
    db = tmp_path / "tracking.db"
    uri = f"sqlite:///{db.absolute()}"
    mlflow.set_tracking_uri(uri)
    return uri


MODEL_CONFIG = {
    "name": "ddm",
    "params": ["v", "a", "z", "t"],
    "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
}


def make_training_pickle(path: Path, features_key="lan_data", label_key="lan_labels"):
    """A minimal training-data pickle shaped like ssm-simulators output."""
    n_samples, n_features = 64, 6
    data = {
        features_key: np.random.randn(n_samples, n_features).astype(np.float32),
        label_key: np.random.randn(n_samples).astype(np.float32),
        "generator_config": {"model": "ddm", "generator_approach": "lan"},
        "model_config": MODEL_CONFIG,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


class TestLogTrainingRunIdentity:
    def _log_and_fetch(self, tmp_tracking, tmp_path, **overrides):
        config_yaml = tmp_path / "train.yaml"
        config_yaml.write_text("NETWORK_TYPE: lan\nMODEL: ddm\n")

        dataset = SimpleNamespace(input_dim=6, data_model_config=MODEL_CONFIG)

        kwargs = {
            "model": "ddm",
            "network_type": "lan",
            "backend": "jax",
            "run_uuid": "abc123",
            "config_path": config_yaml,
            "training_data_folder": tmp_path / "data",
            "n_training_files": 42,
            "dataset": dataset,
        }
        kwargs.update(overrides)

        mlflow.set_experiment("identity-train-test")
        with mlflow.start_run() as run:
            log_training_run_identity(**kwargs)
            run_id = run.info.run_id
        return mlflow.tracking.MlflowClient().get_run(run_id)

    def test_identity_params(self, tmp_tracking, tmp_path):
        run = self._log_and_fetch(tmp_tracking, tmp_path)
        p = run.data.params

        assert p["model"] == "ddm"
        assert p["network_type"] == "lan"
        assert p["backend"] == "jax"
        assert p["run_uuid"] == "abc123"
        assert p["n_training_files"] == "42"
        assert p["input_dim"] == "6"
        assert json.loads(p["param_space"]) == ["v", "a", "z", "t"]
        assert len(p["config_sha256"]) == 64
        assert "lanfactory_version" in p

    def test_param_bounds_json_and_sha_consistent(self, tmp_tracking, tmp_path):
        run = self._log_and_fetch(tmp_tracking, tmp_path)
        p = run.data.params

        bounds = json.loads(p["param_bounds_json"])
        assert bounds == MODEL_CONFIG["param_bounds"]
        expected_sha = hashlib.sha256(p["param_bounds_json"].encode()).hexdigest()
        assert p["param_bounds_sha256"] == expected_sha

    def test_schema_tags(self, tmp_tracking, tmp_path):
        run = self._log_and_fetch(tmp_tracking, tmp_path)
        assert run.data.tags["schema_version"] == "1"
        assert run.data.tags["phase"] == "train"

    def test_no_dataset_still_logs_core_identity(self, tmp_tracking, tmp_path):
        run = self._log_and_fetch(tmp_tracking, tmp_path, dataset=None)
        p = run.data.params
        assert p["model"] == "ddm"
        assert "input_dim" not in p
        assert "param_bounds_json" not in p

    def test_dataset_without_model_config_skips_bounds(self, tmp_tracking, tmp_path):
        # DatasetTorch defaults data_model_config to the string "None" when the
        # training files carry no model_config — must not crash or log junk.
        dataset = SimpleNamespace(input_dim=6, data_model_config="None")
        run = self._log_and_fetch(tmp_tracking, tmp_path, dataset=dataset)
        p = run.data.params
        assert p["input_dim"] == "6"
        assert "param_bounds_json" not in p

    def test_noop_without_active_run(self, tmp_tracking, tmp_path):
        # Must be a silent no-op, not an error.
        assert mlflow.active_run() is None
        log_training_run_identity(
            model="ddm",
            network_type="lan",
            backend="jax",
            run_uuid="x",
            config_path=None,
            training_data_folder=None,
            n_training_files=1,
        )


class TestDatasetModelConfigRetention:
    def test_model_config_retained_from_training_file(self, tmp_path):
        f = make_training_pickle(tmp_path / "training_data_x.pickle")
        dataset = lanfactory.trainers.DatasetTorch(
            file_ids=[f], batch_size=16, features_key="lan_data", label_key="lan_labels"
        )
        assert dataset.data_model_config == MODEL_CONFIG
        assert dataset.data_generator_config == {
            "model": "ddm",
            "generator_approach": "lan",
        }

    def test_absent_model_config_leaves_default(self, tmp_path):
        f = tmp_path / "training_data_y.pickle"
        with open(f, "wb") as fh:
            pickle.dump(
                {
                    "lan_data": np.random.randn(64, 6).astype(np.float32),
                    "lan_labels": np.random.randn(64).astype(np.float32),
                },
                fh,
            )
        dataset = lanfactory.trainers.DatasetTorch(
            file_ids=[f], batch_size=16, features_key="lan_data", label_key="lan_labels"
        )
        assert dataset.data_model_config == "None"


class TestDataDetailsCarriesModelConfig:
    def test_torch_save_data_details_includes_model_config(self, tmp_path):
        from lanfactory.trainers.torch_mlp import ModelTrainerTorchMLP

        stub = SimpleNamespace(
            dataset=SimpleNamespace(
                data_generator_config={"model": "ddm"},
                data_model_config=MODEL_CONFIG,
                file_ids=["a.pickle"],
            )
        )
        out = tmp_path / "details_data_details.pickle"
        ModelTrainerTorchMLP._save_data_details(stub, stub, str(out))

        with open(out, "rb") as f:
            details = pickle.load(f)
        assert details["train_data_model_config"] == MODEL_CONFIG
        assert details["valid_data_model_config"] == MODEL_CONFIG


class TestJaxNetworkTypePassthrough:
    def _train_tiny(self, tmp_path, network_type_arg):
        """One-epoch micro-training run; returns the produced filenames."""
        from torch.utils.data import DataLoader

        f = make_training_pickle(
            tmp_path / "training_data_z.pickle",
            features_key="opn_data",
            label_key="opn_labels",
        )
        dataset = lanfactory.trainers.DatasetTorch(
            file_ids=[f], batch_size=16, features_key="opn_data", label_key="opn_labels"
        )
        dl = DataLoader(dataset, batch_size=None)

        net = lanfactory.trainers.JaxMLPFactory(
            network_config={
                "layer_sizes": [8, 1],
                "activations": ["tanh", "linear"],
                "train_output_type": "logits",
                "network_type": "opn",
            },
            train=True,
        )
        # n_epochs must be >= 2: the warmup_cosine_decay schedule uses
        # warmup_steps = len(dataset) and decay_steps = len * n_epochs, and
        # optax requires decay_steps > warmup_steps.
        trainer = lanfactory.trainers.ModelTrainerJaxMLP(
            train_config={
                "n_epochs": 2,
                "loss": "bcelogit",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "lr_scheduler": None,
                "lr_scheduler_params": {},
                "weight_decay": 0.0,
                "train_output_type": "logits",
            },
            train_dl=dl,
            valid_dl=dl,
            model=net,
            seed=42,
        )
        out_dir = tmp_path / "nets"
        trainer.train_and_evaluate(
            output_folder=out_dir,
            output_file_id="ddm",
            run_id="runx",
            mlflow_on=False,
            save_outputs=True,
            verbose=0,
            network_type=network_type_arg,
        )
        return [p.name for p in out_dir.iterdir()]

    def test_explicit_opn_names_files_opn(self, tmp_path):
        names = self._train_tiny(tmp_path, network_type_arg="opn")
        assert names, "no output files produced"
        assert all("_opn_" in n for n in names), names

    def test_fallback_infers_cpn_for_logits(self, tmp_path):
        # Documents the legacy inference: without an explicit network_type,
        # logits-trained networks are labeled cpn — the bug the passthrough
        # exists to avoid.
        names = self._train_tiny(tmp_path, network_type_arg=None)
        assert names, "no output files produced"
        assert all("_cpn_" in n for n in names), names
