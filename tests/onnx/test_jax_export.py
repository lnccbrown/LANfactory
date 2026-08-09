"""Tests for the jax -> ONNX exporter (feat/jax-onnx-export).

The parity test is the load-bearing one: it guards the ecosystem's
single-trial ONNX contract against silent changes in jax2onnx (0.x, pinned
but moving) and in our own wrapper.
"""

import pickle
from pathlib import Path

import lanfactory
import numpy as np
import pytest
from lanfactory.onnx import transform_jax_to_onnx
from torch.utils.data import DataLoader

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

INPUT_DIM = 6

# Calibrated against jax2onnx 0.15 lowering: relu -> Max, logsigmoid eval
# head -> Neg/Exp/Add/Log (+ Constant for literals). All implemented by
# jaxonnxruntime, HSSM's consumer.
ALLOWED_OPS = {
    "Gemm",
    "MatMul",
    "Add",
    "Tanh",
    "Max",
    "Sigmoid",
    "Neg",
    "Exp",
    "Log",
    "Constant",
}

NETWORK_CONFIG = {
    "layer_sizes": [16, 8, 1],
    "activations": ["tanh", "tanh", "linear"],
    "train_output_type": "logprob",
    "network_type": "lan",
}


def make_training_pickle(path: Path, features_key="lan_data", label_key="lan_labels"):
    """A minimal training-data pickle shaped like ssm-simulators output."""
    with open(path, "wb") as f:
        pickle.dump(
            {
                features_key: np.random.randn(64, INPUT_DIM).astype(np.float32),
                label_key: np.random.randn(64).astype(np.float32),
                "generator_config": {"model": "ddm"},
                "model_config": {"params": ["v", "a", "z", "t"]},
            },
            f,
        )
    return path


def _train_tiny_jax_network(tmp_path: Path) -> dict:
    """Train a micro JaxMLP and return paths to its artifacts."""
    data_file = tmp_path / "training_data.pickle"
    with open(data_file, "wb") as f:
        pickle.dump(
            {
                "lan_data": np.random.randn(64, INPUT_DIM).astype(np.float32),
                "lan_labels": np.random.randn(64).astype(np.float32),
                "generator_config": {"model": "ddm"},
                "model_config": {"params": ["v", "a", "z", "t"]},
            },
            f,
        )

    dataset = lanfactory.trainers.DatasetTorch(
        file_ids=[data_file],
        batch_size=16,
        features_key="lan_data",
        label_key="lan_labels",
    )
    dl = DataLoader(dataset, batch_size=None)

    net = lanfactory.trainers.JaxMLPFactory(
        network_config=dict(NETWORK_CONFIG), train=True
    )
    trainer = lanfactory.trainers.ModelTrainerJaxMLP(
        train_config={
            "n_epochs": 2,  # scheduler requires decay_steps > warmup_steps
            "loss": "huber",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "lr_scheduler": None,
            "lr_scheduler_params": {},
            "weight_decay": 0.0,
            "train_output_type": "logprob",
        },
        train_dl=dl,
        valid_dl=dl,
        model=net,
        seed=42,
    )
    out_dir = tmp_path / "nets"
    state = trainer.train_and_evaluate(
        output_folder=out_dir,
        output_file_id="ddm",
        run_id="runx",
        mlflow_on=False,
        save_outputs=True,
        verbose=0,
        network_type="lan",
    )

    def find(suffix):
        matches = [p for p in out_dir.iterdir() if p.name.endswith(suffix)]
        assert len(matches) == 1, (suffix, sorted(p.name for p in out_dir.iterdir()))
        return matches[0]

    return {
        "state_file": find("_train_state.jax"),
        "onnx_file": find("_model.onnx"),
        "trainer": trainer,
        "state": state,
        "out_dir": out_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    return _train_tiny_jax_network(tmp_path_factory.mktemp("jax_export"))


class TestContract:
    """The exported graph must satisfy the ecosystem ONNX contract."""

    def test_all_input_dims_concrete(self, trained):
        model = onnx.load(str(trained["onnx_file"]))
        for inp in model.graph.input:
            dims = inp.type.tensor_type.shape.dim
            for d in dims:
                assert d.HasField("dim_value"), (
                    f"symbolic dim {d.dim_param!r} in {inp.name} — "
                    "dynamic axes are forbidden (HSSM rejects them at load)"
                )

    def test_input_shape_is_one_by_input_dim(self, trained):
        model = onnx.load(str(trained["onnx_file"]))
        dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        assert dims == [1, INPUT_DIM]

    def test_op_profile_is_mlp_only(self, trained):
        # Gemm + elementwise ops only: the profile HSSM's rank-1 + vmap
        # consumption is known to handle. Calibrated against what jax2onnx
        # 0.15 actually emits: relu lowers to Max (not Relu), and the eval
        # logsigmoid head of logits networks adds Neg/Exp/Add/Log.
        model = onnx.load(str(trained["onnx_file"]))
        ops = {n.op_type for n in model.graph.node}
        assert ops <= ALLOWED_OPS, ops


class TestParity:
    def test_ort_matches_jax_forward(self, trained):
        """ONNX output == live jax forward, 1000 draws, float32 tolerance."""
        from functools import partial

        from lanfactory.trainers.jax_mlp import JaxMLP

        sess = ort.InferenceSession(str(trained["onnx_file"]))
        iname = sess.get_inputs()[0].name
        live = trained["trainer"].model
        eval_model = JaxMLP(
            layer_sizes=live.layer_sizes,
            activations=live.activations,
            train_output_type=live.train_output_type,
            train=False,
        )
        fwd = partial(eval_model.apply, trained["state"].params)

        import jax.numpy as jnp

        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, INPUT_DIM)).astype(np.float32)
        jax_out = np.asarray(fwd(jnp.asarray(X)))
        max_err = 0.0
        for i in range(X.shape[0]):
            o = sess.run(None, {iname: X[i : i + 1]})[0]
            max_err = max(max_err, float(np.max(np.abs(o - jax_out[i]))))
        assert max_err < 1e-4, f"parity violated: max|ORT - jax| = {max_err}"

    def test_file_based_transform_matches_in_trainer_export(self, trained):
        """transform_jax_to_onnx on saved artifacts == the in-trainer export."""
        network_config_file = trained["tmp_path"] / "network_config.pickle"
        with open(network_config_file, "wb") as f:
            pickle.dump(dict(NETWORK_CONFIG), f)

        onnx_roundtrip = trained["tmp_path"] / "roundtrip.onnx"
        transform_jax_to_onnx(
            network_config_file=str(network_config_file),
            state_file=str(trained["state_file"]),
            input_shape=INPUT_DIM,
            output_onnx_file=str(onnx_roundtrip),
        )

        sess_a = ort.InferenceSession(str(trained["onnx_file"]))
        sess_b = ort.InferenceSession(str(onnx_roundtrip))
        rng = np.random.default_rng(1)
        for _ in range(50):
            x = rng.standard_normal((1, INPUT_DIM)).astype(np.float32)
            oa = sess_a.run(None, {sess_a.get_inputs()[0].name: x})[0]
            ob = sess_b.run(None, {sess_b.get_inputs()[0].name: x})[0]
            np.testing.assert_allclose(oa, ob, atol=1e-6)


class TestTrainerFlag:
    def test_no_export_onnx_skips_the_artifact(self, tmp_path):
        artifacts = _train_tiny_jax_network(tmp_path)
        # retrain into a fresh dir with export disabled
        out_dir2 = tmp_path / "nets2"
        artifacts["trainer"].train_and_evaluate(
            output_folder=out_dir2,
            output_file_id="ddm",
            run_id="runy",
            mlflow_on=False,
            save_outputs=True,
            verbose=0,
            network_type="lan",
            export_onnx=False,
        )
        names = [p.name for p in out_dir2.iterdir()]
        assert names, "no artifacts produced"
        assert not any(n.endswith(".onnx") for n in names), names


class TestLogitsHead:
    """CPN/OPN exports must emit the eval logsigmoid head, not raw logits.

    The raw-head export was a blocker: HSSM treats network output as
    element-wise log-likelihood, so raw logits silently corrupt every logp
    by +log(1+exp(-logit)).
    """

    def test_logits_export_equals_logsigmoid_of_raw_head(self, tmp_path):
        from functools import partial

        import jax.numpy as jnp
        from lanfactory.trainers.jax_mlp import JaxMLP

        f = make_training_pickle(
            tmp_path / "training_data_cpn.pickle",
            features_key="cpn_data",
            label_key="cpn_labels",
        )
        dataset = lanfactory.trainers.DatasetTorch(
            file_ids=[f], batch_size=16, features_key="cpn_data", label_key="cpn_labels"
        )
        dl = DataLoader(dataset, batch_size=None)

        config = {
            "layer_sizes": [8, 1],
            "activations": ["tanh", "linear"],
            "train_output_type": "logits",
            "network_type": "cpn",
        }
        net = lanfactory.trainers.JaxMLPFactory(network_config=config, train=True)
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
        state = trainer.train_and_evaluate(
            output_folder=out_dir,
            output_file_id="ddm",
            run_id="runc",
            mlflow_on=False,
            save_outputs=True,
            verbose=0,
            network_type="cpn",
        )

        onnx_files = [p for p in out_dir.iterdir() if p.suffix == ".onnx"]
        assert len(onnx_files) == 1
        sess = ort.InferenceSession(str(onnx_files[0]))
        iname = sess.get_inputs()[0].name

        raw_head = partial(trainer.model.apply, state.params)  # train=True
        eval_head = partial(
            JaxMLP(
                layer_sizes=trainer.model.layer_sizes,
                activations=trainer.model.activations,
                train_output_type="logits",
                train=False,
            ).apply,
            state.params,
        )

        rng = np.random.default_rng(2)
        for _ in range(100):
            x = rng.standard_normal((1, INPUT_DIM)).astype(np.float32)
            onnx_out = sess.run(None, {iname: x})[0]
            raw = np.asarray(raw_head(jnp.asarray(x)))
            expected = np.asarray(eval_head(jnp.asarray(x)))
            # eval head == logsigmoid of raw head, and the export matches it
            np.testing.assert_allclose(
                expected, -np.log1p(np.exp(-raw)), rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(onnx_out, expected, rtol=1e-4, atol=1e-5)
            # and decisively: the export is NOT the raw head
            assert np.max(np.abs(onnx_out - raw)) > 1e-3
