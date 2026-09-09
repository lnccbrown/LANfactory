"""Tests for CLI utilities."""

from unittest.mock import patch, mock_open

import pytest

from lanfactory.cli.utils import (
    _get_train_network_config,
    _make_train_network_configs,
    resolve_mlflow_artifact_location,
    resolve_mlflow_tracking_enabled,
)


class TestResolveMlflowTrackingEnabled:
    _off = dict(
        mlflow_run_name=None, mlflow_run_id=None, data_generation_experiment_id=None
    )

    def test_nothing_set_is_disabled(self):
        assert resolve_mlflow_tracking_enabled(mlflow_flag=None, **self._off) is False

    @pytest.mark.parametrize(
        "override",
        [
            {"mlflow_run_name": "r"},
            {"mlflow_run_id": "abc"},
            {"data_generation_experiment_id": "1"},
        ],
    )
    def test_any_tracking_option_enables(self, override):
        kwargs = {**self._off, **override}
        assert resolve_mlflow_tracking_enabled(mlflow_flag=None, **kwargs) is True

    def test_tracking_uri_env_enables(self):
        assert (
            resolve_mlflow_tracking_enabled(
                mlflow_flag=None, tracking_uri_env="http://mlflow:5000", **self._off
            )
            is True
        )

    def test_empty_env_does_not_enable(self):
        assert (
            resolve_mlflow_tracking_enabled(
                mlflow_flag=None, tracking_uri_env="", **self._off
            )
            is False
        )

    def test_explicit_flag_wins_both_ways(self):
        assert (
            resolve_mlflow_tracking_enabled(
                mlflow_flag=False,
                tracking_uri_env="http://x",
                mlflow_run_name="r",
                mlflow_run_id=None,
                data_generation_experiment_id=None,
            )
            is False
        )
        assert resolve_mlflow_tracking_enabled(mlflow_flag=True, **self._off) is True


class TestResolveMlflowArtifactLocation:
    def test_none_and_empty_pass_through(self):
        assert resolve_mlflow_artifact_location(None) is None
        assert resolve_mlflow_artifact_location("") is None

    def test_relative_path_becomes_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_mlflow_artifact_location("mlruns") == str(tmp_path / "mlruns")

    def test_absolute_path_unchanged(self, tmp_path):
        assert resolve_mlflow_artifact_location(str(tmp_path)) == str(tmp_path)

    @pytest.mark.parametrize(
        "uri",
        [
            "s3://bucket/prefix",
            "gs://bucket/prefix",
            "mlflow-artifacts:/",
            "file:///oscar/data/lab/mlflow/artifacts",
            "hdfs://nn:8020/mlflow",
        ],
    )
    def test_uris_pass_through_untouched(self, uri):
        assert resolve_mlflow_artifact_location(uri) == uri


def test_make_train_network_configs_with_dict_args():
    """Test _make_train_network_configs with dictionary arguments."""
    train_arg_dict = {"n_epochs": 10, "learning_rate": 0.001}
    network_arg_dict = {"layer_sizes": [100, 100], "activations": ["tanh", "tanh"]}

    result = _make_train_network_configs(
        training_data_folder="/fake/data",
        train_val_split=0.9,
        save_folder=".",
        network_arg_dict=network_arg_dict,
        train_arg_dict=train_arg_dict,
        save_name=None,
    )

    assert result["config_dict"] is not None
    assert "train_config" in result["config_dict"]
    assert "network_config" in result["config_dict"]
    assert result["config_dict"]["train_config"]["n_epochs"] == 10
    assert result["config_dict"]["network_config"]["layer_sizes"] == [100, 100]


def test_make_train_network_configs_without_save_name():
    """Test _make_train_network_configs without save_name (no file written)."""
    result = _make_train_network_configs(
        training_data_folder="/fake/data",
        train_val_split=0.9,
        save_folder=".",
        network_arg_dict=None,
        train_arg_dict=None,
        save_name=None,
    )

    assert result["config_dict"] is not None
    assert result["config_file_name"] is None


def test_make_train_network_configs_with_save_name(tmp_path):
    """Test _make_train_network_configs with save_name (file written)."""
    save_name = "test_config.pickle"

    with patch("builtins.open", mock_open()), patch("pickle.dump") as mock_dump:
        result = _make_train_network_configs(
            training_data_folder="/fake/data",
            train_val_split=0.9,
            save_folder=str(tmp_path),
            network_arg_dict=None,
            train_arg_dict=None,
            save_name=save_name,
        )

        assert mock_dump.called
        assert result["config_file_name"] == tmp_path / save_name


def test_get_train_network_config_lan():
    """Test _get_train_network_config with LAN network type."""
    yaml_content = {
        "NETWORK_TYPE": "lan",
        "LAYER_SIZES": [[100, 100, 1]],
        "ACTIVATIONS": [["tanh", "tanh"]],
        "N_EPOCHS": 10,
        "OPTIMIZER_": "adam",
        "N_TRAINING_FILES": 1000,
        "TRAIN_VAL_SPLIT": 0.9,
        "WEIGHT_DECAY": 0.0,
        "CPU_BATCH_SIZE": 128,
        "GPU_BATCH_SIZE": 256,
        "SHUFFLE": True,
        "LABELS_LOWER_BOUND": "np.log(1e-7)",
        "LEARNING_RATE": 0.001,
        "LR_SCHEDULER": "reduce_on_plateau",
        "LR_SCHEDULER_PARAMS": {"patience": 2},
        "TRAINING_DATA_FOLDER": "/fake/data",
        "MODEL": "ddm",
    }

    with (
        patch("builtins.open", mock_open()),
        patch("yaml.safe_load", return_value=yaml_content),
    ):
        result = _get_train_network_config(yaml_config_path="fake.yaml", net_index=0)

        assert result["config_dict"]["network_config"]["train_output_type"] == "logprob"
        assert result["config_dict"]["train_config"]["loss"] == "huber"
        assert result["config_dict"]["train_config"]["features_key"] == "lan_data"
        assert result["config_dict"]["train_config"]["label_key"] == "lan_labels"
        assert result["extra_fields"]["model"] == "ddm"


def test_get_train_network_config_cpn():
    """Test _get_train_network_config with CPN network type."""
    yaml_content = {
        "NETWORK_TYPE": "cpn",
        "LAYER_SIZES": [[100, 100, 1]],
        "ACTIVATIONS": [["tanh", "tanh"]],
        "N_EPOCHS": 10,
        "OPTIMIZER_": "adam",
        "N_TRAINING_FILES": 1000,
        "TRAIN_VAL_SPLIT": 0.9,
        "WEIGHT_DECAY": 0.0,
        "CPU_BATCH_SIZE": 128,
        "GPU_BATCH_SIZE": 256,
        "SHUFFLE": True,
        "LABELS_LOWER_BOUND": "np.log(1e-7)",
        "LEARNING_RATE": 0.001,
        "LR_SCHEDULER": "reduce_on_plateau",
        "LR_SCHEDULER_PARAMS": {"patience": 2},
        "TRAINING_DATA_FOLDER": "/fake/data",
        "MODEL": "ddm",
    }

    with (
        patch("builtins.open", mock_open()),
        patch("yaml.safe_load", return_value=yaml_content),
    ):
        result = _get_train_network_config(yaml_config_path="fake.yaml", net_index=0)

        assert result["config_dict"]["network_config"]["train_output_type"] == "logits"
        assert result["config_dict"]["train_config"]["loss"] == "bcelogit"
        assert result["config_dict"]["train_config"]["features_key"] == "cpn_data"
        assert result["config_dict"]["train_config"]["label_key"] == "cpn_labels"


def test_get_train_network_config_opn():
    """Test _get_train_network_config with OPN network type."""
    yaml_content = {
        "NETWORK_TYPE": "opn",
        "LAYER_SIZES": [[100, 100, 1]],
        "ACTIVATIONS": [["tanh", "tanh"]],
        "N_EPOCHS": 10,
        "OPTIMIZER_": "adam",
        "N_TRAINING_FILES": 1000,
        "TRAIN_VAL_SPLIT": 0.9,
        "WEIGHT_DECAY": 0.0,
        "CPU_BATCH_SIZE": 128,
        "GPU_BATCH_SIZE": 256,
        "SHUFFLE": True,
        "LABELS_LOWER_BOUND": "np.log(1e-7)",
        "LEARNING_RATE": 0.001,
        "LR_SCHEDULER": "reduce_on_plateau",
        "LR_SCHEDULER_PARAMS": {"patience": 2},
        "TRAINING_DATA_FOLDER": "/fake/data",
        "MODEL": "ddm",
    }

    with (
        patch("builtins.open", mock_open()),
        patch("yaml.safe_load", return_value=yaml_content),
    ):
        result = _get_train_network_config(yaml_config_path="fake.yaml", net_index=0)

        assert result["config_dict"]["network_config"]["train_output_type"] == "logits"
        assert result["config_dict"]["train_config"]["loss"] == "bcelogit"
        assert result["config_dict"]["train_config"]["features_key"] == "opn_data"
        assert result["config_dict"]["train_config"]["label_key"] == "opn_labels"


def test_get_train_network_config_no_path():
    """Test _get_train_network_config raises ValueError when no path provided."""
    with pytest.raises(ValueError, match="No YAML config path provided"):
        _get_train_network_config(yaml_config_path=None)


def test_get_train_network_config_with_net_index():
    """Test _get_train_network_config with different net_index."""
    yaml_content = {
        "NETWORK_TYPE": "lan",
        "LAYER_SIZES": [[100, 100, 1], [120, 120, 1]],
        "ACTIVATIONS": [["tanh", "tanh"], ["relu", "relu"]],
        "N_EPOCHS": 10,
        "OPTIMIZER_": "adam",
        "N_TRAINING_FILES": 1000,
        "TRAIN_VAL_SPLIT": 0.9,
        "WEIGHT_DECAY": 0.0,
        "CPU_BATCH_SIZE": 128,
        "GPU_BATCH_SIZE": 256,
        "SHUFFLE": True,
        "LABELS_LOWER_BOUND": "np.log(1e-7)",
        "LEARNING_RATE": 0.001,
        "LR_SCHEDULER": "reduce_on_plateau",
        "LR_SCHEDULER_PARAMS": {"patience": 2},
        "TRAINING_DATA_FOLDER": "/fake/data",
        "MODEL": "ddm",
    }

    with (
        patch("builtins.open", mock_open()),
        patch("yaml.safe_load", return_value=yaml_content),
    ):
        result = _get_train_network_config(yaml_config_path="fake.yaml", net_index=1)

        # layer_sizes comes directly from YAML (not modified)
        assert result["config_dict"]["network_config"]["layer_sizes"] == [120, 120, 1]
        # activations has output layer activation appended
        assert result["config_dict"]["network_config"]["activations"] == [
            "relu",
            "relu",
            "linear",
        ]
