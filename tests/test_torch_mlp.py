"""Tests for the DatasetTorch class and related components."""

import pickle
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from lanfactory.trainers.torch_mlp import (
    DatasetTorch,
    ModelTrainerTorchMLP,
    LoadTorchMLPInfer,
    TorchMLP,
)


@pytest.fixture
def create_mock_data_files(tmp_path):
    """Create multiple real pickle files for testing."""

    def _create_files(n_files=3):
        file_list = []
        for i in range(n_files):
            file_path = tmp_path / f"training_data_{i}.pickle"
            # Create slightly different data for each file
            data = {
                "lan_data": np.random.randn(1000, 6).astype(np.float32),
                "lan_labels": np.random.randn(1000).astype(np.float32),
                "generator_config": {"model": "ddm", "n_samples": 1000},
            }
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            file_list.append(str(file_path))
        return file_list

    return _create_files


def test_dataset_torch_init(create_mock_data_files):  # pylint: disable=redefined-outer-name
    """Test DatasetTorch initialization."""
    file_list = create_mock_data_files(n_files=2)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=100,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Verify attributes are set
    assert dataset.batch_size == 100
    assert len(dataset.file_ids) == 2
    assert dataset.input_dim == 6
    assert dataset.batches_per_file == 1000 // 100  # 10 batches per file


def test_dataset_torch_len(create_mock_data_files):  # pylint: disable=redefined-outer-name
    """Test __len__ returns correct number of batches."""
    file_list = create_mock_data_files(n_files=2)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=100,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # With 2 files of 1000 samples each, batch size 100
    # Expected: 2 files * 10 batches = 20 batches
    expected_len = 20
    assert len(dataset) == expected_len


def test_dataset_torch_getitem_single_batch(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test __getitem__ returns correct batch shape."""
    file_list = create_mock_data_files(n_files=1)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=100,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Get first batch
    X, y = dataset[0]

    assert X.shape == (100, 6)  # batch_size x features
    assert y.shape == (100, 1)  # batch_size x 1 (expanded)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_dataset_torch_getitem_loads_file_on_first_access(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test that __getitem__ loads file on first access."""
    file_list = create_mock_data_files(n_files=2)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=100,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # First access should load file and populate tmp_data
    assert dataset.tmp_data == {}  # Initially empty

    X, y = dataset[0]

    # Now tmp_data should be populated
    assert dataset.tmp_data != {}
    assert "lan_data" in dataset.tmp_data
    assert "lan_labels" in dataset.tmp_data


def test_dataset_torch_getitem_multiple_batches_same_file(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test accessing multiple batches from the same file."""
    file_list = create_mock_data_files(n_files=1)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=200,  # 1000 samples / 200 = 5 batches per file
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Access multiple batches from the same file
    X0, y0 = dataset[0]
    X1, y1 = dataset[1]
    X2, y2 = dataset[2]

    # All should have correct shape
    assert X0.shape == (200, 6)
    assert X1.shape == (200, 6)
    assert X2.shape == (200, 6)

    # Batches should be different (very unlikely to be identical with random data)
    assert not np.array_equal(X0, X1)
    assert not np.array_equal(X1, X2)


def test_dataset_torch_getitem_crosses_file_boundary(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test that __getitem__ loads new file when crossing file boundary."""
    file_list = create_mock_data_files(n_files=3)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=250,  # 4 batches per file
        features_key="lan_data",
        label_key="lan_labels",
    )

    # batches_per_file = 1000 / 250 = 4
    # So indices 0-3 are from file 0, 4-7 from file 1, 8-11 from file 2

    # Access batch from first file
    X0, y0 = dataset[0]
    first_file_data = dataset.tmp_data["lan_data"].copy()

    # Access batch from same file
    X3, y3 = dataset[3]
    assert np.array_equal(dataset.tmp_data["lan_data"], first_file_data)

    # Access first batch from second file - should trigger file load
    X4, y4 = dataset[4]
    second_file_data = dataset.tmp_data["lan_data"]

    # Data should be different (new file loaded)
    assert not np.array_equal(second_file_data, first_file_data)

    # Access first batch from third file
    X8, y8 = dataset[8]
    third_file_data = dataset.tmp_data["lan_data"]

    # Should be different from second file
    assert not np.array_equal(third_file_data, second_file_data)


def test_dataset_torch_getitem_with_label_bounds(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test that label bounds are applied correctly."""
    file_list = create_mock_data_files(n_files=1)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=100,
        label_lower_bound=-10.0,
        label_upper_bound=10.0,
        features_key="lan_data",
        label_key="lan_labels",
    )

    X, y = dataset[0]

    # Labels should be clipped to bounds
    assert np.all(y >= -10.0)
    assert np.all(y <= 10.0)


def test_dataset_torch_with_2d_labels(tmp_path):
    """Test DatasetTorch handles 2D labels correctly."""
    # Create data with 2D labels
    data = {
        "lan_data": np.random.randn(1000, 6).astype(np.float32),
        "lan_labels": np.random.randn(1000, 3).astype(np.float32),  # 2D labels
    }

    file_path = tmp_path / "training_data_2d.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    dataset = DatasetTorch(
        file_ids=[str(file_path)],
        batch_size=100,
        features_key="lan_data",
        label_key="lan_labels",
    )

    X, y = dataset[0]

    # 2D labels should remain 2D
    assert X.shape == (100, 6)
    assert y.shape == (100, 3)


def test_dataset_torch_sequential_access_pattern(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test sequential access through multiple files (realistic DataLoader pattern)."""
    file_list = create_mock_data_files(n_files=3)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=500,  # 2 batches per file
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Simulate DataLoader iterating through dataset
    total_batches = len(dataset)  # Should be 3 files * 2 batches = 6

    for i in range(total_batches):
        X, y = dataset[i]

        # Each batch should have correct shape
        assert X.shape == (500, 6)
        assert y.shape == (500, 1)


def test_dataset_torch_empty_tmp_data_triggers_load(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test that empty tmp_data triggers file load regardless of index."""
    file_list = create_mock_data_files(n_files=2)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=200,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Access batch 2 (not 0), but tmp_data is empty
    X, y = dataset[2]

    # Should work - file was loaded due to empty tmp_data
    assert X.shape == (200, 6)
    assert dataset.tmp_data != {}


def test_dataset_torch_with_jax_output(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test DatasetTorch with jax output framework."""
    file_list = create_mock_data_files(n_files=1)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=100,
        features_key="lan_data",
        label_key="lan_labels",
        out_framework="jax",
    )

    X, y = dataset[0]

    # Should still return numpy arrays (conversion happens elsewhere)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_dataset_torch_batch_ids_calculation(
    create_mock_data_files,  # pylint: disable=redefined-outer-name
):
    """Test that batch_ids are calculated correctly for different indices."""
    file_list = create_mock_data_files(n_files=1)

    dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=100,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Access different batches and verify they use different data ranges
    X0, _ = dataset[0]  # Should use indices 0-99
    X1, _ = dataset[1]  # Should use indices 100-199
    X9, _ = dataset[9]  # Should use indices 900-999

    # All should be different
    assert not np.array_equal(X0, X1)
    assert not np.array_equal(X1, X9)


def test_dataset_torch_label_bounds(tmp_path):
    """Test DatasetTorch applies label bounds correctly."""
    # Create data with labels outside bounds
    file_path = tmp_path / "training_data.pickle"
    data = {
        "lan_data": np.random.randn(1000, 6).astype(np.float32),
        "lan_labels": np.random.randn(1000).astype(np.float32) * 20,  # Large values
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    dataset = DatasetTorch(
        file_ids=[str(file_path)],
        batch_size=100,
        label_lower_bound=-10.0,
        label_upper_bound=10.0,
        features_key="lan_data",
        label_key="lan_labels",
    )

    X, y = dataset[0]

    # Labels should be clipped to bounds
    assert np.all(y >= -10.0)
    assert np.all(y <= 10.0)


def test_dataset_torch_3d_labels_raises_error(tmp_path):
    """Test DatasetTorch raises ValueError for 3D labels."""
    file_path = tmp_path / "training_data.pickle"
    data = {
        "lan_data": np.random.randn(1000, 6).astype(np.float32),
        "lan_labels": np.random.randn(1000, 3, 2).astype(np.float32),  # 3D labels
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    dataset = DatasetTorch(
        file_ids=[str(file_path)],
        batch_size=100,
        features_key="lan_data",
        label_key="lan_labels",
    )

    with pytest.raises(ValueError, match="Label data has unexpected shape"):
        X, y = dataset[0]


def test_model_trainer_torch_mlp_init_with_dict():
    """Test ModelTrainerTorchMLP initialization with dict train_config."""
    train_config = {
        "n_epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "loss": "huber",
        "lr_scheduler": "reduce_on_plateau",
        "lr_scheduler_params": {"patience": 2},
    }

    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create mock model and dataloaders
    mock_model = TorchMLP(
        network_config=network_config, input_shape=6, generative_model_id=None
    )
    mock_train_dl = MagicMock()
    mock_valid_dl = MagicMock()

    trainer = ModelTrainerTorchMLP(
        train_config=train_config,
        model=mock_model,
        train_dl=mock_train_dl,
        valid_dl=mock_valid_dl,
    )

    assert trainer.train_config == train_config
    assert trainer.model is not None


def test_model_trainer_torch_mlp_init_with_path(tmp_path):
    """Test ModelTrainerTorchMLP initialization with path train_config."""
    train_config = {
        "n_epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "loss": "huber",
        "lr_scheduler": "reduce_on_plateau",
        "lr_scheduler_params": {"patience": 2},
    }

    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Save train_config to file
    config_file = tmp_path / "train_config.pickle"
    with open(config_file, "wb") as f:
        pickle.dump(train_config, f)

    # Create mock model and dataloaders
    mock_model = TorchMLP(
        network_config=network_config, input_shape=6, generative_model_id=None
    )
    mock_train_dl = MagicMock()
    mock_valid_dl = MagicMock()

    trainer = ModelTrainerTorchMLP(
        train_config=str(config_file),
        model=mock_model,
        train_dl=mock_train_dl,
        valid_dl=mock_valid_dl,
    )

    assert trainer.train_config["n_epochs"] == 10
    assert trainer.train_config["learning_rate"] == 0.001


def test_load_torch_mlp_infer_with_dict_config(tmp_path):
    """Test LoadTorchMLPInfer with dict network_config."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create a simple model and save its state dict
    model = TorchMLP(
        network_config=network_config, input_shape=6, generative_model_id=None
    )
    model_file = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_file)

    # Load with LoadTorchMLPInfer
    infer_model = LoadTorchMLPInfer(
        model_file_path=str(model_file),
        network_config=network_config,
        input_dim=6,
    )

    assert infer_model.network_config == network_config
    assert infer_model.input_dim == 6


def test_load_torch_mlp_infer_with_string_config(tmp_path):
    """Test LoadTorchMLPInfer with string path network_config."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Save network_config to file
    config_file = tmp_path / "network_config.pickle"
    with open(config_file, "wb") as f:
        pickle.dump(network_config, f)

    # Create a simple model and save its state dict
    model = TorchMLP(
        network_config=network_config, input_shape=6, generative_model_id=None
    )
    model_file = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_file)

    # Load with LoadTorchMLPInfer using string path for config
    infer_model = LoadTorchMLPInfer(
        model_file_path=str(model_file),
        network_config=str(config_file),
        input_dim=6,
    )

    assert infer_model.network_config == network_config
    assert infer_model.input_dim == 6


def test_load_torch_mlp_infer_call_method(tmp_path):
    """Test LoadTorchMLPInfer.__call__() method."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create a simple model and save its state dict
    model = TorchMLP(
        network_config=network_config, input_shape=6, generative_model_id=None
    )
    model_file = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_file)

    # Load with LoadTorchMLPInfer
    infer_model = LoadTorchMLPInfer(
        model_file_path=str(model_file),
        network_config=network_config,
        input_dim=6,
    )

    # Test call method
    test_input = torch.randn(10, 6)
    output = infer_model(test_input)

    assert output.shape == (10, 1)
    assert isinstance(output, torch.Tensor)


def test_load_torch_mlp_infer_predict_on_batch(tmp_path):
    """Test LoadTorchMLPInfer.predict_on_batch() method."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create a simple model and save its state dict
    model = TorchMLP(
        network_config=network_config, input_shape=6, generative_model_id=None
    )
    model_file = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_file)

    # Load with LoadTorchMLPInfer
    infer_model = LoadTorchMLPInfer(
        model_file_path=str(model_file),
        network_config=network_config,
        input_dim=6,
    )

    # Test predict_on_batch method
    test_input = np.random.randn(10, 6).astype(np.float32)
    output = infer_model.predict_on_batch(test_input)

    assert output.shape == (10, 1)
    assert isinstance(output, np.ndarray)


def test_load_torch_mlp_with_dict_config(tmp_path):
    """Test LoadTorchMLP initialization with dictionary config."""
    from lanfactory.trainers.torch_mlp import LoadTorchMLP, TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create and save a model
    model = TorchMLP(network_config=network_config, input_shape=6)
    model_file = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_file)

    # Load with LoadTorchMLP using dict config
    loaded_model = LoadTorchMLP(
        model_file_path=str(model_file),
        network_config=network_config,
        input_dim=6,
    )

    assert loaded_model.input_dim == 6
    assert loaded_model.network_config == network_config
    assert isinstance(loaded_model.net, TorchMLP)


def test_load_torch_mlp_with_string_config(tmp_path):
    """Test LoadTorchMLP initialization with string path to config."""
    from lanfactory.trainers.torch_mlp import LoadTorchMLP, TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create and save a model
    model = TorchMLP(network_config=network_config, input_shape=6)
    model_file = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_file)

    # Save config to pickle file
    config_file = tmp_path / "config.pickle"
    with open(config_file, "wb") as f:
        pickle.dump(network_config, f)

    # Load with LoadTorchMLP using string path
    loaded_model = LoadTorchMLP(
        model_file_path=str(model_file),
        network_config=str(config_file),
        input_dim=6,
    )

    assert loaded_model.input_dim == 6
    assert loaded_model.network_config == network_config
    assert isinstance(loaded_model.net, TorchMLP)


def test_load_torch_mlp_call_method(tmp_path):
    """Test LoadTorchMLP __call__ method."""
    from lanfactory.trainers.torch_mlp import LoadTorchMLP, TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create and save a model
    model = TorchMLP(network_config=network_config, input_shape=6)
    model_file = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_file)

    # Load model
    loaded_model = LoadTorchMLP(
        model_file_path=str(model_file),
        network_config=network_config,
        input_dim=6,
    )

    # Test __call__ method
    test_input = torch.randn(10, 6)
    output = loaded_model(test_input)

    assert output.shape == (10, 1)
    assert isinstance(output, torch.Tensor)


def test_load_torch_mlp_predict_on_batch(tmp_path):
    """Test LoadTorchMLP predict_on_batch method."""
    from lanfactory.trainers.torch_mlp import LoadTorchMLP, TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create and save a model
    model = TorchMLP(network_config=network_config, input_shape=6)
    model_file = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_file)

    # Load model
    loaded_model = LoadTorchMLP(
        model_file_path=str(model_file),
        network_config=network_config,
        input_dim=6,
    )

    # Test predict_on_batch method
    test_input = np.random.randn(10, 6).astype(np.float32)
    output = loaded_model.predict_on_batch(test_input)

    assert output.shape == (10, 1)
    assert isinstance(output, np.ndarray)


def test_model_trainer_torch_mlp_with_mse_loss(create_mock_data_files):
    """Test ModelTrainerTorchMLP with MSE loss function."""
    from lanfactory.trainers.torch_mlp import (
        DatasetTorch,
        ModelTrainerTorchMLP,
        TorchMLP,
    )

    file_list = create_mock_data_files(n_files=2)

    train_config = {
        "layer_sizes": [10, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
        "cpu_batch_size": 16,
        "gpu_batch_size": 16,
        "n_epochs": 1,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "lr_scheduler": None,
        "weight_decay": 0.0,
        "loss": "mse",  # MSE loss
        "save_history": False,
        "n_training_files": 2,
        "train_val_split": 0.8,
        "shuffle_files": True,
        "label_lower_bound": -16.0,
        "features_key": "data",
        "label_key": "labels",
    }

    network_config = {
        "layer_sizes": [10, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create datasets
    train_dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=20,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )
    valid_dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=20,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=None)

    model = TorchMLP(network_config=network_config, input_shape=5)
    trainer = ModelTrainerTorchMLP(
        model=model,
        train_config=train_config,
        train_dl=train_dl,
        valid_dl=valid_dl,
    )

    assert trainer.loss_fun == torch.nn.functional.mse_loss


def test_model_trainer_torch_mlp_with_bce_loss(create_mock_data_files):
    """Test ModelTrainerTorchMLP with BCE loss function."""
    from lanfactory.trainers.torch_mlp import (
        DatasetTorch,
        ModelTrainerTorchMLP,
        TorchMLP,
    )

    file_list = create_mock_data_files(n_files=2)

    train_config = {
        "layer_sizes": [10, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
        "cpu_batch_size": 16,
        "gpu_batch_size": 16,
        "n_epochs": 1,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "lr_scheduler": None,
        "weight_decay": 0.0,
        "loss": "bce",  # BCE loss
        "save_history": False,
        "n_training_files": 2,
        "train_val_split": 0.8,
        "shuffle_files": True,
        "label_lower_bound": -16.0,
        "features_key": "data",
        "label_key": "labels",
    }

    network_config = {
        "layer_sizes": [10, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create datasets
    train_dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=20,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )
    valid_dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=20,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=None)

    model = TorchMLP(network_config=network_config, input_shape=5)
    trainer = ModelTrainerTorchMLP(
        model=model,
        train_config=train_config,
        train_dl=train_dl,
        valid_dl=valid_dl,
    )

    assert trainer.loss_fun == torch.nn.functional.binary_cross_entropy


def test_model_trainer_torch_mlp_with_sgd_optimizer(create_mock_data_files):
    """Test ModelTrainerTorchMLP with SGD optimizer."""
    from lanfactory.trainers.torch_mlp import (
        DatasetTorch,
        ModelTrainerTorchMLP,
        TorchMLP,
    )

    file_list = create_mock_data_files(n_files=2)

    train_config = {
        "layer_sizes": [10, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
        "cpu_batch_size": 16,
        "gpu_batch_size": 16,
        "n_epochs": 1,
        "optimizer": "sgd",  # SGD optimizer
        "learning_rate": 0.001,
        "lr_scheduler": None,
        "weight_decay": 0.0,
        "loss": "huber",
        "save_history": False,
        "n_training_files": 2,
        "train_val_split": 0.8,
        "shuffle_files": True,
        "label_lower_bound": -16.0,
        "features_key": "data",
        "label_key": "labels",
    }

    network_config = {
        "layer_sizes": [10, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Create datasets
    train_dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=20,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )
    valid_dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=20,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=None)

    model = TorchMLP(network_config=network_config, input_shape=5)
    trainer = ModelTrainerTorchMLP(
        model=model,
        train_config=train_config,
        train_dl=train_dl,
        valid_dl=valid_dl,
    )

    assert isinstance(trainer.optimizer, torch.optim.SGD)


def test_torch_mlp_without_train_output_type():
    """Test TorchMLP initialization without train_output_type in config."""
    from lanfactory.trainers.torch_mlp import TorchMLP

    # Config without train_output_type
    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
    }

    model = TorchMLP(network_config=network_config, input_shape=5)

    # Should default to "logprob"
    assert model.train_output_type == "logprob"


def test_torch_mlp_with_explicit_network_type():
    """Test TorchMLP initialization with explicit network_type parameter."""
    from lanfactory.trainers.torch_mlp import TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = TorchMLP(network_config=network_config, input_shape=5, network_type="cpn")

    # Should use the explicitly provided network_type
    assert model.network_type == "cpn"


def test_torch_mlp_forward_with_logits_output():
    """Test TorchMLP forward pass with logits output in inference mode."""
    from lanfactory.trainers.torch_mlp import TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logits",
    }

    model = TorchMLP(network_config=network_config, input_shape=5)
    model.eval()  # Set to inference mode

    test_input = torch.randn(2, 5)
    output = model(test_input)

    assert output.shape == (2, 1)
    assert isinstance(output, torch.Tensor)


def test_torch_mlp_forward_with_other_output_type():
    """Test TorchMLP forward pass with non-standard output type in inference mode."""
    from lanfactory.trainers.torch_mlp import TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "other",  # Non-standard type to trigger else branch
    }

    model = TorchMLP(network_config=network_config, input_shape=5)
    model.eval()  # Set to inference mode

    test_input = torch.randn(2, 5)
    output = model(test_input)

    assert output.shape == (2, 1)
    assert isinstance(output, torch.Tensor)


def test_torch_mlp_with_non_linear_output_activation():
    """Test TorchMLP with non-linear output activation."""
    from lanfactory.trainers.torch_mlp import TorchMLP

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "sigmoid"],  # Non-linear output activation
        "train_output_type": "logprob",
    }

    model = TorchMLP(network_config=network_config, input_shape=5)

    # Check that sigmoid activation was added
    # Layers: input->hidden1, act1, hidden1->hidden2, act2, hidden2->output, act_output
    assert len(model.layers) == 6
    assert isinstance(model.layers[-1], torch.nn.Sigmoid)


def test_model_trainer_torch_mlp_with_none_train_config(create_mock_data_files):
    """Test ModelTrainerTorchMLP raises error when train_config is None."""
    from lanfactory.trainers.torch_mlp import (
        DatasetTorch,
        ModelTrainerTorchMLP,
        TorchMLP,
    )
    import pytest

    file_list = create_mock_data_files(n_files=1)

    network_config = {
        "layer_sizes": [10, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
    }

    train_dataset = DatasetTorch(
        file_ids=file_list,
        batch_size=20,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    model = TorchMLP(network_config=network_config, input_shape=5)

    with pytest.raises(ValueError, match="train_config is passed as None"):
        ModelTrainerTorchMLP(
            model=model,
            train_config=None,
            train_dl=train_dl,
            valid_dl=train_dl,
        )
