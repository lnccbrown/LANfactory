"""Tests for the DatasetTorch class."""

import pickle
import numpy as np
import pytest
from lanfactory.trainers.torch_mlp import DatasetTorch


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
        batch_size=128,
        label_lower_bound=-16.0,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Verify attributes are set
    assert dataset.batch_size == 128
    assert len(dataset.file_ids) == 2
    assert dataset.input_dim == 6
    assert dataset.batches_per_file == 1000 // 128  # 7 batches per file


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
        batch_size=128,
        features_key="lan_data",
        label_key="lan_labels",
    )

    # Get first batch
    X, y = dataset[0]

    assert X.shape == (128, 6)  # batch_size x features
    assert y.shape == (128, 1)  # batch_size x 1 (expanded)
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
        batch_size=128,
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
        batch_size=128,
        features_key="lan_data",
        label_key="lan_labels",
    )

    X, y = dataset[0]

    # 2D labels should remain 2D
    assert X.shape == (128, 6)
    assert y.shape == (128, 3)


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
        batch_size=128,
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
