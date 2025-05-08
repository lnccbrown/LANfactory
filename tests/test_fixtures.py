import numpy as np
import torch


def test_data_fixture(sample_data_small):
    """Test that the sample data fixture works correctly."""
    X, y = sample_data_small
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (100, 5)
    assert y.shape == (100, 1)


def test_torch_device(torch_device):
    """Test that the torch device fixture works."""
    assert isinstance(torch_device, torch.device)


def test_random_seed(random_seed):
    """Test reproducibility with the random seed."""
    np.random.seed(random_seed)
    a = np.random.randn(10)
    np.random.seed(random_seed)
    b = np.random.randn(10)
    np.testing.assert_array_equal(a, b)
