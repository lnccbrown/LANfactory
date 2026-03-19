"""Tests for the ONNX transformation module."""

import pickle
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from lanfactory.onnx.transform_onnx import transform_to_onnx, main


@pytest.fixture
def mock_network_config():
    """Fixture providing a mock network configuration."""
    return {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }


@pytest.fixture
def mock_state_dict():
    """Fixture providing a mock state dictionary."""
    return {
        "layers.0.weight": torch.randn(100, 6),
        "layers.0.bias": torch.randn(100),
        "layers.2.weight": torch.randn(100, 100),
        "layers.2.bias": torch.randn(100),
        "layers.4.weight": torch.randn(1, 100),
        "layers.4.bias": torch.randn(1),
    }


def test_transform_to_onnx_success(mock_network_config, mock_state_dict):
    """Test successful ONNX transformation."""
    config_file = "/fake/network_config.pickle"
    state_dict_file = "/fake/state_dict.pt"
    output_file = "/fake/model.onnx"

    # Mock all file operations
    with (
        patch("builtins.open", mock_open(read_data=pickle.dumps(mock_network_config))),
        patch("lanfactory.onnx.transform_onnx.pickle.load") as mock_pickle_load,
        patch("lanfactory.onnx.transform_onnx.torch.load") as mock_torch_load,
        patch("lanfactory.onnx.transform_onnx.TorchMLP") as mock_torch_mlp,
        patch("lanfactory.onnx.transform_onnx.torch.onnx.export") as mock_export,
    ):
        mock_pickle_load.return_value = mock_network_config
        mock_torch_load.return_value = mock_state_dict
        mock_model = MagicMock()
        mock_torch_mlp.return_value = mock_model

        transform_to_onnx(
            network_config_file=config_file,
            state_dict_file=state_dict_file,
            input_shape=6,
            output_onnx_file=output_file,
        )

        # Assert torch.onnx.export was called once
        assert mock_export.call_count == 1

        # Verify the arguments passed to export
        call_args = mock_export.call_args
        assert call_args[0][2] == output_file  # output file path
        assert call_args[1]["dynamo"] is False  # dynamo disabled


def test_transform_to_onnx_loads_network_config(mock_network_config):
    """Test that transform_to_onnx correctly loads the network config."""
    config_file = "/fake/network_config.pickle"

    # Mock all file operations and dependencies
    with (
        patch("builtins.open", mock_open()),
        patch("lanfactory.onnx.transform_onnx.pickle.load") as mock_pickle_load,
        patch("lanfactory.onnx.transform_onnx.TorchMLP") as mock_torch_mlp,
        patch("lanfactory.onnx.transform_onnx.torch.load"),
        patch("lanfactory.onnx.transform_onnx.torch.onnx.export"),
    ):
        mock_pickle_load.return_value = mock_network_config
        mock_model = MagicMock()
        mock_torch_mlp.return_value = mock_model

        transform_to_onnx(
            network_config_file=config_file,
            state_dict_file="/fake/state.pt",
            input_shape=6,
            output_onnx_file="/fake/output.onnx",
        )

        # Verify pickle.load was called
        mock_pickle_load.assert_called_once()

        # Verify TorchMLP was called with the loaded config
        mock_torch_mlp.assert_called_once()
        call_kwargs = mock_torch_mlp.call_args[1]
        assert call_kwargs["network_config"] == mock_network_config
        assert call_kwargs["input_shape"] == 6


def test_transform_to_onnx_loads_state_dict(mock_network_config, mock_state_dict):
    """Test that transform_to_onnx correctly loads the state dict."""
    config_file = "/fake/network_config.pickle"
    state_dict_file = "/fake/state_dict.pt"

    # Mock all file operations
    with (
        patch("builtins.open", mock_open()),
        patch("lanfactory.onnx.transform_onnx.pickle.load") as mock_pickle_load,
        patch("lanfactory.onnx.transform_onnx.torch.load") as mock_torch_load,
        patch("lanfactory.onnx.transform_onnx.TorchMLP") as mock_torch_mlp,
        patch("lanfactory.onnx.transform_onnx.torch.onnx.export"),
    ):
        mock_pickle_load.return_value = mock_network_config
        mock_model = MagicMock()
        mock_torch_mlp.return_value = mock_model
        mock_torch_load.return_value = mock_state_dict

        transform_to_onnx(
            network_config_file=config_file,
            state_dict_file=state_dict_file,
            input_shape=6,
            output_onnx_file="/fake/output.onnx",
        )

        # Verify torch.load was called with correct arguments
        mock_torch_load.assert_called_once_with(
            state_dict_file, map_location=torch.device("cpu")
        )

        # Verify load_state_dict was called on the model
        mock_model.load_state_dict.assert_called_once_with(mock_state_dict)


def test_transform_to_onnx_creates_correct_input_tensor(mock_network_config):
    """Test that transform_to_onnx creates input tensor with correct shape."""
    config_file = "/fake/network_config.pickle"

    with (
        patch("builtins.open", mock_open()),
        patch("lanfactory.onnx.transform_onnx.pickle.load") as mock_pickle_load,
        patch("lanfactory.onnx.transform_onnx.TorchMLP"),
        patch("lanfactory.onnx.transform_onnx.torch.load"),
        patch("lanfactory.onnx.transform_onnx.torch.randn") as mock_randn,
        patch("lanfactory.onnx.transform_onnx.torch.onnx.export"),
    ):
        mock_pickle_load.return_value = mock_network_config
        mock_tensor = MagicMock()
        mock_randn.return_value = mock_tensor

        input_shape = 10
        transform_to_onnx(
            network_config_file=config_file,
            state_dict_file="/fake/state.pt",
            input_shape=input_shape,
            output_onnx_file="/fake/output.onnx",
        )

        # Verify randn was called with correct shape
        mock_randn.assert_called_once_with(1, input_shape, requires_grad=True)


def test_transform_to_onnx_missing_config_file():
    """Test that transform_to_onnx raises error for missing config file."""
    # Mock open to raise FileNotFoundError
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            transform_to_onnx(
                network_config_file="/nonexistent/config.pickle",
                state_dict_file="/fake/state.pt",
                input_shape=6,
                output_onnx_file="/fake/output.onnx",
            )


def test_transform_to_onnx_missing_state_dict_file(mock_network_config):
    """Test that transform_to_onnx raises error for missing state dict file."""
    config_file = "/fake/network_config.pickle"

    with (
        patch("builtins.open", mock_open()),
        patch("lanfactory.onnx.transform_onnx.pickle.load") as mock_pickle_load,
        patch("lanfactory.onnx.transform_onnx.TorchMLP"),
        patch(
            "lanfactory.onnx.transform_onnx.torch.load",
            side_effect=FileNotFoundError("State dict not found"),
        ),
    ):
        mock_pickle_load.return_value = mock_network_config

        with pytest.raises(FileNotFoundError):
            transform_to_onnx(
                network_config_file=config_file,
                state_dict_file="/nonexistent/state.pt",
                input_shape=6,
                output_onnx_file="/fake/output.onnx",
            )


def test_transform_to_onnx_invalid_pickle_file():
    """Test that transform_to_onnx handles invalid pickle file."""
    config_file = "/fake/invalid.pickle"

    # Mock pickle.load to raise UnpicklingError
    with (
        patch("builtins.open", mock_open()),
        patch(
            "lanfactory.onnx.transform_onnx.pickle.load",
            side_effect=pickle.UnpicklingError("Invalid pickle"),
        ),
    ):
        with pytest.raises(pickle.UnpicklingError):
            transform_to_onnx(
                network_config_file=config_file,
                state_dict_file="/fake/state.pt",
                input_shape=6,
                output_onnx_file="/fake/output.onnx",
            )


def test_main_calls_transform_to_onnx():
    """Test that main() CLI command calls transform_to_onnx with correct args."""
    config_file = "/fake/config.pickle"
    state_file = "/fake/state.pt"
    output_file = "/fake/output.onnx"
    input_shape = 8

    with patch("lanfactory.onnx.transform_onnx.transform_to_onnx") as mock_transform:
        main(
            network_config_file=config_file,
            state_dict_file=state_file,
            input_shape=input_shape,
            output_onnx_file=output_file,
        )

        # Verify transform_to_onnx was called with correct arguments
        mock_transform.assert_called_once_with(
            config_file,
            state_file,
            input_shape,
            output_file,
        )


def test_transform_to_onnx_with_different_input_shapes(mock_network_config):
    """Test transform_to_onnx works with various input shapes."""
    config_file = "/fake/network_config.pickle"

    for input_shape in [1, 5, 10, 100]:
        with (
            patch("builtins.open", mock_open()),
            patch("lanfactory.onnx.transform_onnx.pickle.load") as mock_pickle_load,
            patch("lanfactory.onnx.transform_onnx.TorchMLP"),
            patch("lanfactory.onnx.transform_onnx.torch.load"),
            patch("lanfactory.onnx.transform_onnx.torch.onnx.export") as mock_export,
        ):
            mock_pickle_load.return_value = mock_network_config

            transform_to_onnx(
                network_config_file=config_file,
                state_dict_file="/fake/state.pt",
                input_shape=input_shape,
                output_onnx_file="/fake/output.onnx",
            )

            # Should complete without error
            assert mock_export.called
