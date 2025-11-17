"""Tests for the JAX MLP components."""

import pickle

import pytest
import jax
import jax.numpy as jnp

from lanfactory.trainers.jax_mlp import MLPJaxFactory, MLPJax


def test_mlp_jax_factory_with_dict():
    """Test MLPJaxFactory with dict network_config."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)

    assert isinstance(model, MLPJax)
    assert model.layer_sizes == [100, 100, 1]
    assert model.activations == ["tanh", "tanh", "linear"]


def test_mlp_jax_factory_with_string_path(tmp_path):
    """Test MLPJaxFactory with string path to network_config."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    # Save config to file
    config_file = tmp_path / "network_config.pickle"
    with open(config_file, "wb") as f:
        pickle.dump(network_config, f)

    # Load using string path
    model = MLPJaxFactory(network_config=str(config_file), train=True)

    assert isinstance(model, MLPJax)
    assert model.layer_sizes == [100, 100, 1]


def test_mlp_jax_factory_raises_value_error():
    """Test MLPJaxFactory raises ValueError for invalid network_config type."""
    with pytest.raises(ValueError, match="network_config argument is not passed"):
        MLPJaxFactory(network_config=123, train=True)  # Invalid type


def test_mlp_jax_class_initialization():
    """Test MLPJax class initialization."""
    model = MLPJax(
        layer_sizes=[100, 100, 1],
        activations=["tanh", "tanh", "linear"],
        train_output_type="logprob",
        train=True,
    )

    assert model.layer_sizes == [100, 100, 1]
    assert model.activations == ["tanh", "tanh", "linear"]
    assert model.train_output_type == "logprob"
    assert model.train is True


def test_mlp_jax_forward_pass():
    """Test MLPJax forward pass."""
    model = MLPJax(
        layer_sizes=[10, 10, 1],
        activations=["tanh", "tanh", "linear"],
        train_output_type="logprob",
        train=True,
    )

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    test_input = jnp.ones((5, 6))  # Batch of 5, input dim 6
    params = model.init(rng, test_input)

    # Forward pass
    output = model.apply(params, test_input)

    assert output.shape == (5, 1)
    assert isinstance(output, jnp.ndarray)


def test_mlp_jax_with_different_activations():
    """Test MLPJax with different activation functions."""
    model = MLPJax(
        layer_sizes=[10, 10, 1],
        activations=["relu", "sigmoid", "linear"],
        train_output_type="logits",
        train=True,
    )

    assert model.activations == ["relu", "sigmoid", "linear"]
    assert model.train_output_type == "logits"


def test_mlp_jax_factory_train_false():
    """Test MLPJaxFactory with train=False."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=False)

    assert isinstance(model, MLPJax)
    assert model.train is False
