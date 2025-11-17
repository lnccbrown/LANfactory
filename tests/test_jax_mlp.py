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


def test_mlp_jax_forward_with_non_linear_output_activation():
    """Test MLPJax forward pass with non-linear output activation."""
    from lanfactory.trainers.jax_mlp import MLPJaxFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "tanh"],  # Non-linear output activation
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 5))

    # Initialize model state
    key1, key2 = jax.random.split(key)
    state = model.init(key1, x)

    # Forward pass
    output = model.apply(state, x)

    assert output.shape == (5, 1)


def test_mlp_jax_inference_mode_with_logits():
    """Test MLPJax forward pass in inference mode with logits output."""
    from lanfactory.trainers.jax_mlp import MLPJaxFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logits",
    }

    model = MLPJaxFactory(network_config=network_config, train=False)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 5))

    # Initialize model state
    key1, key2 = jax.random.split(key)
    state = model.init(key1, x)

    # Forward pass in inference mode
    output = model.apply(state, x)

    assert output.shape == (5, 1)


def test_mlp_jax_load_state_from_file_error():
    """Test MLPJax load_state_from_file raises error when file_path is None."""
    from lanfactory.trainers.jax_mlp import MLPJaxFactory
    import pytest

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)

    with pytest.raises(ValueError, match="file_path argument needs to be specified"):
        model.load_state_from_file(seed=42, input_dim=5, file_path=None)


def test_mlp_jax_load_state_from_file_without_input_dim(tmp_path):
    """Test MLPJax load_state_from_file without providing input_dim."""
    from lanfactory.trainers.jax_mlp import MLPJaxFactory
    import flax.serialization

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 5))

    # Create initial state
    state = model.init(key, x)

    # Save state to file
    state_file = tmp_path / "model_state.jax"
    with open(state_file, "wb") as f:
        f.write(flax.serialization.to_bytes(state))

    # Load state without input_dim
    loaded_state = model.load_state_from_file(
        seed=42, input_dim=None, file_path=str(state_file)
    )

    assert loaded_state is not None


def test_mlp_jax_make_forward_partial_with_dict_state(tmp_path):
    """Test MLPJax make_forward_partial with dict state."""
    from lanfactory.trainers.jax_mlp import MLPJaxFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 5))

    # Create initial state
    state = model.init(key, x)

    # Make forward partial with dict state
    net_forward, net_forward_jitted = model.make_forward_partial(
        seed=42, input_dim=5, state=state, add_jitted=True
    )

    # Test forward pass
    output = net_forward(x)
    assert output.shape == (5, 1)

    # Test jitted forward pass
    output_jitted = net_forward_jitted(x)
    assert output_jitted.shape == (5, 1)


def test_mlp_jax_make_forward_partial_without_jit(tmp_path):
    """Test MLPJax make_forward_partial without JIT compilation."""
    from lanfactory.trainers.jax_mlp import MLPJaxFactory
    import flax.serialization

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 5))

    # Create and save state
    state = model.init(key, x)
    state_file = tmp_path / "model_state.jax"
    with open(state_file, "wb") as f:
        f.write(flax.serialization.to_bytes(state))

    # Make forward partial without JIT
    net_forward, net_forward_jitted = model.make_forward_partial(
        seed=42, input_dim=5, state=str(state_file), add_jitted=False
    )

    # Test forward pass
    output = net_forward(x)
    assert output.shape == (5, 1)

    # net_forward_jitted should be None
    assert net_forward_jitted is None


def test_mlp_jax_make_forward_partial_invalid_state_type():
    """Test MLPJax make_forward_partial raises error with invalid state type."""
    from lanfactory.trainers.jax_mlp import MLPJaxFactory
    import pytest

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)

    # Test with invalid state type (list instead of dict or string)
    with pytest.raises(
        ValueError, match="state argument has to be a dictionary or a string"
    ):
        model.make_forward_partial(
            seed=42, input_dim=5, state=[1, 2, 3], add_jitted=True
        )


def test_mlp_jax_with_logits_inference():
    """Test MLPJax forward pass with logits output in inference mode."""
    import jax.numpy as jnp

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logits",
    }

    model = MLPJaxFactory(network_config=network_config, train=False)

    # Create dummy input
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (5, 10))

    # Initialize parameters
    params = model.init(rng, x)

    # Forward pass
    output = model.apply(params, x)

    assert output.shape == (5, 1)
    assert isinstance(output, jnp.ndarray)


def test_mlp_jax_with_non_linear_output_activation():
    """Test MLPJax with non-linear output activation."""
    import jax.numpy as jnp

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "sigmoid"],  # Non-linear output
        "train_output_type": "logprob",
    }

    model = MLPJaxFactory(network_config=network_config, train=True)

    # Create dummy input
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (5, 10))

    # Initialize parameters and forward pass
    params = model.init(rng, x)
    output = model.apply(params, x)

    assert output.shape == (5, 1)
    assert isinstance(output, jnp.ndarray)
