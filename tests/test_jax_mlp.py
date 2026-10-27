"""Tests for the JAX MLP components."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanfactory.trainers.jax_mlp import JaxMLP, JaxMLPFactory

# Independent of JaxMLP.activations_dict, so the alignment test below checks
# the model against plain math rather than against itself.
REFERENCE_ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "linear": lambda z: z,
}


def test_mlp_jax_factory_with_dict():
    """Test JaxMLPFactory with dict network_config."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)

    assert isinstance(model, JaxMLP)
    assert model.layer_sizes == [100, 100, 1]
    assert model.activations == ["tanh", "tanh", "linear"]


def test_mlp_jax_factory_with_string_path(tmp_path):
    """Test JaxMLPFactory with string path to network_config."""
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
    model = JaxMLPFactory(network_config=str(config_file), train=True)

    assert isinstance(model, JaxMLP)
    assert model.layer_sizes == [100, 100, 1]


def test_mlp_jax_factory_raises_value_error():
    """Test JaxMLPFactory raises ValueError for invalid network_config type."""
    with pytest.raises(ValueError, match="network_config argument is not passed"):
        JaxMLPFactory(network_config=123, train=True)  # Invalid type


def test_mlp_jax_class_initialization():
    """Test JaxMLP class initialization."""
    model = JaxMLP(
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
    """Test JaxMLP forward pass."""
    model = JaxMLP(
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


@pytest.mark.parametrize(
    "activations",
    [
        ["tanh", "tanh", "linear"],  # production default
        ["tanh", "linear", "linear"],  # issue #105: IndexError
        ["linear", "tanh", "linear"],
        ["relu", "linear", "sigmoid"],
        # Over-long: jaxtrain appends the output "linear" to a YAML that
        # already lists it. Must keep working (hence no zip(strict=True)).
        ["tanh", "tanh", "linear", "linear"],
        # Over-long with a hidden "linear": main ran these without an error
        # but applied shifted activations (issue #105's silent case).
        ["linear", "tanh", "linear", "tanh"],
        ["tanh", "linear", "linear", "sigmoid"],
    ],
    ids="-".join,
)
def test_mlp_jax_activations_stay_layer_aligned(activations):
    """Each layer applies its own activation, "linear" hidden layers included."""
    layer_sizes = [10, 10, 1]
    model = JaxMLP(
        layer_sizes=layer_sizes,
        activations=activations,
        train_output_type="logprob",
        train=True,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 6))
    params = model.init(jax.random.PRNGKey(0), x)

    expected = x
    # One activation per layer; entries past the last layer are ignored.
    for i in range(len(layer_sizes)):
        layer = params["params"][f"layers_{i}"]
        expected = REFERENCE_ACTIVATIONS[activations[i]](
            expected @ layer["kernel"] + layer["bias"]
        )

    output = model.apply(params, x)

    assert output.shape == (5, 1)
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_mlp_jax_too_few_activations_raises():
    """Fewer activations than layers is a clear ValueError, not an IndexError."""
    # A jaxtrain YAML missing one hidden activation: the CLI appends the
    # output "linear", leaving 3 activations for 4 layers.
    model = JaxMLP(
        layer_sizes=[10, 10, 10, 1],
        activations=["tanh", "tanh", "linear"],
    )
    with pytest.raises(ValueError, match="got 3 activations for 4 layers"):
        model.init(jax.random.PRNGKey(0), jnp.ones((1, 6)))


def test_mlp_jax_with_different_activations():
    """Test JaxMLP with different activation functions."""
    model = JaxMLP(
        layer_sizes=[10, 10, 1],
        activations=["relu", "sigmoid", "linear"],
        train_output_type="logits",
        train=True,
    )

    assert model.activations == ["relu", "sigmoid", "linear"]
    assert model.train_output_type == "logits"


def test_mlp_jax_factory_train_false():
    """Test JaxMLPFactory with train=False."""
    network_config = {
        "layer_sizes": [100, 100, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=False)

    assert isinstance(model, JaxMLP)
    assert model.train is False


def test_mlp_jax_forward_with_non_linear_output_activation():
    """Test JaxMLP forward pass with non-linear output activation."""
    from lanfactory.trainers.jax_mlp import JaxMLPFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "tanh"],  # Non-linear output activation
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 5))

    # Initialize model state
    key1, _key2 = jax.random.split(key)
    state = model.init(key1, x)

    # Forward pass
    output = model.apply(state, x)

    assert output.shape == (5, 1)


def test_mlp_jax_inference_mode_with_logits():
    """Test JaxMLP forward pass in inference mode with logits output."""
    from lanfactory.trainers.jax_mlp import JaxMLPFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logits",
    }

    model = JaxMLPFactory(network_config=network_config, train=False)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 5))

    # Initialize model state
    key1, _key2 = jax.random.split(key)
    state = model.init(key1, x)

    # Forward pass in inference mode
    output = model.apply(state, x)

    assert output.shape == (5, 1)


def test_mlp_jax_load_state_from_file_error():
    """Test JaxMLP load_state_from_file raises error when file_path is None."""
    import pytest

    from lanfactory.trainers.jax_mlp import JaxMLPFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)

    with pytest.raises(ValueError, match="file_path argument needs to be specified"):
        model.load_state_from_file(seed=42, input_dim=5, file_path=None)


def test_mlp_jax_load_state_from_file_without_input_dim(tmp_path):
    """Test JaxMLP load_state_from_file without providing input_dim."""
    import flax.serialization

    from lanfactory.trainers.jax_mlp import JaxMLPFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)
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
    """Test JaxMLP make_forward_partial with dict state."""
    from lanfactory.trainers.jax_mlp import JaxMLPFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)
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
    """Test JaxMLP make_forward_partial without JIT compilation."""
    import flax.serialization

    from lanfactory.trainers.jax_mlp import JaxMLPFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)
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
    """Test JaxMLP make_forward_partial raises error with invalid state type."""
    import pytest

    from lanfactory.trainers.jax_mlp import JaxMLPFactory

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)

    # Test with invalid state type (list instead of dict or string)
    with pytest.raises(
        ValueError, match="state argument has to be a dictionary or a string"
    ):
        model.make_forward_partial(
            seed=42, input_dim=5, state=[1, 2, 3], add_jitted=True
        )


def test_mlp_jax_with_logits_inference():
    """Test JaxMLP forward pass with logits output in inference mode."""
    import jax.numpy as jnp

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "linear"],
        "train_output_type": "logits",
    }

    model = JaxMLPFactory(network_config=network_config, train=False)

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
    """Test JaxMLP with non-linear output activation."""
    import jax.numpy as jnp

    network_config = {
        "layer_sizes": [10, 10, 1],
        "activations": ["tanh", "tanh", "sigmoid"],  # Non-linear output
        "train_output_type": "logprob",
    }

    model = JaxMLPFactory(network_config=network_config, train=True)

    # Create dummy input
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (5, 10))

    # Initialize parameters and forward pass
    params = model.init(rng, x)
    output = model.apply(params, x)

    assert output.shape == (5, 1)
    assert isinstance(output, jnp.ndarray)


def test_trainer_jax_mlp_non_int_seed_raises_value_error(tmp_path):
    """train_and_evaluate rejects a non-int seed with ValueError, not TypeError."""
    from lanfactory.trainers.jax_mlp import ModelTrainerJaxMLP

    network_config = {
        "layer_sizes": [4, 1],
        "activations": ["tanh", "linear"],
        "train_output_type": "logprob",
    }
    model = JaxMLPFactory(network_config=network_config, train=True)
    trainer = ModelTrainerJaxMLP(
        {"loss": "huber", "n_epochs": 1}, model, None, None, seed=1.5
    )

    with pytest.raises(ValueError, match="seed argument is not an integer"):
        trainer.train_and_evaluate(
            output_folder=tmp_path,
            network_type="lan",
            save_outputs=False,
            export_onnx=False,
        )
