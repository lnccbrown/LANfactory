import pytest
import numpy as np
import os
import torch
import jax
import jax.numpy as jnp
import ssms
from copy import deepcopy
import random
import time
import lanfactory
from lanfactory.utils import clean_out_folder
import multiprocessing
from dataclasses import dataclass
from .constants import (
    TEST_GENERATOR_CONSTANTS,
    TEST_TRAIN_CONSTANTS,
    TEST_NETWORK_CONSTANTS_LAN,
)
import logging

multiprocessing.set_start_method("spawn", force=True)

# Set up logging
logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Configure logging for pytest."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@pytest.fixture
def random_seed():
    """Fixture to provide a fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_data_small():
    """Fixture providing a small synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    y = np.random.randn(100, 1)  # 100 samples, 1 target
    return X, y


@pytest.fixture
def torch_device():
    """Fixture to provide a PyTorch device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def jax_random_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def test_run_seed():
    """Fixture providing a fixed seed for test runs."""
    return int(time.time())


@pytest.fixture
def available_models():
    """Fixture providing a list of available models."""
    return list(ssms.config.model_config.keys())


@pytest.fixture
def model_selector(request, available_models, test_run_seed):
    """Flexible model selection fixture."""

    def _select_model(
        mode="random",
        models=available_models,
        run_seed=test_run_seed,
    ):
        if mode == "deterministic":
            # Always same model for this test
            rng = random.Random(request.node.name)
        elif mode == "run_consistent":
            # Same model within a test run, different across runs
            rng = random.Random(f"{run_seed}_{request.node.name}")
        else:  # "random"
            # Different model every time
            rng = random.Random()

        selected_model = rng.choice(models)
        logger.info(
            "Selected model '%s' for test '%s' (mode: %s)",
            selected_model,
            request.node.name,
            mode,
        )
        return selected_model

    return _select_model


@pytest.fixture
def dummy_generator_config(model_selector):
    """Fixture providing a dummy model config for testing."""

    def _dummy_generator_config(mode="random"):

        # Initialize the generator config (for MLP LANs)
        generator_config = deepcopy(ssms.config.data_generator_config["lan"])
        # Specify generative model (one from the list of included models mentioned above)
        generator_config["model"] = model_selector(mode=mode)
        # Specify number of parameter sets to simulate
        generator_config["n_parameter_sets"] = TEST_GENERATOR_CONSTANTS.N_PARAMETER_SETS
        # Specify how many samples a simulation run should entail
        generator_config["n_samples"] = TEST_GENERATOR_CONSTANTS.N_SAMPLES
        # Specify folder in which to save generated data
        generator_config["output_folder"] = TEST_GENERATOR_CONSTANTS.OUT_FOLDER
        generator_config["n_training_samples_by_parameter_set"] = (
            TEST_GENERATOR_CONSTANTS.N_SAMPLES_BY_PARAMETER_SET
        )
        logger.info(f"Generator config from dummy_generator_config: {generator_config}")

        model_config = deepcopy(ssms.config.model_config[generator_config["model"]])
        logger.info(f"Model config from dummy_generator_config: {model_config}")
        return {"generator_config": generator_config, "model_config": model_config}

    return _dummy_generator_config


@pytest.fixture
def dummy_network_train_config_lan():
    """Fixture providing a dummy network train config for testing."""
    network_config = deepcopy(lanfactory.config.network_configs.network_config_mlp)
    network_config["layer_sizes"] = TEST_NETWORK_CONSTANTS_LAN.LAYER_SIZES
    network_config["activations"] = TEST_NETWORK_CONSTANTS_LAN.ACTIVATIONS
    network_config["train_output_type"] = TEST_NETWORK_CONSTANTS_LAN.TRAIN_OUTPUT_TYPE
    logger.info(f"Network config from dummy_network_train_config_lan: {network_config}")

    train_config = deepcopy(lanfactory.config.network_configs.train_config_mlp)
    train_config["n_epochs"] = TEST_TRAIN_CONSTANTS.N_EPOCHS
    train_config["cpu_batch_size"] = TEST_TRAIN_CONSTANTS.CPU_BATCH_SIZE
    train_config["gpu_batch_size"] = TEST_TRAIN_CONSTANTS.GPU_BATCH_SIZE
    train_config["optimizer"] = TEST_TRAIN_CONSTANTS.OPTIMIZER
    train_config["learning_rate"] = TEST_TRAIN_CONSTANTS.LEARNING_RATE
    logger.info(f"Train config from dummy_network_train_config_lan: {train_config}")

    return {"network_config": network_config, "train_config": train_config}
