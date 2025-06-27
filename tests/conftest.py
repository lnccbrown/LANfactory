import logging
import multiprocessing
import os
import random
import time
import uuid
from copy import deepcopy

import jax
import lanfactory
import numpy as np
import pytest
import ssms
import torch

from .constants import (
    TEST_GENERATOR_CONSTANTS,
    TEST_NETWORK_CONSTANTS_CPN,
    TEST_NETWORK_CONSTANTS_LAN,
    TEST_NETWORK_CONSTANTS_OPN,
    TEST_TRAIN_CONSTANTS_CPN,
    TEST_TRAIN_CONSTANTS_LAN,
    TEST_TRAIN_CONSTANTS_OPN,
)
from .utils import clean_out_folder

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
        simulator_param_mapping = True
        while simulator_param_mapping:
            # TODO: #35 use this after ssms v1.0.0 release
            # generator_config = ssms.config.get_default_generator_config("lan")
            # and delete the line below
            generator_config = deepcopy(ssms.config.data_generator_config["lan"])
            # Specify generative model (one from the list of included models mentioned above)
            generator_config["model"] = model_selector(mode=mode)
            # Specify number of parameter sets to simulate
            generator_config["n_parameter_sets"] = TEST_GENERATOR_CONSTANTS.N_PARAMETER_SETS
            # Specify how many samples a simulation run should entail
            generator_config["n_samples"] = TEST_GENERATOR_CONSTANTS.N_SAMPLES
            # Specify folder in which to save generated data
            generator_config["output_folder"] = os.path.join(TEST_GENERATOR_CONSTANTS.OUT_FOLDER, str(uuid.uuid4()))
            generator_config["n_training_samples_by_parameter_set"] = (
                TEST_GENERATOR_CONSTANTS.N_SAMPLES_BY_PARAMETER_SET
            )
            model_config = deepcopy(ssms.config.model_config[generator_config["model"]])

            if "simulator_param_mappings" in model_config:
                simulator_param_mapping = True
            else:
                simulator_param_mapping = False

        logger.info(f"Generator config from dummy_generator_config: {generator_config}")
        logger.info(f"Model config from dummy_generator_config: {model_config}")
        return {"generator_config": generator_config, "model_config": model_config}

    return _dummy_generator_config


@pytest.fixture
def dummy_generator_config_simple_two_choices(model_selector):
    """Fixture providing a dummy model config for testing."""

    # TODO: replace use of ssms.config.data_generator_config with ssms.config.get_default_generator_config
    # after ssms v1.0.0 release
    def _dummy_generator_config_simple_two_choices(mode="random"):
        two_choices = False
        simulator_param_mapping = True
        while (not two_choices) or (simulator_param_mapping):
            # Initialize the generator config (for MLP LANs)
            # TODO: use this after ssms v1.0.0 release
            # generator_config = ssms.config.get_default_generator_config("lan")
            # and delete the line below
            generator_config = deepcopy(ssms.config.data_generator_config["lan"])
            # Specify generative model (one from the list of included models mentioned above)
            generator_config["model"] = model_selector(mode=mode)
            # Specify number of parameter sets to simulate
            generator_config["n_parameter_sets"] = TEST_GENERATOR_CONSTANTS.N_PARAMETER_SETS
            # Specify how many samples a simulation run should entail
            generator_config["n_samples"] = TEST_GENERATOR_CONSTANTS.N_SAMPLES
            # Specify folder in which to save generated data
            generator_config["output_folder"] = os.path.join(TEST_GENERATOR_CONSTANTS.OUT_FOLDER, str(uuid.uuid4()))
            generator_config["n_training_samples_by_parameter_set"] = (
                TEST_GENERATOR_CONSTANTS.N_SAMPLES_BY_PARAMETER_SET
            )
            model_config = deepcopy(ssms.config.model_config[generator_config["model"]])
            if model_config["nchoices"] == 2:
                two_choices = True
            if "simulator_param_mappings" in model_config:
                simulator_param_mapping = True
            else:
                simulator_param_mapping = False

        logger.info(f"Generator config from dummy_generator_config: {generator_config}")
        logger.info(f"Model config from dummy_generator_config: {model_config}")
        return {"generator_config": generator_config, "model_config": model_config}

    return _dummy_generator_config_simple_two_choices


@pytest.fixture
def dummy_network_train_config_lan():
    """Fixture providing a dummy network train config for testing."""
    network_config = deepcopy(lanfactory.config.network_configs.network_config_mlp)
    network_config["layer_sizes"] = TEST_NETWORK_CONSTANTS_LAN.LAYER_SIZES
    network_config["activations"] = TEST_NETWORK_CONSTANTS_LAN.ACTIVATIONS
    network_config["train_output_type"] = TEST_NETWORK_CONSTANTS_LAN.TRAIN_OUTPUT_TYPE
    logger.info("Network config from dummy_network_train_config_lan: %s", network_config)

    train_config = deepcopy(lanfactory.config.network_configs.train_config_mlp)
    train_config["n_epochs"] = TEST_TRAIN_CONSTANTS_LAN.N_EPOCHS
    train_config["cpu_batch_size"] = TEST_TRAIN_CONSTANTS_LAN.CPU_BATCH_SIZE
    train_config["gpu_batch_size"] = TEST_TRAIN_CONSTANTS_LAN.GPU_BATCH_SIZE
    train_config["optimizer"] = TEST_TRAIN_CONSTANTS_LAN.OPTIMIZER
    train_config["learning_rate"] = TEST_TRAIN_CONSTANTS_LAN.LEARNING_RATE
    logger.info("Train config from dummy_network_train_config_lan: %s", train_config)

    return {"network_config": network_config, "train_config": train_config}


@pytest.fixture
def dummy_network_train_config_cpn():
    """Fixture providing a dummy network train config for testing."""
    network_config = deepcopy(lanfactory.config.network_configs.network_config_mlp)
    network_config["layer_sizes"] = TEST_NETWORK_CONSTANTS_CPN.LAYER_SIZES
    network_config["activations"] = TEST_NETWORK_CONSTANTS_CPN.ACTIVATIONS
    network_config["train_output_type"] = TEST_NETWORK_CONSTANTS_CPN.TRAIN_OUTPUT_TYPE
    logger.info("Network config from dummy_network_train_config_cpn: %s", network_config)

    train_config = deepcopy(lanfactory.config.network_configs.train_config_mlp)
    train_config["n_epochs"] = TEST_TRAIN_CONSTANTS_CPN.N_EPOCHS
    train_config["cpu_batch_size"] = TEST_TRAIN_CONSTANTS_CPN.CPU_BATCH_SIZE
    train_config["gpu_batch_size"] = TEST_TRAIN_CONSTANTS_CPN.GPU_BATCH_SIZE
    train_config["optimizer"] = TEST_TRAIN_CONSTANTS_CPN.OPTIMIZER
    train_config["learning_rate"] = TEST_TRAIN_CONSTANTS_CPN.LEARNING_RATE
    train_config["loss"] = TEST_TRAIN_CONSTANTS_CPN.LOSS
    logger.info("Train config from dummy_network_train_config_cpn: %s", train_config)

    return {"network_config": network_config, "train_config": train_config}


@pytest.fixture
def dummy_network_train_config_opn():
    """Fixture providing a dummy network train config for testing."""
    network_config = deepcopy(lanfactory.config.network_configs.network_config_mlp)
    network_config["layer_sizes"] = TEST_NETWORK_CONSTANTS_OPN.LAYER_SIZES
    network_config["activations"] = TEST_NETWORK_CONSTANTS_OPN.ACTIVATIONS
    network_config["train_output_type"] = TEST_NETWORK_CONSTANTS_OPN.TRAIN_OUTPUT_TYPE
    logger.info("Network config from dummy_network_train_config_opn: %s", network_config)

    train_config = deepcopy(lanfactory.config.network_configs.train_config_mlp)
    train_config["n_epochs"] = TEST_TRAIN_CONSTANTS_OPN.N_EPOCHS
    train_config["cpu_batch_size"] = TEST_TRAIN_CONSTANTS_OPN.CPU_BATCH_SIZE
    train_config["gpu_batch_size"] = TEST_TRAIN_CONSTANTS_OPN.GPU_BATCH_SIZE
    train_config["optimizer"] = TEST_TRAIN_CONSTANTS_OPN.OPTIMIZER
    train_config["learning_rate"] = TEST_TRAIN_CONSTANTS_OPN.LEARNING_RATE
    train_config["loss"] = TEST_TRAIN_CONSTANTS_OPN.LOSS
    logger.info("Train config from dummy_network_train_config_opn: %s", train_config)

    return {"network_config": network_config, "train_config": train_config}


@pytest.fixture(autouse=True, scope="session")
def cleanup_afters_tests(request):
    def cleanup():
        logger.info("Cleaning up test data")
        clean_out_folder(folder=TEST_GENERATOR_CONSTANTS.TEST_FOLDER, dry_run=False)

    request.addfinalizer(cleanup)
