import pytest
import ssms
import lanfactory
import os
import numpy as np
from copy import deepcopy
import jax.numpy as jnp
import torch
from .constants import (
    TEST_GENERATOR_CONSTANTS,
    TEST_MODEL_FOLDER_CONSTANTS_JAX,
)

# import logger
import logging

logger = logging.getLogger(__name__)

LEN_FORWARD_PASS_DUMMY = 2000


def dummy_training_data_files(generator_config, model_config, save=True):
    """Fixture providing a dummy training data for testing."""
    os.makedirs(generator_config["output_folder"], exist_ok=True)
    for i in range(TEST_GENERATOR_CONSTANTS.N_DATA_FILES):
        # log progress
        logger.info(
            "Generating training data for file %d of %d",
            i + 1,
            TEST_GENERATOR_CONSTANTS.N_DATA_FILES,
        )
        my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
            generator_config=generator_config, model_config=model_config
        )
        _ = my_dataset_generator.generate_data_training_uniform(save=save)

    return [
        os.path.join(generator_config["output_folder"], file_)
        for file_ in os.listdir(generator_config["output_folder"])
    ]


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    "train_type, config_fixture, generator_config_fixture",
    [
        (
            "cpn",
            "dummy_network_train_config_cpn",
            "dummy_generator_config_simple_two_choices",
        ),
        (
            "opn",
            "dummy_network_train_config_opn",
            "dummy_generator_config_simple_two_choices",
        ),
        (
            "lan",
            "dummy_network_train_config_lan",
            "dummy_generator_config",
        ),
    ],
)
def test_end_to_end_lan_mlp(
    train_type,
    config_fixture,
    request,
    generator_config_fixture,
):
    """End-to-end test for LAN/CPN/OPN MLP models.

    Tests the complete workflow from data generation to model training and evaluation.

    Args:
        train_type: Type of network to train ('lan', 'cpn', or 'opn')
        config_fixture: Fixture name for network and training configuration
        request: Pytest request object for fixture access
        generator_config_fixture: Fixture name for data generator configuration
    """
    if train_type == "lan":
        MODEL_FOLDER = TEST_MODEL_FOLDER_CONSTANTS_JAX.LAN_MODEL_FOLDER
    elif train_type == "cpn":
        MODEL_FOLDER = TEST_MODEL_FOLDER_CONSTANTS_JAX.CPN_MODEL_FOLDER
    elif train_type == "opn":
        MODEL_FOLDER = TEST_MODEL_FOLDER_CONSTANTS_JAX.OPN_MODEL_FOLDER
    else:
        raise ValueError(f"Invalid train type: {train_type}")

    train_config_dict = request.getfixturevalue(config_fixture)
    config_generator = request.getfixturevalue(generator_config_fixture)

    generator_config_dict = config_generator()

    logger.info("Generator config: %s", generator_config_dict)
    generator_config = generator_config_dict["generator_config"]
    logger.info("Generator config: %s", generator_config)
    model_config = generator_config_dict["model_config"]
    logger.info("Model config: %s", model_config)

    logger.info("Train config: %s", train_config_dict)
    network_config = train_config_dict["network_config"]
    train_config = train_config_dict["train_config"]

    file_list_ = dummy_training_data_files(generator_config, model_config)
    logger.info("File list: %s", file_list_)

    logger.info("Testing end-to-end %s MLP with model %s", train_type, model_config["name"])

    # INDEPENDENT TESTS OF DATALOADERS
    # Training dataset
    jax_training_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=file_list_,
        batch_size=(train_config[TEST_GENERATOR_CONSTANTS.DEVICE + "_batch_size"]),
        label_lower_bound=np.log(1e-10),
        features_key=f"{train_type}_data",
        label_key=f"{train_type}_labels",
        out_framework="jax",
    )

    jax_training_dataloader = torch.utils.data.DataLoader(
        jax_training_dataset,
        shuffle=True,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    # Validation dataset
    jax_validation_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=file_list_,
        batch_size=(train_config[TEST_GENERATOR_CONSTANTS.DEVICE + "_batch_size"]),
        label_lower_bound=np.log(1e-10),
        features_key=f"{train_type}_data",
        label_key=f"{train_type}_labels",
        out_framework="jax",
    )

    jax_validation_dataloader = torch.utils.data.DataLoader(
        jax_validation_dataset,
        shuffle=True,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    jax_net = lanfactory.trainers.MLPJaxFactory(network_config=network_config, train=True)

    # Test properties of jax trainer
    jax_trainer = lanfactory.trainers.ModelTrainerJaxMLP(
        train_config=train_config,
        model=jax_net,
        train_dl=jax_training_dataloader,
        valid_dl=jax_validation_dataloader,
        pin_memory=True,
    )

    _ = jax_trainer.train_and_evaluate(
        output_folder=MODEL_FOLDER,
        output_file_id=model_config["name"],
        run_id="jax",
        wandb_on=False,
        wandb_project_id="jax",
        save_outputs=True,
        verbose=1,
    )

    jax_infer = lanfactory.trainers.MLPJaxFactory(
        network_config=network_config,
        train=False,
    )

    # TODO also test this with returned test_state!
    forward_pass, forward_pass_jitted = jax_infer.make_forward_partial(
        seed=42,
        input_dim=(model_config["n_params"] + 2 if train_type == "lan" else model_config["n_params"]),
        state=os.path.join(
            MODEL_FOLDER,
            (
                "jax_"
                + (train_type if train_type == "lan" else "cpn")
                + "_"
                + model_config["name"]
                + "__train_state.jax"
            ),
        ),
        add_jitted=True,
    )

    # Make input metric
    logger.info("Model config: %s", model_config)
    theta = deepcopy(ssms.config.model_config[model_config["name"]]["default_params"])
    logger.info("Theta: %s", theta)

    if train_type == "lan":
        input_mat = jnp.zeros((LEN_FORWARD_PASS_DUMMY, len(theta) + 2))

        # Add rt
        input_mat = input_mat.at[:, len(theta)].set(
            jnp.array(
                np.concatenate(
                    [
                        np.linspace(5, 0, LEN_FORWARD_PASS_DUMMY // 2).astype(np.float32),
                        np.linspace(0, 5, LEN_FORWARD_PASS_DUMMY // 2).astype(np.float32),
                    ]
                )
            )
        )

        # Add choices
        input_mat = input_mat.at[:, len(theta) + 1].set(
            jnp.array(
                np.concatenate(
                    [
                        np.repeat(-1.0, LEN_FORWARD_PASS_DUMMY // 2),
                        np.repeat(1.0, LEN_FORWARD_PASS_DUMMY // 2),
                    ]
                ).astype(np.float32)
            )
        )
    else:
        input_mat = jnp.zeros((LEN_FORWARD_PASS_DUMMY, len(theta)))

    logger.info("Input mat shape: %s", input_mat.shape)

    for i, param in enumerate(theta):
        input_mat = input_mat.at[:, i].set(jnp.ones(LEN_FORWARD_PASS_DUMMY) * param)

    logger.info("Input mat shape: %s", input_mat.shape)
    shape_of_input = jax_infer.load_state_from_file(
        file_path=os.path.join(
            MODEL_FOLDER,
            (
                "jax_"
                + (train_type if train_type == "lan" else "cpn")
                + "_"
                + model_config["name"]
                + "__train_state.jax"
            ),
        )
    )["params"]["layers_0"]["kernel"].shape
    logger.info("Shape of input from loading state: %s", shape_of_input)

    net_out_jitted = forward_pass_jitted(input_mat)
    assert net_out_jitted.shape == (LEN_FORWARD_PASS_DUMMY, 1)

    net_out = forward_pass(input_mat)
    assert net_out.shape == (LEN_FORWARD_PASS_DUMMY, 1)

    # Compare the two outputs
    np.testing.assert_allclose(net_out, net_out_jitted, rtol=1e-4, atol=1e-4)
