import pytest
import ssms
import lanfactory
import os
import numpy as np
from copy import deepcopy
import torch
from .utils import clean_out_folder
import jax.numpy as jnp
from .constants import (
    TEST_GENERATOR_CONSTANTS,
    TEST_TRAIN_CONSTANTS,
    TEST_NETWORK_CONSTANTS_LAN,
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
        os.path.join(TEST_GENERATOR_CONSTANTS.OUT_FOLDER, file_)
        for file_ in os.listdir(TEST_GENERATOR_CONSTANTS.OUT_FOLDER)
    ]


def test_end_to_end_lan_mlp(
    dummy_generator_config,
    dummy_network_train_config_lan,
):
    generator_config_dict = dummy_generator_config()
    logger.info(f"Generator config: {generator_config_dict}")
    generator_config = generator_config_dict["generator_config"]
    logger.info(f"Generator config: {generator_config}")
    model_config = generator_config_dict["model_config"]
    logger.info(f"Model config: {model_config}")

    train_config_dict = dummy_network_train_config_lan
    logger.info(f"Train config: {train_config_dict}")
    network_config = train_config_dict["network_config"]
    train_config = train_config_dict["train_config"]

    file_list_ = dummy_training_data_files(generator_config, model_config)
    logger.info(f"File list: {file_list_}")
    device = TEST_GENERATOR_CONSTANTS.DEVICE

    logger.info(f"Testing end-to-end LAN MLP with model {model_config['name']}")

    # INDEPENDENT TESTS OF DATALOADERS
    # Training dataset
    jax_training_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=file_list_,
        batch_size=(
            train_config[device + "_batch_size"]
            if torch.cuda.is_available()
            else train_config[device + "_batch_size"]
        ),
        label_lower_bound=np.log(1e-10),
        features_key="lan_data",
        label_key="lan_labels",
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
        batch_size=(
            train_config[device + "_batch_size"]
            if torch.cuda.is_available()
            else train_config[device + "_batch_size"]
        ),
        label_lower_bound=np.log(1e-10),
        features_key="lan_data",
        label_key="lan_labels",
        out_framework="jax",
    )

    jax_validation_dataloader = torch.utils.data.DataLoader(
        jax_validation_dataset,
        shuffle=True,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    jax_net = lanfactory.trainers.MLPJaxFactory(
        network_config=network_config, train=True
    )

    # Test properties of jax trainer
    jax_trainer = lanfactory.trainers.ModelTrainerJaxMLP(
        train_config=train_config,
        model=jax_net,
        train_dl=jax_training_dataloader,
        valid_dl=jax_validation_dataloader,
        pin_memory=True,
    )

    train_state = jax_trainer.train_and_evaluate(
        output_folder=TEST_GENERATOR_CONSTANTS.MODEL_FOLDER,
        output_file_id=model_config["name"],
        run_id="jax",
        wandb_on=False,
        wandb_project_id="jax",
        save_data_details=True,
        verbose=1,
        save_all=True,
    )

    jax_infer = lanfactory.trainers.MLPJaxFactory(
        network_config=network_config,
        train=False,
    )

    forward_pass, forward_pass_jitted = jax_infer.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 2,
        state=os.path.join(
            TEST_GENERATOR_CONSTANTS.MODEL_FOLDER,
            "jax_lan_" + model_config["name"] + "__train_state.jax",
        ),
        add_jitted=True,
    )

    # Make input metric
    logger.info(f"Model config: {model_config}")
    theta = deepcopy(ssms.config.model_config[model_config["name"]]["default_params"])
    logger.info(f"Theta: {theta}")
    input_mat = jnp.zeros((LEN_FORWARD_PASS_DUMMY, len(theta) + 2))
    logger.info(f"Input mat shape: {input_mat.shape}")

    for i, param in enumerate(theta):
        input_mat = input_mat.at[:, i].set(jnp.ones(LEN_FORWARD_PASS_DUMMY) * param)

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
    logger.info("Input mat shape: %s", input_mat.shape)
    shape_of_input = jax_infer.load_state_from_file(
        file_path=os.path.join(
            TEST_GENERATOR_CONSTANTS.MODEL_FOLDER,
            "jax_lan_" + model_config["name"] + "__train_state.jax",
        )
    )["params"]["layers_0"]["kernel"].shape
    logger.info("Shape of input from loading state: %s", shape_of_input)

    net_out_jitted = forward_pass_jitted(input_mat)
    assert net_out_jitted.shape == (LEN_FORWARD_PASS_DUMMY, 1)

    net_out = forward_pass(input_mat)
    assert net_out.shape == (LEN_FORWARD_PASS_DUMMY, 1)

    # Compare the two outputs
    np.testing.assert_allclose(net_out, net_out_jitted, rtol=1e-4, atol=1e-4)

    clean_out_folder(folder=TEST_GENERATOR_CONSTANTS.TEST_FOLDER, dry_run=False)
