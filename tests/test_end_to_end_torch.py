import pytest
import ssms
import lanfactory
import os
import numpy as np
from copy import deepcopy
import torch
from .constants import (
    TEST_GENERATOR_CONSTANTS,
    TEST_MODEL_FOLDER_CONSTANTS_TORCH,
)

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
        MODEL_FOLDER = TEST_MODEL_FOLDER_CONSTANTS_TORCH.LAN_MODEL_FOLDER
    elif train_type == "cpn":
        MODEL_FOLDER = TEST_MODEL_FOLDER_CONSTANTS_TORCH.CPN_MODEL_FOLDER
    elif train_type == "opn":
        MODEL_FOLDER = TEST_MODEL_FOLDER_CONSTANTS_TORCH.OPN_MODEL_FOLDER
    else:
        raise ValueError(f"Invalid train type: {train_type}")

    train_config_dict = request.getfixturevalue(config_fixture)
    config_generator = request.getfixturevalue(generator_config_fixture)

    generator_config_dict = config_generator()

    logger.info("Generator config: %s \n", generator_config_dict)
    generator_config = generator_config_dict["generator_config"]
    logger.info("Generator config: %s \n", generator_config)
    model_config = generator_config_dict["model_config"]
    logger.info("Model config: %s \n", model_config)

    logger.info("Train config: %s \n", train_config_dict)
    network_config = train_config_dict["network_config"]
    train_config = train_config_dict["train_config"]

    file_list_ = dummy_training_data_files(generator_config, model_config)
    logger.info("File list: %s \n", file_list_)

    logger.info(
        "Testing end-to-end %s MLP with model %s \n",
        train_type,
        model_config["name"],
    )

    # INDEPENDENT TESTS OF DATALOADERS
    # Training dataset
    torch_training_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=file_list_,
        batch_size=(train_config[TEST_GENERATOR_CONSTANTS.DEVICE + "_batch_size"]),
        label_lower_bound=np.log(1e-10),
        features_key=f"{train_type}_data",
        label_key=f"{train_type}_labels",
        out_framework="torch",
    )

    torch_training_dataloader = torch.utils.data.DataLoader(
        torch_training_dataset,
        shuffle=True,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    # Validation dataset
    torch_validation_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=file_list_,
        batch_size=(train_config[TEST_GENERATOR_CONSTANTS.DEVICE + "_batch_size"]),
        label_lower_bound=np.log(1e-10),
        features_key=f"{train_type}_data",
        label_key=f"{train_type}_labels",
        out_framework="torch",
    )

    torch_validation_dataloader = torch.utils.data.DataLoader(
        torch_validation_dataset,
        shuffle=True,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    torch_net = lanfactory.trainers.TorchMLP(
        network_config=network_config,
        input_shape=torch_training_dataset.input_dim,
        network_type=train_type,
        train=True,
    )

    logger.info(f"torch_net: {torch_net} \n")

    # Test properties of torch trainer
    torch_trainer = lanfactory.trainers.ModelTrainerTorchMLP(
        train_config=train_config,
        model=torch_net,
        train_dl=torch_training_dataloader,
        valid_dl=torch_validation_dataloader,
        pin_memory=True,
    )

    torch_trainer.train_and_evaluate(
        output_folder=MODEL_FOLDER,
        output_file_id=model_config["name"],
        run_id="runid",
        wandb_on=False,
        wandb_project_id="torch",
        verbose=1,
        save_outputs=True,
    )

    network = lanfactory.trainers.LoadTorchMLPInfer(
        model_file_path=os.path.join(
            MODEL_FOLDER,
            f"{model_config['name']}_{train_type}_runid_train_state_dict.pt",
        ),
        network_config=network_config,
        input_dim=torch_training_dataset.input_dim,
    )

    # Make input matrix
    logger.info("Model config: %s \n \n \n", model_config)
    theta = deepcopy(ssms.config.model_config[model_config["name"]]["default_params"])
    logger.info("Theta: %s \n \n \n", theta)

    if train_type == "lan":
        input_mat = torch.from_numpy(np.zeros((LEN_FORWARD_PASS_DUMMY, len(theta) + 2)).astype(np.float32))

        # Add rt
        input_mat[:, len(theta)] = torch.from_numpy(
            np.concatenate(
                [
                    np.linspace(5, 0, LEN_FORWARD_PASS_DUMMY // 2).astype(np.float32),
                    np.linspace(0, 5, LEN_FORWARD_PASS_DUMMY // 2).astype(np.float32),
                ]
            )
        )

        # Add choices
        input_mat[:, len(theta) + 1] = torch.from_numpy(
            np.concatenate(
                [
                    np.repeat(-1.0, LEN_FORWARD_PASS_DUMMY // 2),
                    np.repeat(1.0, LEN_FORWARD_PASS_DUMMY // 2),
                ]
            ).astype(np.float32)
        )
    else:
        input_mat = torch.from_numpy(np.zeros((LEN_FORWARD_PASS_DUMMY, len(theta))).astype(np.float32))

    logger.info("Input mat shape: %s \n \n \n", input_mat.shape)

    for i, param in enumerate(theta):
        input_mat[:, i] = torch.from_numpy((np.ones(LEN_FORWARD_PASS_DUMMY) * param).astype(np.float32))

    logger.info("Input mat shape: %s \n \n \n", input_mat.shape)

    net_out = network(input_mat)
    assert net_out.shape == (LEN_FORWARD_PASS_DUMMY, 1)
