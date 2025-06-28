#!/usr/bin/env -S uv run --script
"""Command-line interface for training PyTorch neural networks.

This module provides a CLI tool for training PyTorch neural networks using configurations
specified in YAML files. It handles dataset loading, model initialization, training,
and saving of model artifacts.

The main functionality includes:
- Loading and validating configuration from YAML files
- Setting up training and validation datasets with DataLoader
- Initializing PyTorch neural networks
- Training models with configurable parameters
- Saving trained models and associated metadata
- Optional logging to Weights & Biases
"""

import random
import logging
from pathlib import Path
import uuid
from copy import deepcopy
import pickle
import torch
import typer
import psutil

import lanfactory
from lanfactory.cli.utils import (
    _get_train_network_config,
)

app = typer.Typer()


@app.command()
def main(
    config_path: Path = typer.Option(..., help="Path to the YAML config file"),
    training_data_folder: Path = typer.Option(..., help="Path to the training data folder"),
    networks_path_base: Path = typer.Option(..., help="Base path for networks"),
    network_id: int = typer.Option(0, help="Network ID to train"),
    dl_workers: int = typer.Option(1, help="Number of workers for DataLoader"),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
        show_default=True,
        rich_help_panel="Logging",
        metavar="LEVEL",
        autocompletion=lambda: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    ),
):
    """Train a JAX neural network using the provided configuration.

    Args:
        config_path: Path to the YAML configuration file containing network and training settings
        training_data_folder: Directory containing the training data files
        network_id: ID of the specific network configuration to use from the config file
        dl_workers: Number of worker processes for data loading. If <= 0, will use CPU count - 2
        networks_path_base: Base directory to save trained networks and configs
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    The function:
    1. Sets up logging and loads configuration
    2. Prepares training and validation datasets
    3. Initializes the JAX network and trainer
    4. Trains the network and saves results
    """

    # Set up logging ------------------------------------------------
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    # -------------------------------------------------------------

    # Set up basic configuration ------------------------------------
    n_workers = dl_workers if dl_workers > 0 else min(12, psutil.cpu_count(logical=False) - 2)
    n_workers = max(1, n_workers)

    logger.info("Number of workers we assign to the DataLoader: %d", n_workers)

    # Load config dict (new)
    if network_id is None:
        config_dict = _get_train_network_config(yaml_config_path=config_path, net_index=0)
    else:
        config_dict = _get_train_network_config(yaml_config_path=str(config_path), net_index=network_id)

    logger.info("config dict keys: %s", config_dict.keys())

    train_config = config_dict["config_dict"]["train_config"]
    network_config = config_dict["config_dict"]["network_config"]
    extra_config = config_dict["extra_fields"]

    logger.info("TRAIN CONFIG: %s", train_config)
    logger.info("NETWORK CONFIG: %s", network_config)
    logger.info("CONFIG_DICT: %s", config_dict)

    valid_file_list = list(training_data_folder.iterdir())

    logger.info("VALID FILE LIST: %s", valid_file_list)

    random.shuffle(valid_file_list)
    n_training_files = min(len(valid_file_list), train_config["n_training_files"])
    val_idx_cutoff = int(config_dict["config_dict"]["train_val_split"] * n_training_files)

    logger.info("NUMBER OF TRAINING FILES FOUND: %d", len(valid_file_list))
    logger.info("NUMBER OF TRAINING FILES USED: %d", n_training_files)

    if torch.cuda.device_count() > 0:
        BATCH_SIZE = train_config["gpu_batch_size"]
        train_config["train_batch_size"] = BATCH_SIZE
    else:
        BATCH_SIZE = train_config["cpu_batch_size"]
        train_config["train_batch_size"] = BATCH_SIZE

    logger.info("CUDA devices: %d", torch.cuda.device_count())
    logger.info("BATCH SIZE CHOSEN: %d", BATCH_SIZE)
    # -------------------------------------------------------------

    # Make the dataloaders -------------------------------------------
    train_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=valid_file_list[:val_idx_cutoff],
        batch_size=BATCH_SIZE,
        label_lower_bound=train_config["label_lower_bound"],
        features_key=train_config["features_key"],
        label_key=train_config["label_key"],
        out_framework="torch",
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=train_config["shuffle_files"],
        batch_size=None,
        num_workers=n_workers,
        pin_memory=True,
    )

    val_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=valid_file_list[val_idx_cutoff:],
        batch_size=BATCH_SIZE,
        label_lower_bound=train_config["label_lower_bound"],
        features_key=train_config["features_key"],
        label_key=train_config["label_key"],
        out_framework="torch",
    )

    dataloader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=train_config["shuffle_files"],
        batch_size=None,
        num_workers=n_workers,
        pin_memory=True,
    )
    # -------------------------------------------------------------

    # Training and Saving -----------------------------------------
    RUN_ID = uuid.uuid1().hex

    # wandb_project_id
    wandb_project_id = "_".join([extra_config["model"], network_config["network_type"]])

    # save network config for this run
    networks_path = Path(networks_path_base) / network_config["network_type"] / extra_config["model"]
    networks_path.mkdir(parents=True, exist_ok=True)

    file_name_suffix = "_".join(
        [
            RUN_ID,
            network_config["network_type"],
            extra_config["model"],
            "network_config.pickle",
        ]
    )

    pickle.dump(
        network_config,
        open(
            networks_path / file_name_suffix,
            "wb",
        ),
    )

    # Load network
    net = lanfactory.trainers.TorchMLP(
        network_config=deepcopy(network_config),
        input_shape=train_dataset.input_dim,
        network_type=network_config["network_type"],
    )

    # Load model trainer
    model_trainer = lanfactory.trainers.ModelTrainerTorchMLP(
        train_config=deepcopy(train_config),
        model=net,
        train_dl=dataloader_train,
        valid_dl=dataloader_val,
        allow_abs_path_folder_generation=True,
        pin_memory=True,
        seed=None,
    )

    # Train model
    model_trainer.train_and_evaluate(
        output_folder=str(networks_path),
        output_file_id=extra_config["model"],
        run_id=RUN_ID,
        wandb_on=False,
        wandb_project_id=wandb_project_id,
        save_outputs=True,
    )
    # -------------------------------------------------------------


if __name__ == "__main__":
    app()
