# Adapted from https://github.com/lnccbrown/LAN_pipeline_minimal  authored by Alexander Fengler

import argparse
import logging
import os
import pickle  # convert to dill later
import random
import uuid
from copy import deepcopy
from pathlib import Path
from pprint import pformat

import jax
import numpy as np
import lanfactory
import psutil
import torch
import yaml


def non_negative_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer")

    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} must be a non-negative integer")
    return ivalue


def _make_train_network_configs(
    training_data_folder: str | Path = None,
    train_val_split: float = 0.9,
    save_folder: str | Path = ".",
    network_arg_dict: dict | None = None,
    train_arg_dict: dict | None = None,
    save_name: str | Path | None = None,
):
    # Load basic configs and update with provided arguments
    train_config = lanfactory.config.train_config_mlp
    train_config.update(train_arg_dict)
    network_config = lanfactory.config.network_config_mlp
    network_config.update(network_arg_dict)

    config_dict = {
        "network_config": network_config,
        "train_config": train_config,
        "training_data_folder": training_data_folder,
        "train_val_split": train_val_split,
    }

    # Serialize the configuration dictionary to a file if a save name is provided
    # TODO: Where is save_name specified? It should be passed as an argument
    if save_name:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_name = save_folder / save_name
        pickle.dump(config_dict, open(save_name, "wb"))
        print(f"Saved to: {save_name}")
    else:
        print("No save name provided, config not saved to file.")

    return {"config_dict": config_dict, "config_file_name": save_name}


def _get_train_network_config(yaml_config_path: str | Path | None = None, net_index=0):
    basic_config = yaml.safe_load(open(yaml_config_path, "rb"))
    network_type = basic_config["NETWORK_TYPE"]

    # Train output type specifies what the network output node
    # 'represents' (e.g. log-probabilities / logprob, logits, probabilities / prob)

    # Specifically for cpn, we train on logit outputs for numerical stability, then transform outputs
    # to log-probabilities when running the model in evaluation / inference mode
    train_output_type_dict = {"lan": "logprob", "cpn": "logits", "opn": "logits", "gonogo": "logits", "cpn_bce": "prob"}

    # Last layer activation depending on train output type
    output_layer_dict = {"logits": "linear", "logprob": "linear", "prob": "sigmoid"}

    # LOSS
    # 'bce' (for binary-cross-entropy), use when train output is 'prob'
    # 'bcelogit' (for binary-cross-entropy with inputs representing logits) use when train output type is 'logits', (this is standard for cpns)
    # 'huber' (usually) used when train output is 'logprob'

    train_loss_dict = {"logprob": "huber", "logits": "bcelogit", "prob": "bce"}

    data_key_dict = {
        "lan": {"features_key": "lan_data", "label_key": "lan_labels"},
        "cpn": {"features_key": "cpn_data", "label_key": "cpn_labels"},
        "opn": {"features_key": "opn_data", "label_key": "opn_labels"},
        "gonogo": {"features_key": "gonogo_data", "label_key": "gonogo_labels"},
    }

    # Network architectures
    layer_sizes = basic_config["LAYER_SIZES"][net_index]
    activations = basic_config["ACTIVATIONS"][net_index]
    activations.append(output_layer_dict[train_output_type_dict[network_type]])
    # Append last layer (type of layer depends on type of network as per train_output_type_dict dictionary above)

    # Number is set to 10000 here (an upper bound), for training on all available data (usually roughly 300 files, but has never been more than 1000)
    # For numerical experiments, one may want to artificially constraint the number of training files to teest the impact on network performance

    network_arg_dict = {"train_output_type": train_output_type_dict[network_type], "network_type": network_type}

    network_arg_dict["layer_sizes"] = layer_sizes
    network_arg_dict["activations"] = activations

    # initial train_arg_dict
    # refined in for loop in next cell
    train_arg_dict = {
        "n_epochs": basic_config["N_EPOCHS"],
        "loss": train_loss_dict[train_output_type_dict[network_type]],
        "optimizer": basic_config["OPTIMIZER_"],
        "train_output_type": train_output_type_dict[network_type],
        "n_training_files": basic_config["N_TRAINING_FILES"],
        "train_val_split": basic_config["TRAIN_VAL_SPLIT"],
        "weight_decay": basic_config["WEIGHT_DECAY"],
        "cpu_batch_size": basic_config["CPU_BATCH_SIZE"],
        "gpu_batch_size": basic_config["GPU_BATCH_SIZE"],
        "shuffle_files": basic_config["SHUFFLE"],
        "label_lower_bound": eval(basic_config["LABELS_LOWER_BOUND"], {"np": np}),
        "layer_sizes": layer_sizes,
        "activations": activations,
        "learning_rate": basic_config["LEARNING_RATE"],
        "features_key": data_key_dict[network_type]["features_key"],
        "label_key": data_key_dict[network_type]["label_key"],
        "save_history": True,
        "lr_scheduler": basic_config["LR_SCHEDULER"],
        "lr_scheduler_params": basic_config["LR_SCHEDULER_PARAMS"],
    }

    config = _make_train_network_configs(
        training_data_folder=basic_config["TRAINING_DATA_FOLDER"],
        train_val_split=basic_config["TRAIN_VAL_SPLIT"],
        save_name=None,
        train_arg_dict=train_arg_dict,
        network_arg_dict=network_arg_dict,
    )
    # Add some extra fields to our config dictionary (other scripts might need these)
    config["extra_fields"] = {"model": basic_config["MODEL"]}

    return config


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--config-path", type=Path, default=None, help="Path to the YAML config file")
    CLI.add_argument("--training-data-folder", type=Path, default=None, help="Path to the training data folder")
    CLI.add_argument(
        "--network-id", type=non_negative_int, default=0, help="Network ID to train"
    )  # can it be named net_index as used below?
    CLI.add_argument("--dl-workers", type=non_negative_int, default=1, help="Number of workers for DataLoader")
    CLI.add_argument("--networks-path-base", type=Path, default=None, help="Base path for networks")
    CLI.add_argument("--log-level", type=str, default="DEBUG", help="Logging level")

    args = CLI.parse_args()

    # Set up logging
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # handlers=[
        #     logging.StreamHandler(),
        #     logging.FileHandler("training_script.log"),
        # ],
    )
    logger = logging.getLogger(__name__)

    logger.info("Arguments passed:\n %s", pformat(vars(args)))
    n_workers = args.dl_workers if args.dl_workers > 0 else min(12, psutil.cpu_count(logical=False) - 2)
    n_workers = max(0, n_workers)

    logger.info("Number of workers we assign to the DataLoader: %d", n_workers)

    # Load config dict (new)
    config_dict = _get_train_network_config(yaml_config_path=args.config_path, net_index=args.network_id)

    logger.info("config dict keys: %s", config_dict.keys())
    train_config = config_dict["config_dict"]["train_config"]
    network_config = config_dict["config_dict"]["network_config"]
    extra_config = config_dict["extra_fields"]

    logger.info("TRAIN CONFIG: %s", train_config)
    logger.info("NETWORK CONFIG: %s", network_config)

    # Get training and validation data files
    file_list = os.listdir(args.training_data_folder)

    logger.info("TRAINING DATA FILES: %s", file_list)

    # TODO: this is weird. Improve this later
    valid_file_list = [str(args.training_data_folder) + "/" + file_ for file_ in file_list]

    logger.info("VALID FILE LIST: %s", valid_file_list)

    random.shuffle(valid_file_list)
    n_training_files = min(len(valid_file_list), train_config["n_training_files"])
    val_idx_cutoff = int(config_dict["config_dict"]["train_val_split"] * n_training_files)

    logger.info("NUMBER OF TRAINING FILES FOUND: %d", len(valid_file_list))
    logger.info("NUMBER OF TRAINING FILES USED: %d", n_training_files)

    # Check if gpu is available
    backend = jax.default_backend()
    batch_size = train_config["gpu_batch_size"] if backend == "gpu" else train_config["cpu_batch_size"]
    train_config["train_batch_size"] = batch_size

    logger.info("CUDA devices: %s", jax.devices())
    logger.info("BATCH SIZE CHOSEN: %d", batch_size)

    # Make the dataloaders
    train_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=valid_file_list[:val_idx_cutoff],
        batch_size=batch_size,
        label_lower_bound=train_config["label_lower_bound"],
        features_key=train_config["features_key"],
        label_key=train_config["label_key"],
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, shuffle=train_config["shuffle_files"], batch_size=None, num_workers=n_workers, pin_memory=True
    )

    val_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=valid_file_list[val_idx_cutoff:],
        batch_size=batch_size,
        label_lower_bound=train_config["label_lower_bound"],
        features_key=train_config["features_key"],
        label_key=train_config["label_key"],
    )

    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, shuffle=train_config["shuffle_files"], batch_size=None, num_workers=n_workers, pin_memory=True
    )

    # Load network
    net = lanfactory.trainers.MLPJaxFactory(
        network_config=deepcopy(network_config),
        train=True,
    )

    # Load model trainer
    model_trainer = lanfactory.trainers.ModelTrainerJaxMLP(
        train_config=deepcopy(train_config),
        train_dl=dataloader_train,
        valid_dl=dataloader_val,
        model=net,
        allow_abs_path_folder_generation=True,
    )

    # run_id
    run_id = uuid.uuid1().hex

    # wandb_project_id
    wandb_project_id = extra_config["model"] + "_" + network_config["network_type"]

    # save network config for this run
    networks_path = args.networks_path_base / network_config["network_type"] / extra_config["model"]
    networks_path.mkdir(parents=True, exist_ok=True)

    # try_gen_folder(folder=networks_path, allow_abs_path_folder_generation=True)
    pickle.dump(
        network_config,
        open(
            networks_path
            / (
                run_id
                + "_"
                + network_config["network_type"]
                + "_"
                + extra_config["model"]
                + "_"
                + "network_config.pickle"
            ),
            "wb",
        ),
    )

    # Train model
    model_trainer.train_and_evaluate(
        save_history=train_config["save_history"],
        output_folder=networks_path,
        output_file_id=extra_config["model"],
        run_id=run_id,
        wandb_on=False,  # TODO: make this optional
        wandb_project_id=wandb_project_id,  # TODO: make this optional
        save_all=True,
        verbose=1,
    )


if __name__ == "__main__":
    main()
