# import argparse
import logging
from pathlib import Path
import pickle
import yaml
import numpy as np
import lanfactory

logger = logging.getLogger(__name__)


# def non_negative_int(value):
#     """Convert string value to non-negative integer.

#     Args:
#         value: String value to convert to integer.

#     Returns:
#         int: Non-negative integer value.

#     Raises:
#         argparse.ArgumentTypeError: If value cannot be converted to integer or is negative.
#     """
#     try:
#         ivalue = int(value)
#     except ValueError:
#         raise argparse.ArgumentTypeError(f"{value} is not an integer")

#     if ivalue < 0:
#         raise argparse.ArgumentTypeError(f"{value} must be a non-negative integer")
#     return ivalue


# def none_or_str(value: str) -> str | None:
#     """Convert "None" string to None, otherwise return string value unchanged.

#     Args:
#         value: String value to check.

#     Returns:
#         Optional[str]: None if input is "None", otherwise returns input string unchanged.
#     """
#     if value == "None":
#         return None
#     return value


# def none_or_int(value: str) -> int | None:
#     """Convert "None" string to None, otherwise convert to integer.

#     Args:
#         value: String value to convert.

#     Returns:
#         Optional[int]: None if input is "None", otherwise returns integer value.
#     """
#     if value == "None":
#         return None
#     return int(value)


def _make_train_network_configs(
    training_data_folder: str | Path | None = None,
    train_val_split: float = 0.9,
    save_folder: str | Path = ".",
    network_arg_dict: dict | None = None,
    train_arg_dict: dict | None = None,
    save_name: str | Path | None = None,
):
    # Load basic configs and update with provided arguments
    train_config = lanfactory.config.train_config_mlp
    if train_arg_dict is not None:
        train_config.update(train_arg_dict)

    network_config = lanfactory.config.network_config_mlp
    if network_arg_dict is not None:
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
    if yaml_config_path is not None:
        basic_config = yaml.safe_load(open(yaml_config_path, "rb"))
        network_type = basic_config["NETWORK_TYPE"]
    else:
        raise ValueError("No YAML config path provided")

    # Train output type specifies what the network output node
    # 'represents' (e.g. log-probabilities / logprob, logits, probabilities / prob)

    # Specifically for cpn, we train on logit outputs for numerical stability, then transform outputs
    # to log-probabilities when running the model in evaluation / inference mode
    train_output_type_dict = {
        "lan": "logprob",
        "cpn": "logits",
        "opn": "logits",
        "gonogo": "logits",
        "cpn_bce": "prob",
    }

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

    network_arg_dict = {
        "train_output_type": train_output_type_dict[network_type],
        "network_type": network_type,
    }

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
