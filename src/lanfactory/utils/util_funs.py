"""Some utility functions for the lanfactory package."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _create_training_history(n_epochs: int) -> pd.DataFrame:
    """Create an empty training history with stable column dtypes."""
    return pd.DataFrame(
        {
            "epoch": np.zeros(n_epochs, dtype=int),
            "val_loss": np.zeros(n_epochs),
        }
    )


def save_configs(
    model_id: str | None = None,
    save_folder: str | Path | None = None,
    network_config: dict | None = None,
    train_config: dict | None = None,
) -> None:
    """Function to save the network and training configurations to a folder.

    Arguments
    ---------
        model_id (str):
            The id of the model.
        save_folder (str):
            The folder to save the configurations to.
        network_config (dict):
            The network configuration dictionary.
        train_config (dict):
            The training configuration dictionary.
    """

    # Generate save_folder if it doesn't yet exist
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Save network config
    pickle.dump(
        network_config,
        open(Path(save_folder) / f"{model_id}_network_config.pickle", "wb"),
    )
    print("Saved network config")
    # Save train config
    pickle.dump(
        train_config, open(Path(save_folder) / f"{model_id}_train_config.pickle", "wb")
    )
    print("Saved train config")
