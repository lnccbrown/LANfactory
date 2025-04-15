import os
import pickle
import shutil
import glob
from typing import List, Tuple
import warnings


"""Some utility functions for the lanfactory package."""


def try_gen_folder(
    folder: str | None = None, allow_abs_path_folder_generation: bool = True
) -> None:
    """Function to generate a folder from a string. If the folder already exists, it will not be generated.

    Arguments
    ---------
        folder (str):
            The folder string to generate.
        allow_abs_path_folder_generation (bool):
            If True, the folder string is treated as an absolute path.
            If False, the folder string is treated as a relative path.

    """
    folder_list = folder.split("/")

    # Check if folder string supplied defines a relative or absolute path
    if not folder_list[0]:
        if not allow_abs_path_folder_generation:
            warnings.warn(
                "Absolute folder path provided, "
                "but setting allow_abs_path_folder_generation = False. \n"
                "No folders will be generated."
            )
            return
        else:
            rel_folder = True
            i = 1
    else:
        rel_folder = False
        i = 0

    #
    while i < len(folder_list):
        if not folder_list[i]:
            folder_list.pop(i)
        else:
            i += 1

    if rel_folder:
        folder_list[1] = "/" + folder_list[1]
        folder_list.pop(0)

    tmp_dir_str = ""
    i = 0

    while i < len(folder_list):
        if i == 0:
            tmp_dir_str += folder_list[i]
        else:
            tmp_dir_str += "/" + folder_list[i]

        if not os.path.exists(tmp_dir_str):
            print("Did not find folder: ", tmp_dir_str)
            print("Creating it...")
            try:
                os.makedirs(tmp_dir_str)
            except Exception as e:
                print(e)
                print("Some problem occured when creating the directory ", tmp_dir_str)
        else:
            print("Found folder: ", tmp_dir_str)
            print("Moving on...")
        i += 1

    return


def save_configs(
    model_id: str | None = None,
    save_folder: str | None = None,
    network_config: dict | None = None,
    train_config: dict | None = None,
    allow_abs_path_folder_generation: bool = True,
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
        allow_abs_path_folder_generation (bool):
            If True, the folder string is treated as an absolute path.
            If False, the folder string is treated as a relative path.
    """

    # Generate save_folder if it doesn't yet exist
    try_gen_folder(
        folder=save_folder,
        allow_abs_path_folder_generation=allow_abs_path_folder_generation,
    )

    # Save network config
    pickle.dump(
        network_config,
        open(save_folder + "/" + model_id + "_network_config.pickle", "wb"),
    )
    print("Saved network config")
    # Save train config
    pickle.dump(
        train_config, open(save_folder + "/" + model_id + "_train_config.pickle", "wb")
    )
    print("Saved train config")
    return


def clean_out_folder(folder="tests/test_data", dry_run=True) -> List[Tuple[str, str]]:
    """Safely remove all contents of a folder with optional dry-run mode.

    Args:
        folder (str): Path to the folder to clean out
        dry_run (bool): If True, only prints what would be removed without actually deleting

    Returns:
        List[Tuple[str, str]]: List of tuples containing (item_path, action_type)
        where action_type is either 'file' or 'directory'

    Example:
        # Preview what would be deleted
        items = clean_out_folder("test_data", dry_run=True)
        for path, item_type in items:
            print(f"Would remove {item_type}: {path}")

        # Actually delete the items
        clean_out_folder("test_data", dry_run=False)
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return []

    items_to_remove = []

    # Collect all items that would be removed
    for item in glob.glob(os.path.join(folder, "*")):
        if os.path.isfile(item):
            items_to_remove.append((item, "file"))
        elif os.path.isdir(item):
            items_to_remove.append((item, "directory"))

    # Actually remove the items
    for item_path, item_type in items_to_remove:
        try:
            if item_type == "file":
                if dry_run:
                    print(f"Would remove file: {os.path.basename(item_path)}")
                else:
                    os.remove(item_path)
            else:
                if dry_run:
                    print(f"Would remove directory: {os.path.basename(item_path)}")
                else:
                    shutil.rmtree(item_path)
        except PermissionError:
            print(f"Permission denied when trying to remove {item_type}: {item_path}")
        except Exception as e:
            print(f"Error removing {item_type} {item_path}: {str(e)}")

    return items_to_remove
