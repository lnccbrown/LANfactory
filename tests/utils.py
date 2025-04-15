import os
import glob
import shutil
import pathlib

import logging

logger = logging.getLogger(__name__)


def clean_out_folder(
    folder: str | pathlib.Path | None = None, dry_run=True
) -> list[tuple[str, str]]:
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
    if folder is None:
        raise ValueError("Folder must be provided.")

    folder = pathlib.Path(folder)
    if not folder.exists():
        print(f"Folder '{folder}' does not exist.")
        return []

    # Collect all items that would be removed
    if dry_run:
        logger.info(f"Would clean out folder: {folder}")
    else:
        logger.info(f"Cleaning out folder: {folder}")

    logger.info("Contents:")
    print_tree(folder)

    # items_to_remove: list[tuple[str, str]] = []
    # for item in glob.glob(os.path.join(folder, "*")):
    # if os.path.isfile(item):
    #     items_to_remove.append((item, "file"))
    # elif os.path.isdir(item):
    #     items_to_remove.append((item, "directory"))

    if folder.exists():
        try:
            if dry_run:
                logger.info(f"Would remove folder: {folder}, via shutil.rmtree")
            else:
                shutil.rmtree(folder)
        except PermissionError:
            logger.error(f"Permission denied when trying to remove folder '{folder}'.")
        except Exception as e:
            logger.error(f"Error removing folder '{folder}': {str(e)}")
    else:
        logger.error(f"Folder '{folder}' does not exist.")

    # Actually remove the items
    # for item_path, item_type in items_to_remove:
    #     try:
    #         if item_type == "file":
    #             if dry_run:
    #                 print(f"Would remove file: {os.path.basename(item_path)}")
    #             else:
    #                 os.remove(item_path)
    #         else:
    #             if dry_run:
    #                 print(f"Would remove directory: {os.path.basename(item_path)}")
    #             else:
    #                 shutil.rmtree(item_path)
    #     except PermissionError:
    #         print(f"Permission denied when trying to remove {item_type}: {item_path}")
    #     except Exception as e:
    #         print(f"Error removing {item_type} {item_path}: {str(e)}")

    return items_to_remove


def print_tree(
    path: pathlib.Path | str,
    prefix: str = "",
    logger: logging.Logger | None = logger,
    out_str_list: list[str] = [],
) -> list[str]:
    """Print a directory tree structure starting from the given path.

    This function recursively traverses the directory structure and prints
    a visual representation of the file hierarchy.

    Arguments
    ---------
        path (Path):
            The directory path to start printing from
        prefix (str, optional):
            Prefix string used for indentation in recursive calls.
            Defaults to an empty string.

    Returns
    -------
        None

    Example
    -------
        from pathlib import Path
        print_tree(Path("./my_directory"))
    """
    path = pathlib.Path(path)
    contents = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for index, item in enumerate(contents):
        connector = "└── " if index == len(contents) - 1 else "├── "
        out_str = prefix + connector + item.name
        if logger is not None:
            logger.info(out_str)
        else:
            print(out_str)
            out_str_list.append(out_str)
        if item.is_dir():
            extension = "    " if index == len(contents) - 1 else "│   "
            out_str_list = print_tree(item, prefix + extension, logger, out_str_list)
    return out_str_list


def print_str_list(str_list: list[str], logger: logging.Logger | None = logger):
    """Print each string in a list of strings.

    This function iterates through a list of strings and prints each one,
    either using a logger if provided or standard print output.

    Arguments
    ---------
        str_list (list[str]):
            The list of strings to print
        logger (logging.Logger, optional):
            Logger to use for printing. If None, standard print is used.
            Defaults to the module's logger.

    Returns
    -------
        None

    Example
    -------
        print_str_list(["Line 1", "Line 2"])
    """
    for str_ in str_list:
        if logger is not None:
            logger.info(str_)
        else:
            print(str_)
