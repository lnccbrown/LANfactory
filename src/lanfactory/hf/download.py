"""Download utilities for HuggingFace Hub.

This module provides functions to download LANfactory models
from HuggingFace Hub.
"""

import logging
import shutil
from pathlib import Path

from lanfactory.hf import DEFAULT_REPO_ID, VALID_NETWORK_TYPES

logger = logging.getLogger(__name__)


def download_model(
    network_type: str,
    model_name: str,
    output_folder: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    token: str | None = None,
    force: bool = False,
) -> Path:
    """Download a model from HuggingFace Hub.

    Parameters
    ----------
    network_type : str
        Network type (e.g., "lan", "cpn", "opn").
    model_name : str
        Model name (e.g., "ddm", "angle").
    output_folder : Path
        Local destination folder.
    repo_id : str
        HuggingFace repository ID (default: "franklab/HSSM").
    revision : str | None
        Specific branch/tag/commit to download (default: main).
    include_patterns : list[str] | None
        Glob patterns for files to include.
    exclude_patterns : list[str] | None
        Glob patterns for files to exclude.
    token : str | None
        HuggingFace API token for private repos.
    force : bool
        Whether to overwrite existing files.

    Returns
    -------
    Path
        Path to the downloaded model folder.

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
    ValueError
        If network_type is not valid.
    FileExistsError
        If output_folder exists and force is False.
    """
    # Validate inputs
    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError(
            f"network_type must be one of {list(VALID_NETWORK_TYPES)}, got: {network_type}"
        )

    output_folder = Path(output_folder)

    # Check if output folder exists
    if output_folder.exists() and not force:
        raise FileExistsError(
            f"Output folder already exists: {output_folder}. Use --force to overwrite."
        )

    try:  # pragma: no cover (requires huggingface_hub optional dependency)
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HuggingFace downloads. "
            "Install it with: pip install lanfactory[hf]"
        ) from exc

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Build the path prefix for this model
    path_prefix = f"{network_type}/{model_name}/"

    # List files in the repository
    try:
        all_files = list_repo_files(
            repo_id=repo_id,
            revision=revision,
            token=token,
        )
    except Exception as e:
        logger.error(f"Failed to list repository files: {e}")
        raise

    # Filter files by path prefix
    model_files = [f for f in all_files if f.startswith(path_prefix)]

    if not model_files:
        raise FileNotFoundError(
            f"No files found at {repo_id}/{path_prefix}. "
            f"Available paths: {set(f.split('/')[0] for f in all_files if '/' in f)}"
        )

    # Apply include/exclude patterns
    if include_patterns:
        filtered_files = []
        for f in model_files:
            filename = Path(f).name
            for pattern in include_patterns:
                if Path(filename).match(pattern):
                    filtered_files.append(f)
                    break
        model_files = filtered_files

    if exclude_patterns:
        filtered_files = []
        for f in model_files:
            filename = Path(f).name
            excluded = False
            for pattern in exclude_patterns:
                if Path(filename).match(pattern):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append(f)
        model_files = filtered_files

    if not model_files:
        raise FileNotFoundError(
            f"No files matching patterns found at {repo_id}/{path_prefix}"
        )

    logger.info(f"Downloading {len(model_files)} files from {repo_id}/{path_prefix}")

    # Download each file
    downloaded_files = []
    for file_path in model_files:
        filename = Path(file_path).name
        logger.info(f"  Downloading: {filename}")

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                revision=revision,
                token=token,
            )

            # Copy to output folder
            dest_path = output_folder / filename
            shutil.copy2(local_path, dest_path)
            downloaded_files.append(dest_path)

        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")
            raise

    logger.info(f"Downloaded {len(downloaded_files)} files to {output_folder}")
    print("\nDownload successful!")
    print(f"Model saved to: {output_folder}")
    print(f"Files downloaded: {len(downloaded_files)}")

    return output_folder
