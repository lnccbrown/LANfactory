"""Upload utilities for HuggingFace Hub.

This module provides functions to upload trained LANfactory models
to HuggingFace Hub with proper organization and metadata.
"""

import logging
import shutil
import tempfile
from pathlib import Path

from lanfactory.hf import DEFAULT_REPO_ID, VALID_NETWORK_TYPES

logger = logging.getLogger(__name__)

# Default file patterns to include in uploads
DEFAULT_INCLUDE_PATTERNS = [
    "*.onnx",
    "*.pt",
    "*.jax",
    "*_config.pickle",
    "*.csv",
    "model_card.yaml",
]


def upload_model(
    model_folder: Path,
    network_type: str,
    model_name: str,
    repo_id: str = DEFAULT_REPO_ID,
    commit_message: str = "Upload model",
    private: bool = False,
    create_repo: bool = False,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
) -> str | None:
    """Upload a trained model to HuggingFace Hub.

    Parameters
    ----------
    model_folder : Path
        Path to the folder containing trained model artifacts.
    network_type : str
        Network type (e.g., "lan", "cpn", "opn").
    model_name : str
        Model name (e.g., "ddm", "angle").
    repo_id : str
        HuggingFace repository ID (default: "franklab/HSSM").
    commit_message : str
        Git commit message for the upload.
    private : bool
        Whether to create a private repository.
    create_repo : bool
        Whether to create the repository if it doesn't exist.
    include_patterns : list[str] | None
        Glob patterns for files to include.
    exclude_patterns : list[str] | None
        Glob patterns for files to exclude.
    revision : str | None
        Branch or tag name for versioning.
    token : str | None
        HuggingFace API token.
    dry_run : bool
        If True, show what would be uploaded without uploading.

    Returns
    -------
    str | None
        URL of the uploaded model, or None if dry_run is True.

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
    FileNotFoundError
        If model_folder doesn't exist or is missing required files.
    ValueError
        If network_type is not valid.
    """
    # Validate inputs
    model_folder = Path(model_folder)
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")

    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError(
            f"network_type must be one of {list(VALID_NETWORK_TYPES)}, got: {network_type}"
        )

    # Check for model_card.yaml
    model_card_path = model_folder / "model_card.yaml"
    if not model_card_path.exists():
        raise FileNotFoundError(
            f"model_card.yaml not found in {model_folder}. "
            "Please create a model_card.yaml file with model metadata."
        )

    # Use default patterns if not specified
    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE_PATTERNS

    # Collect files to upload
    files_to_upload = _collect_files(model_folder, include_patterns, exclude_patterns)

    if not files_to_upload:
        raise FileNotFoundError(
            f"No files matching patterns {include_patterns} found in {model_folder}"
        )

    # Log what will be uploaded
    path_in_repo = f"{network_type}/{model_name}"
    logger.info(f"Upload destination: {repo_id}/{path_in_repo}")
    logger.info(f"Files to upload ({len(files_to_upload)}):")
    for f in files_to_upload:
        logger.info(f"  - {f.name}")

    if dry_run:
        logger.info("DRY RUN: No files were uploaded.")
        print(
            f"\nDRY RUN: Would upload {len(files_to_upload)} files to {repo_id}/{path_in_repo}"
        )
        for f in files_to_upload:
            print(f"  - {f.name}")
        return None

    return _upload_to_hf(  # pragma: no cover
        model_folder=model_folder,
        model_name=model_name,
        files_to_upload=files_to_upload,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
        create_repo=create_repo,
        revision=revision,
        token=token,
    )


def _upload_to_hf(  # pragma: no cover
    model_folder: Path,
    model_name: str,
    files_to_upload: list[Path],
    path_in_repo: str,
    repo_id: str,
    commit_message: str,
    private: bool,
    create_repo: bool,
    revision: str | None,
    token: str | None,
) -> str:
    """HF-dependent implementation of upload_model."""
    try:
        from huggingface_hub import HfApi, create_repo as hf_create_repo
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HuggingFace uploads. "
            "Install it with: pip install lanfactory[hf]"
        ) from exc

    api = HfApi(token=token)

    if create_repo:
        try:
            hf_create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
                token=token,
            )
            logger.info(f"Repository created/verified: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise

    from lanfactory.hf.model_card import load_model_card_yaml, write_readme

    config = load_model_card_yaml(model_folder)
    readme_path = write_readme(model_folder, config, model_name)
    files_to_upload.append(readme_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for file_path in files_to_upload:
            dest = tmp_path / file_path.name
            shutil.copy2(file_path, dest)

        try:
            api.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
                revision=revision,
                token=token,
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    url = f"https://huggingface.co/{repo_id}/tree/{revision or 'main'}/{path_in_repo}"
    logger.info(f"Upload successful: {url}")
    print("\nUpload successful!")
    print(f"View your model at: {url}")

    return url


def _collect_files(
    folder: Path,
    include_patterns: list[str],
    exclude_patterns: list[str] | None,
) -> list[Path]:
    """Collect files matching include patterns and not matching exclude patterns.

    Parameters
    ----------
    folder : Path
        Folder to search for files.
    include_patterns : list[str]
        Glob patterns for files to include.
    exclude_patterns : list[str] | None
        Glob patterns for files to exclude.

    Returns
    -------
    list[Path]
        List of file paths to upload.
    """
    files = set()

    # Collect files matching include patterns
    for pattern in include_patterns:
        files.update(folder.glob(pattern))

    # Remove files matching exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            excluded = set(folder.glob(pattern))
            files -= excluded

    # Filter to only regular files (not directories)
    files = [f for f in files if f.is_file()]

    return sorted(files)
