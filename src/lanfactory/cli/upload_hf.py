#!/usr/bin/env python
"""Command-line interface for uploading models to HuggingFace Hub.

This module provides a CLI tool for uploading trained LANfactory models
to HuggingFace Hub with proper organization and metadata.

Usage:
    upload-hf --model-folder ./networks/lan/ddm/ --network-type lan --model-name ddm
"""

import logging
from pathlib import Path

import typer

from lanfactory.hf import DEFAULT_REPO_ID, VALID_NETWORK_TYPES

app = typer.Typer()


@app.command()
def main(
    model_folder: Path = typer.Option(
        ...,
        "--model-folder",
        help="Path to the folder containing trained model artifacts (should contain model_card.yaml).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    network_type: str = typer.Option(
        ...,
        "--network-type",
        help="Network type: lan, cpn, or opn.",
    ),
    model_name: str = typer.Option(
        ...,
        "--model-name",
        help="Model name (e.g., ddm, angle).",
    ),
    repo_id: str = typer.Option(
        DEFAULT_REPO_ID,
        "--repo-id",
        help=f"HuggingFace repository ID (default: {DEFAULT_REPO_ID}).",
    ),
    commit_message: str = typer.Option(
        "Upload model",
        "--commit-message",
        help="Git commit message for the upload.",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Create a private repository.",
        is_flag=True,
    ),
    create_repo: bool = typer.Option(
        False,
        "--create-repo",
        help="Create the repository if it doesn't exist.",
        is_flag=True,
    ),
    include_patterns: str = typer.Option(
        None,
        "--include-patterns",
        help="Comma-separated glob patterns for files to include.",
    ),
    exclude_patterns: str = typer.Option(
        None,
        "--exclude-patterns",
        help="Comma-separated glob patterns for files to exclude.",
    ),
    revision: str = typer.Option(
        None,
        "--revision",
        help="Branch or tag name for versioning.",
    ),
    token: str = typer.Option(
        None,
        "--token",
        envvar="HF_TOKEN",
        help="HuggingFace API token (defaults to HF_TOKEN env var).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be uploaded without uploading.",
        is_flag=True,
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
    ),
):
    """Upload a trained LANfactory model to HuggingFace Hub.

    This command uploads model artifacts to a HuggingFace repository at the path
    {network_type}/{model_name}/ (e.g., lan/ddm/).

    The model folder must contain a model_card.yaml file with model metadata.
    This YAML file is converted to a README.md for HuggingFace.

    Example:
        upload-hf --model-folder ./networks/lan/ddm/ --network-type lan --model-name ddm

    This uploads to franklab/HSSM at path lan/ddm/ by default.
    """
    # Set up logging
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate network_type
    if network_type not in VALID_NETWORK_TYPES:
        raise typer.BadParameter(
            f"network_type must be one of {list(VALID_NETWORK_TYPES)}, got: {network_type}"
        )

    # Parse patterns
    include_list = None
    if include_patterns:
        include_list = [p.strip() for p in include_patterns.split(",")]

    exclude_list = None
    if exclude_patterns:
        exclude_list = [p.strip() for p in exclude_patterns.split(",")]

    # Import here to provide better error message if huggingface_hub not installed
    try:
        from lanfactory.hf import upload_model
    except ImportError as e:
        logger.error(
            "huggingface_hub is required for HuggingFace uploads. "
            "Install it with: pip install lanfactory[hf]"
        )
        raise typer.Exit(code=1) from e

    # Show upload destination
    path_in_repo = f"{network_type}/{model_name}"
    typer.echo(f"Upload destination: {repo_id}/{path_in_repo}")

    try:
        url = upload_model(
            model_folder=model_folder,
            network_type=network_type,
            model_name=model_name,
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            create_repo=create_repo,
            include_patterns=include_list,
            exclude_patterns=exclude_list,
            revision=revision,
            token=token,
            dry_run=dry_run,
        )

        if url and not dry_run:
            typer.echo(f"\nView your model at: {url}")

    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
