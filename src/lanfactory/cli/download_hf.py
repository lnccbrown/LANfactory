#!/usr/bin/env python
"""Command-line interface for downloading models from HuggingFace Hub.

This module provides a CLI tool for downloading LANfactory models
from HuggingFace Hub.

Usage:
    download-hf --network-type lan --model-name ddm --output-folder ./models/ddm/
"""

import logging
from pathlib import Path

import typer

from lanfactory.hf import DEFAULT_REPO_ID, VALID_NETWORK_TYPES

app = typer.Typer()


@app.command()
def main(
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
    output_folder: Path = typer.Option(
        ...,
        "--output-folder",
        help="Local destination folder.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    repo_id: str = typer.Option(
        DEFAULT_REPO_ID,
        "--repo-id",
        help=f"HuggingFace repository ID (default: {DEFAULT_REPO_ID}).",
    ),
    revision: str = typer.Option(
        None,
        "--revision",
        help="Specific branch, tag, or commit to download (default: main).",
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
    token: str = typer.Option(
        None,
        "--token",
        envvar="HF_TOKEN",
        help="HuggingFace API token for private repos (defaults to HF_TOKEN env var).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing files.",
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
    """Download a LANfactory model from HuggingFace Hub.

    This command downloads model artifacts from a HuggingFace repository at the path
    {network_type}/{model_name}/ (e.g., lan/ddm/).

    Example:
        download-hf --network-type lan --model-name ddm --output-folder ./models/ddm/

    This downloads from franklab/HSSM at path lan/ddm/ by default.
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
        from lanfactory.hf import download_model
    except ImportError as e:
        logger.error(
            "huggingface_hub is required for HuggingFace downloads. "
            "Install it with: pip install lanfactory[hf]"
        )
        raise typer.Exit(code=1) from e

    # Show download source
    path_in_repo = f"{network_type}/{model_name}"
    typer.echo(f"Download source: {repo_id}/{path_in_repo}")
    typer.echo(f"Output folder: {output_folder}")

    try:
        result_path = download_model(
            network_type=network_type,
            model_name=model_name,
            output_folder=output_folder,
            repo_id=repo_id,
            revision=revision,
            include_patterns=include_list,
            exclude_patterns=exclude_list,
            token=token,
            force=force,
        )

        typer.echo(f"\nModel downloaded to: {result_path}")

    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(code=1) from e
    except FileExistsError as e:
        logger.error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error("Download failed: %s", e)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
