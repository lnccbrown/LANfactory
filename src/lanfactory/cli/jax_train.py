#!/usr/bin/env -S uv run --script
"""Command-line interface for training JAX neural networks.

This module provides a CLI tool for training JAX neural networks using configurations
specified in YAML files. It handles dataset loading, model initialization, training,
and saving of model artifacts.

The main functionality includes:
- Loading and validating configuration from YAML files
- Setting up training and validation datasets with DataLoader
- Initializing JAX neural networks
- Training models with configurable parameters
- Saving trained models and associated metadata
- Optional logging to Weights & Biases
"""

import logging
import pickle  # convert to dill later
import random
import uuid
from copy import deepcopy
from importlib.resources import as_file, files
from pathlib import Path

import jax
import lanfactory
import psutil
import typer
from lanfactory.cli.utils import (
    _get_train_network_config,
)
from torch.utils.data import DataLoader

app = typer.Typer()


@app.command()
def main(
    config_path: Path = typer.Option(None, help="Path to the YAML config file"),
    training_data_folder: Path = typer.Option(
        None,
        help="Path to the training data folder. "
        "Optional if --data-generation-experiment-id is provided (will be derived from MLflow). "
        "If both are provided, validates that MLflow files exist in this folder.",
    ),
    network_id: int = typer.Option(0, help="Network ID to train"),
    dl_workers: int = typer.Option(1, help="Number of workers for DataLoader"),
    networks_path_base: Path = typer.Option(..., help="Base path for networks"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the pipeline without training. Useful for testing configurations.",
        is_flag=True,
    ),
    mlflow_run_name: str = typer.Option(
        None,
        "--mlflow-run-name",
        help="MLflow run name. If provided, enables MLflow tracking.",
    ),
    mlflow_experiment_name: str = typer.Option(
        None,
        "--mlflow-experiment-name",
        help="MLflow experiment name. Defaults to MLFLOW_EXPERIMENT_NAME env var.",
    ),
    mlflow_run_id: str = typer.Option(
        None,
        "--mlflow-run-id",
        help="(Advanced) MLflow Run ID to resume. If provided, will continue logging to existing run.",
    ),
    data_generation_experiment_id: str = typer.Option(
        None,
        "--data-generation-experiment-id",
        help="MLflow Experiment ID of the data generation experiment. "
        "All runs in this experiment generated the training data. "
        "If provided, training data location will be derived from MLflow and lineage will be logged.",
    ),
    mlflow_tracking_uri: str = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="MLflow tracking URI (e.g., 'sqlite:///path/to/mlflow.db' or 'http://mlflow-server:5000'). "
        "Defaults to MLFLOW_TRACKING_URI env var, then 'sqlite:///mlflow.db'.",
    ),
    mlflow_artifact_location: str = typer.Option(
        None,
        "--mlflow-artifact-location",
        help="Root directory for MLflow artifacts. "
        "Defaults to MLFLOW_ARTIFACT_LOCATION env var, then './mlruns'.",
    ),
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
    """Train a JAX neural network using the provided configuration."""

    # Set up logging ------------------------------------------------
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    # -------------------------------------------------------------

    # Set up basic configuration ------------------------------------
    n_workers = (
        dl_workers if dl_workers > 0 else min(12, psutil.cpu_count(logical=False) - 2)
    )
    n_workers = max(1, n_workers)

    logger.info("Number of workers we assign to the DataLoader: %d", n_workers)

    if config_path is None:
        logger.warning("No config path provided, using default configuration.")
        with as_file(
            files("lanfactory.cli") / "config_network_training_lan.yaml"
        ) as default_config:
            config_path = default_config

    config_dict = _get_train_network_config(
        yaml_config_path=str(config_path),
        net_index=network_id,
    )

    logger.info("config dict keys: %s", config_dict.keys())

    train_config = config_dict["config_dict"]["train_config"]
    network_config = config_dict["config_dict"]["network_config"]
    extra_config = config_dict["extra_fields"]

    logger.info("TRAIN CONFIG: %s", train_config)
    logger.info("NETWORK CONFIG: %s", network_config)
    logger.info("CONFIG_DICT: %s", config_dict)

    # Validate input arguments and determine training data source
    # Three modes:
    # 1. MLflow-first: data_generation_experiment_id provided, derive training_data_folder
    # 2. Validation: Both provided, verify MLflow files exist in training_data_folder
    # 3. Traditional: Only training_data_folder provided

    if data_generation_experiment_id is None and training_data_folder is None:
        raise ValueError(
            "Must provide either --data-generation-experiment-id or --training-data-folder. "
            "Cannot proceed without a data source."
        )

    # Determine if MLflow tracking should be enabled
    # Enable if: --mlflow-run-name provided OR --mlflow-run-id provided OR --data-generation-experiment-id provided
    mlflow_tracking_enabled = (
        mlflow_run_name is not None
        or mlflow_run_id is not None
        or data_generation_experiment_id is not None
    )

    # Initialize MLflow if needed
    mlflow_lineage_info = None
    tracking_uri = None

    if mlflow_tracking_enabled:
        try:
            import mlflow
            import os

            # Set tracking URI with priority: CLI arg > env var > default
            if mlflow_tracking_uri:
                tracking_uri = mlflow_tracking_uri
            else:
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking URI: %s", tracking_uri)

            # Set experiment with optional artifact location
            # Priority: CLI arg > env var
            mlflow_experiment = mlflow_experiment_name or os.getenv(
                "MLFLOW_EXPERIMENT_NAME"
            )
            if mlflow_experiment:
                # Determine artifact location with priority: CLI arg > env var > None (default)
                if mlflow_artifact_location:
                    artifact_location = mlflow_artifact_location
                else:
                    artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION", None)

                if artifact_location:
                    # Ensure artifact location is absolute path
                    artifact_location = str(Path(artifact_location).absolute())

                    # Try to get existing experiment, or create new one with artifact location
                    experiment = mlflow.get_experiment_by_name(mlflow_experiment)
                    if experiment is None:
                        # Create new experiment with artifact location
                        mlflow.create_experiment(
                            mlflow_experiment, artifact_location=artifact_location
                        )
                        logger.info(
                            "Created MLflow experiment: %s (artifacts: %s)",
                            mlflow_experiment,
                            artifact_location,
                        )
                    mlflow.set_experiment(mlflow_experiment)
                else:
                    mlflow.set_experiment(mlflow_experiment)
                    logger.info(
                        "Set MLflow experiment: %s (default artifact location)",
                        mlflow_experiment,
                    )

            # Start/resume run if provided, otherwise create new run
            if mlflow_run_id:
                mlflow.start_run(run_id=mlflow_run_id)
                logger.info("Resumed MLflow run: %s", mlflow_run_id)
            else:
                run = mlflow.start_run(run_name=mlflow_run_name)
                logger.info(
                    "Started new MLflow run: %s (name: %s)",
                    run.info.run_id,
                    mlflow_run_name,
                )
        except Exception as e:
            logger.error("Failed to initialize MLflow: %s", e)
            mlflow_tracking_enabled = False

    # Get data lineage information if experiment ID provided
    if data_generation_experiment_id:
        try:
            from lanfactory.utils import get_files_from_data_generation_experiment

            if not mlflow_tracking_enabled:
                # Need to initialize MLflow just for querying
                import mlflow
                import os

                # Use same logic as above for tracking URI
                if mlflow_tracking_uri:
                    tracking_uri = mlflow_tracking_uri
                else:
                    tracking_uri = os.getenv(
                        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
                    )
                mlflow.set_tracking_uri(tracking_uri)

            # Get file information from MLflow
            logger.info("Querying MLflow experiment: %s", data_generation_experiment_id)
            mlflow_lineage_info = get_files_from_data_generation_experiment(
                experiment_id=data_generation_experiment_id, tracking_uri=tracking_uri
            )

            logger.info(
                "MLflow reports %d runs with %d total files",
                mlflow_lineage_info["num_runs"],
                mlflow_lineage_info["total_files"],
            )

            # Get the data output folder from the first run (they should all be the same)
            if mlflow_lineage_info["runs_info"]:
                first_run_info = mlflow_lineage_info["runs_info"][0]
                mlflow_data_folder = first_run_info.get("data_output_folder")

                if mlflow_data_folder:
                    logger.info("MLflow recorded data folder: %s", mlflow_data_folder)

                    # Mode 1: MLflow-first (no training_data_folder provided)
                    if training_data_folder is None:
                        training_data_folder = Path(mlflow_data_folder)
                        logger.info(
                            "Using training data folder from MLflow: %s",
                            training_data_folder,
                        )
                    # Mode 2: Validation (both provided)
                    else:
                        logger.info(
                            "Validation mode: Checking MLflow files against provided folder"
                        )

            # If still no training_data_folder, we have a problem
            if training_data_folder is None:
                raise ValueError(
                    f"Could not determine training_data_folder from MLflow experiment {data_generation_experiment_id}. "
                    "Please provide --training-data-folder explicitly."
                )

        except Exception as e:
            logger.error("Failed to query MLflow: %s", e)
            if training_data_folder is None:
                raise ValueError(
                    f"Failed to get training data from MLflow and no --training-data-folder provided: {e}"
                ) from e
            else:
                logger.warning(
                    "MLflow query failed, falling back to training_data_folder only"
                )

    # Get actual files from training_data_folder
    if not training_data_folder.exists():
        raise FileNotFoundError(
            f"Training data folder does not exist: {training_data_folder}"
        )

    valid_file_list = [
        f for f in training_data_folder.iterdir() if f.suffix == ".pickle"
    ]

    logger.info("NUMBER OF TRAINING FILES FOUND in folder: %d", len(valid_file_list))

    # Mode 2: Validation - verify MLflow files exist in training_data_folder
    if mlflow_lineage_info and training_data_folder:
        expected_files = set(mlflow_lineage_info["all_files"])
        actual_files = set(f.name for f in valid_file_list)

        missing_files = expected_files - actual_files
        extra_files = actual_files - expected_files

        if missing_files:
            raise FileNotFoundError(
                f"MLflow experiment {data_generation_experiment_id} expects {len(expected_files)} files, "
                f"but {len(missing_files)} are missing from {training_data_folder}. "
                f"Missing files: {list(missing_files)[:10]}{'...' if len(missing_files) > 10 else ''}"
            )

        if extra_files:
            logger.warning(
                "Found %d extra files in training folder not tracked by MLflow: %s",
                len(extra_files),
                list(extra_files)[:5],
            )

        logger.info(
            "✓ Validation passed: All %d MLflow-tracked files found in training folder",
            len(expected_files),
        )

        # Log detailed lineage
        from lanfactory.utils import log_training_data_lineage

        n_training_files = min(len(valid_file_list), train_config["n_training_files"])
        log_training_data_lineage(
            data_generation_experiment_id=data_generation_experiment_id,
            training_data_folder=training_data_folder,
            valid_file_list=valid_file_list,
            n_training_files=n_training_files,
            tracking_uri=tracking_uri,
        )

    # NOW shuffle and prepare for training
    random.shuffle(valid_file_list)
    n_training_files = min(len(valid_file_list), train_config["n_training_files"])
    val_idx_cutoff = int(
        config_dict["config_dict"]["train_val_split"] * n_training_files
    )

    logger.info("NUMBER OF TRAINING FILES USED: %d", n_training_files)
    logger.info("TRAIN/VAL SPLIT AT INDEX: %d", val_idx_cutoff)

    # Check if gpu is available
    backend = jax.default_backend()
    BATCH_SIZE = (
        train_config["gpu_batch_size"]
        if backend == "gpu"
        else train_config["cpu_batch_size"]
    )
    train_config["train_batch_size"] = BATCH_SIZE

    logger.info("CUDA devices: %s", jax.devices())
    logger.info("BATCH SIZE CHOSEN: %d", BATCH_SIZE)
    # ----------------------------------------------------------------

    # Make the dataloaders -------------------------------------------
    train_dataset = lanfactory.trainers.DatasetTorch(
        file_ids=valid_file_list[:val_idx_cutoff],
        batch_size=BATCH_SIZE,
        label_lower_bound=train_config["label_lower_bound"],
        features_key=train_config["features_key"],
        label_key=train_config["label_key"],
    )

    dataloader_train = DataLoader(
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
    )

    dataloader_val = DataLoader(
        val_dataset,
        shuffle=train_config["shuffle_files"],
        batch_size=None,
        num_workers=n_workers,
        pin_memory=True,
    )
    # -------------------------------------------------------------

    # Dry run validation ------------------------------------------
    if dry_run:
        logger.info("DRY RUN: Validating training pipeline...")
        try:
            # Validate we can load a batch from the dataloader
            batch = next(iter(dataloader_train))
            logger.info("✓ DRY RUN: DataLoader validation successful!")
            logger.info(
                "✓ DRY RUN: Batch shape: features=%s, labels=%s",
                batch[0].shape,
                batch[1].shape,
            )
            logger.info("✓ DRY RUN: Would train with %d training files", val_idx_cutoff)
            logger.info(
                "✓ DRY RUN: Would train with %d validation files",
                n_training_files - val_idx_cutoff,
            )
            logger.info("✓ DRY RUN: Network config: %s", network_config)
            logger.info("✓ DRY RUN: Train config: %s", train_config)

            if mlflow_tracking_enabled:
                import mlflow

                mlflow.log_param("dry_run", True)
                mlflow.end_run()

            logger.info("DRY RUN complete - no training was performed")
            return
        except Exception as e:
            logger.error("✗ DRY RUN: Pipeline validation failed: %s", e)
            if mlflow_tracking_enabled:
                import mlflow

                mlflow.end_run(status="FAILED")
            raise typer.Exit(code=1)
    # -------------------------------------------------------------

    # Training and Saving -----------------------------------------
    # Generate unique run_id for file naming (independent of MLflow run_id)
    RUN_ID = uuid.uuid1().hex

    # save network config for this run
    networks_path = (
        Path(networks_path_base)
        / network_config["network_type"]
        / extra_config["model"]
    )
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
    net = lanfactory.trainers.JaxMLPFactory(
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

    # Train model
    model_trainer.train_and_evaluate(
        output_folder=networks_path,
        output_file_id=extra_config["model"],
        run_id=RUN_ID,
        mlflow_on=mlflow_tracking_enabled,
        save_outputs=True,
        verbose=1,
    )
    # -------------------------------------------------------------


if __name__ == "__main__":
    app()
