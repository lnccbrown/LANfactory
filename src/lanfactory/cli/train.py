"""Command-line interface for training LANfactory networks.

One command, ``train``, trains a network from a YAML configuration with either
backend, selected by ``--backend torch`` or ``--backend jax``. It handles
dataset discovery (optionally via an MLflow data-generation experiment), model
initialization, training, and saving of model artifacts.

The two backends share the whole pipeline and differ only in device detection
for the batch size and in the network and trainer classes. Both write the
HSSM-consumable ONNX artifact next to their native state unless
``--no-export-onnx`` is given.
"""

import logging
import os
import pickle
import random
import uuid
from copy import deepcopy
from enum import Enum
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import psutil
import typer
from torch.utils.data import DataLoader

import lanfactory
from lanfactory.cli.utils import (
    _get_train_network_config,
    log_training_run_identity,
)

logger = logging.getLogger(__name__)

app = typer.Typer()


class Backend(str, Enum):
    """Training backend."""

    torch = "torch"
    jax = "jax"


def _resolve_tracking_uri(mlflow_tracking_uri: str | None) -> str:
    """CLI arg > ``MLFLOW_TRACKING_URI`` env var > local sqlite file."""
    if mlflow_tracking_uri:
        return mlflow_tracking_uri
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _start_mlflow_run(
    *,
    tracking_uri: str,
    mlflow_experiment_name: str | None,
    mlflow_artifact_location: str | None,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
) -> None:
    """Point MLflow at the tracking server, select the experiment, start/resume a run."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    # Experiment: CLI arg > env var
    mlflow_experiment = mlflow_experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME")
    if mlflow_experiment:
        # Artifact location: CLI arg > env var > MLflow default
        artifact_location = mlflow_artifact_location or os.getenv(
            "MLFLOW_ARTIFACT_LOCATION"
        )
        if artifact_location:
            artifact_location = str(Path(artifact_location).absolute())
            if mlflow.get_experiment_by_name(mlflow_experiment) is None:
                mlflow.create_experiment(
                    mlflow_experiment, artifact_location=artifact_location
                )
                logger.info(
                    "Created MLflow experiment: %s (artifacts: %s)",
                    mlflow_experiment,
                    artifact_location,
                )
        else:
            logger.info(
                "Set MLflow experiment: %s (default artifact location)",
                mlflow_experiment,
            )
        mlflow.set_experiment(mlflow_experiment)

    if mlflow_run_id:
        mlflow.start_run(run_id=mlflow_run_id)
        logger.info("Resumed MLflow run: %s", mlflow_run_id)
    else:
        run = mlflow.start_run(run_name=mlflow_run_name)
        logger.info(
            "Started new MLflow run: %s (name: %s)", run.info.run_id, mlflow_run_name
        )


def _select_batch_size(backend: Backend, train_config: dict) -> int:
    """GPU batch size when the backend sees a GPU, CPU batch size otherwise."""
    if backend is Backend.torch:
        import torch

        n_cuda = torch.cuda.device_count()
        logger.info("CUDA devices: %d", n_cuda)
        on_gpu = n_cuda > 0
    else:
        import jax

        logger.info("JAX devices: %s", jax.devices())
        on_gpu = jax.default_backend() == "gpu"
    return train_config["gpu_batch_size"] if on_gpu else train_config["cpu_batch_size"]


def _build_and_train(
    *,
    backend: Backend,
    network_config: dict,
    train_config: dict,
    input_dim: int,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    networks_path: Path,
    output_file_id: str,
    run_id: str,
    mlflow_on: bool,
    export_onnx: bool,
) -> None:
    """Construct the backend's network and trainer, then train and save."""
    if backend is Backend.torch:
        net = lanfactory.trainers.TorchMLP(
            network_config=deepcopy(network_config),
            input_shape=input_dim,
            network_type=network_config["network_type"],
        )
        torch_trainer = lanfactory.trainers.ModelTrainerTorchMLP(
            train_config=deepcopy(train_config),
            model=net,
            train_dl=dataloader_train,
            valid_dl=dataloader_val,
            allow_abs_path_folder_generation=True,
            pin_memory=True,
            seed=None,
        )
        torch_trainer.train_and_evaluate(
            output_folder=str(networks_path),
            output_file_id=output_file_id,
            run_id=run_id,
            mlflow_on=mlflow_on,
            save_outputs=True,
            export_onnx=export_onnx,
        )
        return

    jax_net = lanfactory.trainers.JaxMLPFactory(
        network_config=deepcopy(network_config),
        train=True,
    )
    jax_trainer = lanfactory.trainers.ModelTrainerJaxMLP(
        train_config=deepcopy(train_config),
        train_dl=dataloader_train,
        valid_dl=dataloader_val,
        model=jax_net,
        allow_abs_path_folder_generation=True,
    )
    jax_trainer.train_and_evaluate(
        output_folder=networks_path,
        output_file_id=output_file_id,
        run_id=run_id,
        mlflow_on=mlflow_on,
        save_outputs=True,
        verbose=1,
        # Pass explicitly: inference from train_output_type mislabels OPN as
        # cpn (both train on logits).
        network_type=network_config["network_type"],
        export_onnx=export_onnx,
    )


def run_training(
    *,
    backend: Backend | str,
    networks_path_base: Path,
    config_path: Path | None = None,
    training_data_folder: Path | None = None,
    network_id: int = 0,
    dl_workers: int = 1,
    dry_run: bool = False,
    export_onnx: bool = True,
    mlflow_run_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_id: str | None = None,
    data_generation_experiment_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_artifact_location: str | None = None,
) -> None:
    """Train a network with the given backend; see ``train --help``.

    Training data comes from ``training_data_folder``, from the MLflow
    experiment ``data_generation_experiment_id``, or from both — in which case
    the folder is validated against the files MLflow recorded.
    """
    backend = Backend(backend)

    # DataLoader workers ------------------------------------------
    n_workers = (
        dl_workers if dl_workers > 0 else min(12, psutil.cpu_count(logical=False) - 2)
    )
    n_workers = max(1, n_workers)
    logger.info("Number of workers we assign to the DataLoader: %d", n_workers)

    # Configuration -----------------------------------------------
    if config_path is None:
        logger.warning("No config path provided, using default configuration.")
        with as_file(
            files("lanfactory.cli") / "config_network_training_lan.yaml"
        ) as default_config:
            config_path = default_config

    config_dict = _get_train_network_config(
        yaml_config_path=str(config_path), net_index=network_id
    )

    train_config = config_dict["config_dict"]["train_config"]
    network_config = config_dict["config_dict"]["network_config"]
    extra_config = config_dict["extra_fields"]

    logger.info("TRAIN CONFIG: %s", train_config)
    logger.info("NETWORK CONFIG: %s", network_config)
    logger.info("CONFIG_DICT: %s", config_dict)

    # Training data source ----------------------------------------
    # Three modes:
    # 1. MLflow-first: data_generation_experiment_id provided, derive training_data_folder
    # 2. Validation: Both provided, verify MLflow files exist in training_data_folder
    # 3. Traditional: Only training_data_folder provided
    if data_generation_experiment_id is None and training_data_folder is None:
        raise ValueError(
            "Must provide either --data-generation-experiment-id or --training-data-folder. "
            "Cannot proceed without a data source."
        )

    # Enable tracking if a run name, a run to resume, or a data-generation
    # experiment is given.
    mlflow_tracking_enabled = (
        mlflow_run_name is not None
        or mlflow_run_id is not None
        or data_generation_experiment_id is not None
    )

    mlflow_lineage_info: dict[str, Any] | None = None
    tracking_uri = _resolve_tracking_uri(mlflow_tracking_uri)

    if mlflow_tracking_enabled:
        try:
            _start_mlflow_run(
                tracking_uri=tracking_uri,
                mlflow_experiment_name=mlflow_experiment_name,
                mlflow_artifact_location=mlflow_artifact_location,
                mlflow_run_name=mlflow_run_name,
                mlflow_run_id=mlflow_run_id,
            )
        except Exception as e:
            logger.error("Failed to initialize MLflow: %s", e)
            mlflow_tracking_enabled = False

    if data_generation_experiment_id:
        try:
            from lanfactory.utils import get_files_from_data_generation_experiment

            if not mlflow_tracking_enabled:
                # Initialize MLflow just for querying
                import mlflow

                mlflow.set_tracking_uri(tracking_uri)

            logger.info("Querying MLflow experiment: %s", data_generation_experiment_id)
            mlflow_lineage_info = get_files_from_data_generation_experiment(
                experiment_id=data_generation_experiment_id, tracking_uri=tracking_uri
            )
            logger.info(
                "MLflow reports %d runs with %d total files",
                mlflow_lineage_info["num_runs"],
                mlflow_lineage_info["total_files"],
            )

            # All runs write to the same folder; take it from the first.
            if mlflow_lineage_info["runs_info"]:
                first_run_info = mlflow_lineage_info["runs_info"][0]
                mlflow_data_folder = first_run_info.get("data_output_folder")

                if mlflow_data_folder:
                    logger.info("MLflow recorded data folder: %s", mlflow_data_folder)
                    if training_data_folder is None:  # Mode 1
                        training_data_folder = Path(mlflow_data_folder)
                        logger.info(
                            "Using training data folder from MLflow: %s",
                            training_data_folder,
                        )
                    else:  # Mode 2
                        logger.info(
                            "Validation mode: Checking MLflow files against provided folder"
                        )

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
            logger.warning(
                "MLflow query failed, falling back to training_data_folder only"
            )

    assert training_data_folder is not None  # guaranteed by the checks above
    training_data_folder = Path(training_data_folder)
    if not training_data_folder.exists():
        raise FileNotFoundError(
            f"Training data folder does not exist: {training_data_folder}"
        )

    valid_file_list = [
        f for f in training_data_folder.iterdir() if f.suffix == ".pickle"
    ]
    logger.info("NUMBER OF TRAINING FILES FOUND in folder: %d", len(valid_file_list))

    # Mode 2: verify the MLflow-tracked files exist in training_data_folder
    if mlflow_lineage_info and data_generation_experiment_id:
        expected_files = set(mlflow_lineage_info["all_files"])
        actual_files = {f.name for f in valid_file_list}

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

        from lanfactory.utils import log_training_data_lineage

        log_training_data_lineage(
            data_generation_experiment_id=data_generation_experiment_id,
            training_data_folder=training_data_folder,
            valid_file_list=valid_file_list,
            n_training_files=min(
                len(valid_file_list), train_config["n_training_files"]
            ),
            tracking_uri=tracking_uri,
        )

    # Split ---------------------------------------------------------
    random.shuffle(valid_file_list)
    n_training_files = min(len(valid_file_list), train_config["n_training_files"])
    val_idx_cutoff = int(
        config_dict["config_dict"]["train_val_split"] * n_training_files
    )
    logger.info("NUMBER OF TRAINING FILES USED: %d", n_training_files)
    logger.info("TRAIN/VAL SPLIT AT INDEX: %d", val_idx_cutoff)

    batch_size = _select_batch_size(backend, train_config)
    train_config["train_batch_size"] = batch_size
    logger.info("BATCH SIZE CHOSEN: %d", batch_size)

    # Dataloaders -----------------------------------------------------
    def _loader(file_ids: list[Path]) -> tuple[Any, DataLoader]:
        dataset = lanfactory.trainers.DatasetTorch(
            file_ids=file_ids,
            batch_size=batch_size,
            label_lower_bound=train_config["label_lower_bound"],
            features_key=train_config["features_key"],
            label_key=train_config["label_key"],
        )
        loader = DataLoader(
            dataset,
            shuffle=train_config["shuffle_files"],
            batch_size=None,
            num_workers=n_workers,
            pin_memory=True,
        )
        return dataset, loader

    train_dataset, dataloader_train = _loader(valid_file_list[:val_idx_cutoff])
    _, dataloader_val = _loader(valid_file_list[val_idx_cutoff:])

    # Dry run ---------------------------------------------------------
    if dry_run:
        logger.info("DRY RUN: Validating training pipeline...")
        try:
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

    # Train and save --------------------------------------------------
    # Unique run id for file naming (independent of the MLflow run id)
    run_uuid = uuid.uuid1().hex

    networks_path = (
        Path(networks_path_base)
        / network_config["network_type"]
        / extra_config["model"]
    )
    networks_path.mkdir(parents=True, exist_ok=True)

    config_file_name = "_".join(
        [
            run_uuid,
            network_config["network_type"],
            extra_config["model"],
            "network_config.pickle",
        ]
    )
    with open(networks_path / config_file_name, "wb") as config_file:
        pickle.dump(network_config, config_file)

    if mlflow_tracking_enabled:
        # Identity params/tags: make the run self-describing (model, bounds,
        # run_uuid join key). Best-effort inside the helper.
        log_training_run_identity(
            model=extra_config["model"],
            network_type=network_config["network_type"],
            backend=backend.value,
            run_uuid=run_uuid,
            config_path=config_path,
            training_data_folder=training_data_folder,
            n_training_files=n_training_files,
            dataset=train_dataset,
        )
        try:
            import mlflow

            # The network config is required to reconstruct the network; log it
            # so the MLflow artifact set alone can rebuild one.
            mlflow.log_artifact(
                str(networks_path / config_file_name), artifact_path="training_output"
            )
            if config_path is not None and Path(config_path).is_file():
                mlflow.log_artifact(str(config_path), artifact_path="training_output")
        except Exception as e:
            logger.error("Failed to log config artifacts to MLflow: %s", e)

    _build_and_train(
        backend=backend,
        network_config=network_config,
        train_config=train_config,
        input_dim=train_dataset.input_dim,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        networks_path=networks_path,
        output_file_id=extra_config["model"],
        run_id=run_uuid,
        mlflow_on=mlflow_tracking_enabled,
        export_onnx=export_onnx,
    )


@app.command()
def main(
    backend: Backend = typer.Option(
        ...,
        "--backend",
        "-b",
        case_sensitive=False,
        help="Training backend: torch or jax.",
    ),
    config_path: Path = typer.Option(None, help="Path to the YAML config file"),
    training_data_folder: Path = typer.Option(
        None,
        help="Path to the training data folder. "
        "Optional if --data-generation-experiment-id is provided (will be derived from MLflow). "
        "If both are provided, validates that MLflow files exist in this folder.",
    ),
    networks_path_base: Path = typer.Option(..., help="Base path for networks"),
    network_id: int = typer.Option(0, help="Network ID to train"),
    dl_workers: int = typer.Option(1, help="Number of workers for DataLoader"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the pipeline without training. Useful for testing configurations.",
        is_flag=True,
    ),
    export_onnx: bool = typer.Option(
        True,
        "--export-onnx/--no-export-onnx",
        help="Export the trained network to ONNX (single-trial contract) "
        "alongside the backend's native state. ONNX is the artifact HSSM consumes.",
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
    """Train a LAN/CPN/OPN network with the torch or jax backend."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_training(
        backend=backend,
        config_path=config_path,
        training_data_folder=training_data_folder,
        networks_path_base=networks_path_base,
        network_id=network_id,
        dl_workers=dl_workers,
        dry_run=dry_run,
        export_onnx=export_onnx,
        mlflow_run_name=mlflow_run_name,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_id=mlflow_run_id,
        data_generation_experiment_id=data_generation_experiment_id,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_artifact_location=mlflow_artifact_location,
    )


if __name__ == "__main__":
    app()
