"""Utilities for MLflow tracking and lineage in LANfactory."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_files_from_data_generation_experiment(
    experiment_id: str, tracking_uri: str = "sqlite:///mlflow.db"
) -> dict:
    """Get all files generated across all runs in a data generation experiment.

    This function queries all runs in a data generation experiment and collects
    the complete inventory of generated files across all distributed runs.

    Arguments
    ---------
        experiment_id : str
            The MLflow experiment ID for data generation
        tracking_uri : str
            MLflow tracking URI (default: "sqlite:///mlflow.db")

    Returns
    -------
        dict
            Dictionary with keys:
            - num_runs: number of runs in the experiment
            - total_files: total number of files generated
            - all_files: list of all filenames
            - runs_info: list of dicts with run details

    Raises
    ------
        ImportError
            If mlflow is not installed
        Exception
            If experiment cannot be accessed or has no runs
    """
    try:
        import mlflow
    except ImportError:
        logger.error("mlflow package not installed")
        raise ImportError("mlflow is required for this function")

    mlflow.set_tracking_uri(tracking_uri)

    # Get all runs in the data generation experiment
    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",  # Get all runs
            order_by=["start_time DESC"],
        )
    except Exception as e:
        logger.error(f"Failed to search runs in experiment {experiment_id}: {e}")
        raise

    if len(runs) == 0:
        logger.warning(f"No runs found in experiment {experiment_id}")
        return {"num_runs": 0, "total_files": 0, "all_files": [], "runs_info": []}

    all_files = []
    runs_info = []

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    for _, run in runs.iterrows():
        run_id = run["run_id"]

        try:
            # Download the file inventory artifact from each run
            inventory_path = client.download_artifacts(
                run_id, "generated_files_inventory.json"
            )

            with open(inventory_path, "r") as f:
                inventory = json.load(f)

            # Collect filenames from this run
            run_files = [file_info["filename"] for file_info in inventory["files"]]
            all_files.extend(run_files)

            runs_info.append(
                {
                    "run_id": run_id,
                    "run_name": run.get("tags.mlflow.runName", "unknown"),
                    "num_files": inventory["num_files"],
                    "total_size_mb": inventory["total_size_mb"],
                    "files": run_files,
                }
            )

        except Exception as e:
            # Log but don't fail if one run is missing inventory
            logger.warning(f"Could not get inventory for run {run_id}: {e}")
            continue

    return {
        "num_runs": len(runs_info),
        "total_files": len(all_files),
        "all_files": all_files,
        "runs_info": runs_info,
    }


def log_training_data_lineage(
    data_generation_experiment_id: str,
    training_data_folder: Path,
    valid_file_list: list,
    n_training_files: int,
    tracking_uri: str = None,
) -> dict:
    """Log training data lineage information to MLflow.

    This function retrieves the expected files from the data generation experiment,
    compares them with actual files found, and logs the lineage information.

    Arguments
    ---------
        data_generation_experiment_id : str
            MLflow experiment ID of the data generation experiment
        training_data_folder : Path
            Path to the folder containing training data
        valid_file_list : list
            List of Path objects for available training files
        n_training_files : int
            Number of files actually used for training
        tracking_uri : str, optional
            MLflow tracking URI (uses current if None)

    Returns
    -------
        dict
            Dictionary with lineage information including:
            - data_generation_experiment_id
            - expected_files
            - actual_files_used
            - missing_files
            - extra_files
    """
    try:
        import mlflow
        import os
    except ImportError:
        logger.error("mlflow package not installed")
        return {}

    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    try:
        # Get expected files from data generation experiment
        expected_files_info = get_files_from_data_generation_experiment(
            data_generation_experiment_id, tracking_uri
        )

        # Log data lineage as tags/params
        mlflow.set_tag("data_generation_experiment_id", data_generation_experiment_id)
        mlflow.log_param("data_generation_num_runs", expected_files_info["num_runs"])
        mlflow.log_param(
            "data_generation_total_files", expected_files_info["total_files"]
        )

        # Verify we have all the files we expect
        expected_file_names = set(expected_files_info["all_files"])
        actual_file_names = set(f.name for f in valid_file_list)

        missing_files = expected_file_names - actual_file_names
        extra_files = actual_file_names - expected_file_names

        if missing_files:
            logger.warning(
                f"Missing {len(missing_files)} expected files from data generation"
            )
            mlflow.log_param("missing_files_count", len(missing_files))

        if extra_files:
            logger.info(
                f"Found {len(extra_files)} extra files not in data generation manifest"
            )
            mlflow.log_param("extra_files_count", len(extra_files))

        # Create detailed file inventory
        file_inventory = {
            "data_generation_experiment_id": data_generation_experiment_id,
            "expected_files": expected_files_info["all_files"],
            "expected_num_runs": expected_files_info["num_runs"],
            "training_data_folder": str(training_data_folder),
            "actual_files_found": [f.name for f in valid_file_list],
            "actual_files_used": [f.name for f in valid_file_list[:n_training_files]],
            "num_files_used": n_training_files,
            "missing_files": list(missing_files),
            "extra_files": list(extra_files),
            "data_gen_runs_info": expected_files_info["runs_info"],
        }

        # Log as artifact
        mlflow.log_dict(file_inventory, "training_data_lineage.json")

        logger.info(
            f"Logged training data lineage for experiment {data_generation_experiment_id}"
        )
        logger.info(
            f"Expected {len(expected_file_names)} files, found {len(actual_file_names)}, using {n_training_files}"
        )

        return file_inventory

    except Exception as e:
        logger.error(f"Failed to log training data lineage: {e}")
        mlflow.log_param("data_lineage_error", str(e))
        return {}
