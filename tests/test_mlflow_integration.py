"""Tests for MLflow integration in LANfactory trainers and utilities."""

import pickle
import shutil
from pathlib import Path

import pytest
import numpy as np

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


from lanfactory.utils import (
    get_files_from_data_generation_experiment,
    log_training_data_lineage,
)

# Only run tests if mlflow is available
pytestmark = pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="mlflow not installed")


def set_experiment_with_artifact_location(experiment_name, artifact_location):
    """Helper to set experiment with artifact location (MLflow compatible)."""
    # Check if experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Create new experiment with artifact location
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow():
    """Clean up any MLflow artifacts after each test."""
    # Store original tracking URI
    original_uri = mlflow.get_tracking_uri()

    yield

    # Clean up any stray mlruns directory
    mlruns_path = Path.cwd() / "mlruns"
    if mlruns_path.exists():
        shutil.rmtree(mlruns_path)

    # Reset to original URI (or default)
    try:
        mlflow.set_tracking_uri(original_uri)
    except Exception:
        pass


@pytest.fixture
def test_mlflow_dir(tmp_path):
    """Create a temporary MLflow tracking directory with SQLite backend."""
    # Create isolated directories
    mlflow_root = tmp_path / "mlflow_test"
    mlflow_root.mkdir()

    mlflow_db = mlflow_root / "tracking.db"
    artifact_dir = mlflow_root / "artifacts"
    artifact_dir.mkdir()

    # Use absolute path for SQLite
    tracking_uri = f"sqlite:///{mlflow_db.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)

    # Store artifact location for tests that need it
    artifact_location = str(artifact_dir.absolute())

    # Return tracking URI, artifact location, and tmp_path
    yield {
        "tracking_uri": tracking_uri,
        "artifact_location": artifact_location,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def mock_data_generation_experiment(test_mlflow_dir):
    """Create a mock data generation experiment with multiple runs."""
    artifact_location = test_mlflow_dir["artifact_location"]

    set_experiment_with_artifact_location("test-data-generation", artifact_location)
    experiment = mlflow.get_experiment_by_name("test-data-generation")

    # Create multiple runs simulating distributed data generation
    run_ids = []
    all_files = []

    for i in range(3):
        with mlflow.start_run(run_name=f"data-gen-run-{i}"):
            run_id = mlflow.active_run().info.run_id
            run_ids.append(run_id)

            # Simulate file inventory for this run
            files_for_this_run = [
                f"training_data_file_{i}_{j}.pickle"
                for j in range(5)  # 5 files per run
            ]
            all_files.extend(files_for_this_run)

            file_inventory = {
                "num_files": len(files_for_this_run),
                "total_size_mb": len(files_for_this_run) * 2.5,
                "files": [
                    {
                        "filename": fname,
                        "relative_path": f"data/training_data/{fname}",
                        "size_bytes": 2621440,
                        "size_mb": 2.5,
                    }
                    for fname in files_for_this_run
                ],
            }

            mlflow.log_dict(file_inventory, "generated_files_inventory.json")
            mlflow.log_param("run_index", i)

    return {
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.name,
        "run_ids": run_ids,
        "all_files": all_files,
        "num_runs": 3,
        "total_files": 15,  # 3 runs * 5 files each
    }


@pytest.fixture
def mock_training_data_folder(tmp_path, mock_data_generation_experiment):
    """Create a mock training data folder with some pickle files."""
    data_folder = tmp_path / "training_data"
    data_folder.mkdir()

    # Create subset of the files that were "generated"
    files_to_create = mock_data_generation_experiment["all_files"][:10]  # Use 10 of 15

    for fname in files_to_create:
        file_path = data_folder / fname
        # Create a minimal pickle file
        with open(file_path, "wb") as f:
            pickle.dump({"data": np.random.randn(10, 5)}, f)

    return data_folder


class TestMLflowUtils:
    """Test suite for MLflow utility functions."""

    def test_get_files_from_data_generation_experiment(
        self, test_mlflow_dir, mock_data_generation_experiment
    ):
        """Test retrieving files from a data generation experiment."""
        tracking_uri = test_mlflow_dir["tracking_uri"]

        result = get_files_from_data_generation_experiment(
            experiment_id=mock_data_generation_experiment["experiment_id"],
            tracking_uri=tracking_uri,
        )

        # Verify structure
        assert "num_runs" in result
        assert "total_files" in result
        assert "all_files" in result
        assert "runs_info" in result

        # Verify counts
        assert result["num_runs"] == 3
        assert result["total_files"] == 15
        assert len(result["all_files"]) == 15
        assert len(result["runs_info"]) == 3

        # Verify file names match what we created
        expected_files = set(mock_data_generation_experiment["all_files"])
        actual_files = set(result["all_files"])
        assert expected_files == actual_files

    def test_get_files_from_empty_experiment(self, test_mlflow_dir):
        """Test behavior when experiment has no runs."""
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]

        set_experiment_with_artifact_location("empty-experiment", artifact_location)
        experiment = mlflow.get_experiment_by_name("empty-experiment")

        result = get_files_from_data_generation_experiment(
            experiment_id=experiment.experiment_id, tracking_uri=tracking_uri
        )

        assert result["num_runs"] == 0
        assert result["total_files"] == 0
        assert result["all_files"] == []

    def test_log_training_data_lineage(
        self,
        test_mlflow_dir,
        mock_data_generation_experiment,
        mock_training_data_folder,
    ):
        """Test logging training data lineage."""
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]

        set_experiment_with_artifact_location("test-training", artifact_location)

        # Create a training run
        with mlflow.start_run(run_name="test-training-run"):
            run_id = mlflow.active_run().info.run_id

            # Get file list
            valid_file_list = list(mock_training_data_folder.iterdir())

            # Log lineage
            lineage = log_training_data_lineage(
                data_generation_experiment_id=mock_data_generation_experiment[
                    "experiment_id"
                ],
                training_data_folder=mock_training_data_folder,
                valid_file_list=valid_file_list,
                n_training_files=len(valid_file_list),
                tracking_uri=tracking_uri,
            )

        # Verify lineage was logged
        assert "data_generation_experiment_id" in lineage
        assert (
            lineage["data_generation_experiment_id"]
            == mock_data_generation_experiment["experiment_id"]
        )

        # Verify file tracking
        assert "expected_files" in lineage
        assert "actual_files_used" in lineage
        assert len(lineage["expected_files"]) == 15  # All files from data gen
        assert len(lineage["actual_files_used"]) == 10  # Files we actually created

        # Verify missing files detected
        assert "missing_files" in lineage
        assert len(lineage["missing_files"]) == 5  # 15 expected - 10 actual

        # Verify artifact was logged to MLflow
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "training_data_lineage.json" in artifact_names

        # Verify tags were set
        run = client.get_run(run_id)
        assert "data_generation_experiment_id" in run.data.tags

    def test_lineage_with_extra_files(
        self, test_mlflow_dir, mock_data_generation_experiment, tmp_path
    ):
        """Test lineage logging when extra files exist in training folder."""
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]
        set_experiment_with_artifact_location("test-training", artifact_location)

        # Create training data folder with extra files
        data_folder = tmp_path / "training_data"
        data_folder.mkdir()

        # Create expected files
        for fname in mock_data_generation_experiment["all_files"][:5]:
            with open(data_folder / fname, "wb") as f:
                pickle.dump({"data": "test"}, f)

        # Create EXTRA files not from data generation
        extra_files = ["extra_file_1.pickle", "extra_file_2.pickle"]
        for fname in extra_files:
            with open(data_folder / fname, "wb") as f:
                pickle.dump({"data": "extra"}, f)

        # Log lineage
        with mlflow.start_run(run_name="test-extra-files"):
            valid_file_list = list(data_folder.iterdir())

            lineage = log_training_data_lineage(
                data_generation_experiment_id=mock_data_generation_experiment[
                    "experiment_id"
                ],
                training_data_folder=data_folder,
                valid_file_list=valid_file_list,
                n_training_files=len(valid_file_list),
                tracking_uri=tracking_uri,
            )

        # Verify extra files detected
        assert "extra_files" in lineage
        assert len(lineage["extra_files"]) == 2


class TestMLflowIntegrationWithTrainers:
    """Test MLflow integration with actual training."""

    def test_jax_trainer_mlflow_logging(
        self,
        test_mlflow_dir,
        dummy_generator_config_simple_two_choices,
        dummy_network_train_config_lan,
        tmp_path,
    ):
        """Test that JAX trainer logs to MLflow correctly."""
        from torch.utils.data import DataLoader
        from lanfactory.trainers.jax_mlp import JaxMLPFactory, ModelTrainerJaxMLP
        from lanfactory.trainers.torch_mlp import DatasetTorch

        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]
        set_experiment_with_artifact_location("test-jax-training", artifact_location)

        # Generate minimal training data
        gen_configs = dummy_generator_config_simple_two_choices()
        generator_config = gen_configs["generator_config"]
        model_config = gen_configs["model_config"]

        # Set small output folder
        output_folder = tmp_path / "data"
        output_folder.mkdir()
        generator_config["output_folder"] = str(output_folder)
        generator_config["n_parameter_sets"] = 20  # Must be >= n_subruns (default 10)
        generator_config["n_subruns"] = 2  # Reduce subruns for faster test

        # Generate one file
        import ssms

        gen = ssms.dataset_generators.lan_mlp.data_generator(
            generator_config=generator_config, model_config=model_config
        )
        gen.generate_data_training_uniform(save=True, verbose=False)

        # Set up training (dummy_network_train_config_lan is already a dict, not a function)
        network_config = dummy_network_train_config_lan["network_config"]
        train_config = dummy_network_train_config_lan["train_config"]
        train_config["n_epochs"] = 2  # Fast test
        train_config["cpu_batch_size"] = 1000  # Small batch size for test data

        # Create dataloaders
        file_list = list(output_folder.rglob("*.pickle"))
        train_dataset = DatasetTorch(
            file_ids=file_list,
            batch_size=train_config["cpu_batch_size"],
            label_lower_bound=train_config.get("label_lower_bound", -16.0),
            features_key=train_config.get("features_key", "lan_data"),
            label_key=train_config.get("label_key", "lan_labels"),
        )

        dataloader_train = DataLoader(train_dataset, shuffle=False, batch_size=None)
        dataloader_val = DataLoader(train_dataset, shuffle=False, batch_size=None)

        # Create model and trainer
        net = JaxMLPFactory(network_config=network_config, train=True)

        trainer = ModelTrainerJaxMLP(
            train_config=train_config,
            train_dl=dataloader_train,
            valid_dl=dataloader_val,
            model=net,
            allow_abs_path_folder_generation=True,
        )

        # Train with MLflow
        networks_output = tmp_path / "networks"
        networks_output.mkdir()

        with mlflow.start_run(run_name="test-jax-training") as run:
            run_id = run.info.run_id

            trainer.train_and_evaluate(
                output_folder=str(networks_output),
                output_file_id="test_model",
                run_id="test_run_id",
                mlflow_on=True,
                save_outputs=True,
                verbose=0,
            )

        # Verify MLflow logged artifacts
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        artifacts = client.list_artifacts(run_id, path="training_output")
        artifact_names = [a.path for a in artifacts]

        # Check for expected artifacts
        assert any("training_history" in name for name in artifact_names)
        assert any("train_state" in name for name in artifact_names)
        assert any("train_config" in name for name in artifact_names)

    def test_pytorch_trainer_mlflow_logging(
        self,
        test_mlflow_dir,
        dummy_generator_config_simple_two_choices,
        dummy_network_train_config_lan,
        tmp_path,
    ):
        """Test that PyTorch trainer logs to MLflow correctly."""
        from torch.utils.data import DataLoader
        from lanfactory.trainers.torch_mlp import (
            TorchMLP,
            ModelTrainerTorchMLP,
            DatasetTorch,
        )

        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]
        set_experiment_with_artifact_location("test-torch-training", artifact_location)

        # Generate minimal training data
        gen_configs = dummy_generator_config_simple_two_choices()
        generator_config = gen_configs["generator_config"]
        model_config = gen_configs["model_config"]

        # Set small output folder
        output_folder = tmp_path / "data"
        output_folder.mkdir()
        generator_config["output_folder"] = str(output_folder)
        generator_config["n_parameter_sets"] = 20  # Must be >= n_subruns (default 10)
        generator_config["n_subruns"] = 2  # Reduce subruns for faster test

        # Generate one file
        import ssms

        gen = ssms.dataset_generators.lan_mlp.data_generator(
            generator_config=generator_config, model_config=model_config
        )
        gen.generate_data_training_uniform(save=True, verbose=False)

        # Set up training (dummy_network_train_config_lan is already a dict, not a function)
        network_config = dummy_network_train_config_lan["network_config"]
        train_config = dummy_network_train_config_lan["train_config"]
        train_config["n_epochs"] = 2  # Fast test
        train_config["cpu_batch_size"] = 1000  # Small batch size for test data

        # Create dataloaders
        file_list = list(output_folder.rglob("*.pickle"))
        train_dataset = DatasetTorch(
            file_ids=file_list,
            batch_size=train_config["cpu_batch_size"],
            label_lower_bound=train_config.get("label_lower_bound", -16.0),
            features_key=train_config.get("features_key", "lan_data"),
            label_key=train_config.get("label_key", "lan_labels"),
        )

        dataloader_train = DataLoader(train_dataset, shuffle=False, batch_size=None)
        dataloader_val = DataLoader(train_dataset, shuffle=False, batch_size=None)

        # Create model and trainer
        # Determine input_dim from actual data shape (not assumptions)
        # Get one batch to inspect the data shape
        sample_batch = next(iter(dataloader_train))
        actual_input_dim = sample_batch[0].shape[1]  # [batch_size, input_dim]

        input_dim = train_config.get("input_dim", actual_input_dim)
        net = TorchMLP(
            network_config=network_config, input_shape=input_dim, network_type="mlp"
        )

        trainer = ModelTrainerTorchMLP(
            train_config=train_config,
            train_dl=dataloader_train,
            valid_dl=dataloader_val,
            model=net,
            allow_abs_path_folder_generation=True,
        )

        # Train with MLflow
        networks_output = tmp_path / "networks"
        networks_output.mkdir()

        with mlflow.start_run(run_name="test-torch-training") as run:
            run_id = run.info.run_id

            trainer.train_and_evaluate(
                output_folder=str(networks_output),
                output_file_id="test_model",
                run_id="test_run_id",
                mlflow_on=True,
                save_outputs=True,
                verbose=0,
            )

        # Verify MLflow logged artifacts
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        artifacts = client.list_artifacts(run_id, path="training_output")
        artifact_names = [a.path for a in artifacts]

        # Check for expected artifacts
        assert any("training_history" in name for name in artifact_names)
        assert any(
            "train_state" in name or "model_state" in name for name in artifact_names
        )
        assert any("train_config" in name for name in artifact_names)


class TestDataLineageTracking:
    """Test suite for data lineage tracking functionality."""

    def test_complete_lineage_workflow(
        self,
        test_mlflow_dir,
        mock_data_generation_experiment,
        mock_training_data_folder,
    ):
        """Test the complete lineage workflow from data gen to training."""
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]
        set_experiment_with_artifact_location(
            "test-lineage-workflow", artifact_location
        )

        with mlflow.start_run(run_name="training-with-lineage") as run:
            run_id = run.info.run_id

            # Get files
            valid_file_list = list(mock_training_data_folder.iterdir())

            # Log lineage
            lineage = log_training_data_lineage(
                data_generation_experiment_id=mock_data_generation_experiment[
                    "experiment_id"
                ],
                training_data_folder=mock_training_data_folder,
                valid_file_list=valid_file_list,
                n_training_files=len(valid_file_list),
                tracking_uri=tracking_uri,
            )

        # Verify lineage structure
        assert lineage["num_files_used"] == 10
        assert lineage["expected_num_runs"] == 3
        assert len(lineage["expected_files"]) == 15
        assert len(lineage["actual_files_used"]) == 10

        # Verify MLflow tags
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        assert "data_generation_experiment_id" in run.data.tags
        assert (
            run.data.tags["data_generation_experiment_id"]
            == mock_data_generation_experiment["experiment_id"]
        )

        # Verify parameters
        assert "data_generation_num_runs" in run.data.params
        assert run.data.params["data_generation_num_runs"] == "3"
        assert run.data.params["data_generation_total_files"] == "15"

    def test_lineage_missing_files_detection(
        self, test_mlflow_dir, mock_data_generation_experiment, tmp_path
    ):
        """Test that missing files are properly detected and logged."""
        tracking_uri = test_mlflow_dir["tracking_uri"]
        artifact_location = test_mlflow_dir["artifact_location"]
        set_experiment_with_artifact_location("test-missing-files", artifact_location)

        # Create folder with only 3 of the expected 15 files
        data_folder = tmp_path / "incomplete_data"
        data_folder.mkdir()

        for fname in mock_data_generation_experiment["all_files"][:3]:
            with open(data_folder / fname, "wb") as f:
                pickle.dump({"data": "test"}, f)

        with mlflow.start_run(run_name="incomplete-data-training"):
            run_id = mlflow.active_run().info.run_id

            valid_file_list = list(data_folder.iterdir())

            lineage = log_training_data_lineage(
                data_generation_experiment_id=mock_data_generation_experiment[
                    "experiment_id"
                ],
                training_data_folder=data_folder,
                valid_file_list=valid_file_list,
                n_training_files=len(valid_file_list),
                tracking_uri=tracking_uri,
            )

        # Verify missing files detected
        assert len(lineage["missing_files"]) == 12  # 15 - 3

        # Verify logged to MLflow
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        assert "missing_files_count" in run.data.params
        assert run.data.params["missing_files_count"] == "12"


def test_mlflow_experiment_organization(test_mlflow_dir):
    """Test that experiments are properly organized."""
    tracking_uri = test_mlflow_dir["tracking_uri"]
    artifact_location = test_mlflow_dir["artifact_location"]

    # Create data generation experiment
    set_experiment_with_artifact_location("model-a-data-generation", artifact_location)
    data_exp = mlflow.get_experiment_by_name("model-a-data-generation")
    with mlflow.start_run(run_name="data-gen-1"):
        mlflow.log_param("test", "value")

    # Create training experiment
    set_experiment_with_artifact_location("model-a-training", artifact_location)
    train_exp = mlflow.get_experiment_by_name("model-a-training")
    with mlflow.start_run(run_name="training-1") as run:
        mlflow.set_tag("data_generation_experiment_id", data_exp.experiment_id)
        mlflow.log_param("test", "value")
        training_run_id = run.info.run_id

    # Verify separation
    assert data_exp.experiment_id != train_exp.experiment_id

    # Verify training run links to data generation
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    training_run = client.get_run(training_run_id)
    assert (
        training_run.data.tags["data_generation_experiment_id"]
        == data_exp.experiment_id
    )

    # Verify we can navigate the lineage
    data_runs = mlflow.search_runs(experiment_ids=[data_exp.experiment_id])
    assert len(data_runs) == 1


class TestMLflowEdgeCases:
    """Test edge cases and error handling in MLflow integration."""

    def test_trainer_without_mlflow(
        self,
        dummy_generator_config_simple_two_choices,
        dummy_network_train_config_lan,
        tmp_path,
    ):
        """Test that trainers work correctly when MLflow is disabled."""
        from torch.utils.data import DataLoader
        from lanfactory.trainers.torch_mlp import (
            TorchMLP,
            ModelTrainerTorchMLP,
            DatasetTorch,
        )

        # Generate minimal training data
        gen_configs = dummy_generator_config_simple_two_choices()
        generator_config = gen_configs["generator_config"]
        model_config = gen_configs["model_config"]

        output_folder = tmp_path / "data"
        output_folder.mkdir()
        generator_config["output_folder"] = str(output_folder)
        generator_config["n_parameter_sets"] = 20
        generator_config["n_subruns"] = 2

        import ssms

        gen = ssms.dataset_generators.lan_mlp.data_generator(
            generator_config=generator_config, model_config=model_config
        )
        gen.generate_data_training_uniform(save=True, verbose=False)

        # Set up training
        network_config = dummy_network_train_config_lan["network_config"]
        train_config = dummy_network_train_config_lan["train_config"]
        train_config["n_epochs"] = 2  # Need at least 2 epochs for proper training
        train_config["cpu_batch_size"] = 1000  # Small batch size for test data

        # Create dataloaders
        file_list = list(output_folder.rglob("*.pickle"))
        train_dataset = DatasetTorch(
            file_ids=file_list,
            batch_size=train_config["cpu_batch_size"],
            label_lower_bound=-16.0,
            features_key="lan_data",
            label_key="lan_labels",
        )

        dataloader_train = DataLoader(train_dataset, shuffle=False, batch_size=None)
        dataloader_val = DataLoader(train_dataset, shuffle=False, batch_size=None)

        # Create model and trainer - determine input_dim from data
        sample_batch = next(iter(dataloader_train))
        actual_input_dim = sample_batch[0].shape[1]

        net = TorchMLP(
            network_config=network_config,
            input_shape=actual_input_dim,
            network_type="mlp",
        )

        trainer = ModelTrainerTorchMLP(
            train_config=train_config,
            train_dl=dataloader_train,
            valid_dl=dataloader_val,
            model=net,
            allow_abs_path_folder_generation=True,
        )

        # Train WITHOUT MLflow
        networks_output = tmp_path / "networks"
        networks_output.mkdir()

        trainer.train_and_evaluate(
            output_folder=str(networks_output),
            output_file_id="test_model_no_mlflow",
            run_id="test_run_id",
            mlflow_on=False,  # Explicitly disabled
            save_outputs=True,
            verbose=0,
        )

        # Verify outputs were saved
        assert len(list(networks_output.iterdir())) > 0

    def test_lineage_with_invalid_experiment_id(self, test_mlflow_dir, tmp_path):
        """Test graceful handling of invalid data generation experiment ID."""
        from lanfactory.utils import get_files_from_data_generation_experiment

        tracking_uri = test_mlflow_dir["tracking_uri"]

        # Try to get files from non-existent experiment
        # Should return empty result, not crash
        result = get_files_from_data_generation_experiment(
            experiment_id="999999999999",  # Non-existent
            tracking_uri=tracking_uri,
        )

        # Should handle gracefully
        assert result["num_runs"] == 0
        assert result["total_files"] == 0
