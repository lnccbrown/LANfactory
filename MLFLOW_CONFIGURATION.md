# MLflow Configuration for LANfactory

This document describes the MLflow configuration options available in LANfactory training CLI commands.

## Overview

LANfactory provides **optional MLflow integration** for tracking network training experiments, logging metrics, parameters, and artifacts. The package supports SQLite-based tracking backend with configurable artifact storage.

## Installation

MLflow is an **optional dependency**. To use MLflow tracking features:

```bash
# Install with MLflow support
pip install lanfactory[mlflow]

# Or with uv
uv pip install lanfactory[mlflow]
```

Without the `[mlflow]` extra, the package works normally but MLflow tracking will be disabled even if `--mlflow-on` is provided.

## Configuration Options

### CLI Arguments

Both `jaxtrain` and `torchtrain` commands support the following MLflow-related arguments:

#### Tracking and Storage

- `--mlflow-tracking-uri TEXT`: MLflow tracking URI for metadata storage
  - Examples: `sqlite:///mlflow.db`, `sqlite:////shared/path/mlflow.db`, `http://mlflow-server:5000`
  - Default: Falls back to `MLFLOW_TRACKING_URI` env var, then `sqlite:///mlflow.db`

- `--mlflow-artifact-location TEXT`: Root directory for MLflow artifacts
  - Example: `/shared/storage/mlflow_artifacts`
  - Default: Falls back to `MLFLOW_ARTIFACT_LOCATION` env var, then MLflow default location

#### Run Control

- `--mlflow-on`: Explicitly enable MLflow tracking for this training run

- `--mlflow-run-id TEXT`: (Advanced) Resume logging to an existing MLflow run
  - Automatically enables `--mlflow-on`

#### Data Lineage

- `--data-generation-experiment-id TEXT`: Link this training run to a data generation experiment
  - Enables automatic data folder discovery from MLflow
  - Logs complete data lineage information
  - Automatically enables `--mlflow-on`

### Environment Variables

The CLI will respect the following environment variables:

- `MLFLOW_TRACKING_URI`: Default tracking URI if not specified via CLI
- `MLFLOW_ARTIFACT_LOCATION`: Default artifact location if not specified via CLI
- `MLFLOW_EXPERIMENT_NAME`: MLflow experiment to log runs to

### Configuration Priority

For each setting, the priority order is:

1. CLI argument (highest priority)
2. Environment variable
3. Default value (lowest priority)

## Usage Examples

### Local Development (SQLite)

```bash
# Simple local training with SQLite backend
jaxtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-on \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --mlflow-artifact-location ./mlflow_artifacts
```

### Shared Cluster Storage (SQLite)

```bash
# Training on cluster with shared filesystem
export MLFLOW_TRACKING_URI="sqlite:////shared/storage/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/shared/storage/mlflow/artifacts"
export MLFLOW_EXPERIMENT_NAME="model-v2-training"

torchtrain \
  --config-path config.yaml \
  --training-data-folder /shared/data \
  --networks-path-base /shared/networks \
  --mlflow-on
```

### With Data Lineage

```bash
# Training with automatic data lineage tracking
jaxtrain \
  --config-path config.yaml \
  --data-generation-experiment-id "123456789" \
  --networks-path-base ./networks \
  --mlflow-tracking-uri sqlite:///mlflow.db
```

Note: When `--data-generation-experiment-id` is provided, the training data folder can be automatically discovered from MLflow.

### Resume Existing Run

```bash
# Continue logging to an existing run
torchtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-id "abc123def456"
```

## Data Management Modes

LANfactory supports three modes for managing training data:

### 1. MLflow-First Mode

Provide only `--data-generation-experiment-id`. Training data folder is derived from MLflow.

```bash
jaxtrain \
  --data-generation-experiment-id "123456789" \
  --networks-path-base ./networks
```

### 2. Validation Mode

Provide both `--data-generation-experiment-id` and `--training-data-folder`. LANfactory validates that all MLflow-tracked files exist in the folder.

```bash
jaxtrain \
  --data-generation-experiment-id "123456789" \
  --training-data-folder ./data \
  --networks-path-base ./networks
```

### 3. Traditional Mode

Provide only `--training-data-folder`. No MLflow lineage tracking.

```bash
jaxtrain \
  --training-data-folder ./data \
  --networks-path-base ./networks
```

## Experiment Creation and Artifact Location

LANfactory uses the correct MLflow API for experiment management:

- If an experiment doesn't exist and `artifact_location` is specified, LANfactory calls:
  ```python
  mlflow.create_experiment(name, artifact_location=artifact_location)
  ```

- Then it sets the experiment as active:
  ```python
  mlflow.set_experiment(name)
  ```

Note: `mlflow.set_experiment()` does not accept `artifact_location` parameter. The artifact location must be specified when creating the experiment.

## Best Practices

### For Local Development

Use relative SQLite paths and local artifact directories:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="./mlflow_artifacts"
```

### For Cluster Environments

Use absolute paths on shared storage:

```bash
export MLFLOW_TRACKING_URI="sqlite:////shared/storage/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/shared/storage/mlflow/artifacts"
```

### For Production

Consider using a centralized MLflow server:

```bash
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
# Artifacts can be stored on S3, Azure Blob, etc.
export MLFLOW_ARTIFACT_LOCATION="s3://my-bucket/mlflow-artifacts"
```

## Migration from Filesystem Backend

LANfactory previously defaulted to `./mlruns` (filesystem backend, now deprecated). The new default is `sqlite:///mlflow.db`.

To maintain existing tracking data, you can:

1. Keep using the filesystem backend explicitly:
   ```bash
   export MLFLOW_TRACKING_URI="./mlruns"
   ```

2. Or migrate to SQLite using MLflow's migration tools (recommended).

## Troubleshooting

### "mlruns" directory still created

Check that you're not setting `MLFLOW_TRACKING_URI` to a filesystem path. Use SQLite URI format:
```bash
# Wrong
export MLFLOW_TRACKING_URI="./mlruns"

# Correct
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

### Artifacts stored in unexpected location

Check the experiment's artifact location in MLflow UI or via API:
```python
import mlflow
experiment = mlflow.get_experiment_by_name("your-experiment")
print(experiment.artifact_location)
```

### Missing data files in validation mode

Ensure all files tracked in the data generation experiment exist in your training data folder. Check the error message for the list of missing files.

## See Also

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLFLOW_INTEGRATION_SUMMARY.md](../../LAN_pipeline_minimal/MLFLOW_INTEGRATION_SUMMARY.md) - Overall integration guide
