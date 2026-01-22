# MLflow Tutorial for LANfactory

Track and manage your network training experiments with MLflow.

## 📚 What is MLflow?

MLflow helps you:
- 📊 Track training experiments and hyperparameters
- 🔍 Compare model configurations and performance
- 📁 Organize trained networks and artifacts
- 🔄 Reproduce training runs exactly
- 🔗 Link training to data generation (lineage tracking)


## 🚀 Quick Start (5 minutes)

**1. Install:**
```bash
pip install lanfactory[mlflow]
```

**2. Train with tracking:**
```bash
jaxtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "my-first-run"
```

**3. View results:**
```bash
mlflow ui
# Open http://localhost:5000
```

## 💡 What Gets Tracked?

**Automatically logged:**
- Configuration: network architecture, learning rate, batch size, epochs
- Metrics: training loss, validation loss per epoch
- Artifacts: trained model state, training history, config files
- Lineage: link to data generation experiment (optional)

## 📖 Usage Examples

### Example 1: Basic JAX Training

```bash
jaxtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "ddm-baseline" \
  --mlflow-experiment-name "ddm-experiments"
```

### Example 2: PyTorch Training

```bash
torchtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "ddm-torch-v1" \
  --mlflow-experiment-name "ddm-experiments"
```

### Example 3: Dry-Run Validation

Validate the pipeline without training:

```bash
# Validate config before committing to a long training run
jaxtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --dry-run \
  --mlflow-run-name "validation-test"

# Then run for real
jaxtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "production-run"
```

### Example 4: Training with Data Lineage

Link training to a data generation experiment for full reproducibility:

```bash
# Training with automatic data lineage tracking
jaxtrain \
  --config-path config.yaml \
  --data-generation-experiment-id "123456789" \
  --networks-path-base ./networks \
  --mlflow-run-name "train-with-lineage"
```

Note: When `--data-generation-experiment-id` is provided, the training data folder can be automatically discovered from MLflow.

### Example 5: Cluster with Shared Filesystem

```bash
#!/bin/bash
#SBATCH --job-name=lan-training

# Use shared filesystem
export MLFLOW_TRACKING_URI="sqlite:////nfs/project/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/nfs/project/mlflow/artifacts"

jaxtrain \
  --config-path config.yaml \
  --training-data-folder /nfs/project/data \
  --networks-path-base /nfs/project/networks \
  --mlflow-run-name "cluster-job-${SLURM_JOB_ID}" \
  --mlflow-experiment-name "production-training"
```

**Why absolute paths?** All nodes can access the same tracking database and artifacts.

## 🔧 Configuration

Three layers of configuration (priority: CLI > Environment > Defaults):

**1. Defaults (no configuration):**
```bash
jaxtrain --config-path config.yaml --training-data-folder ./data \
  --networks-path-base ./networks --mlflow-run-name "test"
# Uses: sqlite:///mlflow.db
```

**2. Environment variables (set once):**
```bash
export MLFLOW_TRACKING_URI="sqlite:///~/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="~/mlflow/artifacts"
export MLFLOW_EXPERIMENT_NAME="my-project"
```

**3. CLI arguments (per-run override):**
```bash
jaxtrain \
  --mlflow-tracking-uri "sqlite:////shared/mlflow.db" \
  --mlflow-artifact-location "/shared/artifacts" \
  --mlflow-experiment-name "override-experiment" \
  --mlflow-run-name "run-001" \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks
```

## 🗂️ Data Management Modes

LANfactory supports three modes for managing training data:

### MLflow-First Mode
Provide only `--data-generation-experiment-id`. Training data folder is derived from MLflow.

```bash
jaxtrain \
  --data-generation-experiment-id "123456789" \
  --networks-path-base ./networks \
  --mlflow-run-name "train-from-mlflow"
```

### Validation Mode
Provide both options. LANfactory validates that all MLflow-tracked files exist.

```bash
jaxtrain \
  --data-generation-experiment-id "123456789" \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "train-with-validation"
```

### Traditional Mode
Provide only `--training-data-folder`. No MLflow tracking unless `--mlflow-run-name` is also provided.

```bash
# Without MLflow tracking
jaxtrain \
  --training-data-folder ./data \
  --networks-path-base ./networks

# With MLflow tracking
jaxtrain \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "traditional-mode-run"
```

## 📊 Using the MLflow UI

```bash
mlflow ui
# Opens http://localhost:5000

# Sets up UI with tracking from .db
mlflow server --backend-store-uri <path/to/tracking.db>
```


## 💾 File Storage

**MLflow stores two types of data:**

| Type | What | Location |
|------|------|----------|
| **Metadata** | Experiment/run names, parameters, metrics | `--mlflow-tracking-uri` (SQLite DB) |
| **Artifacts** | Config files, training history, model states | `--mlflow-artifact-location` |
| **Networks** | Your trained model files | `--networks-path-base` |

**Example structure:**
```
project/
├── mlflow/
│   ├── tracking.db          ← Metadata (lightweight)
│   └── artifacts/           ← Configs, histories
└── networks/                ← Your trained models
    └── lan/
        └── ddm/
            ├── model_state.jax
            └── training_history.csv
```

## 🗄️ Working with the SQLite Database

### View and Query

**Python API:**
```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Search all runs
runs = mlflow.search_runs()
print(runs)

# Search specific experiment
runs = mlflow.search_runs(experiment_names=["my-training"])

# Filter by parameters
runs = mlflow.search_runs(
    filter_string="params.network_type = 'lan'"
)

# Export to CSV
runs.to_csv("training_history.csv")
```

**Command line:**
```bash
# Direct SQLite queries (advanced)
sqlite3 mlflow.db "SELECT name FROM experiments;"
```

### Backup and Migration

```bash
# Backup database
cp mlflow.db mlflow-backup-$(date +%Y%m%d).db

# Move to new machine
tar -czf mlflow-export.tar.gz mlflow/
scp mlflow-export.tar.gz newmachine:~/project/
# Extract and set MLFLOW_TRACKING_URI on new machine
```

## 🎯 Common Use Cases

### Find Runs with Specific Config

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

runs = mlflow.search_runs(
    filter_string="params.network_type = 'lan' AND metrics.val_loss < 0.1"
)
print(f"Found {len(runs)} matching runs")
```

### Resume an Existing Run

```bash
# Continue logging to an existing run
torchtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-id "abc123def456"
```

### Compare Model Versions

```bash
# Version 1
jaxtrain --config-path config_v1.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks/v1 \
  --mlflow-run-name "architecture-v1" \
  --mlflow-experiment-name "architecture-comparison"

# Version 2 (improved architecture)
jaxtrain --config-path config_v2.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks/v2 \
  --mlflow-run-name "architecture-v2" \
  --mlflow-experiment-name "architecture-comparison"

# Compare in UI to see improvements
mlflow ui
```

## ⚙️ Best Practices

### Project Organization

**Recommended structure:**
```bash
# Create organized directories
mkdir -p ~/projects/my-project/mlflow/artifacts
mkdir -p ~/projects/my-project/networks

# Set environment (add to ~/.bashrc)
export MLFLOW_TRACKING_URI="sqlite:////$HOME/projects/my-project/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="$HOME/projects/my-project/mlflow/artifacts"
export MLFLOW_EXPERIMENT_NAME="my-training-project"
```

### Naming Conventions

- **Experiments**: Group related work (`"ddm-training-v2"` not `"exp1"`)
- **Runs**: Include version/iteration (`"baseline-v1.0"`)
- **Use dry-run**: Validate before large training runs
- **Use lineage**: Link training to data generation experiments

### Cluster Usage

```bash
# Always use absolute paths on shared filesystems
export MLFLOW_TRACKING_URI="sqlite:////nfs/shared/mlflow.db"  # 4 slashes!
export MLFLOW_ARTIFACT_LOCATION="/nfs/shared/artifacts"
```

## 🚀 Quick Reference

```bash
# Minimal JAX training with MLflow
jaxtrain --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "my-run"

# Minimal PyTorch training with MLflow
torchtrain --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "my-run"

# Dry run validation
jaxtrain --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --dry-run \
  --mlflow-run-name "validation"

# Full command with all MLflow options
jaxtrain \
  --config-path config.yaml \
  --training-data-folder ./data \
  --networks-path-base ./networks \
  --mlflow-run-name "production-run" \
  --mlflow-experiment-name "my-project" \
  --mlflow-tracking-uri "sqlite:///mlflow.db" \
  --mlflow-artifact-location "./mlflow_artifacts" \
  --data-generation-experiment-id "123456789"

# View experiments
mlflow ui

# Sets up UI with tracking from .db
mlflow server --backend-store-uri <path/to/tracking.db>

# Python queries
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
print(mlflow.search_runs())
"

# Backup
cp mlflow.db mlflow-backup-$(date +%Y%m%d).db
```

## 📚 Complete Workflow Example

```bash
# Setup
export MLFLOW_TRACKING_URI="sqlite:///project_mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="./mlflow_artifacts"

# 1. Generate training data (using ssm-simulators)
generate --config-path data_config.yaml --output ./data/train \
  --n-files 80 \
  --mlflow-run-name "train-data" \
  --mlflow-experiment-name "data-generation"

# 2. Validate training pipeline
jaxtrain \
  --config-path network_config.yaml \
  --training-data-folder ./data/train \
  --networks-path-base ./networks \
  --dry-run \
  --mlflow-run-name "validation"

# 3. Train LAN network
jaxtrain \
  --config-path network_config.yaml \
  --training-data-folder ./data/train \
  --networks-path-base ./networks \
  --mlflow-run-name "lan-production" \
  --mlflow-experiment-name "lan-training"

# 4. Review in UI
mlflow ui

# Sets up UI with tracking from .db
mlflow server --backend-store-uri <path/to/tracking.db>
```

## 🔧 Troubleshooting

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

### MLflow revision error
```bash
alembic.util.exc.CommandError: Can't locate revision identified by <revision number>
```
### Solution:
```bash
pip install --upgrade mlflow
```

## 📚 See Also

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
