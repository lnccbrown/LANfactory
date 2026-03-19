# Using HuggingFace Hub

LANfactory provides CLI commands for uploading trained models to and downloading models from HuggingFace Hub.

## Installation

HuggingFace support requires the optional `hf` dependencies:

```bash
pip install lanfactory[hf]
```

Or install all optional dependencies:

```bash
pip install lanfactory[all]
```

## Authentication

Before uploading, authenticate with HuggingFace:

```bash
# Option 1: Login interactively
huggingface-cli login

# Option 2: Set environment variable
export HF_TOKEN="your_token_here"

# Option 3: Pass token via CLI
upload-hf ... --token "your_token_here"
```

## Uploading Models

### 1. Create a `model_card.yaml` file

In your trained model folder, create a `model_card.yaml` file with model metadata:

```yaml
# Required metadata (HuggingFace frontmatter)
tags:
  - lan
  - ssm
  - ddm
  - hssm
library_name: onnx
license: mit

# Model information
title: "LAN Model for DDM"
description: "Likelihood Approximation Network trained on DDM (Drift Diffusion Model) simulations."

# Optional: Network architecture (auto-extracted from config.pickle if not provided)
architecture:
  layer_sizes: [100, 100, 1]
  activations: [tanh, tanh, linear]
  network_type: lan

# Optional: Training details
training:
  epochs: 20
  optimizer: adam
  learning_rate: 0.001

# Usage example (shown in README)
usage_example: |
  import hssm
  model = hssm.HSSM(data=my_data, model="ddm", loglik_kind="approx_differentiable")
```

### 2. Upload using the CLI

```bash
upload-hf \
  --model-folder ./networks/lan/ddm/ \
  --network-type lan \
  --model-name ddm \
  --commit-message "Initial upload"
```

This uploads to `franklab/HSSM` (default) at path `lan/ddm/`.

### CLI Options

| Option | Required | Description |
|--------|----------|-------------|
| `--model-folder` | Yes | Path to folder with trained model artifacts |
| `--network-type` | Yes | Network type: `lan`, `cpn`, or `opn` |
| `--model-name` | Yes | Model name (e.g., `ddm`, `angle`) |
| `--repo-id` | No | HuggingFace repo ID (default: `franklab/HSSM`) |
| `--commit-message` | No | Git commit message (default: "Upload model") |
| `--private` | No | Create a private repository |
| `--create-repo` | No | Create repository if it doesn't exist |
| `--include-patterns` | No | Comma-separated glob patterns to include |
| `--exclude-patterns` | No | Comma-separated glob patterns to exclude |
| `--revision` | No | Branch or tag name for versioning |
| `--token` | No | HuggingFace API token |
| `--dry-run` | No | Show what would be uploaded without uploading |

### Dry Run

To preview what will be uploaded without actually uploading:

```bash
upload-hf \
  --model-folder ./networks/lan/ddm/ \
  --network-type lan \
  --model-name ddm \
  --dry-run
```

## Downloading Models

### Download using the CLI

```bash
download-hf \
  --network-type lan \
  --model-name ddm \
  --output-folder ./models/ddm/
```

This downloads from `franklab/HSSM` at path `lan/ddm/`.

### CLI Options

| Option | Required | Description |
|--------|----------|-------------|
| `--network-type` | Yes | Network type: `lan`, `cpn`, or `opn` |
| `--model-name` | Yes | Model name (e.g., `ddm`, `angle`) |
| `--output-folder` | Yes | Local destination folder |
| `--repo-id` | No | HuggingFace repo ID (default: `franklab/HSSM`) |
| `--revision` | No | Branch, tag, or commit to download (default: main) |
| `--include-patterns` | No | Comma-separated glob patterns to include |
| `--exclude-patterns` | No | Comma-separated glob patterns to exclude |
| `--token` | No | HuggingFace API token (for private repos) |
| `--force` | No | Overwrite existing files |

## Repository Structure

Models are organized in the repository using the following structure:

```
franklab/HSSM/
├── lan/
│   ├── ddm/
│   │   ├── model.onnx
│   │   ├── network_config.pickle
│   │   ├── train_config.pickle
│   │   └── README.md
│   ├── angle/
│   │   └── ...
│   └── weibull/
│       └── ...
├── cpn/
│   └── ...
└── opn/
    └── ...
```

## Using Downloaded Models with HSSM

After downloading a model, you can use it with HSSM:

```python
import hssm

# HSSM will look for models in the franklab/HSSM repository
model = hssm.HSSM(
    data=my_data,
    model="ddm",
    loglik_kind="approx_differentiable"
)
```

## Programmatic Usage

You can also use the upload/download functions directly in Python:

```python
from pathlib import Path
from lanfactory.hf import upload_model, download_model

# Upload
upload_model(
    model_folder=Path("./networks/lan/ddm/"),
    network_type="lan",
    model_name="ddm",
    commit_message="v1.0.0 release",
)

# Download
download_model(
    network_type="lan",
    model_name="ddm",
    output_folder=Path("./models/ddm/"),
)
```
