# LANfactory — Project Context for Claude

## What is LANfactory?

Lightweight Python package for training Likelihood Approximation Networks (LANs), Choice Probability Networks (CPNs), and Option Probability Networks (OPNs) using PyTorch or JAX/Flax backends. Trained networks are exported to ONNX format and uploaded to HuggingFace for consumption by HSSM. This package sits in the middle of the HSSM ecosystem: it depends on ssm-simulators for training data and produces the neural network artifacts that HSSM uses at inference time. For ecosystem-wide context, see the HSSMSpine repo.

## Project Structure

```
src/lanfactory/                # Main package
  cli/                         # Typer CLIs: jaxtrain, torchtrain, transform-onnx, upload-hf, download-hf
  config/                      # Default network and training configs (LAN, CPN, OPN)
  trainers/                    # Training implementations (torch_mlp.py, jax_mlp.py)
  onnx/                        # PyTorch → ONNX export
  hf/                          # HuggingFace Hub integration (upload, download, model cards)
  utils/                       # Config save/load, MLflow utilities
tests/                         # pytest suite (trainers, CLI, ONNX, HuggingFace, E2E)
docs/                          # MkDocs documentation + tutorial notebooks
notebooks/                     # Test notebooks
```

## Build & Tooling

- **Build system:** setuptools (pure Python, no compiled extensions)
- **Package manager:** uv (with `uv.lock`)
- **Python:** >3.10, <3.14 (classifiers target 3.11, 3.12, 3.13)
- **Linting:** ruff (line length 88, via pre-commit)
- **Type checking:** mypy
- **No system dependencies** — unlike ssm-simulators, this is pure Python + PyTorch/Flax

## Common Commands

```bash
# Install all dependency groups (e.g. dev)
uv sync --all-groups

# Run tests
uv run pytest tests/

# Lint & format
uv run ruff check src/lanfactory && uv run ruff format --check .

# Build docs
uv run mkdocs build
uv run mkdocs serve

# Train a network (PyTorch)
uv run torchtrain --config-path <yaml> --training-data-folder <dir> --networks-path-base <dir>

# Train a network (JAX)
uv run jaxtrain --config-path <yaml> --training-data-folder <dir> --networks-path-base <dir>

# Export PyTorch model to ONNX
uv run transform-onnx --network-config-file config.pickle --state-dict-file model.pt \
  --input-shape 6 --output-onnx-file model.onnx

# Upload trained models to HuggingFace
uv run upload-hf --model-folder <dir> --network-type lan --model-name ddm

# Download models from HuggingFace
uv run download-hf --network-type lan --model-name ddm --output-folder <dir>
```

## Key Architecture Patterns

### Network Types

| Type | Full Name | Output | Loss | Use Case |
|------|-----------|--------|------|----------|
| LAN | Likelihood Approximation Network | logprob | Huber | Log-likelihood approximation |
| CPN | Choice Probability Network | logits | BCE with logits | Choice probability estimation |
| OPN | Option Probability Network | logits | BCE with logits | Option probability estimation |

All three use the same MLP architecture (`[100, 100, 1]` default, tanh activations)
but differ in output type and loss function.

### Training Backends

- **PyTorch** (`torchtrain` CLI, `trainers/torch_mlp.py`) — primary backend.
  Supports CUDA, ONNX export, full training loop with validation.
- **JAX/Flax** (`jaxtrain` CLI, `trainers/jax_mlp.py`) — alternative backend.
  Uses optax optimizers. No native ONNX export (train in JAX, convert via PyTorch if needed).

### ONNX Export Pipeline

PyTorch model → `torch.onnx.export()` → `.onnx` file. This is the format
HSSM consumes at runtime. Only PyTorch models can be directly exported to ONNX.

### HuggingFace Integration

- **Upload:** `lanfactory.hf.upload_model()` — uploads `.onnx`, `.pt`, config pickles,
  and auto-generated README to `franklab/HSSM` on HuggingFace.
  Requires `model_card.yaml` in the model folder.
- **Download:** `lanfactory.hf.download_model()` — downloads by network type + model name.
- **Default repo:** `franklab/HSSM`
- **Optional dependency:** `huggingface-hub>=0.20.0` (install via `uv sync --extra hf`)

### Config System

Training configs are YAML files parsed by the CLI. Key fields:
- `NETWORK_TYPE`: `lan`, `cpn`, or `opn`
- `layer_sizes`, `activations`: network architecture
- `n_epochs`, `learning_rate`, `loss`, `optimizer`: training hyperparams
- `cpu_batch_size`, `gpu_batch_size`: device-specific batch sizes

Default configs available in `lanfactory.config.network_configs`.

### MLflow Integration

Optional experiment tracking via MLflow. CLI flags: `--mlflow-run-name`, `--mlflow-experiment-name`,
`--mlflow-tracking-uri`, `--mlflow-artifact-location`. Supports resuming runs via `--mlflow-run-id`.

## CLI Entry Points

| Command | Module | Purpose |
|---------|--------|---------|
| `torchtrain` | `lanfactory.cli.torch_train` | Train PyTorch networks from YAML config |
| `jaxtrain` | `lanfactory.cli.jax_train` | Train JAX networks from YAML config |
| `transform-onnx` | `lanfactory.onnx.transform_onnx` | Convert PyTorch model → ONNX |
| `upload-hf` | `lanfactory.cli.upload_hf` | Upload trained models to HuggingFace |
| `download-hf` | `lanfactory.cli.download_hf` | Download models from HuggingFace |

## CI Workflows

| Workflow | Purpose |
|----------|---------|
| `run_tests.yml` | Tests on Python 3.11/3.12/3.13 + ruff lint/format + codecov |
| `build_wheels.yml` | Build sdist, upload to TestPyPI → PyPI on release publish |

## Known Issues

- `__init__.py` version (`0.5.3`) is out of sync with `pyproject.toml` (`0.6.1`)

## Compaction

When compacting, preserve: file list of modified files, the three network types
(LAN/CPN/OPN) and their differences, CLI entry points, ONNX export flow,
HuggingFace upload/download interface, and all test commands.
