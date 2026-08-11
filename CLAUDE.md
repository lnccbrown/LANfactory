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
- **Python:** >=3.12, <3.15 (classifiers target 3.12, 3.13, 3.14)
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
  Uses optax optimizers. Exports ONNX directly via `jax2onnx` (`onnx/jax_export.py`);
  `jaxtrain` writes the artifact by default (`--no-export-onnx` to skip).

### ONNX Export Pipeline

Four exporters, all in `src/lanfactory/onnx/`:
- **LAN/CPN/OPN MLPs (torch)** — `transform_onnx.py` (`transform-onnx` CLI)
- **LAN/CPN/OPN MLPs (jax)** — `jax_export.py` (`transform-jax-onnx` CLI), via `jax2onnx`
- **sbi** posterior/likelihood/ratio estimators — `sbi.py` (`transform_sbi_to_onnx`)
- **bayesflow** networks — `bayesflow.py` (`transform_bayesflow_to_onnx`)

All follow the single-trial contract: **every input dim concrete, no
`dynamic_axes`**; HSSM batches per-trial via `jax.vmap`.

The *rank* is not part of the contract — it follows from how your tracer lowers
a dense layer. The MLP exporters (torch and jax) trace `(1, D)` and lower to
`Gemm`, whose ONNX spec requires rank 2; the sbi and bayesflow exporters trace
rank-1 `(D,)` because `torch.onnx.export` lowers `Linear` to rank-agnostic
`MatMul`+`Add`. Both load in HSSM and, measured under `vmap`+`jit`, run
identically. The production networks on franklab/HSSM are `(1, D)` Gemm.
`assert_single_trial_contract` in `onnx/contract.py` is the executable version
of this paragraph — call it from any new exporter's tests.

### HuggingFace Integration

- **Upload:** `lanfactory.hf.upload_model()` — uploads `.onnx`, `.pt`, config pickles,
  and auto-generated README to `franklab/HSSM` on HuggingFace. Publishes the
  canonical ONNX at the repo *root* under the filename HSSM downloads, plus the
  full artifact set under `{network_type}/{model}/`, plus a root `manifest.json`
  — in one atomic commit. Refuses to replace an existing root network without
  `--overwrite-root`. `model_card.yaml` is generated when absent
  (`--require-model-card` to demand one).
- **Download:** `lanfactory.hf.download_model()` — downloads by network type + model name.
- **Default repo:** `franklab/HSSM`
- **Optional dependency:** `huggingface-hub>=0.20.0` (install via `uv sync --extra hf`)

### Config System

Training configs are YAML files parsed by the CLI. Key fields:
- `NETWORK_TYPE`: `lan`, `cpn`, `opn`, or `gonogo`
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
| `transform-jax-onnx` | `lanfactory.onnx.jax_export` | Convert a jaxtrain network → ONNX |
| `upload-hf` | `lanfactory.cli.upload_hf` | Upload trained models to HuggingFace |
| `download-hf` | `lanfactory.cli.download_hf` | Download models from HuggingFace |

## CI Workflows

| Workflow | Purpose |
|----------|---------|
| `run_tests.yml` | Tests on Python 3.12/3.13/3.14 + ruff lint/format + codecov |
| `build_wheels.yml` | Build sdist, upload to TestPyPI → PyPI on release publish |

## Compaction

When compacting, preserve: file list of modified files, the three network types
(LAN/CPN/OPN) and their differences, CLI entry points, ONNX export flow,
HuggingFace upload/download interface, and all test commands.
