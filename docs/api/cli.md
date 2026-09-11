# Command-line reference

LANfactory installs seven commands. The tables below describe the complete
command-line surface; run `COMMAND --help` to inspect the version installed in
your environment. Training commands consume LANfactory configuration files,
export commands convert saved trainer artifacts, Hub commands move reviewed
artifacts to or from a Hugging Face repository, and `derive-aux` turns a
trained LAN into an auxiliary-network training corpus.

## `jaxtrain`

Train a JAX network. Either `--training-data-folder` or
`--data-generation-experiment-id` must identify the training data.

| Option | Required/default | Contract |
| --- | --- | --- |
| `--config-path PATH` | bundled configuration | YAML training configuration |
| `--training-data-folder PATH` | unset | Training data directory; optional when an MLflow data-generation experiment is supplied |
| `--network-id INTEGER` | `0` | Network entry selected from the configuration |
| `--dl-workers INTEGER` | `1` | DataLoader worker count; non-positive values request automatic sizing |
| `--networks-path-base PATH` | required | Base directory for saved network artifacts |
| `--dry-run` | off | Validate configuration and data discovery without training |
| `--export-onnx` / `--no-export-onnx` | on | Export the HSSM-consumable ONNX artifact after training |
| `--mlflow-run-name TEXT` | unset | Enable tracking under this run name |
| `--mlflow-experiment-name TEXT` | `MLFLOW_EXPERIMENT_NAME` or unset | MLflow experiment name |
| `--mlflow-run-id TEXT` | unset | Resume an existing MLflow run |
| `--data-generation-experiment-id TEXT` | unset | Derive the data location and lineage from an MLflow experiment |
| `--mlflow-tracking-uri TEXT` | `MLFLOW_TRACKING_URI` or `sqlite:///mlflow.db` | MLflow tracking backend |
| `--mlflow-artifact-location TEXT` | `MLFLOW_ARTIFACT_LOCATION` or `./mlruns` | MLflow artifact root |
| `--log-level LEVEL`, `-l LEVEL` | `WARNING` | Logging threshold |

## `torchtrain`

Train a PyTorch network. Its data-discovery and MLflow options match
`jaxtrain`; PyTorch artifacts can be converted with `transform-onnx`.

| Option | Required/default | Contract |
| --- | --- | --- |
| `--config-path PATH` | bundled configuration | YAML training configuration |
| `--training-data-folder PATH` | unset | Training data directory; optional when an MLflow data-generation experiment is supplied |
| `--networks-path-base PATH` | required | Base directory for saved network artifacts |
| `--network-id INTEGER` | `0` | Network entry selected from the configuration |
| `--dl-workers INTEGER` | `1` | DataLoader worker count; non-positive values request automatic sizing |
| `--dry-run` | off | Validate configuration and data discovery without training |
| `--mlflow-run-name TEXT` | unset | Enable tracking under this run name |
| `--mlflow-experiment-name TEXT` | `MLFLOW_EXPERIMENT_NAME` or unset | MLflow experiment name |
| `--mlflow-run-id TEXT` | unset | Resume an existing MLflow run |
| `--data-generation-experiment-id TEXT` | unset | Derive the data location and lineage from an MLflow experiment |
| `--mlflow-tracking-uri TEXT` | `MLFLOW_TRACKING_URI` or `sqlite:///mlflow.db` | MLflow tracking backend |
| `--mlflow-artifact-location TEXT` | `MLFLOW_ARTIFACT_LOCATION` or `./mlruns` | MLflow artifact root |
| `--log-level LEVEL`, `-l LEVEL` | `WARNING` | Logging threshold |

## `transform-onnx`

Convert a saved PyTorch `TorchMLP` configuration and state dictionary to ONNX.

| Option | Required/default | Contract |
| --- | --- | --- |
| `--network-config-file TEXT` | required | Pickled network configuration |
| `--state-dict-file TEXT` | required | Saved PyTorch state dictionary |
| `--input-shape INTEGER` | required | Concrete single-trial input width |
| `--output-onnx-file TEXT` | required | Destination ONNX file |

## `transform-jax-onnx`

Convert a saved `jaxtrain` network configuration and Flax state to ONNX.

| Option | Required/default | Contract |
| --- | --- | --- |
| `--network-config-file TEXT` | required | Pickled network configuration |
| `--state-file TEXT` | required | `*_train_state.jax` Flax parameter bytes |
| `--input-shape INTEGER` | required | Concrete single-trial input width |
| `--output-onnx-file TEXT` | required | Destination ONNX file |
| `--opset INTEGER` | `17` | Target ONNX opset |

Both transform commands produce concrete-shape, single-trial artifacts. Rank is
exporter-specific; HSSM owns the consumer contract and trial-wise vectorization.
See the [sbi](../exporting_sbi_models.md) and
[BayesFlow](../exporting_bayesflow_models.md) exporter references for the same
cross-package boundary.

## `upload-hf`

Publish a trained artifact set, its optional model card, canonical root alias,
and manifest entry.

| Option | Required/default | Contract |
| --- | --- | --- |
| `--model-folder PATH` | required | Folder containing the trained artifacts; `model_card.yaml` is optional |
| `--network-type TEXT` | required | One of `lan`, `cpn`, `opn`, or `gonogo` |
| `--model-name TEXT` | required | Model identifier used in the folder and root filename |
| `--repo-id TEXT` | `franklab/HSSM` | Target Hub repository |
| `--commit-message TEXT` | `Upload model` | Hub commit message |
| `--private` | off | Create a private repository when creating the target |
| `--create-repo` | off | Create the target repository if absent |
| `--include-patterns TEXT` | unset | Comma-separated filename globs to include |
| `--exclude-patterns TEXT` | unset | Comma-separated filename globs to exclude |
| `--revision TEXT` | unset | Target branch or tag |
| `--token TEXT` | `HF_TOKEN` or unset | Explicit token or environment fallback |
| `--dry-run` | off | Print the publication plan without uploading or mutating files |
| `--publish-root-alias` / `--no-publish-root-alias` | on | Publish the canonical root filename consumed by HSSM |
| `--update-manifest` / `--no-update-manifest` | on | Read-modify-write the root `manifest.json` |
| `--require-model-card` | off | Reject a missing `model_card.yaml` instead of generating metadata |
| `--canonical-onnx PATH` | inferred | Select the ONNX file copied to the repository root |
| `--overwrite-root` | off | Permit replacement of an existing HSSM-facing root artifact |
| `--log-level LEVEL`, `-l LEVEL` | `WARNING` | Logging threshold |

## `download-hf`

Retrieve one `{network-type}/{model-name}/` folder from a Hub repository.

| Option | Required/default | Contract |
| --- | --- | --- |
| `--network-type TEXT` | required | One of `lan`, `cpn`, `opn`, or `gonogo` |
| `--model-name TEXT` | required | Model folder to retrieve |
| `--output-folder PATH` | required | Local destination; must be absent unless `--force` is set |
| `--repo-id TEXT` | `franklab/HSSM` | Source Hub repository |
| `--revision TEXT` | unset (Hub default: `main`) | Branch, tag, or commit to retrieve |
| `--include-patterns TEXT` | unset | Comma-separated filename globs to include |
| `--exclude-patterns TEXT` | unset | Comma-separated filename globs to exclude |
| `--token TEXT` | `HF_TOKEN` or unset | Explicit token or environment fallback for private repositories |
| `--force` | off | Replace an existing destination |
| `--log-level LEVEL`, `-l LEVEL` | `WARNING` | Logging threshold |

## `derive-aux`

Derive a CPN, OPN, or go/no-go training corpus from a trained LAN by
integrating its density over reaction time (see
[Deriving auxiliary networks](../network_types.md#deriving-auxiliary-networks-from-a-trained-lan)).
Writes `--n-files` pickles plus `derive_manifest.json` to `--output-folder`,
in the layout `torchtrain` and `jaxtrain` read, and prints the manifest path.
The option names follow the Hub commands; `--type`, `--model`, and `--out` are
short aliases.

| Option | Required/default | Contract |
| --- | --- | --- |
| `--from-onnx PATH` | required | The trained LAN; a `(1, n_params + 2)` ONNX artifact |
| `--network-type TEXT`, `--type TEXT` | required | Auxiliary network type to derive: one of `cpn`, `opn`, or `gonogo` |
| `--model-name TEXT`, `--model TEXT` | required | ssms model name the LAN was trained for (e.g. `ddm`) |
| `--output-folder PATH`, `--out PATH` | required | Destination folder; created if absent |
| `--n-files INTEGER` | `100` | Number of pickles to write; at least 2 |
| `--n-theta-per-file INTEGER` | `4096` | Parameter vectors per file; rows per file for `opn`/`gonogo`, times the number of choices for `cpn` |
| `--onset-param TEXT` | `t` | Non-decision-time parameter the per-parameter-vector onset grid is refined around; an empty string integrates on a uniform grid instead |
| `--grid-points INTEGER` | `1000` | Points of the uniform grid; used only when no onset grid applies (`--onset-param ''` or a model without that parameter) |
| `--max-t FLOAT` | `20.0` | Upper edge of the integration grid in seconds; the density is never evaluated past it |
| `--deadline-quantile-frac FLOAT` | `0.7` | Share of `opn`/`gonogo` deadlines drawn from the LAN's own RT quantiles; the rest are uniform on the deadline bounds |
| `--seed INTEGER` | `0` | Base seed; file `i` is generated from `[seed, i]` |
| `--source-run-uuid TEXT` | parsed from the filename | Training run uuid recorded in the provenance |
| `--source-hf-repo TEXT` | unset | Hub repository the LAN was downloaded from |
| `--source-hf-revision TEXT` | unset | Hub revision of that download, recorded as `source_lan_hf_commit` |

For task-oriented workflows, see [Track training with MLflow](../using_mlflow.md)
and [Share trained networks on Hugging Face Hub](../using_huggingface.md). The
[Python API reference](hf.md) documents the Hub helpers and constants called by
the two Hub entry points.
