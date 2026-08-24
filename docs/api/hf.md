:::lanfactory.hf

## Public constants

| Export | Value | Contract |
| --- | --- | --- |
| `DEFAULT_REPO_ID` | `franklab/HSSM` | Default artifact repository used by Hub helpers |
| `DEFAULT_LICENSE` | `bsd-2-clause` | License metadata used for generated model cards |
| `VALID_NETWORK_TYPES` | `lan`, `cpn`, `opn`, `gonogo` | Network types accepted by Hub publication helpers |

## Public helpers

- `load_model_card_yaml` reads model-card metadata.
- `generate_readme` renders a model card from metadata.
- `ModelCardConfig` stores model-card metadata and defaults.
- `upload_model` publishes a trained artifact and its metadata.
- `download_model` retrieves a published network artifact.

The installed `upload-hf` and `download-hf` entry points, including every flag
and safety default, are documented in the [command-line reference](cli.md).

For the task-oriented publication sequence, see
[Share trained networks on Hugging Face Hub](../using_huggingface.md).
