# Share trained networks on Hugging Face Hub

Use LANfactory's Hub commands to publish a trained artifact set, make its
canonical ONNX file discoverable by released HSSM versions, or retrieve the
folder for inspection and local reuse.

!!! info "Execution status"

    Package tests exercise placement, manifest, overwrite, and download logic
    against a fake Hub. Documentation CI never authenticates, uploads, or
    downloads. Preview every publication locally with `--dry-run`, then verify
    the real commit and root alias in the target repository.

## Install and authenticate

Install the optional Hub dependency:

```bash
pip install 'lanfactory[hf]'
```

From a source checkout, use `uv sync --extra hf` instead. Authenticate with
the Hugging Face CLI or provide `HF_TOKEN` through your shell or secret manager.
Avoid putting a token directly in a shared command or shell history.

## Preview the publication

Point `upload-hf` at one trained-model folder. The accepted network types are
`lan`, `cpn`, `opn`, and `gonogo`.

```bash
upload-hf \
  --model-folder ./networks/lan/ddm/ \
  --network-type lan \
  --model-name ddm \
  --dry-run
```

By default, LANfactory plans three coordinated placements:

1. the complete artifact set under `lan/ddm/`;
2. the canonical ONNX file at the repository root as `ddm.onnx`; and
3. an updated root `manifest.json` entry.

The other root aliases are `{model}_cpn.onnx`, `{model}_opn.onnx`, and
`{model}_gonogo.onnx`. Released HSSM versions resolve these root filenames;
publishing only the folder does not make a network consumable by HSSM.

If the folder has no unambiguous ONNX filename for the requested model, pass
`--canonical-onnx PATH`. A `model_card.yaml` is optional by default: LANfactory
can generate one from the saved configuration, using the artifact repository's
`bsd-2-clause` license metadata. Use `--require-model-card` when publication
policy requires a reviewed, hand-authored card.

## Publish and verify

Remove `--dry-run` only after the plan identifies the intended artifact and
root filename:

```bash
upload-hf \
  --model-folder ./networks/lan/ddm/ \
  --network-type lan \
  --model-name ddm \
  --commit-message "Publish validated DDM network"
```

The command refuses to replace an existing root network unless you explicitly
pass `--overwrite-root`. That guard is consequential: released HSSM versions
download the root file from `main` without pinning a revision. Test a candidate
in a staging repository and complete its validation before authorizing a root
replacement.

After upload, verify that the Hub commit contains the folder, root alias, and
manifest update together. `--no-publish-root-alias` and
`--no-update-manifest` are specialized escape hatches; do not use them for a
normal HSSM-facing publication.

## Retrieve a published folder

`download-hf` copies the selected `{network-type}/{model-name}/` folder into a
local directory:

```bash
download-hf \
  --network-type lan \
  --model-name ddm \
  --output-folder ./models/ddm/
```

An existing destination is rejected unless `--force` is set. The download
command does not reconfigure HSSM. To use a downloaded ONNX file directly,
pass its local path through HSSM's
[approximate-differentiable ONNX route](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/).
When a validated root alias is published in `franklab/HSSM`, HSSM's built-in
model configuration can resolve it from the Hub instead.

## Inspect the exact interface

Run `upload-hf --help` and `download-hf --help` for the installed version. The
[command-line reference](api/cli.md#upload-hf) records every flag and safety
default. The [Hub Python API reference](api/hf.md) documents the public helpers
and constants owned by LANfactory.
