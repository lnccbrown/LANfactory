# `lanfactory.derive`

Integrate a trained LAN's density over reaction time so that choice
probabilities and deadline (omission) mass can be derived from the LAN
instead of simulated.

::: lanfactory.derive.integrate
    options:
      members: false

::: lanfactory.derive.load_onnx_predictor

::: lanfactory.derive.OnnxPredictor

::: lanfactory.derive.Predictor

::: lanfactory.derive.OnsetGrid

::: lanfactory.derive.choice_mass

::: lanfactory.derive.ChoiceMass

::: lanfactory.derive.survey

## Derived corpora

`lanfactory.derive.corpus` turns those masses into training corpora the
trainers read unchanged (`{type}_data`, `{type}_labels`, `generator_config`,
`model_config` per pickle), plus a `derive_manifest.json` sidecar. The
`derive-aux` [command](cli.md#derive-aux) wraps `derive_aux_corpus`; the
[network types](../network_types.md#deriving-auxiliary-networks-from-a-trained-lan)
page explains what is derived, the renormalisation and the fallback, the
recorded statistics and provenance, and how to train on the result.

Every row is `n_params + 1` wide, in ssms parameter order:

| type | training row | label |
| --- | --- | --- |
| `cpn` | `[theta..., choice]` — one row per choice code in `model_config["choices"]` | `mass(choice) / total` |
| `opn` | `[theta..., deadline]` (deadline last) | `1 − F(deadline) / total`, `F(d)` = `mass_before(d)` summed over choices |

**Batch size.** `DatasetTorch` requires the batch size to divide the rows
per file: `n_theta_per_file` rows for `opn`, `n_theta_per_file
× n_choices` for `cpn`. With the default 4096 thetas that is 4096 and 8192
rows for a two-choice model; `512` divides both.

When `torchtrain` or `jaxtrain` trains on a derived corpus with MLflow
tracking on, the training run carries those keys as params —
`source_lan_run_id` only when known, the run uuid and Hub commit as `""`
when unknown — the `derive_stats` as tags (a `None` statistic as `""`),
and the tag `data_origin=derived`; see
[Derived corpora](../using_mlflow.md#derived-corpora) under *Track training
runs with MLflow* for the exact set and what an empty value means
downstream.

::: lanfactory.derive.derive_aux_corpus

::: lanfactory.derive.NETWORK_TYPES

::: lanfactory.derive.AUX_CATEGORY

::: lanfactory.derive.DERIVATION_METHOD

::: lanfactory.derive.MANIFEST_NAME
