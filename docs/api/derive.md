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
page explains what is derived, the row layouts, the renormalisation and the
fallback, the recorded statistics and provenance, and how to train on the
result.

::: lanfactory.derive.derive_aux_corpus

::: lanfactory.derive.NETWORK_TYPES

::: lanfactory.derive.AUX_CATEGORY

::: lanfactory.derive.DERIVATION_METHOD

::: lanfactory.derive.MANIFEST_NAME
