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
page explains what is derived and how to train on the result.

Every row is `n_params + 1` wide, in ssms parameter order:

| type | training row | label |
| --- | --- | --- |
| `cpn` | `[theta..., choice]` — one row per choice code in `model_config["choices"]` | `mass(choice) / total` |
| `opn` | `[theta..., deadline]` (deadline last) | `1 − F(deadline) / total`, `F(d)` = `mass_before(d)` summed over choices |
| `gonogo` | `[theta..., deadline]` | `nogo_before(deadline) / total + (1 − F(deadline) / total)`, `nogo` = every choice but the largest code (ssms' `nogo_p`) |

Labels are renormalised by the network's own total (`ChoiceMass.total`, the
mass on `[t_min, max_t]` summed over choices); the total is recorded, never
hidden — each pickle's `generator_config["derive_stats"]` and the manifest
carry its mean / min / max. The LAN's error against simulation is mostly
scale, so the division removes most of it (Hub ddm LAN: cpn mean error 0.035
→ 0.004, max 0.17 → 0.033); what remains sits where the total is off, which
is what the simulation fallback of `derive_aux_corpus` covers. The float32
clip to `[0, 1]` is kept as a safety net, and the omission term is formed
once so `gonogo == nogo_before / total + opn` holds row by row.
A derived `cpn` corpus is one column wider than ssms' simulated CPN corpus
(`[theta...]` with a single `P(choice = 1)` label); see the
[network types](../network_types.md#deriving-auxiliary-networks-from-a-trained-lan)
page.

**Batch size.** `DatasetTorch` requires the batch size to divide the rows
per file: `n_theta_per_file` rows for `opn` / `gonogo`, `n_theta_per_file
× n_choices` for `cpn`. With the default 4096 thetas that is 4096 and 8192
rows for a two-choice model; `512` divides both.

**Fallback.** A theta whose total lies outside `fallback_window` (default
`(0.98, 1.03)`; a total equal to either endpoint is trusted) is labelled by
`simulate_labels` on ssms rather
than by the LAN; `fallback_window=None` disables it. Each pickle's
`generator_config["derive_stats"]` and the manifest's `derive_stats` carry the
flat keys `derive_total_mass_mean`, `derive_total_mass_min`,
`derive_total_mass_max`, `derive_fallback_frac`, `derive_sim_past_max_t_max`
(`None` when nothing fell back) and `derive_leak_below_onset_p99` (`None`
without an onset grid: on a uniform grid the leak would mostly be the grid's
own quadrature error, and the manifest's `lan_survey` records none there
either); the manifest also carries the source LAN's
whole-box `survey` under `lan_survey`. The
[network types](../network_types.md#renormalisation-the-window-and-the-fallback)
page gives the measurements behind the policy.

**Provenance.** `generator_config["source"]` is the flat dict returned by
`SourceLAN.provenance`, with exactly the keys `derivation_method`
(`"derived-from-lan"`), `aux_category` (`choice` / `omission` / `nogo`),
`source_lan_run_uuid`, `source_lan_sha256`, `source_lan_hf_commit`,
`source_lan_run_id` (`None`; filled in by the tool that knows the MLflow
run), `integration_grid`, `integration_max_t`. Downstream tools read these
by name.

::: lanfactory.derive.derive_aux_corpus

::: lanfactory.derive.sample_theta

::: lanfactory.derive.sample_deadlines

::: lanfactory.derive.cpn_labels

::: lanfactory.derive.opn_labels

::: lanfactory.derive.gonogo_labels

::: lanfactory.derive.simulate_labels

::: lanfactory.derive.SourceLAN

::: lanfactory.derive.grid_description

::: lanfactory.derive.NETWORK_TYPES

::: lanfactory.derive.AUX_CATEGORY

::: lanfactory.derive.DERIVATION_METHOD

::: lanfactory.derive.MANIFEST_NAME
