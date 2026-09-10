# `lanfactory.derive`

Integrate a trained LAN's density over reaction time so that choice
probabilities and deadline (omission) mass can be derived from the LAN
instead of simulated.

## Grid policy

The density is integrated with the trapezoid rule over the full support
`[t_min, max_t]` (`max_t` = 20 s, the LANs' training support; nothing is
extrapolated past it). A LAN's density is not smooth: it rises from zero
within a few milliseconds of the non-decision time `t`. On the Hub ddm LAN a
uniform 1000-point `IntegrationGrid` carries 15–25 % quadrature error at
`a < 0.5` (totals down to 0.66 that are pure grid artefacts), so the grid of
record is the per-parameter-vector `OnsetGrid`: 32 points on
`[t_min, t − 0.05]`, 128 on `[t − 0.05, t]`, 600 on `[t, t + 1]` and 400 on
`[t + 1, max_t]`, which matches a 16 000-point uniform grid to 5 × 10⁻⁵ at
~1160 points. On that grid, over 20 000 uniform draws from the ddm box, the
total mass has median `|total − 1|` 0.0037, p99 0.080, min 0.877 / max 1.219,
with 1.9 % of the box beyond 0.05. Integrating from `t` only was tested and
rejected (mean error against simulation 0.0083 vs 0.0042): the LAN leaks
mass below `t` (mean 0.0035, p99 0.052) and that leak is part of what it
predicts, so `ChoiceMass.leak_below` reports it rather than dropping it.

## Tail policy

The per-choice masses are never renormalised; they carry the density the LAN
puts past `max_t` and the network's own scale error (ssms returns
un-terminated trials at `rt ≈ max_t + t` with a choice rather than as
omissions, so its `choice_p` sums to one while these masses need not).
Renormalisation is the corpus's job; `ChoiceMass.total` is always recorded
alongside the per-choice masses, and `survey` reports how it behaves over a
whole training box.

::: lanfactory.derive.load_onnx_predictor

::: lanfactory.derive.OnnxPredictor

::: lanfactory.derive.Predictor

::: lanfactory.derive.IntegrationGrid

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
| `cpn` | `[theta..., choice]` — one row per choice code in `model_config["choices"]` | `mass(choice)` |
| `opn` | `[theta..., deadline]` (deadline last) | `1 - mass_before(deadline)`, summed over choices |
| `gonogo` | `[theta..., deadline]` | `mass_before(deadline, nogo) + (1 - mass_before(deadline))`, `nogo` = every choice but the largest code (ssms' `nogo_p`) |

Labels are clipped to `[0, 1]` (the LAN's total mass can sit a few
thousandths above one); they are never renormalised. The omission term
`1 - mass_before(deadline)` is clipped once, before it enters the `opn` and
`gonogo` labels, so `gonogo == mass_before(deadline, nogo) + opn` holds in
the written corpus. Each pickle's `generator_config["derive_stats"]` and the
manifest record the total-mass mean / min / max so the deficit stays visible.
A derived `cpn` corpus is one column wider than ssms' simulated CPN corpus
(`[theta...]` with a single `P(choice = 1)` label); see the
[network types](../network_types.md#deriving-auxiliary-networks-from-a-trained-lan)
page.

**Batch size.** `DatasetTorch` requires the batch size to divide the rows
per file: `n_theta_per_file` rows for `opn` / `gonogo`, `n_theta_per_file
× n_choices` for `cpn`. With the default 4096 thetas that is 4096 and 8192
rows for a two-choice model; `512` divides both.

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

::: lanfactory.derive.SourceLAN

::: lanfactory.derive.NETWORK_TYPES

::: lanfactory.derive.AUX_CATEGORY

::: lanfactory.derive.DERIVATION_METHOD

::: lanfactory.derive.MANIFEST_NAME
