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
