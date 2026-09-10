# `lanfactory.derive`

Integrate a trained LAN's density over reaction time so that choice
probabilities and deadline (omission) mass can be derived from the LAN
instead of simulated. The density is integrated only on `[t_min, max_t]`
(`max_t` = 20 s, the LANs' training support) and never renormalised, so the
per-choice masses fall short of one by the density past `max_t` (ssms returns
such trials at `rt ≈ max_t + t` with a choice rather than as omissions, so its
`choice_p` sums to one); callers record `ChoiceMass.total` alongside the
per-choice masses.

::: lanfactory.derive.load_onnx_predictor

::: lanfactory.derive.OnnxPredictor

::: lanfactory.derive.Predictor

::: lanfactory.derive.IntegrationGrid

::: lanfactory.derive.choice_mass

::: lanfactory.derive.ChoiceMass
