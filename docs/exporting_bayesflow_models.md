# bayesflow export: architectures and constraints

> The artifact rules every exporter here satisfies are collected in
> [The ONNX likelihood contract](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/).

LANfactory's [`transform_bayesflow_to_onnx`](api/onnx.md) is the bayesflow
sibling of [`transform_sbi_to_onnx`](exporting_sbi_models.md). It wraps a
trained [`bayesflow`](https://github.com/bayesflow-org/bayesflow)
`ContinuousApproximator` (NLE) or `RatioApproximator` (NRE) and writes a
single-trial ONNX file. This page owns exporter architecture support,
constraints, and numerical guarantees. Follow the runnable
[bayesflow export tutorial](tutorials/exporting_bayesflow_to_onnx.ipynb) for
installation, training, export, and cross-backend verification. HSSM owns the
downstream [ONNX likelihood contract](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/)
and model-loading procedure.

The classifier logit is `log p(x|θ)/p(x) = log p(x|θ) − log p(x)`. The
θ-independent `log p(x)` term drops out under MCMC, so the raw logit is the
log-likelihood up to a constant. No Jacobian correction is needed — ratios
are invariant to z-score standardization.

## Known constraints (v1)

The constraints below were uncovered by the C-series validation spike. They
fall into four buckets.

### 1. KERAS_BACKEND must be `torch`

ONNX export goes through `torch.onnx.export`. Under `KERAS_BACKEND=jax` the
network weights live in JAX; tracing them with torch's exporter is not
supported. The exporter raises `RuntimeError` with a corrective hint.

### 2. CouplingFlow knobs

`bf.networks.CouplingFlow` has a few defaults that don't survive ONNX export
at opset 17/20. Override them at training time:

| Knob | Required value | Why |
|---|---|---|
| `permutation` | `None` | `FixedPermutation` uses `keras.ops.take`, which lowers to `aten::ravel`. Neither opset 17 nor 20 implements it. |
| `use_actnorm` | `False` | Not validated in v1. May work; not tested. |
| `transform` | `AffineTransform(clamp=False)` (explicit instance) | Default `clamp=True` emits `ops.arcsinh`, which exports as `aten::asinh`. Unsupported in opset 17/20. Pass an explicit instance — bayesflow's `find_transform("affine")` silently drops `transform_kwargs` (upstream bug). |

### 3. Subnet activation

The default coupling MLP activation is `"hard_silu"` (HardSwish, the
piecewise-linear approximation to SiLU). PyTorch exports HardSwish as a
single fused ONNX op (`HardSwish`, added in opset 14) preserving the
efficiency motivation behind the function. jaxonnxruntime does not yet
implement a handler for that op.

**Workaround**: use `"silu"` (the smooth Swish, `x · σ(x)`). It decomposes
to `Sigmoid + Mul` on export — primitive ops every runtime supports. The
two functions differ by at most ~0.14 across the real line (max around
`|x| ≈ 3`) and are interchangeable for SBI accuracy. Set:

```python
subnet_kwargs={"widths": (...), "activation": "silu", "dropout": None}
```

`dropout=None` is recommended for a cleaner inference-time trace; the
trained weights are unchanged by this.

### 4. Adapter must be identity

The exporter raises `ValueError` if `approximator.adapter` contains any
transforms. The bayesflow `Adapter` pipeline is implemented in numpy
(dict reshuffling, log/sqrt transforms, scale, concat, etc.) and cannot
be baked into an ONNX graph in v1.

**What you can use without an adapter**: the in-network `Standardize`
layer (via `standardize="inference_variables"` or `"all"`) IS tensor-based
and gets baked into the exported graph automatically, including the
correct Jacobian correction for absolute log-probability values.

**What you cannot use**: `Adapter().log("rt").standardize(...).concatenate(...)`
style chains. Move pointwise transforms (log/sqrt of observations) into your
simulator output and apply them externally to your HSSM data before
sampling.

## Explicitly out of scope (v1)

| Excluded | Reason |
|---|---|
| Discrete + continuous observations (MNLE-style) | bayesflow has no MNLE-equivalent approximator; would need new network types and training objectives. |
| Non-identity adapters | Numpy-only operations can't be baked into ONNX; see Constraint 4 above. Pointwise tensor adapter ops (log, sqrt, scale) are a candidate for v1.x. |
| Transformer / attention summary networks | Contain `LayerNormalization` (no jaxonnxruntime handler) and dynamic-shape attention. |
| FlowMatching, DiffusionModel, ConsistencyModel inference networks | `log_prob` requires ODE integration, not ONNX-exportable. |
| `KERAS_BACKEND=jax` workflows | Use the bayesflow LRE-style in-memory JAX callable path (see [`bayesflow_lre_integration.ipynb`](https://lnccbrown.github.io/HSSM/tutorials/bayesflow_lre_integration/) in HSSM). |

## Numerical guarantees

The bayesflow regression tests (`tests/test_bayesflow_*_export.py`) assert:

- Forward pass: torch reference wrapper, `onnxruntime`, and
  `jaxonnxruntime` all agree to `atol=1e-5` on fixed inputs.
- Gradients: `jax.grad` of the translated graph agrees with
  `torch.autograd.grad` on the wrapped network to `atol=1e-4`.

If you observe drift larger than these thresholds, please open an issue
with a minimal reproducer.

## HSSM handoff

LANfactory owns the portable ONNX exporter. HSSM owns the choice between a
file-backed likelihood and an in-memory JAX callable, together with its model
configuration and sampling behavior. Continue with HSSM's rendered ONNX
contract instead of duplicating that consumer procedure here.

## Related API

- [`lanfactory.onnx.transform_bayesflow_to_onnx`](api/onnx.md) — this exporter.
- [`lanfactory.onnx.transform_sbi_to_onnx`](api/onnx.md) — the sbi sibling.
- [`lanfactory.onnx.transform_to_onnx`](api/onnx.md) — the LAN-MLP exporter.
- [Export a bayesflow model to ONNX](tutorials/exporting_bayesflow_to_onnx.ipynb)
  — the executable task guide for this reference.
