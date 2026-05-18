# Exporting bayesflow-trained networks to ONNX

LANfactory's [`transform_bayesflow_to_onnx`](api/onnx.md) is the bayesflow
sibling of [`transform_sbi_to_onnx`](exporting_sbi_models.md). It wraps a
trained [`bayesflow`](https://github.com/bayesflow-org/bayesflow)
`ContinuousApproximator` (NLE) or `RatioApproximator` (NRE) and writes a
single-trial ONNX file that HSSM's `loglik_kind="approx_differentiable"`
path can consume exactly like an sbi export. Same user gesture, same file
format, same HSSM-side loader — regardless of which training framework you
came from.

## Installation

```bash
pip install lanfactory[bayesflow]
```

The `bayesflow` extra pulls `bayesflow>=2.0.8` and `keras>=3.12`. For both
libraries side-by-side use `pip install lanfactory[all]`.

## Critical: set the Keras backend before importing

`torch.onnx.export` cannot trace a JAX-backed Keras model. You **must** set
`KERAS_BACKEND=torch` *before* importing keras or bayesflow:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
# On Apple silicon, also pin to CPU — the orthogonal initializer needs
# torch.linalg.qr which MPS does not implement.
os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import bayesflow as bf   # now safe
```

The exporter checks this and raises a clear `RuntimeError` if the backend is
anything other than `torch` at export time.

## Quick start (NLE)

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import bayesflow as bf
import keras
from bayesflow.datasets import OfflineDataset
from bayesflow.networks.inference.coupling.transforms import AffineTransform
from lanfactory.onnx import transform_bayesflow_to_onnx

# 1. Build an ONNX-friendly ContinuousApproximator.
#    NLE convention: inference_variables=x (obs), inference_conditions=θ.
approximator = bf.ContinuousApproximator(
    inference_network=bf.networks.CouplingFlow(
        depth=4,
        subnet_kwargs={"widths": (64, 64), "activation": "silu", "dropout": None},
        permutation=None,                     # see Known constraints below
        use_actnorm=False,
        transform=AffineTransform(clamp=False),
    ),
    standardize="inference_variables",
)
approximator.build({
    "inference_variables": (None, x_dim),
    "inference_conditions": (None, theta_dim),
})
approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4))

# 2. Train on your simulator output.
#    `x` is observations, `theta` is parameters — numpy float32 arrays.
dataset = OfflineDataset(
    data={"inference_variables": x, "inference_conditions": theta},
    batch_size=200, adapter=None,
)
approximator.fit(dataset=dataset, epochs=30, verbose=0)

# 3. Export to a single ONNX file.
transform_bayesflow_to_onnx(
    approximator,
    "ddm_nle.onnx",
    mode="nle",
    example_theta_dim=theta_dim,
    example_x_dim=x_dim,
)

# 4. Hand it to HSSM exactly like an sbi or LAN file.
import hssm
model = hssm.HSSM(
    data=obs_data,
    model="ddm",
    loglik_kind="approx_differentiable",
    loglik="ddm_nle.onnx",
    p_outlier=0,
)
idata = model.sample(sampler="numpyro", draws=500, tune=500, chains=2)
```

## Quick start (NRE)

```python
approximator = bf.RatioApproximator(
    inference_network=bf.networks.MLP(
        widths=(64, 64),
        activation="silu",
        residual=False,
        dropout=None,
    ),
    standardize="inference_variables",
)
# NRE convention: inference_variables=θ, inference_conditions=x.
approximator.build({
    "inference_variables": (None, theta_dim),
    "inference_conditions": (None, x_dim),
})
# ... train as above with the OfflineDataset keys flipped ...

transform_bayesflow_to_onnx(
    approximator,
    "ddm_nre.onnx",
    mode="nre",
    example_theta_dim=theta_dim,
    example_x_dim=x_dim,
)
```

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

### 5. Enable JAX x64 in the consuming process

Same caveat as the sbi exporter — ONNX graphs from `torch.onnx.export` carry
int64 shape/index tensors. With JAX's default 32-bit mode, those get
silently truncated to int32, producing wrong log-prob outputs. Before
importing JAX in the consuming process:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

HSSM's `onnx2jax` consumer sets the related
`jaxort_only_allow_initializers_as_static_args = False` flag
automatically. The x64 setting is process-wide and must be opted into by
the caller.

## Explicitly out of scope (v1)

| Excluded | Reason |
|---|---|
| Discrete + continuous observations (MNLE-style) | bayesflow has no MNLE-equivalent approximator; would need new network types and training objectives. |
| Non-identity adapters | Numpy-only operations can't be baked into ONNX; see Constraint 4 above. Pointwise tensor adapter ops (log, sqrt, scale) are a candidate for v1.x. |
| Transformer / attention summary networks | Contain `LayerNormalization` (no jaxonnxruntime handler) and dynamic-shape attention. |
| FlowMatching, DiffusionModel, ConsistencyModel inference networks | `log_prob` requires ODE integration, not ONNX-exportable. |
| `KERAS_BACKEND=jax` workflows | Use the bayesflow LRE-style in-memory JAX callable path (see [`bayesflow_lre_integration.ipynb`](https://github.com/lnccbrown/HSSM/blob/main/docs/tutorials/bayesflow_lre_integration.ipynb) in HSSM). |

## Numerical guarantees

The bayesflow regression tests (`tests/test_bayesflow_*_export.py`) assert:

- Forward pass: torch reference wrapper, `onnxruntime`, and
  `jaxonnxruntime` all agree to `atol=1e-5` on fixed inputs.
- Gradients: `jax.grad` of the translated graph agrees with
  `torch.autograd.grad` on the wrapped network to `atol=1e-4`.

If you observe drift larger than these thresholds, please open an issue
with a minimal reproducer.

## Two paths into HSSM, side by side

| Path | Source library | Mechanism | When to use |
|---|---|---|---|
| `loglik="file.onnx"` | sbi or bayesflow | ONNX file, framework-agnostic | Portability, reproducibility, sharing trained surrogates |
| `loglik=<jax_callable>` | bayesflow (LRE tutorial) | In-memory JAX callable | Fast iteration during model development; bayesflow-only |

The two paths produce numerically equivalent results on the same trained
network. The ONNX path is what you'd ship; the JAX-callable path is what you'd
prototype with.

## Related API

- [`lanfactory.onnx.transform_bayesflow_to_onnx`](api/onnx.md) — this exporter.
- [`lanfactory.onnx.transform_sbi_to_onnx`](api/onnx.md) — the sbi sibling.
- [`lanfactory.onnx.transform_to_onnx`](api/onnx.md) — the LAN-MLP exporter.
