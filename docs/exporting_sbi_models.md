# Exporting sbi-trained networks to ONNX

LANfactory's [`transform_sbi_to_onnx`](api/onnx.md) wraps a trained
[`sbi`](https://github.com/sbi-dev/sbi) estimator and writes a single-trial
ONNX file that HSSM's `loglik_kind="approx_differentiable"` path can consume
exactly like a LAN export. Use it to bring sbi-trained NLE density estimators
or NRE ratio classifiers into a [HSSM](https://github.com/lnccbrown/HSSM) model.

## Installation

```bash
pip install lanfactory[all]
```

The `all` extra pulls `sbi>=0.26` and `nflows>=0.14` in addition to LANfactory's
other optional integrations.

## Quick start (NLE)

```python
import torch
from sbi.inference import NLE_A
from sbi.utils import BoxUniform
from lanfactory.onnx import transform_sbi_to_onnx

# 1. Train a likelihood estimator (your simulator + prior here).
prior = BoxUniform(low=torch.tensor([-3.0, -3.0]), high=torch.tensor([3.0, 3.0]))
inference = NLE_A(prior=prior, density_estimator="maf")
theta = prior.sample((5_000,))
x = my_simulator(theta)                       # shape: (5000, x_dim)
estimator = inference.append_simulations(theta, x).train()

# 2. Export to a HSSM-compatible ONNX file.
transform_sbi_to_onnx(
    estimator,
    "ddm_nle.onnx",
    mode="nle",
    example_theta_dim=theta.shape[-1],
    example_x_dim=x.shape[-1],
)

# 3. Hand it to HSSM exactly like a LAN file.
import hssm
model = hssm.HSSM(
    data=obs_data,
    model="ddm",
    model_config=my_model_config,
    loglik_kind="approx_differentiable",
    loglik="ddm_nle.onnx",
    p_outlier=0,
)
idata = model.sample(sampler="numpyro", draws=500, tune=500, chains=2)
```

## Quick start (NRE)

```python
from sbi.inference import NRE_A
inference = NRE_A(prior=prior)
classifier = inference.append_simulations(theta, x).train()
transform_sbi_to_onnx(
    classifier,
    "ddm_nre.onnx",
    mode="nre",
    example_theta_dim=theta.shape[-1],
    example_x_dim=x.shape[-1],
)
```

The classifier logit is `log p(x, θ) / p(x) p(θ) = log p(x | θ) − log p(x)`. The
θ-independent `log p(x)` term drops out under MCMC and under HSSM's posterior
path, so the raw logit is consumed as the log-likelihood (up to a constant). No
Jacobian correction is needed — ratios are invariant to z-score
standardization.

## Supported architectures (v1)

| Method | Density / classifier | Embedding nets | Status |
|--------|---------------------|----------------|--------|
| **NLE_A** | MAF | none, FC on θ | ✅ supported |
| **NLE_A** | MDN, MoG | none, FC on θ | ✅ supported (untested at v1, expected to work) |
| **NRE_A / B / C / BNRE** | MLP classifier (with `norm_layer=nn.Identity`) | none, FCEmbedding, CNNEmbedding | ✅ supported |

## Explicitly out of scope (v1)

| Excluded | Reason |
|----------|--------|
| Neural Spline Flows (NSF coupling, NSF autoregressive) | `jaxonnxruntime` is missing the `SearchSorted` op. Targeted for a future upstream PR. |
| FMPE (flow-matching), NPSE (score-based) | `log_prob` requires ODE integration; not ONNX-exportable. |
| NPE / SNPE | Posterior-shaped, not likelihood-shaped. The HSSM ecosystem's current scope is neural likelihood surrogates. |
| TabPFN / NPE-PFN | Transformer with in-context inputs; awkward shape handling. Deferred. |

The exporter rejects estimators whose class name is in the unsupported set with a
clear `ValueError`. If you encounter an unsupported architecture, please open an issue.

## Known constraints

Three constraints arose during validation and apply to anyone training their
own sbi estimators for export:

1. **Use ≥2D for both θ and x.** sbi's `density_estimator="maf"` collapses to
   a degenerate Gaussian path in 1D that emits zero-width Gemm contractions
   `jaxonnxruntime` cannot translate. Use 2D or higher (this is the realistic
   case anyway).

2. **Disable LayerNorm in NRE MLP classifiers.** `jaxonnxruntime` does not
   implement the `LayerNormalization` op. When using `classifier_nn(model="mlp", ...)`,
   pass `norm_layer=nn.Identity` to skip it:

   ```python
   from torch import nn
   from sbi.neural_nets import classifier_nn

   classifier_builder = classifier_nn(
       model="mlp",
       embedding_net_x=my_embedding,
       norm_layer=nn.Identity,    # <-- required for ONNX export
   )
   ```

3. **Enable JAX x64 before importing JAX in the consuming process.** ONNX
   graphs from `torch.onnx.export` carry int64 shape/index tensors. With JAX's
   default 32-bit mode, those get silently truncated to int32, producing
   ~0.5-unit drift in log-prob outputs. Set:

   ```python
   import jax
   jax.config.update("jax_enable_x64", True)
   # ...subsequent imports of jaxonnxruntime, hssm, etc.
   ```

   HSSM's `onnx2jax` consumer sets the related `jaxort_only_allow_initializers_as_static_args = False`
   flag automatically, but the x64 setting is process-wide and must be opted
   into by the caller.

## Numerical guarantees

The C2–C5 regression tests assert:

- Forward pass: torch reference, `onnxruntime`, and `jaxonnxruntime` all agree
  to `atol=1e-5` on fixed inputs.
- Gradients: `jax.grad` of the translated graph agrees with `torch.autograd.grad`
  on the original estimator to `atol=1e-4`.

If you run into precision issues smaller than these thresholds, please open
an issue with a minimal repro.

## Float precision

ONNX exports default to float32. PyMC defaults to float64. When sampling, either:

- Cast at the JAX boundary, or
- Set `pytensor.config.floatX = "float32"` for the whole model.

HSSM handles this consistently in its `approx_differentiable` path; if you're
hand-rolling a model with `pm.CustomDist` you'll need to do this yourself.

## Related API

- [`lanfactory.onnx.transform_sbi_to_onnx`](api/onnx.md) — the exporter.
- [`lanfactory.onnx.transform_to_onnx`](api/onnx.md) — the LAN-MLP exporter.
  Same family, different network source.
