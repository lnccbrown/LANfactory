# sbi export: architectures and constraints

LANfactory's [`transform_sbi_to_onnx`](api/onnx.md) wraps a trained
[`sbi`](https://github.com/sbi-dev/sbi) estimator and writes a single-trial
ONNX file that satisfies the same single-trial artifact contract as a LAN
export. This page owns exporter architecture support, constraints, and
numerical guarantees. Follow the runnable
[sbi export tutorial](tutorials/exporting_sbi_to_onnx.ipynb) for installation,
training, export, and cross-backend verification. HSSM owns the downstream
[ONNX likelihood contract](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/)
and model-loading procedure.

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

Two constraints arose during validation and apply to anyone training their
own sbi estimators for export:

1. **For NLE with `density_estimator="maf"`, use ≥2D for both θ and x.** A 1D
   MAF in sbi collapses to a degenerate Gaussian path that emits zero-width Gemm
   contractions `jaxonnxruntime` cannot translate. This is a training-time
   limitation of sbi/nflows, **not** something `transform_sbi_to_onnx` enforces,
   and it is MAF-specific — NRE ratio classifiers export fine in 1D, and other
   density estimators (MDN, MoG) may not share it (untested at v1). Use 2D or
   higher for MAF NLE (this is the realistic case anyway).

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

## Numerical guarantees

The C2–C5 regression tests assert:

- Forward pass: torch reference, `onnxruntime`, and `jaxonnxruntime` all agree
  to `atol=1e-5` on fixed inputs.
- Gradients: `jax.grad` of the translated graph agrees with `torch.autograd.grad`
  on the original estimator to `atol=1e-4`.

If you run into precision issues smaller than these thresholds, please open
an issue with a minimal repro.

## Float precision

The exporter writes float32 graphs. Consumer-side dtype configuration belongs
to HSSM; follow its linked ONNX contract rather than copying a PyMC boundary
recipe from this exporter reference.

## Related API

- [`lanfactory.onnx.transform_sbi_to_onnx`](api/onnx.md) — the exporter.
- [`lanfactory.onnx.transform_to_onnx`](api/onnx.md) — the LAN-MLP exporter.
  Same family, different network source.
- [Export an sbi model to ONNX](tutorials/exporting_sbi_to_onnx.ipynb) — the
  executable task guide for this reference.
