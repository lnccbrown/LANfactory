# marimo source for the "Exporting an sbi model to ONNX for HSSM" tutorial.
#
# Regenerate the rendered docs notebook (with outputs) after editing:
#   uv run marimo export ipynb notebooks/exporting_sbi_to_onnx.py \
#     -o docs/tutorials/exporting_sbi_to_onnx.ipynb --include-outputs
#
# Edit interactively with:  uv run marimo edit notebooks/exporting_sbi_to_onnx.py

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Exporting an `sbi` model to ONNX for HSSM

    [`sbi`](https://github.com/sbi-dev/sbi) trains neural likelihood (NLE) and
    ratio (NRE) estimators. HSSM can use them as differentiable likelihoods —
    *if* they are exported to a single-trial ONNX graph. LANfactory's
    `transform_sbi_to_onnx` does exactly that, so the user gesture into HSSM is
    identical to a native LAN file:

    ```python
    hssm.HSSM(loglik="model.onnx", loglik_kind="approx_differentiable")
    ```

    This notebook runs the full **train → export → verify** loop end to end on
    a tiny toy, then points you at HSSM for the consumption side. For the
    supported-architecture matrix and constraints, see the reference guide
    [Exporting sbi Models](../exporting_sbi_models.md).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Setup

    One ordering rule matters: **enable JAX x64 before anything touches JAX
    dtypes.** ONNX graphs from `torch.onnx.export` carry `int64` shape/index
    tensors; JAX's default 32-bit mode silently truncates them inside
    `jaxonnxruntime`, producing wrong log-probs (~0.5 drift on a MAF).
    """)
    return


@app.cell
def _():
    import logging
    import warnings

    warnings.filterwarnings("ignore")  # keep the rendered tutorial output clean
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)  # silence TPU-probe log

    import jax

    jax.config.update("jax_enable_x64", True)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from jaxonnxruntime import call_onnx, config
    from sbi.inference import NLE_A
    from sbi.utils import BoxUniform

    from lanfactory.onnx import transform_sbi_to_onnx

    # torch.onnx.export emits Reshape shapes as Constant nodes. HSSM's onnx2jax
    # sets this for its consumers; standalone jaxonnxruntime use must set it too.
    config.update("jaxort_only_allow_initializers_as_static_args", False)

    THETA_DIM, X_DIM = 2, 2
    return (
        BoxUniform,
        NLE_A,
        THETA_DIM,
        X_DIM,
        call_onnx,
        jax,
        np,
        onnx,
        ort,
        torch,
        transform_sbi_to_onnx,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Train a tiny NLE

    A 2D Gaussian toy, `x | θ ~ N(θ, I)`, gives a closed-form likelihood to
    sanity-check against. The budget is deliberately small so the notebook
    runs in seconds — bump `n_train` / `max_num_epochs` for a real model.

    > **Why 2D?** A 1D MAF in `sbi` collapses to a degenerate Gaussian path
    > that emits a zero-width `Gemm` contraction `jaxonnxruntime` can't handle.
    > Keep θ and x at ≥2D for MAF NLE.
    """)
    return


@app.cell
def _(BoxUniform, NLE_A, THETA_DIM, torch):
    torch.manual_seed(0)

    prior = BoxUniform(
        low=torch.full((THETA_DIM,), -3.0),
        high=torch.full((THETA_DIM,), 3.0),
    )
    _inference = NLE_A(prior=prior, density_estimator="maf")
    _theta = prior.sample((2000,))
    _x = _theta + torch.randn_like(_theta)  # x | θ ~ N(θ, I)

    estimator = _inference.append_simulations(_theta, _x).train(
        training_batch_size=200,
        max_num_epochs=15,
    )
    estimator.eval()
    estimator
    return (estimator,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Export to ONNX

    `transform_sbi_to_onnx` wraps the trained estimator into a **rank-1**
    single-trial graph (parameters first, observations second; opset 17). The
    rank-1 contract is what lets HSSM `vmap` the graph over trials.
    """)
    return


@app.cell
def _(THETA_DIM, X_DIM, estimator, transform_sbi_to_onnx):
    import os
    import tempfile

    _onnx_dir = tempfile.mkdtemp(prefix="sbi_onnx_")
    onnx_path = os.path.join(_onnx_dir, "ddm_nle.onnx")

    transform_sbi_to_onnx(
        estimator,
        onnx_path,
        mode="nle",
        example_theta_dim=THETA_DIM,
        example_x_dim=X_DIM,
    )
    print("✓ exported ddm_nle.onnx")
    return (onnx_path,)


@app.cell
def _(THETA_DIM, X_DIM, call_onnx, jax, np, onnx, onnx_path, ort):
    # Load the exported graph into onnxruntime and the jax-translated runner once.
    _ort_session = ort.InferenceSession(onnx_path)
    _input_name = _ort_session.get_inputs()[0].name

    _onnx_model = onnx.load(onnx_path)
    _trace_input = np.zeros(THETA_DIM + X_DIM, dtype=np.float32)  # [θ.., x..]
    _model_func, _weights = call_onnx.call_onnx_model(
        _onnx_model, {_input_name: _trace_input}
    )
    jax_run = jax.tree_util.Partial(_model_func, _weights)

    def eval_backends(theta, x):
        """Return (ort, jax) scalar log-probs for a θ/x point (length-2 each)."""
        combined = np.asarray([*theta, *x], dtype=np.float32)
        y_ort = float(np.asarray(_ort_session.run(None, {_input_name: combined})[0]).flatten()[0])
        y_jax = float(np.asarray(jax_run({_input_name: combined})[0]).flatten()[0])
        return y_ort, y_jax

    return (eval_backends,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Verify the three backends agree

    The exported graph must compute the *same* log-likelihood whether run by
    the original torch estimator, `onnxruntime`, or the `jaxonnxruntime`
    translation HSSM uses. Move the sliders — the check re-runs reactively at
    the new (θ, x) point (no retraining).
    """)
    return


@app.cell
def _(mo):
    theta_ui = mo.ui.array(
        [mo.ui.slider(-3.0, 3.0, 0.1, value=v, label=f"θ[{i}]") for i, v in enumerate((0.5, -0.2))]
    )
    x_ui = mo.ui.array(
        [mo.ui.slider(-3.0, 3.0, 0.1, value=v, label=f"x[{i}]") for i, v in enumerate((0.7, 0.3))]
    )
    mo.hstack([mo.vstack(["**θ (parameters)**", *theta_ui]), mo.vstack(["**x (observation)**", *x_ui])])
    return theta_ui, x_ui


@app.cell
def _(estimator, eval_backends, mo, np, theta_ui, torch, x_ui):
    _theta = theta_ui.value
    _x = x_ui.value

    with torch.no_grad():
        _y_torch = float(
            estimator.log_prob(
                torch.tensor([_x], dtype=torch.float32),
                condition=torch.tensor([_theta], dtype=torch.float32),
            )
            .detach()
            .numpy()
            .flatten()[0]
        )
    _y_ort, _y_jax = eval_backends(_theta, _x)
    _max_delta = max(abs(_y_torch - _y_ort), abs(_y_torch - _y_jax), abs(_y_ort - _y_jax))

    mo.vstack(
        [
            mo.md(f"**log p(x | θ)** at θ={np.round(_theta, 2).tolist()}, x={np.round(_x, 2).tolist()}"),
            mo.md(
                f"| backend | log-prob |\n|---|---|\n"
                f"| torch (sbi) | `{_y_torch:.6f}` |\n"
                f"| onnxruntime | `{_y_ort:.6f}` |\n"
                f"| jaxonnxruntime | `{_y_jax:.6f}` |"
            ),
            mo.md(
                f"max pairwise |Δ| = `{_max_delta:.2e}` "
                + ("✅ agree (< 1e-4)" if _max_delta < 1e-4 else "⚠️ disagree")
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Consume it in HSSM

    The `.onnx` file drops into HSSM exactly like a LAN export — HSSM handles
    the `vmap` over trials and (recent versions) the x64 flag:

    ```python
    import jax
    jax.config.update("jax_enable_x64", True)  # if your HSSM version doesn't self-manage it

    import hssm
    model = hssm.HSSM(
        data=obs_data,                      # DataFrame with rt / response columns
        model="ddm",
        loglik_kind="approx_differentiable",
        loglik="ddm_nle.onnx",
        p_outlier=0,
    )
    idata = model.sample(sampler="numpyro", draws=500, tune=500, chains=2)
    ```

    For the consumption side end to end — defining the likelihood, building the
    model, sampling — see HSSM's
    [Build HSSM models starting from ONNX files](https://github.com/lnccbrown/HSSM/blob/main/docs/tutorials/blackbox_contribution_onnx_example.ipynb)
    tutorial. The **NRE** path is identical: train a `RatioApproximator`-style
    classifier and pass `mode="nre"` to `transform_sbi_to_onnx`.
    """)
    return


if __name__ == "__main__":
    app.run()
