# marimo source for the "Exporting a bayesflow model to ONNX for HSSM" tutorial.
#
# Regenerate the rendered docs notebook (with outputs) after editing:
#   uv run marimo export ipynb notebooks/exporting_bayesflow_to_onnx.py \
#     -o docs/tutorials/exporting_bayesflow_to_onnx.ipynb --include-outputs
#
# Edit interactively with:  uv run marimo edit notebooks/exporting_bayesflow_to_onnx.py

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
    # Exporting a `bayesflow` model to ONNX for HSSM

    [`bayesflow`](https://github.com/bayesflow-org/bayesflow) trains amortized
    neural likelihood (`ContinuousApproximator`) and ratio (`RatioApproximator`)
    estimators. LANfactory's `transform_bayesflow_to_onnx` exports them to the
    same single-trial ONNX graph HSSM consumes from any source:

    ```python
    hssm.HSSM(loglik="model.onnx", loglik_kind="approx_differentiable")
    ```

    This is the bayesflow sibling of the [sbi tutorial](exporting_sbi_to_onnx.ipynb):
    same **train → export → verify** loop, with bayesflow's Keras-backed specifics.
    For the supported-architecture matrix and constraints, see the reference guide
    [Exporting bayesflow Models](../exporting_bayesflow_models.md).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Setup

    Two ordering rules matter here:

    1. **`KERAS_BACKEND=torch` must be set before `keras`/`bayesflow` are imported.**
       `torch.onnx.export` cannot trace a JAX-backed Keras model. On Apple silicon,
       `KERAS_TORCH_DEVICE=cpu` avoids an MPS missing-op error in the orthogonal
       initializer. Both are set in the import cell below.
    2. **Enable JAX x64 before anything touches JAX dtypes** (ONNX `int64` indices
       are silently truncated under JAX's default 32-bit mode).
    """)
    return


@app.cell
def _():
    import os

    # MUST precede any keras / bayesflow import.
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")

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

    import bayesflow as bf
    import keras
    from bayesflow.datasets import OfflineDataset
    from bayesflow.networks.inference.coupling.transforms import AffineTransform

    from lanfactory.onnx import transform_bayesflow_to_onnx

    # bayesflow disables autograd at import under the torch backend; restore it.
    torch.set_grad_enabled(True)
    # torch.onnx.export emits Reshape shapes as Constant nodes; standalone
    # jaxonnxruntime use must opt in (HSSM's onnx2jax does this for its consumers).
    config.update("jaxort_only_allow_initializers_as_static_args", False)

    THETA_DIM, X_DIM = 2, 2
    return (
        AffineTransform,
        OfflineDataset,
        THETA_DIM,
        X_DIM,
        bf,
        call_onnx,
        jax,
        keras,
        np,
        onnx,
        ort,
        torch,
        transform_bayesflow_to_onnx,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Train a tiny NLE

    A 2D Gaussian toy, `x | θ ~ N(θ, 0.5²)`. The `CouplingFlow` knobs below are the
    v1 ONNX-exportable settings (see the guide for *why* each is required):
    `permutation=None`, `transform=AffineTransform(clamp=False)`, a smooth
    `activation="silu"`, and `use_actnorm=False`. Budget is tiny for speed.
    """)
    return


@app.cell
def _(AffineTransform, OfflineDataset, THETA_DIM, X_DIM, bf, keras, np):
    keras.utils.set_random_seed(0)
    _rng = np.random.default_rng(0)

    _n_train = 2000
    _theta = _rng.uniform(-3.0, 3.0, size=(_n_train, THETA_DIM)).astype(np.float32)
    _x = (_theta + 0.5 * _rng.standard_normal(size=(_n_train, X_DIM))).astype(np.float32)

    approximator = bf.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(
            depth=4,
            subnet_kwargs={"widths": (32, 32), "activation": "silu", "dropout": None},
            permutation=None,
            use_actnorm=False,
            transform=AffineTransform(clamp=False),
        ),
        standardize="inference_variables",  # standardize x (the observation)
    )
    approximator.build(
        {
            "inference_variables": (None, X_DIM),
            "inference_conditions": (None, THETA_DIM),
        }
    )
    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    _dataset = OfflineDataset(
        data={"inference_variables": _x, "inference_conditions": _theta},
        batch_size=128,
        adapter=None,
    )
    approximator.fit(dataset=_dataset, epochs=20, verbose=0)
    print("trained ContinuousApproximator")
    return (approximator,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Export to ONNX

    `transform_bayesflow_to_onnx` bakes the standardizer's accumulated mean/std
    into the graph as constants (sidestepping the dynamic-shape ops the live Keras
    layer would emit) and writes a **rank-1** single-trial graph, opset 17 — the
    same contract as the sbi and LAN exporters, so HSSM consumes it identically.
    """)
    return


@app.cell
def _(THETA_DIM, X_DIM, approximator, transform_bayesflow_to_onnx):
    import tempfile

    _onnx_dir = tempfile.mkdtemp(prefix="bf_onnx_")
    onnx_path = f"{_onnx_dir}/ddm_bayesflow_nle.onnx"

    transform_bayesflow_to_onnx(
        approximator,
        onnx_path,
        mode="nle",
        example_theta_dim=THETA_DIM,
        example_x_dim=X_DIM,
    )
    print("✓ exported ddm_bayesflow_nle.onnx")
    return (onnx_path,)


@app.cell
def _(
    THETA_DIM,
    X_DIM,
    approximator,
    call_onnx,
    jax,
    np,
    onnx,
    onnx_path,
    ort,
    torch,
):
    # Load the exported graph once into onnxruntime + the jax-translated runner,
    # and build the in-memory wrapper for the torch reference. (We bypass
    # approximator.log_prob, which runs the numpy adapter and re-standardizes.)
    # Import the wrapper here (it is _-prefixed → cell-local in marimo, so it
    # cannot be shared from the setup cell).
    from lanfactory.onnx.bayesflow import _BayesflowNLELogProbWrapper

    _ort_session = ort.InferenceSession(onnx_path)
    _input_name = _ort_session.get_inputs()[0].name

    _onnx_model = onnx.load(onnx_path)
    _trace_input = np.zeros(THETA_DIM + X_DIM, dtype=np.float32)
    _model_func, _weights = call_onnx.call_onnx_model(
        _onnx_model, {_input_name: _trace_input}
    )
    _jax_run = jax.tree_util.Partial(_model_func, _weights)

    _wrapper = _BayesflowNLELogProbWrapper(approximator, THETA_DIM, X_DIM)
    _wrapper.eval()

    def eval_all(theta, x):
        """torch-wrapper / onnxruntime / jaxonnxruntime log-probs at a (θ, x) point."""
        combined = np.asarray([*theta, *x], dtype=np.float32)
        with torch.no_grad():
            y_torch = float(_wrapper(torch.tensor(combined)).item())
        y_ort = float(np.asarray(_ort_session.run(None, {_input_name: combined})[0]).flatten()[0])
        y_jax = float(np.asarray(_jax_run({_input_name: combined})[0]).flatten()[0])
        return y_torch, y_ort, y_jax

    return (eval_all,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Verify the three backends agree

    The exported graph must compute the same log-likelihood whether run by the
    in-memory torch wrapper, `onnxruntime`, or the `jaxonnxruntime` translation
    HSSM uses. Move the sliders — the check re-runs reactively (no retraining).
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
def _(eval_all, mo, np, theta_ui, x_ui):
    _theta = theta_ui.value
    _x = x_ui.value
    _y_torch, _y_ort, _y_jax = eval_all(_theta, _x)
    _max_delta = max(abs(_y_torch - _y_ort), abs(_y_torch - _y_jax), abs(_y_ort - _y_jax))

    mo.vstack(
        [
            mo.md(f"**log p(x | θ)** at θ={np.round(_theta, 2).tolist()}, x={np.round(_x, 2).tolist()}"),
            mo.md(
                f"| backend | log-prob |\n|---|---|\n"
                f"| torch (wrapper) | `{_y_torch:.6f}` |\n"
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

    The `.onnx` file drops into HSSM exactly like a LAN or sbi export:

    ```python
    import jax
    jax.config.update("jax_enable_x64", True)  # if your HSSM version doesn't self-manage it

    import hssm
    model = hssm.HSSM(
        data=obs_data,
        model="ddm",
        loglik_kind="approx_differentiable",
        loglik="ddm_bayesflow_nle.onnx",
        p_outlier=0,
    )
    idata = model.sample(sampler="numpyro", draws=500, tune=500, chains=2)
    ```

    For the consumption side end to end, see HSSM's
    [Build HSSM models starting from ONNX files](https://github.com/lnccbrown/HSSM/blob/main/docs/tutorials/blackbox_contribution_onnx_example.ipynb)
    tutorial. The **NRE** path is analogous: train a `RatioApproximator` and pass
    `mode="nre"` to `transform_bayesflow_to_onnx`.
    """)
    return


if __name__ == "__main__":
    app.run()
