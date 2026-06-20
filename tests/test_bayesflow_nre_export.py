"""Verify ``transform_bayesflow_to_onnx`` for the NRE (RatioApproximator) path.

NRE convention (opposite of NLE): inference_variables=θ, inference_conditions=x.
The classifier logit IS log p(x|θ) - log p(x); HSSM only cares about the
θ-dependent part, so the constant log p(x) is irrelevant. No Jacobian
correction is needed — ratios are invariant under z-score standardization.
"""

# KERAS_BACKEND must precede any keras / bayesflow import. See nle test for
# why; same reasoning here.
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")

from pathlib import Path  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jaxonnxruntime import call_onnx, config  # noqa: E402

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
from bayesflow.datasets import OfflineDataset  # noqa: E402

from lanfactory.onnx import transform_bayesflow_to_onnx  # noqa: E402

# bayesflow under KERAS_BACKEND=torch globally disables autograd at import to
# avoid excessive memory in long training loops. Restore the global default so
# that subsequent tests (e.g. test_sbi_*) that rely on autograd-by-default work.
torch.set_grad_enabled(True)

config.update("jaxort_only_allow_initializers_as_static_args", False)

_THETA_DIM = 2
_X_DIM = 2


def _build_nre_approximator() -> bf.RatioApproximator:
    """RatioApproximator with v1 ONNX-friendly knobs.

    The inference network is a small MLP classifier; we override to silu
    (default ``mish`` exports as a fused op missing in jaxonnxruntime) and
    disable residual/dropout for a clean inference-time trace.
    """
    return bf.RatioApproximator(
        inference_network=bf.networks.MLP(
            widths=(32, 32),
            activation="silu",
            residual=False,
            dropout=None,
        ),
        standardize="inference_variables",
        K=4,
    )


@pytest.fixture(scope="module")
def trained_nre() -> bf.RatioApproximator:
    """Train a tiny RatioApproximator on the same 2D Gaussian toy."""
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    n_train = 2000
    theta = rng.uniform(-3.0, 3.0, size=(n_train, _THETA_DIM)).astype(np.float32)
    x = (theta + 0.5 * rng.standard_normal(size=(n_train, _X_DIM))).astype(np.float32)

    approximator = _build_nre_approximator()
    approximator.build(
        {
            "inference_variables": (None, _THETA_DIM),  # NRE: θ here
            "inference_conditions": (None, _X_DIM),  # NRE: x here
        }
    )
    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    dataset = OfflineDataset(
        data={"inference_variables": theta, "inference_conditions": x},
        batch_size=128,
        adapter=None,
    )
    approximator.fit(dataset=dataset, epochs=30, verbose=0)
    return approximator


def _load_jax_runner(onnx_path: Path, combined: np.ndarray):
    onnx_model = onnx.load(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    model_func, model_weights = call_onnx.call_onnx_model(
        onnx_model, {input_name: combined}
    )
    run_func = jax.tree_util.Partial(model_func, model_weights)
    return run_func, input_name


def _wrapper_reference_log_ratio(
    approximator: bf.RatioApproximator,
    theta: torch.Tensor,
    x: torch.Tensor,
) -> float:
    from lanfactory.onnx.bayesflow import _BayesflowNRELogRatioWrapper

    wrapper = _BayesflowNRELogRatioWrapper(approximator, _THETA_DIM, _X_DIM)
    wrapper.eval()
    combined = torch.cat([theta.flatten(), x.flatten()], dim=-1)
    with torch.no_grad():
        return float(wrapper(combined).item())


@pytest.mark.flaky(reruns=2)
def test_nre_export_three_way_numerical_agreement(
    trained_nre: bf.RatioApproximator, tmp_path: Path
) -> None:
    onnx_path = tmp_path / "bayesflow_nre.onnx"
    transform_bayesflow_to_onnx(
        trained_nre,
        str(onnx_path),
        mode="nre",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    theta_t = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    x_t = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    combined = torch.cat([theta_t, x_t], dim=-1).squeeze(0).numpy()

    y_torch = _wrapper_reference_log_ratio(trained_nre, theta_t, x_t)

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    y_ort = float(np.asarray(sess.run(None, {input_name: combined})[0]).flatten()[0])

    run_func, jax_input_name = _load_jax_runner(onnx_path, combined)
    y_jax = float(np.asarray(run_func({jax_input_name: combined})[0]).flatten()[0])

    atol = 1e-5
    assert np.isclose(y_torch, y_ort, atol=atol), (
        f"torch wrapper vs onnxruntime: |Δ| = {abs(y_torch - y_ort)}"
    )
    assert np.isclose(y_torch, y_jax, atol=atol), (
        f"torch wrapper vs jaxonnxruntime: |Δ| = {abs(y_torch - y_jax)}"
    )
    assert np.isclose(y_ort, y_jax, atol=atol), (
        f"onnxruntime vs jaxonnxruntime: |Δ| = {abs(y_ort - y_jax)}"
    )


@pytest.mark.flaky(reruns=2)
def test_nre_export_gradient_agreement(
    trained_nre: bf.RatioApproximator, tmp_path: Path
) -> None:
    """jax.grad of the translated graph matches torch.autograd.grad of the wrapper."""
    onnx_path = tmp_path / "bayesflow_nre_grad.onnx"
    transform_bayesflow_to_onnx(
        trained_nre,
        str(onnx_path),
        mode="nre",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    from lanfactory.onnx.bayesflow import _BayesflowNRELogRatioWrapper

    wrapper = _BayesflowNRELogRatioWrapper(trained_nre, _THETA_DIM, _X_DIM)
    wrapper.eval()

    theta_init = torch.tensor([0.5, -0.2], dtype=torch.float32)
    x_init = torch.tensor([0.7, 0.3], dtype=torch.float32)
    combined_t = torch.cat([theta_init, x_init], dim=-1).clone().requires_grad_(True)
    with torch.enable_grad():
        y = wrapper(combined_t)
    (grad_torch,) = torch.autograd.grad(y, combined_t)
    grad_theta_torch = grad_torch[:_THETA_DIM].detach().numpy()

    combined_init = torch.cat([theta_init, x_init], dim=-1).numpy().astype(np.float32)
    run_func, input_name = _load_jax_runner(onnx_path, combined_init)

    def jax_logr_of_theta(theta_arr: jnp.ndarray) -> jnp.ndarray:
        combined = jnp.concatenate([theta_arr, jnp.asarray(x_init.numpy())], axis=-1)
        return run_func({input_name: combined})[0].sum()

    grad_theta_jax = np.asarray(
        jax.grad(jax_logr_of_theta)(jnp.asarray(theta_init.numpy()))
    )

    atol = 1e-4
    assert np.allclose(grad_theta_torch, grad_theta_jax, atol=atol), (
        f"torch vs jax theta-gradient mismatch: max |Δ| = "
        f"{np.abs(grad_theta_torch - grad_theta_jax).max()} "
        f"(torch={grad_theta_torch}, jax={grad_theta_jax})"
    )


def test_nre_log_ratio_ordering_makes_sense(
    trained_nre: bf.RatioApproximator,
) -> None:
    """Trained NRE: log r(x | θ_near) > log r(x | θ_far) when x is near the true θ.

    The classifier learns higher logits for compatible (θ, x) pairs.
    """
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    theta_near = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    theta_far = torch.tensor([[2.5, 2.5]], dtype=torch.float32)

    lr_near = _wrapper_reference_log_ratio(trained_nre, theta_near, x)
    lr_far = _wrapper_reference_log_ratio(trained_nre, theta_far, x)

    assert lr_near > lr_far, (
        f"trained log-ratio should be higher for compatible (θ, x): "
        f"lr_near={lr_near}, lr_far={lr_far}"
    )


def test_nre_approximator_in_nle_mode_rejected(
    trained_nre: bf.RatioApproximator, tmp_path: Path
) -> None:
    """RatioApproximator in mode='nle' should fail: inference_network has no .log_prob."""
    with pytest.raises(TypeError, match=r"\.log_prob"):
        transform_bayesflow_to_onnx(
            trained_nre,
            str(tmp_path / "should_not_exist.onnx"),
            mode="nle",
            example_theta_dim=_THETA_DIM,
            example_x_dim=_X_DIM,
        )


class _FakeStandardizeLayer:
    """Minimal stand-in for a keras Standardize layer (moving_mean / moving_std)."""

    def __init__(self, dim: int, std: float = 2.0) -> None:
        self.moving_mean = [np.zeros(dim, dtype=np.float32)]
        self._std = np.full(dim, std, dtype=np.float32)

    def moving_std(self, _index):
        return self._std


def test_nre_wrapper_standardizes_conditions_only() -> None:
    """Cover the NRE branch where x (conditions) is standardized but θ is not.

    The trained fixture standardizes only ``inference_variables`` (θ for NRE);
    this exercises the opposite combination with a stand-in approximator.
    """
    from lanfactory.onnx.bayesflow import _BayesflowNRELogRatioWrapper

    class _Network:
        def __call__(self, classifier_input, training=False):
            return classifier_input  # echo so the assertion can check x scaling

    class _Projector:
        def __call__(self, hidden):
            return hidden.sum().reshape(1, 1)

    class _Standardizer:
        standardize_layers = {"inference_conditions": _FakeStandardizeLayer(_X_DIM)}

    class _Approx:
        def __init__(self):
            self.inference_network = _Network()
            self.projector = _Projector()
            self.standardizer = _Standardizer()

    wrapper = _BayesflowNRELogRatioWrapper(_Approx(), _THETA_DIM, _X_DIM)
    wrapper.eval()

    # NRE: inference_variables=θ (not standardized here), inference_conditions=x.
    assert wrapper._th_mean is None
    assert wrapper._x_mean is not None

    theta = torch.tensor([2.0, 4.0])  # raw
    x = torch.tensor([2.0, 2.0])  # standardized by /2 → [1.0, 1.0]
    out = wrapper(torch.cat([theta, x]))

    assert out.ndim == 0
    # echo: θ.sum()=6 (raw) + x_std.sum()=2 → 8.
    assert torch.isclose(out, torch.tensor(8.0))
