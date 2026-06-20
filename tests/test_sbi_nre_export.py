"""C4 verification: train tiny sbi NRE_A on a Gaussian toy, export to ONNX,
and verify (i) three-way numerical agreement and (ii) gradient agreement
between torch and the jax-translated graph.

NRE classifier output IS the log-ratio log p(x | theta) / p(x), so up to a
theta-independent constant it serves as the log-likelihood. No Jacobian
correction is needed (ratio invariance under z-score standardization).
"""

from pathlib import Path

import jax

# x64 required before any JAX import — see test_sbi_nle_export.py for details.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jaxonnxruntime import call_onnx, config  # noqa: E402
from sbi.inference import NRE_A  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402

from lanfactory.onnx import transform_sbi_to_onnx  # noqa: E402

config.update("jaxort_only_allow_initializers_as_static_args", False)

_THETA_DIM = 2
_X_DIM = 2


def _gaussian_simulator(theta: torch.Tensor) -> torch.Tensor:
    """x | theta ~ N(theta, I)."""
    return theta + torch.randn_like(theta)


@pytest.fixture(scope="module")
def trained_nre() -> torch.nn.Module:
    """Train a tiny NRE_A on a 2D Gaussian. Small budget keeps CI fast."""
    torch.manual_seed(0)
    prior = BoxUniform(
        low=torch.tensor([-3.0, -3.0]),
        high=torch.tensor([3.0, 3.0]),
    )
    inference = NRE_A(prior=prior)
    theta = prior.sample((2000,))
    x = _gaussian_simulator(theta)
    classifier = inference.append_simulations(theta, x).train(
        training_batch_size=200,
        max_num_epochs=15,
    )
    classifier.eval()
    return classifier


def _load_jax_runner(onnx_path: Path, combined: np.ndarray):
    onnx_model = onnx.load(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    model_func, model_weights = call_onnx.call_onnx_model(
        onnx_model, {input_name: combined}
    )
    run_func = jax.tree_util.Partial(model_func, model_weights)
    return run_func, input_name


@pytest.mark.flaky(reruns=2)
def test_nre_export_three_way_numerical_agreement(
    trained_nre: torch.nn.Module, tmp_path: Path
) -> None:
    onnx_path = tmp_path / "nre.onnx"
    transform_sbi_to_onnx(
        trained_nre,
        str(onnx_path),
        mode="nre",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    theta_t = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    x_t = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    # Exported graph is rank-1; pass a 1D concatenated vector.
    combined = torch.cat([theta_t, x_t], dim=-1).squeeze(0).numpy()

    with torch.no_grad():
        y_torch = trained_nre(theta_t, x_t).detach().numpy()

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    y_ort = sess.run(None, {input_name: combined})[0]

    run_func, jax_input_name = _load_jax_runner(onnx_path, combined)
    y_jax = np.asarray(run_func({jax_input_name: combined})[0])

    atol = 1e-5
    y_torch_flat = y_torch.flatten()
    y_ort_flat = y_ort.flatten()
    y_jax_flat = y_jax.flatten()

    assert np.allclose(y_torch_flat, y_ort_flat, atol=atol), (
        f"torch vs onnxruntime: max |Δ| = {np.abs(y_torch_flat - y_ort_flat).max()}"
    )
    assert np.allclose(y_torch_flat, y_jax_flat, atol=atol), (
        f"torch vs jaxonnxruntime: max |Δ| = {np.abs(y_torch_flat - y_jax_flat).max()}"
    )
    assert np.allclose(y_ort_flat, y_jax_flat, atol=atol), (
        f"onnxruntime vs jaxonnxruntime: max |Δ| = "
        f"{np.abs(y_ort_flat - y_jax_flat).max()}"
    )


@pytest.mark.flaky(reruns=2)
def test_nre_export_gradient_agreement(
    trained_nre: torch.nn.Module, tmp_path: Path
) -> None:
    """jax.grad of the translated graph should match torch.autograd.grad."""
    onnx_path = tmp_path / "nre_grad.onnx"
    transform_sbi_to_onnx(
        trained_nre,
        str(onnx_path),
        mode="nre",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    theta_t = torch.tensor([[0.5, -0.2]], dtype=torch.float32, requires_grad=True)
    x_t = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

    logr = trained_nre(theta_t, x_t)
    (grad_torch,) = torch.autograd.grad(logr.sum(), theta_t)
    grad_torch_np = grad_torch.detach().numpy().flatten()

    # Rank-1 vectors throughout — matches the new exporter contract.
    theta_np_1d = theta_t.detach().numpy().squeeze(0)
    x_np_1d = x_t.numpy().squeeze(0)
    combined_init = np.concatenate([theta_np_1d, x_np_1d], axis=-1).astype(np.float32)
    run_func, input_name = _load_jax_runner(onnx_path, combined_init)

    def jax_logr_of_theta(theta_arr: jnp.ndarray) -> jnp.ndarray:
        combined = jnp.concatenate([theta_arr, jnp.asarray(x_np_1d)], axis=-1)
        return run_func({input_name: combined})[0].sum()

    grad_jax = jax.grad(jax_logr_of_theta)(jnp.asarray(theta_np_1d))
    grad_jax_np = np.asarray(grad_jax).flatten()

    atol = 1e-4
    assert np.allclose(grad_torch_np, grad_jax_np, atol=atol), (
        f"torch vs jax gradient mismatch: max |Δ| = "
        f"{np.abs(grad_torch_np - grad_jax_np).max()} "
        f"(torch={grad_torch_np}, jax={grad_jax_np})"
    )


def test_nre_log_ratio_ordering(trained_nre: torch.nn.Module) -> None:
    """Sanity: the log-ratio at theta=mean(x_obs) should exceed that at a
    distant theta. Not a precision test — just confirms training produced a
    surface where the ratio behaves reasonably.
    """
    x_obs = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    theta_near = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    theta_far = torch.tensor([[2.5, 2.5]], dtype=torch.float32)

    with torch.no_grad():
        logr_near = trained_nre(theta_near, x_obs).item()
        logr_far = trained_nre(theta_far, x_obs).item()

    assert logr_near > logr_far, (
        f"log-ratio should be higher near the true theta: "
        f"logr_near={logr_near}, logr_far={logr_far}"
    )
