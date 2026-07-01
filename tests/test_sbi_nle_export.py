"""C3 verification: train tiny sbi NLE_A with MAF on a Gaussian toy, export to
ONNX, and verify (i) three-way numerical agreement and (ii) gradient agreement
between torch and the jax-translated graph.

The Gaussian toy gives us a closed-form likelihood (N(theta, 1)) to sanity-check
that training did something reasonable, separate from the toolchain-equivalence
checks.
"""

from pathlib import Path

import jax

# Enable x64 BEFORE importing anything that may touch JAX dtypes. ONNX graphs
# from torch.onnx.export carry int64 shape/index tensors; JAX's default
# int32 silently truncates them inside jaxonnxruntime translation, producing
# wrong numerical values (~0.5 drift from the torch reference on MAF log_prob).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jaxonnxruntime import call_onnx, config  # noqa: E402
from sbi.inference import NLE_A  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402

from lanfactory.onnx import transform_sbi_to_onnx  # noqa: E402

# Same friction as C2's MAF spike — torch.onnx.export emits Reshape shapes as
# Constant nodes. HSSM's onnx2jax patch (commit 2e76516) sets this globally for
# HSSM consumers; tests here exercise jaxonnxruntime directly and must set it
# themselves.
config.update("jaxort_only_allow_initializers_as_static_args", False)

_THETA_DIM = 2
_X_DIM = 2


def _gaussian_simulator(theta: torch.Tensor) -> torch.Tensor:
    """x | theta ~ N(theta, I) — analytical likelihood available.

    The 2D shape is deliberate: a 1D MAF in sbi collapses to a degenerate
    Gaussian path that emits a zero-width Gemm contraction. jaxonnxruntime
    cannot handle it. 2D keeps the flow non-degenerate.
    """
    return theta + torch.randn_like(theta)


@pytest.fixture(scope="module")
def trained_nle() -> torch.nn.Module:
    """Train a tiny NLE_A on a 2D Gaussian. Small budget keeps CI fast."""
    torch.manual_seed(0)
    prior = BoxUniform(
        low=torch.tensor([-3.0, -3.0]),
        high=torch.tensor([3.0, 3.0]),
    )
    inference = NLE_A(prior=prior, density_estimator="maf")
    theta = prior.sample((2000,))
    x = _gaussian_simulator(theta)
    estimator = inference.append_simulations(theta, x).train(
        training_batch_size=200,
        max_num_epochs=15,
    )
    estimator.eval()
    return estimator


def _load_jax_runner(onnx_path: Path, combined: np.ndarray):
    onnx_model = onnx.load(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    model_func, model_weights = call_onnx.call_onnx_model(
        onnx_model, {input_name: combined}
    )
    run_func = jax.tree_util.Partial(model_func, model_weights)
    return run_func, input_name


@pytest.mark.flaky(reruns=2)
def test_nle_export_three_way_numerical_agreement(
    trained_nle: torch.nn.Module, tmp_path: Path
) -> None:
    onnx_path = tmp_path / "nle.onnx"
    transform_sbi_to_onnx(
        trained_nle,
        str(onnx_path),
        mode="nle",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    theta_t = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    x_t = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    # Exported graph is rank-1; pass a 1D concatenated vector.
    combined = torch.cat([theta_t, x_t], dim=-1).squeeze(0).numpy()

    with torch.no_grad():
        y_torch = trained_nle.log_prob(x_t, condition=theta_t).detach().numpy()

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
def test_nle_export_gradient_agreement(
    trained_nle: torch.nn.Module, tmp_path: Path
) -> None:
    """jax.grad of the translated graph should match torch.autograd.grad."""
    onnx_path = tmp_path / "nle_grad.onnx"
    transform_sbi_to_onnx(
        trained_nle,
        str(onnx_path),
        mode="nle",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    theta_t = torch.tensor([[0.5, -0.2]], dtype=torch.float32, requires_grad=True)
    x_t = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

    logp = trained_nle.log_prob(x_t, condition=theta_t)
    (grad_torch,) = torch.autograd.grad(logp.sum(), theta_t)
    grad_torch_np = grad_torch.detach().numpy().flatten()

    # Rank-1 vectors throughout — matches the new exporter contract.
    theta_np_1d = theta_t.detach().numpy().squeeze(0)
    x_np_1d = x_t.numpy().squeeze(0)
    combined_init = np.concatenate([theta_np_1d, x_np_1d], axis=-1).astype(np.float32)
    run_func, input_name = _load_jax_runner(onnx_path, combined_init)

    def jax_logp_of_theta(theta_arr: jnp.ndarray) -> jnp.ndarray:
        combined = jnp.concatenate([theta_arr, jnp.asarray(x_np_1d)], axis=-1)
        return run_func({input_name: combined})[0].sum()

    grad_jax = jax.grad(jax_logp_of_theta)(jnp.asarray(theta_np_1d))
    grad_jax_np = np.asarray(grad_jax).flatten()

    atol = 1e-4
    assert np.allclose(grad_torch_np, grad_jax_np, atol=atol), (
        f"torch vs jax gradient mismatch: max |Δ| = "
        f"{np.abs(grad_torch_np - grad_jax_np).max()} "
        f"(torch={grad_torch_np}, jax={grad_jax_np})"
    )


def test_nle_log_prob_ordering_matches_analytical_gaussian(
    trained_nle: torch.nn.Module,
) -> None:
    """Sanity: trained N(theta, 1) should rank a near-mean point above a far one.

    Not a precision test — just confirms that training produced a reasonable
    likelihood, not a random surface.
    """
    theta = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    x_near = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    x_far = torch.tensor([[2.5, 2.5]], dtype=torch.float32)

    with torch.no_grad():
        lp_near = trained_nle.log_prob(x_near, condition=theta).item()
        lp_far = trained_nle.log_prob(x_far, condition=theta).item()

    assert lp_near > lp_far, (
        f"trained log-prob should be higher near the mean: "
        f"lp_near={lp_near}, lp_far={lp_far}"
    )


def test_transform_rejects_unsupported_score_estimator(tmp_path: Path) -> None:
    """Estimators in the unsupported set should fail loudly."""

    class ScoreEstimator(torch.nn.Module):  # noqa: D401 - name is the signal
        pass

    with pytest.raises(ValueError, match="does not support"):
        transform_sbi_to_onnx(
            ScoreEstimator(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="nle",
            example_theta_dim=1,
            example_x_dim=1,
        )


def test_transform_rejects_missing_log_prob(tmp_path: Path) -> None:
    """NLE mode without .log_prob should raise a clear TypeError."""

    class NotADensityEstimator(torch.nn.Module):
        pass

    with pytest.raises(TypeError, match=r"\.log_prob\(input, condition\)"):
        transform_sbi_to_onnx(
            NotADensityEstimator(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="nle",
            example_theta_dim=1,
            example_x_dim=1,
        )


def test_nle_estimator_in_nre_mode_rejected(
    trained_nle: torch.nn.Module, tmp_path: Path
) -> None:
    """Passing an NLE density estimator with mode='nre' should raise.

    The presence of .log_prob is the signal that this is a density estimator
    rather than a ratio classifier.
    """
    with pytest.raises(TypeError, match=r"expects a ratio classifier"):
        transform_sbi_to_onnx(
            trained_nle,
            str(tmp_path / "should_not_exist.onnx"),
            mode="nre",
            example_theta_dim=_THETA_DIM,
            example_x_dim=_X_DIM,
        )


def test_transform_rejects_invalid_mode(tmp_path: Path) -> None:
    """An unrecognized mode should raise a clear ValueError, not export."""

    class DummyEstimator(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="mode must be 'nle' or 'nre'"):
        transform_sbi_to_onnx(
            DummyEstimator(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="bogus",  # type: ignore[arg-type]
            example_theta_dim=1,
            example_x_dim=1,
        )


def test_transform_rejects_nonpositive_dims(tmp_path: Path) -> None:
    """Zero or negative example dims should raise a clear ValueError."""

    class DummyEstimator(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="must be positive"):
        transform_sbi_to_onnx(
            DummyEstimator(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="nle",
            example_theta_dim=0,
            example_x_dim=2,
        )


def _max_int64_abs(onnx_model: onnx.ModelProto) -> int:
    """Largest absolute value stored in any int64 tensor in the graph (0 if none)."""
    tensors = list(onnx_model.graph.initializer)
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attr.tensors)
    biggest = 0
    for tensor in tensors:
        if tensor.data_type == onnx.TensorProto.INT64:
            arr = onnx.numpy_helper.to_array(tensor)
            if arr.size:
                biggest = max(biggest, int(np.abs(arr).max()))
    return biggest


def test_export_int64_values_fit_in_int32(
    trained_nle: torch.nn.Module, tmp_path: Path
) -> None:
    """Regression: bounded input slices keep all int64 graph values within int32.

    An open-ended ``combined[theta_dim:]`` slice bakes an ``INT64_MAX`` ``Slice``
    'ends' sentinel into the graph; under ``jax_enable_x64=False`` that int64
    truncates and ``INT64_MAX`` wraps to ``-1`` — silently turning "slice to end"
    into "drop the last element". The NLE and NRE wrappers share the identical
    bounded-slice code, so this guards both modes against a regression.
    """
    onnx_path = tmp_path / "sbi_nle_int_range.onnx"
    transform_sbi_to_onnx(
        trained_nle,
        str(onnx_path),
        mode="nle",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )
    model = onnx.load(str(onnx_path))
    assert _max_int64_abs(model) <= np.iinfo(np.int32).max
