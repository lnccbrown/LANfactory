"""Verify ``transform_bayesflow_to_onnx`` for the NLE (ContinuousApproximator)
path: tiny train, export, three-way numerical agreement, gradient agreement.

Mirrors ``tests/test_sbi_nle_export.py`` so the bayesflow exporter inherits the
same coverage shape as the sbi exporter. The fixture bakes in the v1
constraints (permutation=None, AffineTransform(clamp=False), silu activation,
no actnorm) — see lanfactory.onnx.bayesflow module docstring for the full list.
"""

# KERAS_BACKEND must be set BEFORE importing keras / bayesflow; torch.onnx.export
# cannot trace a JAX-backed Keras model. KERAS_TORCH_DEVICE=cpu avoids the
# Apple-silicon MPS missing-op error in the orthogonal initializer (qr).
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")

from pathlib import Path  # noqa: E402

import jax  # noqa: E402

# Same reason as the sbi test: ONNX shape/index tensors are int64; JAX's default
# int32 silently truncates them inside jaxonnxruntime translation.
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
from bayesflow.networks.inference.coupling.transforms import AffineTransform  # noqa: E402

from lanfactory.onnx import transform_bayesflow_to_onnx  # noqa: E402

# bayesflow under KERAS_BACKEND=torch globally disables autograd at import to
# avoid excessive memory in long training loops. Restore the global default so
# that subsequent tests (e.g. test_sbi_*) that rely on autograd-by-default work.
# Local code that needs gradients should always use ``with torch.enable_grad():``
# explicitly — this re-enable is purely for cross-test hygiene.
torch.set_grad_enabled(True)

config.update("jaxort_only_allow_initializers_as_static_args", False)

_THETA_DIM = 2
_X_DIM = 2


def _gaussian_xs_for_theta(rng: np.random.Generator, theta: np.ndarray) -> np.ndarray:
    """x | theta ~ N(theta, 0.5^2 I) — same toy distribution as the sbi test."""
    return (theta + 0.5 * rng.standard_normal(size=theta.shape)).astype(np.float32)


def _build_nle_approximator() -> bf.ContinuousApproximator:
    """Construct a ContinuousApproximator with the v1 ONNX-friendly knobs."""
    return bf.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(
            depth=4,
            # silu decomposes to Sigmoid + Mul under ONNX; default hard_silu
            # emits a single HardSwish op that jaxonnxruntime can't run.
            subnet_kwargs={
                "widths": (32, 32),
                "activation": "silu",
                "dropout": None,
            },
            # FixedPermutation uses keras.ops.take → aten::ravel, unsupported
            # in ONNX opsets 17/20.
            permutation=None,
            use_actnorm=False,
            # bayesflow's find_transform("affine") silently drops kwargs, so
            # transform_kwargs={"clamp": False} would not take effect. Pass an
            # explicit instance. clamp=False disables ops.arcsinh which
            # neither opset 17 nor 20 supports.
            transform=AffineTransform(clamp=False),
        ),
        standardize="inference_variables",  # standardize x (obs)
    )


@pytest.fixture(scope="module")
def trained_nle() -> bf.ContinuousApproximator:
    """Train the v1-friendly CouplingFlow NLE on the 2D Gaussian toy."""
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)

    n_train = 2000
    theta = rng.uniform(-3.0, 3.0, size=(n_train, _THETA_DIM)).astype(np.float32)
    x = _gaussian_xs_for_theta(rng, theta)

    approximator = _build_nle_approximator()
    approximator.build(
        {
            "inference_variables": (None, _X_DIM),
            "inference_conditions": (None, _THETA_DIM),
        }
    )
    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    dataset = OfflineDataset(
        data={"inference_variables": x, "inference_conditions": theta},
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


def _wrapper_reference_log_prob(
    approximator: bf.ContinuousApproximator,
    theta: torch.Tensor,
    x: torch.Tensor,
) -> float:
    """Compute the same scalar the exported graph emits, using the in-memory
    wrapper. Avoids depending on approximator.log_prob (which goes through the
    numpy adapter path and re-runs the standardizer)."""
    from lanfactory.onnx.bayesflow import _BayesflowNLELogProbWrapper

    wrapper = _BayesflowNLELogProbWrapper(approximator, _THETA_DIM, _X_DIM)
    wrapper.eval()
    combined = torch.cat([theta.flatten(), x.flatten()], dim=-1)
    with torch.no_grad():
        return float(wrapper(combined).item())


@pytest.mark.flaky(reruns=2)
def test_nle_export_three_way_numerical_agreement(
    trained_nle: bf.ContinuousApproximator, tmp_path: Path
) -> None:
    onnx_path = tmp_path / "bayesflow_nle.onnx"
    transform_bayesflow_to_onnx(
        trained_nle,
        str(onnx_path),
        mode="nle",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    theta_t = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    x_t = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    combined = torch.cat([theta_t, x_t], dim=-1).squeeze(0).numpy()

    y_torch = _wrapper_reference_log_prob(trained_nle, theta_t, x_t)

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
def test_nle_export_gradient_agreement(
    trained_nle: bf.ContinuousApproximator, tmp_path: Path
) -> None:
    """jax.grad of the translated graph matches torch.autograd.grad of the wrapper."""
    onnx_path = tmp_path / "bayesflow_nle_grad.onnx"
    transform_bayesflow_to_onnx(
        trained_nle,
        str(onnx_path),
        mode="nle",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )

    from lanfactory.onnx.bayesflow import _BayesflowNLELogProbWrapper

    wrapper = _BayesflowNLELogProbWrapper(trained_nle, _THETA_DIM, _X_DIM)
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

    def jax_logp_of_theta(theta_arr: jnp.ndarray) -> jnp.ndarray:
        combined = jnp.concatenate([theta_arr, jnp.asarray(x_init.numpy())], axis=-1)
        return run_func({input_name: combined})[0].sum()

    grad_theta_jax = np.asarray(
        jax.grad(jax_logp_of_theta)(jnp.asarray(theta_init.numpy()))
    )

    atol = 1e-4
    assert np.allclose(grad_theta_torch, grad_theta_jax, atol=atol), (
        f"torch vs jax theta-gradient mismatch: max |Δ| = "
        f"{np.abs(grad_theta_torch - grad_theta_jax).max()} "
        f"(torch={grad_theta_torch}, jax={grad_theta_jax})"
    )


def test_nle_log_prob_ordering_makes_sense(
    trained_nle: bf.ContinuousApproximator,
) -> None:
    """Sanity: trained N(theta, 0.5) should rank a near-mean point above a far one.

    Not a precision test — just confirms training produced a sensible likelihood
    surface rather than a random one.
    """
    theta = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    x_near = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    x_far = torch.tensor([[2.5, 2.5]], dtype=torch.float32)

    lp_near = _wrapper_reference_log_prob(trained_nle, theta, x_near)
    lp_far = _wrapper_reference_log_prob(trained_nle, theta, x_far)

    assert lp_near > lp_far, (
        f"trained log-prob should be higher near the mean: "
        f"lp_near={lp_near}, lp_far={lp_far}"
    )


def test_transform_rejects_wrong_backend(monkeypatch, tmp_path: Path) -> None:
    """If KERAS_BACKEND != 'torch' at export time, raise a clear error."""

    class _FakeApprox:
        adapter = None

    monkeypatch.setattr(keras.backend, "backend", lambda: "jax")
    with pytest.raises(RuntimeError, match="KERAS_BACKEND='torch'"):
        transform_bayesflow_to_onnx(
            _FakeApprox(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="nle",
            example_theta_dim=1,
            example_x_dim=1,
        )


def test_transform_rejects_non_identity_adapter(tmp_path: Path) -> None:
    """Adapter with any transform → ValueError with the offending name listed."""
    from bayesflow.adapters import Adapter

    class _FakeApprox:
        # .log("foo") attaches a MapTransform; the specific op doesn't matter
        # for the test — we just need a non-identity adapter.
        adapter = Adapter().log("foo")

    with pytest.raises(ValueError, match="identity Adapter"):
        transform_bayesflow_to_onnx(
            _FakeApprox(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="nle",
            example_theta_dim=1,
            example_x_dim=1,
        )


def test_nle_approximator_in_nre_mode_rejected(
    trained_nle: bf.ContinuousApproximator, tmp_path: Path
) -> None:
    """ContinuousApproximator passed with mode='nre' has no .projector → raise."""
    with pytest.raises(TypeError, match=r"\.projector"):
        transform_bayesflow_to_onnx(
            trained_nle,
            str(tmp_path / "should_not_exist.onnx"),
            mode="nre",
            example_theta_dim=_THETA_DIM,
            example_x_dim=_X_DIM,
        )
