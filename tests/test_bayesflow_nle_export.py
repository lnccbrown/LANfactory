"""Verify ``transform_bayesflow_to_onnx`` for the NLE (ContinuousApproximator)
path: tiny train, export, three-way numerical agreement, gradient agreement.

Mirrors ``tests/test_sbi_nle_export.py`` so the bayesflow exporter inherits the
same coverage shape as the sbi exporter. The fixture bakes in the v1
constraints (permutation=None, AffineTransform(clamp=False), silu activation,
no actnorm) — see lanfactory.onnx.bayesflow module docstring for the full list.
"""

# KERAS_BACKEND must be torch and set BEFORE importing keras / bayesflow;
# torch.onnx.export cannot trace a JAX-backed Keras model. Force it (not
# setdefault) so a stray KERAS_BACKEND in the environment can't silently run
# these tests on jax. KERAS_TORCH_DEVICE=cpu (a default) avoids the
# Apple-silicon MPS missing-op error in the orthogonal initializer (qr).
import os

os.environ["KERAS_BACKEND"] = "torch"
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


def test_transform_rejects_invalid_mode(tmp_path: Path) -> None:
    """An unrecognized mode raises ValueError before building a wrapper."""

    class _FakeApprox:
        adapter = None

    with pytest.raises(ValueError, match="mode must be 'nle' or 'nre'"):
        transform_bayesflow_to_onnx(
            _FakeApprox(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="bogus",  # type: ignore[arg-type]
            example_theta_dim=_THETA_DIM,
            example_x_dim=_X_DIM,
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


@pytest.mark.xfail(
    strict=True,
    reason="bayesflow's CouplingFlow emits internal INT64_MAX Constants (from its "
    "open-ended split ops) that LANfactory cannot remove — so bayesflow ONNX is "
    "NOT int32-clean even with bounded wrapper slices. Such exports require x64 on "
    "the consumer side (or a value-aware int cast). This xfail is a tripwire: if "
    "bayesflow ever stops emitting INT64_MAX, it will xpass and flag that the "
    "consumer can drop its x64 requirement. Contrast test_sbi_nle_export, which "
    "IS int32-clean after the bounded-slice fix.",
)
def test_export_int64_values_fit_in_int32(
    trained_nle: bf.ContinuousApproximator, tmp_path: Path
) -> None:
    """Document: bayesflow exports are NOT int32-clean (flow-internal INT64_MAX).

    The bounded-slice wrapper fix removes the wrapper-level ``INT64_MAX`` sentinel,
    but bayesflow's CouplingFlow split ops emit their own — so this assertion fails
    (xfail). The sbi exporter test of the same name passes because nflows/MAF does
    not emit such sentinels.
    """
    onnx_path = tmp_path / "bayesflow_nle_int_range.onnx"
    transform_bayesflow_to_onnx(
        trained_nle,
        str(onnx_path),
        mode="nle",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )
    model = onnx.load(str(onnx_path))
    assert _max_int64_abs(model) <= np.iinfo(np.int32).max


def test_transform_rejects_nonpositive_dims(tmp_path: Path) -> None:
    """Zero or negative example dims raise a clear ValueError."""

    class _FakeApprox:
        adapter = None

    with pytest.raises(ValueError, match="must be positive"):
        transform_bayesflow_to_onnx(
            _FakeApprox(),
            str(tmp_path / "should_not_exist.onnx"),
            mode="nle",
            example_theta_dim=0,
            example_x_dim=_X_DIM,
        )


def test_nle_wrapper_rejects_missing_inference_network() -> None:
    """The NLE wrapper requires an approximator exposing .inference_network."""
    from lanfactory.onnx.bayesflow import _BayesflowNLELogProbWrapper

    class _NoNetwork:
        pass

    with pytest.raises(TypeError, match="inference_network"):
        _BayesflowNLELogProbWrapper(_NoNetwork(), _THETA_DIM, _X_DIM)


class _FakeStandardizeLayer:
    """Minimal stand-in for a keras Standardize layer (moving_mean / moving_std).

    Lets the standardization branches be exercised without training a real
    approximator (which always standardizes only one slot in the fixtures).
    """

    def __init__(self, dim: int, std: float = 2.0) -> None:
        self.moving_mean = [np.zeros(dim, dtype=np.float32)]
        self._std = np.full(dim, std, dtype=np.float32)

    def moving_std(self, _index):
        return self._std


def test_nle_wrapper_standardizes_conditions_only() -> None:
    """Cover the NLE branch where θ (conditions) is standardized but x is not.

    The trained fixture standardizes only ``inference_variables`` (x); this
    exercises the opposite combination with a lightweight stand-in approximator.
    """
    from lanfactory.onnx.bayesflow import _BayesflowNLELogProbWrapper

    class _Network:
        def log_prob(self, samples, conditions):
            # Echo (possibly standardized) inputs so the assertion can confirm
            # standardization was applied to θ but not to x.
            return (samples.sum() + conditions.sum()).reshape(1)

    class _Standardizer:
        standardize_layers = {"inference_conditions": _FakeStandardizeLayer(_THETA_DIM)}

    class _Approx:
        adapter = None

        def __init__(self):
            self.inference_network = _Network()
            self.standardizer = _Standardizer()

    wrapper = _BayesflowNLELogProbWrapper(_Approx(), _THETA_DIM, _X_DIM)
    wrapper.eval()

    # θ standardized → buffer registered; x not → None.
    assert wrapper._th_mean is not None
    assert wrapper._x_mean is None

    theta = torch.tensor([2.0, 4.0])  # standardized by /2 → [1.0, 2.0]
    x = torch.tensor([1.0, 1.0])  # left as-is
    out = wrapper(torch.cat([theta, x]))

    assert out.ndim == 0
    # echo: x.sum()=2 (raw) + θ_std.sum()=3 → 5; x carries no Jacobian term.
    assert torch.isclose(out, torch.tensor(5.0))
