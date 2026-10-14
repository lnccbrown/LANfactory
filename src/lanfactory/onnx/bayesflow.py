"""Export trained bayesflow approximators to ONNX for HSSM consumption.

The single public entry point is :func:`transform_bayesflow_to_onnx`, which
wraps a trained :class:`bayesflow.ContinuousApproximator` (NLE) or
:class:`bayesflow.RatioApproximator` (NRE) and writes a single-trial ONNX
graph that HSSM's ``loglik_kind="approx_differentiable"`` path can load via
``jaxonnxruntime``.

This is the bayesflow sibling of :mod:`lanfactory.onnx.sbi` — same surface
shape, same single-trial rank-1 input convention, so HSSM consumes both
artifacts through the identical ``loglik="model.onnx"`` gesture.

Input/output contract (matches the sbi exporter exactly)
---------------------------------------------------------
The exported graph takes a **rank-1** tensor of shape ``(theta_dim + x_dim,)``
with parameters first, observations second, and returns a **rank-0 scalar**
log-likelihood. HSSM vmaps the graph over trials; tracing with rank-1 dummy
keeps the resulting ``Slice`` ops on ``axes=[0]`` so the vmap path works.

What gets baked into the trace
------------------------------
The wrapper bypasses :meth:`bayesflow.ContinuousApproximator.log_prob` (which
runs the numpy adapter and emits dynamic-shape ``Tile`` / ``Where`` /
``Size`` ops via the ``Standardize`` Keras layer's per-batch log-Jacobian)
and instead:

1. Pre-evaluates the standardizer's accumulated ``moving_mean`` / ``moving_std``
   to ``torch.nn.Module`` buffer constants.
2. Applies standardization inline as a static affine transform.
3. Calls ``approximator.inference_network.log_prob`` directly on torch tensors.
4. Adds the standardizer's constant Jacobian correction so the exported scalar
   is an absolute log-likelihood (matches :meth:`approximator.log_prob` modulo
   adapter ops, which v1 forbids).

Hard constraints for v1
-----------------------
* ``KERAS_BACKEND=torch`` at export time (``torch.onnx.export`` cannot trace a
  JAX-backed Keras model). Set ``KERAS_TORCH_DEVICE=cpu`` on Apple silicon to
  avoid MPS missing-op errors (``aten::linalg_qr.out``).
* The approximator's ``adapter`` must be the default identity ``Adapter()``
  (i.e. ``len(adapter.transforms) == 0``). Numpy-only adapter ops (log, sqrt,
  concat, drop) cannot be baked into ONNX. We raise if any are present.
* The ``CouplingFlow`` inference network must use:
    - ``permutation=None`` — ``FixedPermutation`` calls ``keras.ops.take`` which
      lowers to ``aten::ravel``, unsupported in ONNX opsets 17/20.
    - ``transform=AffineTransform(clamp=False)`` — bayesflow's
      ``find_transform("affine")`` silently drops ``transform_kwargs``
      (upstream bug), and the default ``clamp=True`` emits ``aten::asinh``
      which neither opset 17 nor 20 can export.
    - ``subnet_kwargs={"activation": ..., ...}`` — any smooth activation that
      decomposes to base ONNX ops works (``"silu"`` → ``Sigmoid`` + ``Mul``,
      ``"relu"`` → ``Relu``, ``"tanh"`` → ``Tanh``). The CouplingFlow default
      ``"hard_silu"`` (= HardSwish, a piecewise-linear approximation of SiLU)
      exports as the single ONNX op ``HardSwish`` (added in opset 14), which
      jaxonnxruntime does not yet implement. Workarounds: pick ``"silu"`` at
      training time, or upstream a ``HardSwish`` handler to jaxonnxruntime
      (decomposition: ``x * Clip(x + 3, 0, 6) / 6``).
    - ``use_actnorm=False`` (untested with ActNorm in v1).
* Continuous observations only. MNLE-style discrete + continuous mixes are
  out of v1 scope (mirrors sbi v1 scope).

These constraints are not all *enforced* by this function — some (network
internals) are documented in ``docs/exporting_bayesflow_models.md`` for the
user to satisfy at training time. We enforce what we can introspect cheaply.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch import nn

__all__ = ["transform_bayesflow_to_onnx"]


def transform_bayesflow_to_onnx(
    approximator: Any,
    path: str,
    *,
    mode: Literal["nle", "nre"] = "nle",
    example_theta_dim: int,
    example_x_dim: int,
    opset: int = 17,
) -> None:
    """Export a trained bayesflow approximator to a single-trial ONNX graph.

    Parameters
    ----------
    approximator
        Trained bayesflow approximator. For ``mode="nle"`` this is a
        :class:`bayesflow.ContinuousApproximator` whose ``inference_variables``
        slot was trained on the observation ``x`` (with
        ``inference_conditions`` holding the parameters ``θ``). For
        ``mode="nre"`` this is a :class:`bayesflow.RatioApproximator` trained
        with the opposite convention (``inference_variables=θ``,
        ``inference_conditions=x``).
    path
        Filesystem path to write the ``.onnx`` artifact to.
    mode
        ``"nle"`` exports ``log p(x|θ)`` with the standardizer Jacobian baked
        in. ``"nre"`` exports the classifier logit as the log-likelihood up to
        a θ-independent constant (which drops out in MCMC).
    example_theta_dim
        Parameter-vector dimensionality used to trace the graph.
    example_x_dim
        Observation-vector dimensionality used to trace the graph.
    opset
        ONNX opset version. Pinned to 17 by default for reproducibility against
        ``jaxonnxruntime``.

    Raises
    ------
    RuntimeError
        If ``KERAS_BACKEND`` is not ``"torch"``.
    TypeError
        If ``mode`` is ``"nle"`` and the approximator does not have an
        ``inference_network`` with ``.log_prob``; or if ``mode`` is ``"nre"``
        and the approximator does not have a ``projector``.
    ValueError
        If the approximator's ``adapter`` contains any non-trivial transforms,
        if ``example_theta_dim`` or ``example_x_dim`` is not positive, or if
        ``mode`` is not one of ``"nle"``/``"nre"``.
    """
    if example_theta_dim <= 0 or example_x_dim <= 0:
        raise ValueError(
            "example_theta_dim and example_x_dim must be positive, got "
            f"example_theta_dim={example_theta_dim}, "
            f"example_x_dim={example_x_dim}."
        )

    # The plan only ships KERAS_BACKEND=torch as a hard requirement; the
    # device note is advisory and only matters on Apple silicon.
    import keras  # local import: don't force keras at LANfactory import time

    if keras.backend.backend() != "torch":
        raise RuntimeError(
            "transform_bayesflow_to_onnx requires KERAS_BACKEND='torch' "
            "(got KERAS_BACKEND="
            f"'{keras.backend.backend()}'). torch.onnx.export cannot trace a "
            "JAX-backed Keras model. Set the environment variable BEFORE "
            "importing keras/bayesflow, e.g.:\n"
            "    import os\n"
            "    os.environ['KERAS_BACKEND'] = 'torch'\n"
            "    # on Apple silicon also: os.environ['KERAS_TORCH_DEVICE'] = 'cpu'\n"
            "    import bayesflow as bf"
        )

    _assert_identity_adapter(approximator)

    if mode == "nle":
        wrapper: nn.Module = _BayesflowNLELogProbWrapper(
            approximator, example_theta_dim, example_x_dim
        )
    elif mode == "nre":
        wrapper = _BayesflowNRELogRatioWrapper(
            approximator, example_theta_dim, example_x_dim
        )
    else:
        raise ValueError(f"mode must be 'nle' or 'nre', got {mode!r}")

    wrapper.eval()
    combined_input_dim = example_theta_dim + example_x_dim
    # Rank-1 dummy: see module docstring for why this matters (vmap survival).
    dummy_input = torch.randn(combined_input_dim, requires_grad=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        path,
        dynamo=False,
        opset_version=opset,
    )


def _assert_identity_adapter(approximator: Any) -> None:
    """Raise if the approximator's Adapter has non-trivial transforms.

    The plan's locked-in v1 contract: bake only the tensor-based
    ``standardize_layers`` (which we handle); error on any numpy-only adapter
    op since those cannot live in an ONNX graph.
    """
    adapter = getattr(approximator, "adapter", None)
    if adapter is None:
        return
    transforms = getattr(adapter, "transforms", None)
    if not transforms:
        return
    transform_names = [type(t).__name__ for t in transforms]
    raise ValueError(
        "transform_bayesflow_to_onnx requires an identity Adapter "
        "(approximator.adapter must have no transforms). Found "
        f"{len(transforms)} transform(s): {transform_names}. The numpy-based "
        "Adapter pipeline cannot be exported to ONNX. Apply the adapter's "
        "forward transform externally to your data before sampling, or retrain "
        "the approximator without an adapter."
    )


def _frozen_mean_std(
    standardizer: Any, key: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pre-evaluate a Standardize layer's moving mean/std to torch tensors.

    Going through ``Standardize.call`` at trace time emits ``Where`` / ``If``
    (from ``moving_std``'s ``where(m2>0, …, 1.0)``) and ``Shape`` / ``Size`` /
    ``Tile`` (from the per-batch log-Jacobian tile). jaxonnxruntime cannot
    execute ``Size``. By materializing mean and std now (training is done) we
    sidestep every dynamic-shape construct.
    """
    import keras

    if key not in standardizer.standardize_layers:
        return None, None
    layer = standardizer.standardize_layers[key]
    mean = keras.ops.convert_to_numpy(layer.moving_mean[0])
    std = keras.ops.convert_to_numpy(layer.moving_std(0))
    return (
        torch.as_tensor(mean, dtype=torch.float32),
        torch.as_tensor(std, dtype=torch.float32),
    )


class _BayesflowNLELogProbWrapper(nn.Module):
    """Wrap a bayesflow ``ContinuousApproximator`` so forward(combined) returns log p(x|θ).

    NLE convention assumed: ``inference_variables = x`` (observation),
    ``inference_conditions = θ``. The wrapper bakes the standardizer's
    accumulated ``moving_mean`` / ``moving_std`` as constants and adds the
    constant Jacobian correction so the exported scalar matches
    :meth:`approximator.log_prob` (modulo the adapter, which v1 forbids).
    """

    def __init__(self, approximator: Any, theta_dim: int, x_dim: int) -> None:
        super().__init__()
        if not hasattr(approximator, "inference_network"):
            raise TypeError(
                "NLE mode expects a bayesflow ContinuousApproximator with an "
                f".inference_network attribute; got {type(approximator).__name__}."
            )
        if not hasattr(approximator.inference_network, "log_prob"):
            raise TypeError(
                "NLE mode expects approximator.inference_network to expose "
                f".log_prob(samples, conditions=...); got "
                f"{type(approximator.inference_network).__name__}."
            )

        self.approximator = approximator
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.inference_network = approximator.inference_network

        x_mean, x_std = _frozen_mean_std(
            approximator.standardizer, "inference_variables"
        )
        th_mean, th_std = _frozen_mean_std(
            approximator.standardizer, "inference_conditions"
        )

        if x_mean is not None:
            self.register_buffer("_x_mean", x_mean)
            self.register_buffer("_x_std", x_std)
            # log|det J| of forward standardization x → (x − μ)/σ is −Σ log|σ|,
            # a constant that doesn't depend on the input. We bake it in so the
            # exported scalar is an absolute log p(x|θ).
            self._x_ldj = float(-np.sum(np.log(np.abs(x_std.numpy()))))
        else:
            self._x_mean = None
            self._x_std = None
            self._x_ldj = 0.0

        if th_mean is not None:
            self.register_buffer("_th_mean", th_mean)
            self.register_buffer("_th_std", th_std)
        else:
            self._th_mean = None
            self._th_std = None

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        # combined: rank-1, (theta_dim + x_dim,) — see module docstring.
        theta = combined[: self.theta_dim].unsqueeze(0)
        x_obs = combined[self.theta_dim : self.theta_dim + self.x_dim].unsqueeze(0)

        if self._x_mean is not None:
            x_obs = (x_obs - self._x_mean) / self._x_std
        if self._th_mean is not None:
            theta = (theta - self._th_mean) / self._th_std

        logp = self.inference_network.log_prob(x_obs, conditions=theta)
        return logp.reshape(()) + self._x_ldj


class _BayesflowNRELogRatioWrapper(nn.Module):
    """Wrap a bayesflow ``RatioApproximator`` so forward(combined) returns log r(x, θ).

    NRE convention assumed: ``inference_variables = θ``,
    ``inference_conditions = x``. We mirror the in-library ``logits`` method
    (concat θ and x, push through inference_network, then projector, squeeze),
    bypassing the contrastive-batch / ``log_ratio`` numpy gluing. The
    projector's scalar logit IS log p(x,θ)/(p(x)p(θ)) up to a θ-independent
    constant, which drops out of any MCMC accept ratio HSSM evaluates.
    """

    def __init__(self, approximator: Any, theta_dim: int, x_dim: int) -> None:
        super().__init__()
        if not hasattr(approximator, "projector"):
            raise TypeError(
                "NRE mode expects a bayesflow RatioApproximator with a "
                f".projector attribute; got {type(approximator).__name__}. "
                "If this is a ContinuousApproximator (NLE), use mode='nle'."
            )

        self.approximator = approximator
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.inference_network = approximator.inference_network
        self.projector = approximator.projector

        th_mean, th_std = _frozen_mean_std(
            approximator.standardizer, "inference_variables"
        )
        x_mean, x_std = _frozen_mean_std(
            approximator.standardizer, "inference_conditions"
        )

        if th_mean is not None:
            self.register_buffer("_th_mean", th_mean)
            self.register_buffer("_th_std", th_std)
        else:
            self._th_mean = None
            self._th_std = None

        if x_mean is not None:
            self.register_buffer("_x_mean", x_mean)
            self.register_buffer("_x_std", x_std)
        else:
            self._x_mean = None
            self._x_std = None

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        # combined: rank-1, [theta..., x...]. Internal ordering matches the
        # bayesflow logits() convention: classifier_input = concat([θ, x]).
        theta = combined[: self.theta_dim].unsqueeze(0)
        x_obs = combined[self.theta_dim : self.theta_dim + self.x_dim].unsqueeze(0)

        if self._th_mean is not None:
            theta = (theta - self._th_mean) / self._th_std
        if self._x_mean is not None:
            x_obs = (x_obs - self._x_mean) / self._x_std

        classifier_input = torch.cat([theta, x_obs], dim=-1)
        hidden = self.inference_network(classifier_input, training=False)
        logit = self.projector(hidden)
        # logit shape (1, 1) → scalar. No Jacobian correction: a ratio of
        # standardized densities equals the ratio of unstandardized densities.
        return logit.reshape(())
