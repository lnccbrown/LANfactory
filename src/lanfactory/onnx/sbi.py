"""Export trained sbi estimators to ONNX for HSSM consumption.

The single public entry point is :func:`transform_sbi_to_onnx`, which wraps a
trained sbi density or ratio estimator and writes a single-trial ONNX graph
that HSSM's ``loglik_kind="approx_differentiable"`` path can load via
``jaxonnxruntime``.

This module is intentionally a sibling of :mod:`lanfactory.onnx.transform_onnx`
— the LAN exporter — so that "train a network and emit an ONNX HSSM can read"
stays a single conceptual home in LANfactory regardless of which library
trained the network.

Implementation lands in C3 (NLE path) and C4 (NRE path). See
``plans/sbi-onnx-integration.md`` in HSSMSpine for the full plan.
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = ["transform_sbi_to_onnx"]


def transform_sbi_to_onnx(
    estimator: Any,
    path: str,
    *,
    mode: Literal["nle", "nre"] = "nle",
    example_theta_dim: int | None = None,
    example_x_dim: int | None = None,
    opset: int = 17,
) -> None:
    """Export a trained sbi estimator to a single-trial ONNX graph.

    Parameters
    ----------
    estimator
        A trained sbi estimator. For ``mode="nle"`` this is a
        ``ConditionalDensityEstimator`` (as returned by ``NLE_A.train()``); for
        ``mode="nre"`` it is a ``RatioEstimator`` (from ``NRE_A``/``B``/``C``,
        ``BNRE``).
    path
        Filesystem path to write the ``.onnx`` artifact to.
    mode
        ``"nle"`` exports ``estimator.log_prob`` as the log-likelihood with the
        standardization Jacobian baked in. ``"nre"`` exports the classifier
        logit as the log-likelihood up to a θ-independent constant.
    example_theta_dim
        Parameter-vector dimensionality used to trace the graph. Required.
    example_x_dim
        Observation-vector dimensionality used to trace the graph. Required.
    opset
        ONNX opset version. Pinned to 17 by default for reproducibility against
        ``jaxonnxruntime``.

    Notes
    -----
    Only likelihood-shaped families are supported. NPE/posterior estimators,
    score-based / flow-matching estimators (FMPE, NPSE), TabPFN-based
    estimators, and neural spline flows (blocked on a missing ``SearchSorted``
    op in ``jaxonnxruntime``) are rejected with a clear error at export time.
    """
    raise NotImplementedError(
        "transform_sbi_to_onnx is scaffolded but not yet implemented. "
        "The NLE path lands in commit C3; see plans/sbi-onnx-integration.md."
    )
