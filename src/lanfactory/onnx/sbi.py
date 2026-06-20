"""Export trained sbi estimators to ONNX for HSSM consumption.

The single public entry point is :func:`transform_sbi_to_onnx`, which wraps a
trained sbi density or ratio estimator and writes a single-trial ONNX graph
that HSSM's ``loglik_kind="approx_differentiable"`` path can load via
``jaxonnxruntime``.

This module is a sibling of :mod:`lanfactory.onnx.transform_onnx` (the LAN
exporter): "train a network and emit an ONNX HSSM can read" stays a single
conceptual home in LANfactory regardless of which library trained the network.

The exported graph follows the LAN-and-HSSM convention: a single concatenated
input of **rank 1, shape ``(theta_dim + x_dim,)``**. Inside the graph the
input is split into ``theta`` and ``x``, upranked to ``(1, …)`` to satisfy
sbi's batched ``log_prob`` API, and routed through the trained estimator.
HSSM vmaps this graph over trials, so the per-call input rank from HSSM is 1
— matching the export. Tracing with a 2D ``(1, D)`` dummy would emit ``Slice``
ops with ``axes=[1]`` that fail under HSSM's vmap (``IndexError: list
assignment index out of range`` inside ``jaxonnxruntime`` Slice handler).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

__all__ = ["transform_sbi_to_onnx"]


# Estimator class names that we cannot export. Score-based / flow-matching
# require ODE integration which is not ONNX-exportable; TabPFN has awkward
# in-context input shape; neural-spline-flow estimators are blocked on a
# missing SearchSorted op in jaxonnxruntime (tracked for v1.x upstream PR).
_UNSUPPORTED_ESTIMATORS: frozenset[str] = frozenset(
    {
        "ScoreEstimator",
        "ConditionalScoreEstimator",
        "FlowMatchingEstimator",
        "ConditionalFlowMatchingEstimator",
        "TabPFNEstimator",
    }
)


def transform_sbi_to_onnx(
    estimator: nn.Module,
    path: str,
    *,
    mode: Literal["nle", "nre"] = "nle",
    example_theta_dim: int,
    example_x_dim: int,
    opset: int = 17,
) -> None:
    """Export a trained sbi estimator to a single-trial ONNX graph.

    Parameters
    ----------
    estimator
        A trained sbi estimator. For ``mode="nle"`` this is a
        ``ConditionalDensityEstimator`` (as returned by ``NLE_A.train()``); for
        ``mode="nre"`` it is a ratio-estimator classifier (from ``NRE_A``/``B``/
        ``C``, ``BNRE``).
    path
        Filesystem path to write the ``.onnx`` artifact to.
    mode
        ``"nle"`` exports ``estimator.log_prob`` as the log-likelihood with the
        standardization Jacobian baked in. ``"nre"`` exports the classifier
        logit as the log-likelihood up to a θ-independent constant (lands in
        C4).
    example_theta_dim
        Parameter-vector dimensionality used to trace the graph.
    example_x_dim
        Observation-vector dimensionality used to trace the graph.
    opset
        ONNX opset version. Pinned to 17 by default for reproducibility against
        ``jaxonnxruntime``.

    Notes
    -----
    Only likelihood-shaped families are supported. NPE/posterior estimators are
    rejected by convention (the caller asserts ``mode="nle"`` only for true
    likelihood estimators). Score-based / flow-matching estimators (FMPE,
    NPSE), TabPFN-based estimators, and neural spline flows (blocked on
    missing ``SearchSorted`` in ``jaxonnxruntime``) are rejected with a clear
    error.
    """
    if example_theta_dim <= 0 or example_x_dim <= 0:
        raise ValueError(
            "example_theta_dim and example_x_dim must be positive, got "
            f"example_theta_dim={example_theta_dim}, "
            f"example_x_dim={example_x_dim}."
        )

    estimator_cls = type(estimator).__name__
    if estimator_cls in _UNSUPPORTED_ESTIMATORS:
        raise ValueError(
            f"transform_sbi_to_onnx does not support {estimator_cls}. "
            "Score-based, flow-matching, and TabPFN estimators are out of v1 "
            "scope; neural spline flows are blocked on a missing SearchSorted "
            "op in jaxonnxruntime (queued as a v1.x upstream PR). See the "
            "'Explicitly out of scope (v1)' section of "
            "docs/exporting_sbi_models.md for the full matrix."
        )

    if mode == "nle":
        if not hasattr(estimator, "log_prob"):
            raise TypeError(
                f"NLE mode requires an estimator with "
                f".log_prob(input, condition); got {estimator_cls} which lacks "
                f"it. If this is an NRE ratio classifier, use mode='nre' "
                f"instead."
            )
        wrapper: nn.Module = _NLELogProbWrapper(
            estimator, example_theta_dim, example_x_dim
        )
    elif mode == "nre":
        # NRE classifiers expose forward(theta, x) returning a logit; they do
        # NOT have .log_prob. If the user passes a density estimator with
        # mode="nre", surface the mismatch loudly — silently exporting
        # estimator.forward of an NLE flow would produce a graph that is not
        # a log-ratio.
        if hasattr(estimator, "log_prob"):
            raise TypeError(
                f"NRE mode expects a ratio classifier without .log_prob; "
                f"got {estimator_cls} which has .log_prob. If this is an NLE "
                f"density estimator, use mode='nle' instead."
            )
        wrapper = _NRELogRatioWrapper(estimator, example_theta_dim, example_x_dim)
    else:
        raise ValueError(f"mode must be 'nle' or 'nre', got {mode!r}")

    wrapper.eval()
    combined_input_dim = example_theta_dim + example_x_dim
    # Trace with a rank-1 dummy so the resulting Slice ops use axes=[0],
    # which survives HSSM's per-trial vmap (where the input arrives as 1D).
    dummy_input = torch.randn(combined_input_dim, requires_grad=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        path,
        dynamo=False,
        opset_version=opset,
    )


class _NLELogProbWrapper(nn.Module):
    """Wrap an NLE density estimator so forward(combined) returns log p(x|θ).

    The estimator's standardization stack is baked into the traced graph
    automatically — sbi's ``ConditionalDensityEstimator.log_prob`` already
    applies the z-score Jacobian correction internally on the torch side, so
    tracing the outer ``.log_prob`` call captures the full corrected
    likelihood.
    """

    def __init__(self, estimator: nn.Module, theta_dim: int, x_dim: int) -> None:
        super().__init__()
        self.estimator = estimator
        self.theta_dim = theta_dim
        self.x_dim = x_dim

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        # combined: 1D, shape (theta_dim + x_dim,) — matches HSSM's per-trial
        # vmap input. Split on axis 0 (rank-friendly), then unsqueeze for
        # sbi's batched log_prob contract, and reshape the (1, 1) output back
        # to a scalar so HSSM's downstream .squeeze() leaves it as ().
        theta = combined[: self.theta_dim].unsqueeze(0)
        x = combined[self.theta_dim :].unsqueeze(0)
        return self.estimator.log_prob(x, condition=theta).reshape(())


class _NRELogRatioWrapper(nn.Module):
    """Wrap an NRE ratio classifier so forward(combined) returns log r(x, θ).

    For NRE, the classifier logit IS the log-ratio log p(x, θ) / p(x) p(θ),
    which equals log p(x | θ) − log p(x). The θ-independent term log p(x)
    drops out under MCMC's accept ratios and under HSSM's posterior path, so
    we treat the raw logit as the exportable log-likelihood (up to a constant).
    No Jacobian correction is needed — the ratio is invariant to z-score
    standardization of inputs.
    """

    def __init__(self, estimator: nn.Module, theta_dim: int, x_dim: int) -> None:
        super().__init__()
        self.estimator = estimator
        self.theta_dim = theta_dim
        self.x_dim = x_dim

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        # combined: 1D, shape (theta_dim + x_dim,) — see _NLELogProbWrapper for
        # the rationale around rank-1 tracing and vmap compatibility.
        theta = combined[: self.theta_dim].unsqueeze(0)
        x = combined[self.theta_dim :].unsqueeze(0)
        return self.estimator(theta, x).reshape(())
