"""Integrate a trained LAN's density over reaction time.

A LAN approximates ``log p(rt, choice | theta)``. Integrating the density over
``rt`` gives each choice's probability and the probability of responding
before a deadline: the labels of the auxiliary networks (CPNs, OPNs), here
obtained from the LAN instead of from a fresh simulation.

Grid policy
-----------
The density is integrated with the trapezoid rule over the LANs' training
support ``[t_min, max_t]`` (``max_t = 20`` s, ssms' default) and is never
evaluated past it. It rises from zero within milliseconds of the
non-decision time ``t``, so the one grid, :class:`OnsetGrid`, is refined per
parameter vector around ``t``: on the Hub ddm LAN its 1160 points match a
16 000-point uniform grid to ``5e-5``, where a uniform 1000-point grid is off
by 15–25 % at ``a < 0.5``. The grid starts at ``t_min``, not at ``t``: the
mass a LAN leaks below ``t`` is part of what it predicts. A model without an
onset parameter passes ``onset = 0``.

Tail policy
-----------
Masses are **not** renormalised: they keep the LAN's own scale error and
whatever density it puts past ``max_t``. :attr:`ChoiceMass.total` exposes the
deficit; renormalising is the corpus's job.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = [
    "ChoiceMass",
    "OnnxPredictor",
    "OnsetGrid",
    "Predictor",
    "choice_mass",
    "load_onnx_predictor",
    "survey",
]

#: Parameter vectors per network call: bounds the row buffer of
#: :func:`choice_mass` and the cdf :func:`survey` holds at once.
_CHUNK = 512


class Predictor(Protocol):
    """Maps ``(N, input_width)`` float32 rows to ``(N,)`` log-densities."""

    input_width: int

    def __call__(self, batch: NDArray[np.float32]) -> NDArray[np.floating]: ...


class OnnxPredictor:
    """A ``(1, D)`` ONNX network callable on ``(N, D)`` batches.

    Build with :func:`load_onnx_predictor`.

    Attributes
    ----------
    input_name : str
        Name of the graph input fed by :meth:`__call__`.
    input_width : int
        ``D``, the row width the graph expects (parameters + rt + choice for
        a LAN).
    """

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session: ort.InferenceSession = session
        first_input = session.get_inputs()[0]
        self.input_name: str = first_input.name
        self.input_width: int = int(first_input.shape[-1])

    def __call__(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """Evaluate the network on ``(N, D)`` rows (cast to float32).

        Returns
        -------
        NDArray[np.float32]
            ``(N,)`` network outputs.
        """
        rows = np.ascontiguousarray(batch, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.input_width:
            raise ValueError(
                f"expected a (N, {self.input_width}) batch, got shape {rows.shape}"
            )
        (out,) = self.session.run(None, {self.input_name: rows})
        return np.asarray(out, dtype=np.float32).reshape(rows.shape[0])


def load_onnx_predictor(path: str | Path) -> OnnxPredictor:
    """Load a ``(1, D)`` ONNX network as a batched :class:`OnnxPredictor`.

    The ecosystem's ONNX graphs are exported with a concrete ``(1, D)`` input
    (see ``lanfactory.onnx.contract``), which onnxruntime enforces. The graph's
    batch dimension (dim 0 of every graph input and output) is set to the
    symbolic ``"N"`` on the in-memory protobuf, and the session is built from
    that copy. The file on disk is read, never written, so the single-trial
    contract the artifact was published under is untouched.

    Parameters
    ----------
    path
        Path to the ONNX artifact.

    Returns
    -------
    OnnxPredictor
        Callable on ``(N, D)`` float32 batches.

    Raises
    ------
    ValueError
        If the graph's first input is not rank 2. The sbi and bayesflow
        exporters trace rank-1 ``(D,)`` graphs; those have no batch axis to
        widen, and this loader does not wrap them.
    """
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path))
    io_tensors = [*model.graph.input, *model.graph.output]
    first_dims = io_tensors[0].type.tensor_type.shape.dim
    if len(first_dims) != 2:
        shape = [d.dim_param or d.dim_value for d in first_dims]
        raise ValueError(
            f"{path}: expected a (1, D) input, got rank {len(first_dims)} shape "
            f"{shape}; lanfactory.derive batches (1, D) Gemm graphs only"
        )
    for value_info in io_tensors:
        dims = value_info.type.tensor_type.shape.dim
        if len(dims) >= 2:
            # Assigning dim_param clears dim_value (they share a protobuf oneof).
            dims[0].dim_param = "N"
    # jax2onnx pins intermediate shapes at batch 1; let onnxruntime re-infer
    # them rather than warn (VerifyOutputSizes) on every batched call.
    del model.graph.value_info[:]
    session = ort.InferenceSession(model.SerializeToString())
    return OnnxPredictor(session)


def _segment(
    start: NDArray[np.float64],
    stop: NDArray[np.float64],
    n: int,
    *,
    endpoint: bool,
) -> NDArray[np.float64]:
    """Row-wise ``linspace(start[i], stop[i], n)``; ``(n_rows, n)``."""
    u = np.linspace(0.0, 1.0, n, endpoint=endpoint)
    rows = start[:, None] + (stop - start)[:, None] * u[None, :]
    if endpoint:
        rows[:, -1] = stop  # exact, not ``start + (stop - start)``
    return rows


@dataclass(frozen=True)
class OnsetGrid:
    """Reaction-time grid refined, per parameter vector, around the onset.

    Each row concatenates four ``linspace`` segments around its own onset
    ``t``: ``[t_min, knee)``, ``[knee, t)``, ``[t, t + onset_window)`` and
    ``[t + onset_window, max_t]``. The points concentrate where the density
    moves while the whole support is still integrated. Every row has
    :attr:`n_points` points, is strictly increasing, and contains ``t_min``,
    ``t`` and ``max_t`` exactly.

    Parameters
    ----------
    n_pre, n_knee, n_onset, n_tail
        Points per segment, in that order; only the tail includes its
        endpoint.
    knee
        Width in seconds of the segment just below ``t``, where the LAN blurs
        the onset; half of ``[t_min, t]`` instead when that is shorter.
    onset_window
        Width in seconds of the segment just above ``t``.
    max_t
        Upper edge in seconds: the LANs' training support.
    t_min
        Lower edge in seconds, positive so no network is fed ``rt = 0``.

    Notes
    -----
    Onsets are clipped to ``[t_min + 1e-3, max_t - onset_window - 1e-3]`` so
    that every segment has positive length.
    """

    n_pre: int = 32
    n_knee: int = 128
    knee: float = 0.05
    n_onset: int = 600
    onset_window: float = 1.0
    n_tail: int = 400
    max_t: float = 20.0
    t_min: float = 1e-4

    @property
    def n_points(self) -> int:
        """Points per row: ``n_pre + n_knee + n_onset + n_tail``."""
        return self.n_pre + self.n_knee + self.n_onset + self.n_tail

    def for_theta(self, onset: ArrayLike) -> NDArray[np.float64]:
        """Build one grid row per onset time.

        Parameters
        ----------
        onset
            ``(n_theta,)`` onset times in seconds.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta, n_points)`` strictly increasing rows.
        """
        t = np.asarray(onset, dtype=np.float64)
        if t.ndim != 1:
            raise ValueError(f"onset must be (n_theta,), got shape {t.shape}")
        t = np.clip(t, self.t_min + 1e-3, self.max_t - self.onset_window - 1e-3)
        t_min = np.full_like(t, self.t_min)
        # max(), not a threshold at t_min + knee: the pre segment's width stays
        # continuous in t, so its points never crowd into a sliver that the
        # float32 rows the network sees would collapse.
        knee = np.maximum(t - self.knee, 0.5 * (t_min + t))
        window_end = t + self.onset_window
        return np.concatenate(
            [
                _segment(t_min, knee, self.n_pre, endpoint=False),
                _segment(knee, t, self.n_knee, endpoint=False),
                _segment(t, window_end, self.n_onset, endpoint=False),
                _segment(
                    window_end, np.full_like(t, self.max_t), self.n_tail, endpoint=True
                ),
            ],
            axis=1,
        )


def _interp_rows(
    x: ArrayLike, xp: NDArray[np.float64], fp: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Row-wise ``np.interp(x[i], xp[i], fp[i])`` for non-decreasing ``xp`` rows.

    ``x`` is a scalar or ``(n_rows,)``; outside a row's range the result is
    clamped to that row's first or last ``fp``.
    """
    n_rows, n_points = xp.shape
    x = np.broadcast_to(np.asarray(x, dtype=np.float64), (n_rows,))
    hi = np.clip((xp < x[:, None]).sum(axis=1), 1, n_points - 1)
    rows = np.arange(n_rows)
    x_lo, x_hi = xp[rows, hi - 1], xp[rows, hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.clip((x - x_lo) / (x_hi - x_lo), 0.0, 1.0)
    f_lo, f_hi = fp[rows, hi - 1], fp[rows, hi]
    return f_lo + w * (f_hi - f_lo)


@dataclass(frozen=True)
class ChoiceMass:
    """A LAN's density integrated over reaction time, per parameter vector.

    Attributes
    ----------
    t : NDArray[np.float64]
        ``(n_theta, n_points)`` grid, one :class:`OnsetGrid` row per
        parameter vector.
    cdf : NDArray[np.float64]
        ``(n_theta, n_points)`` cumulative mass along ``t``, summed over
        choices; ``cdf[:, 0] == 0``.
    by_choice : NDArray[np.float64]
        ``(n_theta, n_choices)`` each choice's mass on ``[t_min, max_t]``, in
        the order of the ``choices`` given to :func:`choice_mass`.

    Notes
    -----
    Masses are not renormalised; see the module's tail policy.
    """

    t: NDArray[np.float64]
    cdf: NDArray[np.float64]
    by_choice: NDArray[np.float64]

    @property
    def total(self) -> NDArray[np.float64]:
        """``(n_theta,)`` mass on ``[t_min, max_t]``, summed over choices."""
        return self.cdf[:, -1].copy()  # a view would keep the whole cdf alive

    def mass_before(self, deadline: ArrayLike) -> NDArray[np.float64]:
        """Mass accumulated before ``deadline``, linearly interpolated on the grid.

        Parameters
        ----------
        deadline
            Scalar or ``(n_theta,)`` deadlines in seconds. Values below
            ``t_min`` give 0 and values above ``max_t`` give :attr:`total`.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta,)``.
        """
        return _interp_rows(deadline, self.t, self.cdf)

    def quantile(self, u: ArrayLike) -> NDArray[np.float64]:
        """Time at which the mass reaches ``u`` of :attr:`total`.

        The inverse of :meth:`mass_before`, the same interpolation with the
        axes swapped: ``mass_before(quantile(u))`` is ``u * total``.

        Parameters
        ----------
        u
            Fraction in ``(0, 1)``; scalar or ``(n_theta,)``.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta,)`` times in seconds; NaN where there is no mass to
            invert (a zero or non-finite total).
        """
        u_arr = np.asarray(u, dtype=np.float64)
        if np.any((u_arr <= 0.0) | (u_arr >= 1.0)):
            raise ValueError(f"u must lie in the open interval (0, 1), got {u!r}")
        total = self.total
        q = _interp_rows(u_arr * total, self.cdf, self.t)
        return np.where(total > 0.0, q, np.nan)


def choice_mass(
    predictor: Predictor,
    theta: ArrayLike,
    choices: Sequence[float] | NDArray,
    onset: ArrayLike,
    grid: OnsetGrid = OnsetGrid(),  # noqa: B008 - frozen, so never mutated
) -> ChoiceMass:
    """Integrate a LAN's density over reaction time for every choice.

    Rows are laid out as ``[theta..., rt, choice]``, the input layout the LANs
    are trained on, and evaluated 512 parameter vectors at a time, which
    bounds the row buffer handed to the network.

    Parameters
    ----------
    predictor
        Maps ``(N, n_params + 2)`` rows to ``(N,)`` log-densities; typically
        an :class:`OnnxPredictor`.
    theta
        ``(n_theta, n_params)`` parameter vectors (a single vector is
        promoted).
    choices
        The choice codes the LAN was trained on, e.g. ``[-1, 1]``.
    onset
        Scalar or ``(n_theta,)`` onset (non-decision) times the grid is built
        around, e.g. ``theta[:, t_idx]``; 0 for a model without one.
    grid
        Reaction-time grid (see the module's grid policy).

    Returns
    -------
    ChoiceMass

    Raises
    ------
    ValueError
        If ``predictor.input_width != n_params + 2``, the usual symptom of
        pairing a parameter set with the wrong LAN, or if ``onset`` is not
        finite or does not broadcast to ``(n_theta,)``.
    """
    theta_arr = np.atleast_2d(np.asarray(theta, dtype=np.float32))
    if theta_arr.ndim != 2:
        raise ValueError(f"theta must be (n_theta, n_params), got {theta_arr.shape}")
    n_theta, n_params = theta_arr.shape
    width = n_params + 2
    if predictor.input_width != width:
        raise ValueError(
            f"predictor expects rows of width {predictor.input_width}, but theta "
            f"has {n_params} parameters so rows [theta..., rt, choice] are "
            f"{width} wide; is this the right LAN for these parameters?"
        )
    choice_arr = np.asarray(choices)
    n_choices = choice_arr.shape[0]
    onset_arr = np.broadcast_to(np.asarray(onset, dtype=np.float64), (n_theta,))
    if not np.all(np.isfinite(onset_arr)):
        raise ValueError("onset must be finite; pass 0 for a model without one")
    t = grid.for_theta(onset_arr)
    n_points = t.shape[1]

    by_choice = np.empty((n_theta, n_choices))
    cdf = np.empty((n_theta, n_points))
    for start in range(0, n_theta, _CHUNK):
        stop = min(start + _CHUNK, n_theta)
        t_chunk = t[start:stop, None, :]
        rows = np.empty((stop - start, n_choices, n_points, width), dtype=np.float32)
        rows[..., :n_params] = theta_arr[start:stop, None, None, :]
        # The network sees rt rounded to float32 (~1e-6 s at max_t) while the
        # trapezoid uses the float64 grid: far below the rule's own error.
        rows[..., n_params] = t_chunk
        rows[..., n_params + 1] = choice_arr[None, :, None]
        log_density = np.asarray(predictor(rows.reshape(-1, width)), dtype=np.float64)
        density = np.exp(log_density).reshape(rows.shape[:3])
        # cumulative_trapezoid wants x shaped like the density; a view is free.
        x = np.broadcast_to(t_chunk, density.shape)
        per_choice = cumulative_trapezoid(density, x, axis=-1, initial=0.0)
        by_choice[start:stop] = per_choice[..., -1]
        cdf[start:stop] = per_choice.sum(axis=1)
    return ChoiceMass(t=t, cdf=cdf, by_choice=by_choice)


def _dev_stats(dev: NDArray[np.float64]) -> dict[str, float]:
    """Summary of ``total - 1`` over the surveyed parameter vectors."""
    # A NaN total is a failed integral: infinitely far from one. "higher" takes
    # an order statistic, so an infinite percentile stays inf, not NaN.
    abs_dev = np.where(np.isnan(dev), np.inf, np.abs(dev))
    p50, p90, p99 = np.percentile(abs_dev, [50, 90, 99], method="higher")
    return {
        "mean": float(1.0 + dev.mean()),
        "p50_abs_dev": float(p50),
        "p90_abs_dev": float(p90),
        "p99_abs_dev": float(p99),
        "min": float(1.0 + dev.min()),
        "max": float(1.0 + dev.max()),
        "frac_gt_0.02": float(np.mean(abs_dev > 0.02)),
        "frac_gt_0.05": float(np.mean(abs_dev > 0.05)),
        "frac_gt_0.10": float(np.mean(abs_dev > 0.10)),
    }


def _bin_index(x: NDArray[np.float64], lo: float, hi: float, n_bins: int):
    """Equal-width bin index of ``x`` on ``[lo, hi]``, the top edge inclusive."""
    scaled = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    return np.clip(np.floor(scaled * n_bins).astype(np.intp), 0, n_bins - 1)


def survey(
    predictor: Predictor,
    param_bounds: dict[str, tuple[float, float]],
    params: Sequence[str],
    choices: Sequence[float] | NDArray,
    *,
    n_theta: int = 20_000,
    seed: int = 0,
    onset_param: str | None = "t",
) -> dict:
    """Whole-box statistics of a LAN's integrated mass.

    Draws ``n_theta`` parameter vectors uniformly inside ``param_bounds``,
    integrates each with :func:`choice_mass` (streaming: only the totals are
    kept) and summarises how far the total mass strays from one and where in
    the box it does so. The output is plain numbers (``json.dumps``-able).

    Parameters
    ----------
    predictor
        The LAN, e.g. from :func:`load_onnx_predictor`.
    param_bounds
        ``{name: (lo, hi)}`` for every name in ``params``; the box is
        sampled uniformly (the training box of the LAN, typically).
    params
        Parameter names in the order the LAN's rows expect them.
    choices
        The choice codes the LAN was trained on.
    n_theta
        Number of parameter vectors drawn.
    seed
        Seed of the ``numpy.random.default_rng`` draw.
    onset_param
        Name of the non-decision-time parameter: the onset the grid is built
        around and the edge ``leak_below_onset`` is measured at. ``None``
        builds the grid around onset 0 and reports no leak.

    Returns
    -------
    dict
        ``n_theta``, ``grid`` (its repr), ``seconds`` (wall time),
        ``total`` (mean/min/max of the total and quantiles and exceedance
        fractions of ``|total - 1|``; a NaN total counts as an infinite
        deviation), ``leak_below_onset`` (``mean``/``p99``/``max`` of
        :meth:`ChoiceMass.mass_before` at the onset, or ``None``) and
        ``by_param`` (``{name: [10 bins of lo/hi/mean_dev/max_abs_dev/n]}``;
        empty bins carry ``None``).

    Raises
    ------
    ValueError
        If ``params`` is empty or names a parameter without bounds, if
        ``onset_param`` is not in ``params``, if a bound has ``hi < lo``, or
        if ``n_theta < 1``.
    """
    params = list(params)
    n_params = len(params)
    if n_params == 0:
        raise ValueError("params must name at least one parameter")
    missing = [name for name in params if name not in param_bounds]
    if missing:
        raise ValueError(f"param_bounds lacks bounds for {missing}")
    if onset_param is not None and onset_param not in params:
        raise ValueError(f"onset_param {onset_param!r} is not in params {params}")
    if n_theta < 1:
        raise ValueError(f"n_theta must be >= 1, got {n_theta}")

    lo = np.array([param_bounds[name][0] for name in params], dtype=np.float64)
    hi = np.array([param_bounds[name][1] for name in params], dtype=np.float64)
    if np.any(hi < lo):
        raise ValueError("every bound must satisfy lo <= hi")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(lo, hi, size=(n_theta, n_params))
    onset_idx = params.index(onset_param) if onset_param is not None else None

    total = np.empty(n_theta)
    leak = np.empty(n_theta) if onset_idx is not None else None
    started = time.perf_counter()
    for start in range(0, n_theta, _CHUNK):
        chunk = theta[start : start + _CHUNK]
        stop = start + len(chunk)
        onset = chunk[:, onset_idx] if onset_idx is not None else 0.0
        mass = choice_mass(predictor, chunk, choices, onset)
        total[start:stop] = mass.total
        if leak is not None:
            leak[start:stop] = mass.mass_before(onset)
    seconds = time.perf_counter() - started

    dev = total - 1.0
    by_param: dict[str, list[dict[str, float | int | None]]] = {}
    for j, name in enumerate(params):
        edges = np.linspace(lo[j], hi[j], 11)
        index = _bin_index(theta[:, j], lo[j], hi[j], 10)
        bins = []
        for b in range(10):
            in_bin = dev[index == b]
            bins.append(
                {
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "mean_dev": float(in_bin.mean()) if in_bin.size else None,
                    "max_abs_dev": float(np.abs(in_bin).max()) if in_bin.size else None,
                    "n": int(in_bin.size),
                }
            )
        by_param[name] = bins

    return {
        "n_theta": int(n_theta),
        "grid": repr(OnsetGrid()),
        "seconds": float(seconds),
        "total": _dev_stats(dev),
        "leak_below_onset": (
            None
            if leak is None
            else {
                "mean": float(leak.mean()),
                "p99": float(np.percentile(leak, 99)),
                "max": float(leak.max()),
            }
        ),
        "by_param": by_param,
    }
