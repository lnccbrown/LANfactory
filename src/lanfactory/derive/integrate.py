"""Integrate a trained LAN's density over reaction time.

A LAN approximates ``log p(rt, choice | theta)`` on the support it was trained
on. Integrating that density over ``rt`` for each choice gives the choice
probability and the probability of responding before a deadline — the
quantities auxiliary networks (CPNs, OPNs) are trained on, obtained here from
the LAN instead of from a fresh simulation.

Grid policy
-----------
The density is integrated with the trapezoid rule over the full support
``[t_min, max_t]``: ``max_t = 20`` s is ssms' default ``max_t``, the upper
edge of the LANs' training support, and nothing is extrapolated past it.
Two grids are offered. :class:`IntegrationGrid` is uniform and adequate for
smooth densities; a LAN's density is not smooth — it rises from zero over a
few milliseconds at the non-decision time ``t`` — and on the Hub ddm LAN a
uniform 1000-point grid carries 15–25 % quadrature error at ``a < 0.5``
(totals down to 0.66 that are pure grid artefacts). :class:`OnsetGrid` is
refined per parameter vector around ``t`` (32 points on ``[t_min, t−0.05]``,
128 on ``[t−0.05, t]``, 600 on ``[t, t+1]``, 400 on ``[t+1, max_t]``) and
matches a 16 000-point uniform grid to ``5e-5`` at ~1160 points. Integrating
from ``t`` only was tested and rejected (mean error 0.0083 vs 0.0042 against
simulation): the LAN leaks mass below ``t`` (mean 0.0035, p99 0.052 on the
same LAN) and that leak is part of what it predicts.

Tail policy
-----------
The per-choice masses are **not** renormalised to sum to one; they carry
whatever density the LAN puts past ``max_t`` and the network's own scale
error. Note that ssms does not censor a base (non-deadline) model there: an
un-terminated trial comes back at ``rt ≈ max_t + t`` with its sign-implied
choice, so ssms' own ``choice_p`` sums to one while these masses need not.
Renormalisation is the corpus's job; :attr:`ChoiceMass.total` is always
recorded alongside the per-choice masses so the deficit stays visible.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = [
    "ChoiceMass",
    "IntegrationGrid",
    "OnnxPredictor",
    "OnsetGrid",
    "Predictor",
    "choice_mass",
    "load_onnx_predictor",
    "survey",
]


class Predictor(Protocol):
    """Anything that maps ``(N, D)`` float32 rows to ``(N,)`` log-densities."""

    @property
    def input_width(self) -> int:
        """Row width ``D`` the predictor expects."""
        ...

    def __call__(self, batch: NDArray[np.float32]) -> NDArray[np.floating]:
        """Evaluate the network on a batch of rows.

        Parameters
        ----------
        batch
            ``(N, D)`` float32 rows laid out as the network was trained on.

        Returns
        -------
        NDArray[np.floating]
            ``(N,)`` log-densities.
        """
        ...


class OnnxPredictor:
    """A published single-trial LAN, re-declared with a batch axis in memory.

    Build with :func:`load_onnx_predictor`. The ecosystem's ONNX graphs are
    exported with a concrete ``(1, D)`` input (see
    ``lanfactory.onnx.contract``), which onnxruntime enforces: feeding
    ``(N, D)`` to the file as published raises. This object loads the graph,
    marks the batch dimension of every input and output symbolic *in the
    in-memory protobuf only*, and builds the session from that. The artifact
    on disk is never modified.

    Parameters
    ----------
    session
        An ``onnxruntime.InferenceSession`` whose first input is ``(N, D)``.

    Attributes
    ----------
    input_name : str
        Name of the graph input fed by :meth:`__call__`.
    input_width : int
        ``D``: the per-row width the graph expects (parameters + rt + choice
        for a LAN).
    """

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session: ort.InferenceSession = session
        first_input = session.get_inputs()[0]
        self.input_name: str = first_input.name
        self.input_width: int = int(first_input.shape[-1])

    def __call__(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """Evaluate the network on a batch of rows.

        Parameters
        ----------
        batch
            ``(N, D)`` array; cast to float32 if needed.

        Returns
        -------
        NDArray[np.float32]
            ``(N,)`` network outputs (the trailing ``1`` axis squeezed).
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

    The graph's batch dimension (dim 0 of every graph input and output) is set
    to the symbolic ``"N"`` on the in-memory protobuf, and the session is
    built from the serialized copy. The file on disk keeps its concrete
    ``dim_value == 1`` — it is read, never written — so the single-trial
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
        exporters trace rank-1 ``(D,)`` graphs (see
        ``lanfactory.onnx.contract``); those have no batch axis to widen, and
        this loader does not wrap them.
    """
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path))
    # Defensive for foreign producers that list initializers among the graph
    # inputs (the ecosystem's exporters do not); intentionally untested.
    initializers = {tensor.name for tensor in model.graph.initializer}
    io_tensors = [
        value_info
        for value_info in (*model.graph.input, *model.graph.output)
        if value_info.name not in initializers
    ]
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
    session = ort.InferenceSession(model.SerializeToString())
    return OnnxPredictor(session)


@dataclass(frozen=True)
class IntegrationGrid:
    """Uniform reaction-time grid the density is integrated on.

    Parameters
    ----------
    n_points
        Number of grid points.
    max_t
        Upper edge of the grid, in seconds. Defaults to ssms' default
        ``max_t`` (20 s), the LANs' training support; the density is never
        evaluated past it.
    t_min
        Lower edge, kept strictly positive so ``rt = 0`` is never fed to a
        network whose training data never contains it.
    """

    n_points: int = 1000
    max_t: float = 20.0
    t_min: float = 1e-4

    def __post_init__(self) -> None:
        if self.n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {self.n_points}")
        if not 0 < self.t_min < self.max_t:
            raise ValueError(
                f"need 0 < t_min < max_t, got t_min={self.t_min}, max_t={self.max_t}"
            )

    @property
    def t(self) -> NDArray[np.float64]:
        """The grid, ``np.linspace(t_min, max_t, n_points)``."""
        return np.linspace(self.t_min, self.max_t, self.n_points)


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

    A LAN's density is essentially zero below the non-decision time ``t``
    and rises steeply just above it; a uniform grid either wastes points on
    the flat regions or under-resolves the onset. This grid concatenates
    four ``linspace`` segments per ``t`` — ``[t_min, t − knee]``,
    ``[t − knee, t]``, ``[t, t + onset_window]`` and
    ``[t + onset_window, max_t]`` — so the whole support is still integrated
    (the LAN leaks mass below ``t``, and that leak is part of what it
    predicts) while the points concentrate where the density moves. Every
    row has the same number of points, is strictly increasing, and contains
    ``t_min``, ``t`` and ``max_t`` exactly.

    Parameters
    ----------
    n_pre
        Points on ``[t_min, t − knee]`` (endpoint excluded).
    n_knee
        Points on ``[t − knee, t]`` (endpoint excluded): the LAN blurs the
        onset over a few milliseconds below ``t``, and this segment resolves
        the blur.
    knee
        Width of the knee segment, in seconds. The knee starts at
        ``max(t − knee, (t_min + t) / 2)``: ``knee`` wide unless that would
        leave the pre segment shorter than the knee, in which case the two
        split ``[t_min, t]`` evenly.
    n_onset
        Points on ``[t, t + onset_window]`` (endpoint excluded).
    onset_window
        Width of the onset segment, in seconds.
    n_tail
        Points on ``[t + onset_window, max_t]``, endpoint included.
    max_t
        Upper edge of the grid; see :class:`IntegrationGrid`.
    t_min
        Lower edge, kept strictly positive; see :class:`IntegrationGrid`.

    Notes
    -----
    ``t`` is clipped to ``[t_min + 1e-3, max_t − onset_window − 1e-3]`` so
    that every segment has positive length; the density is still evaluated
    over the full ``[t_min, max_t]`` for such rows.
    """

    n_pre: int = 32
    n_knee: int = 128
    knee: float = 0.05
    n_onset: int = 600
    onset_window: float = 1.0
    n_tail: int = 400
    max_t: float = 20.0
    t_min: float = 1e-4

    #: Margin keeping the outer segments non-degenerate at the box edges.
    margin: ClassVar[float] = 1e-3

    def __post_init__(self) -> None:
        for name in ("n_pre", "n_knee", "n_onset"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.n_tail < 2:
            raise ValueError(f"n_tail must be >= 2, got {self.n_tail}")
        if self.knee <= 0.0 or self.onset_window <= 0.0:
            raise ValueError(
                f"knee and onset_window must be > 0, got knee={self.knee}, "
                f"onset_window={self.onset_window}"
            )
        if self.t_min <= 0.0 or self.onset_bounds[0] >= self.onset_bounds[1]:
            raise ValueError(
                f"need 0 < t_min and t_min + {self.margin} < max_t - onset_window "
                f"- {self.margin}, got t_min={self.t_min}, max_t={self.max_t}, "
                f"onset_window={self.onset_window}"
            )

    @property
    def n_points(self) -> int:
        """Points per row: ``n_pre + n_knee + n_onset + n_tail``."""
        return self.n_pre + self.n_knee + self.n_onset + self.n_tail

    @property
    def onset_bounds(self) -> tuple[float, float]:
        """The interval onsets are clipped to before the row is built."""
        return (
            self.t_min + self.margin,
            self.max_t - self.onset_window - self.margin,
        )

    def for_theta(self, onset: ArrayLike) -> NDArray[np.float64]:
        """Build one grid row per onset time.

        Parameters
        ----------
        onset
            ``(n_theta,)`` onset times in seconds, e.g. ``theta[:, t_idx]``.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta, n_points)`` strictly increasing rows.
        """
        t = np.asarray(onset, dtype=np.float64)
        if t.ndim != 1:
            raise ValueError(f"onset must be (n_theta,), got shape {t.shape}")
        t = np.clip(t, *self.onset_bounds)
        t_min = np.full_like(t, self.t_min)
        # The knee is ``knee`` wide or half of ``[t_min, t]``, whichever is
        # shorter, so the pre segment is never thinner than ``margin / 2``
        # and its width varies continuously in ``t`` (a threshold rule at
        # ``t_min + knee`` would cram ``n_pre`` points into a vanishing
        # interval just above it: distinct in float64, duplicates in the
        # float32 rows the network sees).
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


@dataclass
class ChoiceMass:
    """Cumulative per-choice mass of a LAN's density on a reaction-time grid.

    Attributes
    ----------
    t : NDArray[np.float64]
        The grid the density was integrated on: ``(n_points,)`` when shared
        by every parameter vector (:class:`IntegrationGrid`), or
        ``(n_theta, n_points)`` when refined per parameter vector
        (:class:`OnsetGrid`).
    choices : NDArray
        ``(n_choices,)`` choice codes, in the order of the ``cdf`` axis.
    cdf : NDArray[np.float64]
        ``(n_theta, n_choices, n_points)`` cumulative trapezoid integral of
        the density along ``t``; ``cdf[..., 0] == 0``.

    Notes
    -----
    Masses are not renormalised. ``total`` differs from one by the density
    the LAN puts past ``max_t`` plus any approximation error in the network
    itself (ssms returns such trials at ``rt ≈ max_t + t`` rather than as
    omissions, so its ``choice_p`` does not share this deficit); record it
    with the masses.
    """

    t: NDArray[np.float64]
    choices: NDArray
    cdf: NDArray[np.float64]

    def _choice_index(self, choice: float) -> int:
        matches = np.flatnonzero(self.choices == choice)
        if matches.size == 0:
            raise ValueError(
                f"unknown choice code {choice!r}; known choices: "
                f"{self.choices.tolist()}"
            )
        return int(matches[0])

    def _curve(self, choice: float | None) -> NDArray[np.float64]:
        """``(n_theta, n_points)`` cumulative mass for one choice or all summed."""
        if choice is None:
            return self.cdf.sum(axis=1)
        return self.cdf[:, self._choice_index(choice), :]

    def _t_rows(self) -> NDArray[np.float64]:
        """The grid as ``(n_theta, n_points)``, a broadcast view when shared."""
        n_theta, _, n_points = self.cdf.shape
        return np.broadcast_to(self.t, (n_theta, n_points))

    def _bracket(
        self, d: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
        """Row indices and the grid interval ``[lo, hi]`` holding each ``d``."""
        t_rows = self._t_rows()
        n_theta, n_points = t_rows.shape
        # Per row: ``searchsorted(t_row, d, side="right") - 1``.
        lo = np.clip((t_rows <= d[:, None]).sum(axis=1) - 1, 0, n_points - 2)
        return np.arange(n_theta), lo, lo + 1

    def mass(self, choice: float) -> NDArray[np.float64]:
        """Mass of one choice on ``[t_min, max_t]``.

        Parameters
        ----------
        choice
            A choice code from :attr:`choices`.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta,)``.
        """
        return self.cdf[:, self._choice_index(choice), -1]

    @property
    def total(self) -> NDArray[np.float64]:
        """Mass summed over choices, shape ``(n_theta,)``."""
        return self.cdf[:, :, -1].sum(axis=1)

    def mass_before(
        self, deadline: ArrayLike, choice: float | None = None
    ) -> NDArray[np.float64]:
        """Mass accumulated before ``deadline``, linearly interpolated on the grid.

        Parameters
        ----------
        deadline
            Scalar, or ``(n_theta,)`` per-parameter-vector deadlines, in
            seconds. Values below ``t_min`` give 0; values above ``max_t``
            give the mass at ``max_t`` (no extrapolation).
        choice
            A choice code, or ``None`` to sum over choices.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta,)``.
        """
        curve = self._curve(choice)
        t_rows = self._t_rows()
        d = np.broadcast_to(np.asarray(deadline, dtype=np.float64), (curve.shape[0],))
        d = np.clip(d, t_rows[:, 0], t_rows[:, -1])
        rows, lo, hi = self._bracket(d)
        t_lo, t_hi = t_rows[rows, lo], t_rows[rows, hi]
        w = (d - t_lo) / (t_hi - t_lo)
        return curve[rows, lo] * (1.0 - w) + curve[rows, hi] * w

    def leak_below(self, onset: ArrayLike) -> NDArray[np.float64]:
        """Mass on ``[t_min, onset]`` summed over choices, per parameter vector.

        Under the simulated model no response precedes the non-decision
        time; a LAN nevertheless puts some density there, and the corpus and
        :func:`survey` report that leak through this one definition, which
        is :meth:`mass_before` with ``choice=None``.

        Parameters
        ----------
        onset
            Scalar or ``(n_theta,)`` onset times in seconds.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta,)``.
        """
        return self.mass_before(onset)

    def quantile(
        self, u: ArrayLike, choice: float | None = None
    ) -> NDArray[np.float64]:
        """Time at which the cumulative mass reaches ``u`` of its own total.

        The inverse of :meth:`mass_before` on the same piecewise-linear curve,
        with the target expressed as a fraction of the (per-theta, per-choice)
        mass on ``[t_min, max_t]`` rather than of one.

        Parameters
        ----------
        u
            Fraction in ``(0, 1)``; scalar or ``(n_theta,)``.
        choice
            A choice code, or ``None`` for the mass summed over choices.

        Returns
        -------
        NDArray[np.float64]
            ``(n_theta,)`` times in seconds.
        """
        curve = self._curve(choice)
        n_theta, n_points = curve.shape
        u_arr = np.broadcast_to(np.asarray(u, dtype=np.float64), (n_theta,))
        if np.any((u_arr <= 0.0) | (u_arr >= 1.0)):
            raise ValueError(f"u must lie in the open interval (0, 1), got {u!r}")
        target = u_arr * curve[:, -1]
        # First grid index where the (non-decreasing) curve reaches the target.
        hi = np.clip((curve < target[:, None]).sum(axis=1), 1, n_points - 1)
        lo = hi - 1
        rows = np.arange(n_theta)
        rise = curve[rows, hi] - curve[rows, lo]
        w = np.where(
            rise > 0.0, (target - curve[rows, lo]) / np.where(rise > 0, rise, 1), 0.0
        )
        t_rows = self._t_rows()
        t_lo, t_hi = t_rows[rows, lo], t_rows[rows, hi]
        return t_lo + w * (t_hi - t_lo)


def choice_mass(
    predictor: Predictor,
    theta: ArrayLike,
    choices: Sequence[float] | NDArray,
    grid: IntegrationGrid | OnsetGrid = IntegrationGrid(),
    chunk_size: int = 512,
    *,
    onset: ArrayLike | None = None,
) -> ChoiceMass:
    """Integrate a LAN's density over reaction time for every choice.

    Rows are laid out as ``[theta..., rt, choice]`` — the input layout the
    LANs are trained on — and evaluated in chunks of ``chunk_size`` parameter
    vectors. Chunking bounds the row buffer handed to the network at
    ``chunk_size * n_choices * n_points * (n_params + 2)`` float32 values;
    the returned ``cdf`` itself costs ``n_theta * n_choices * n_points``
    float64 values and is the only full-size array held.

    Parameters
    ----------
    predictor
        Maps ``(N, n_params + 2)`` rows to ``(N,)`` log-densities;
        typically an :class:`OnnxPredictor`.
    theta
        ``(n_theta, n_params)`` parameter vectors (a single ``(n_params,)``
        vector is promoted).
    choices
        The choice codes the LAN was trained on, e.g. ``[-1, 1]``.
    grid
        Reaction-time grid. A uniform :class:`IntegrationGrid` is shared by
        every parameter vector; an :class:`OnsetGrid` is built per parameter
        vector from ``onset`` and is the grid of choice for a LAN (see the
        module notes on grid policy).
    chunk_size
        Number of parameter vectors evaluated per network call.
    onset
        ``(n_theta,)`` onset (non-decision) times, e.g. ``theta[:, t_idx]``.
        Required with an :class:`OnsetGrid`, rejected otherwise.

    Returns
    -------
    ChoiceMass
        Cumulative mass with ``cdf`` of shape ``(n_theta, n_choices, n_points)``
        and ``t`` of shape ``(n_points,)`` (uniform grid) or
        ``(n_theta, n_points)`` (onset grid).

    Raises
    ------
    ValueError
        If ``predictor.input_width != n_params + 2`` — the usual symptom of
        pairing a parameter set with the wrong LAN — or if ``onset`` and
        ``grid`` do not go together.
    """
    theta_arr = np.atleast_2d(np.asarray(theta, dtype=np.float32))
    if theta_arr.ndim != 2:
        raise ValueError(f"theta must be (n_theta, n_params), got {theta_arr.shape}")
    n_theta, n_params = theta_arr.shape
    expected_width = n_params + 2
    if predictor.input_width != expected_width:
        raise ValueError(
            f"predictor expects rows of width {predictor.input_width}, but theta "
            f"has {n_params} parameters so rows [theta..., rt, choice] are "
            f"{expected_width} wide; is this the right LAN for these parameters?"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    choice_arr = np.asarray(choices)
    n_choices = choice_arr.shape[0]

    per_theta = isinstance(grid, OnsetGrid)
    if per_theta != (onset is not None):
        raise ValueError(
            "an OnsetGrid needs onset=theta[:, t_idx] and a uniform "
            "IntegrationGrid takes no onset; got "
            f"grid={type(grid).__name__} with onset={'given' if onset is not None else 'None'}"
        )
    if per_theta:
        onset_arr = np.asarray(onset, dtype=np.float64)
        if onset_arr.shape != (n_theta,):
            raise ValueError(
                f"onset must be (n_theta,) = ({n_theta},), got shape {onset_arr.shape}"
            )
        n_points = grid.n_points
        t = np.empty((n_theta, n_points), dtype=np.float64)
    else:
        t = grid.t
        n_points = t.shape[0]

    cdf = np.empty((n_theta, n_choices, n_points), dtype=np.float64)
    for start in range(0, n_theta, chunk_size):
        chunk = theta_arr[start : start + chunk_size]
        n_chunk = chunk.shape[0]
        rows = np.empty(
            (n_chunk, n_choices, n_points, expected_width), dtype=np.float32
        )
        rows[..., :n_params] = chunk[:, None, None, :]
        # The network sees rt rounded to float32 (~1e-6 s at max_t) while the
        # trapezoid uses the float64 grid; the mismatch is orders of magnitude
        # below the rule's own discretisation error.
        if per_theta:
            t_chunk = grid.for_theta(onset_arr[start : start + n_chunk])
            t[start : start + n_chunk] = t_chunk
            x = t_chunk[:, None, :]
        else:
            x = t
        rows[..., n_params] = x
        rows[..., n_params + 1] = choice_arr[None, :, None]
        log_density = np.asarray(predictor(rows.reshape(-1, expected_width)))
        density = np.exp(
            log_density.reshape(n_chunk, n_choices, n_points).astype(np.float64)
        )
        cdf[start : start + n_chunk] = cumulative_trapezoid(
            density, x, axis=-1, initial=0.0
        )

    return ChoiceMass(t=t, choices=choice_arr, cdf=cdf)


def _dev_stats(dev: NDArray[np.float64]) -> dict[str, float | None]:
    """Summary of ``total - 1`` over a set of parameter vectors."""
    if dev.size == 0:
        keys = (
            "mean", "p50_abs_dev", "p90_abs_dev", "p99_abs_dev", "min", "max",
            "frac_gt_0.02", "frac_gt_0.05", "frac_gt_0.10",
        )  # fmt: skip
        return dict.fromkeys(keys)
    abs_dev = np.abs(dev)
    p50, p90, p99 = np.percentile(abs_dev, [50, 90, 99])
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
    grid: IntegrationGrid | OnsetGrid | None = None,
    chunk_size: int = 512,
    shrink: float = 0.1,
) -> dict:
    """Whole-box statistics of a LAN's integrated mass.

    Draws ``n_theta`` parameter vectors uniformly inside ``param_bounds``,
    integrates the density for each with :func:`choice_mass` (streaming;
    the cdf is not retained) and summarises how far the total mass strays
    from one and where in the box it does so. The output is plain numbers
    (``json.dumps``-able) — the record a corpus keeps next to the artifact
    it was derived from.

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
        Name of the non-decision-time parameter; its column is the onset
        the :class:`OnsetGrid` is built around and the edge
        ``leak_below_onset`` is measured at. ``None`` integrates on a
        uniform :class:`IntegrationGrid` and reports no leak.
    grid
        Explicit grid; defaults to ``OnsetGrid()`` when ``onset_param`` is
        given and ``IntegrationGrid()`` otherwise.
    chunk_size
        Parameter vectors per network call.
    shrink
        Fraction of each parameter's range trimmed at both ends for the
        ``shrunk_box`` statistics (``0.1`` keeps the middle 80 % per axis).

    Returns
    -------
    dict
        ``n_theta``, ``grid`` (its repr), ``seconds`` (wall time),
        ``total`` (mean/min/max of the total and quantiles and exceedance
        fractions of ``|total - 1|``), ``shrunk_box`` (the same over the
        interior of the box plus ``frac_of_theta``), ``leak_below_onset``
        (``mean``/``p99``/``max`` of :meth:`ChoiceMass.leak_below`, or
        ``None``), ``by_param`` (``{name: [10 bins of lo/hi/mean_dev/
        max_abs_dev/n]}``; empty bins carry ``None``) and ``worst_cell``
        (on an 8×8 grid of the two parameters most correlated with the
        deviation, the cell with the largest ``|mean_dev|``; ``None`` with
        fewer than two parameters).
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
    if not 0.0 <= shrink < 0.5:
        raise ValueError(f"shrink must lie in [0, 0.5), got {shrink}")
    if grid is None:
        grid = OnsetGrid() if onset_param is not None else IntegrationGrid()
    if isinstance(grid, OnsetGrid) != (onset_param is not None):
        raise ValueError(
            f"grid {type(grid).__name__} does not go with onset_param={onset_param!r}: "
            "an OnsetGrid needs an onset parameter and a uniform grid takes none"
        )

    lo = np.array([param_bounds[name][0] for name in params], dtype=np.float64)
    hi = np.array([param_bounds[name][1] for name in params], dtype=np.float64)
    if np.any(hi < lo):
        raise ValueError("every bound must satisfy lo <= hi")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(lo, hi, size=(n_theta, n_params))
    onset_idx = params.index(onset_param) if onset_param is not None else None

    total = np.empty(n_theta, dtype=np.float64)
    leak = np.empty(n_theta, dtype=np.float64) if onset_idx is not None else None
    started = time.perf_counter()
    for start in range(0, n_theta, chunk_size):
        chunk = theta[start : start + chunk_size]
        stop = start + chunk.shape[0]
        onset = chunk[:, onset_idx] if onset_idx is not None else None
        mass = choice_mass(
            predictor, chunk, choices, grid=grid, chunk_size=chunk_size, onset=onset
        )
        total[start:stop] = mass.total
        if leak is not None:
            leak[start:stop] = mass.leak_below(onset)
    seconds = time.perf_counter() - started

    dev = total - 1.0
    span = hi - lo
    inside = np.all(
        (theta >= lo + shrink * span) & (theta <= hi - shrink * span), axis=1
    )
    shrunk_box = _dev_stats(dev[inside])
    shrunk_box["frac_of_theta"] = float(inside.mean())

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

    worst_cell = None
    if n_params >= 2 and n_theta >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.array(
                [np.corrcoef(theta[:, j], dev)[0, 1] for j in range(n_params)]
            )
        corr = np.nan_to_num(corr)
        jx, jy = np.argsort(-np.abs(corr))[:2]
        ix = _bin_index(theta[:, jx], lo[jx], hi[jx], 8)
        iy = _bin_index(theta[:, jy], lo[jy], hi[jy], 8)
        flat = ix * 8 + iy
        counts = np.bincount(flat, minlength=64)
        sums = np.bincount(flat, weights=dev, minlength=64)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        cell = int(np.argmax(np.abs(means)))
        cx, cy = divmod(cell, 8)
        x_edges = np.linspace(lo[jx], hi[jx], 9)
        y_edges = np.linspace(lo[jy], hi[jy], 9)
        worst_cell = {
            "param_x": params[jx],
            "param_y": params[jy],
            "x_lo": float(x_edges[cx]),
            "x_hi": float(x_edges[cx + 1]),
            "y_lo": float(y_edges[cy]),
            "y_hi": float(y_edges[cy + 1]),
            "mean_dev": float(means[cell]),
            "n": int(counts[cell]),
        }

    return {
        "n_theta": int(n_theta),
        "grid": repr(grid),
        "seconds": float(seconds),
        "total": _dev_stats(dev),
        "shrunk_box": shrunk_box,
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
        "worst_cell": worst_cell,
    }
