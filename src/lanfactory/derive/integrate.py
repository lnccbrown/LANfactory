"""Integrate a trained LAN's density over reaction time.

A LAN approximates ``log p(rt, choice | theta)`` on the support it was trained
on. Integrating that density over ``rt`` for each choice gives the choice
probability and the probability of responding before a deadline — the
quantities auxiliary networks (CPNs, OPNs) are trained on, obtained here from
the LAN instead of from a fresh simulation.

Tail policy
-----------
The density is integrated only over ``[t_min, max_t]`` with the trapezoid rule
on a uniform grid. ``max_t = 20`` s is ssms' default ``max_t``, the upper edge
of the LANs' training support; nothing is extrapolated past it. The per-choice
masses are **not** renormalised to sum to one, so they fall short by whatever
density the LAN puts past ``max_t``. Note that ssms does not censor a base
(non-deadline) model there: an un-terminated trial comes back at
``rt ≈ max_t + t`` with its sign-implied choice, so ssms' own ``choice_p``
sums to one while these masses do not. Callers record
:attr:`ChoiceMass.total` alongside the per-choice masses so that deficit
stays visible.
"""

from __future__ import annotations

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
    "IntegrationGrid",
    "OnnxPredictor",
    "Predictor",
    "choice_mass",
    "load_onnx_predictor",
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


@dataclass
class ChoiceMass:
    """Cumulative per-choice mass of a LAN's density on a reaction-time grid.

    Attributes
    ----------
    t : NDArray[np.float64]
        ``(n_points,)`` grid the density was integrated on.
    choices : NDArray
        ``(n_choices,)`` choice codes, in the order of the ``cdf`` axis.
    cdf : NDArray[np.float64]
        ``(n_theta, n_choices, n_points)`` cumulative trapezoid integral of
        the density along ``t``; ``cdf[..., 0] == 0``.

    Notes
    -----
    Masses are not renormalised. ``total`` falls short of one by the density
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
        n_theta, n_points = curve.shape
        d = np.broadcast_to(np.asarray(deadline, dtype=np.float64), (n_theta,))
        d = np.clip(d, self.t[0], self.t[-1])
        lo = np.clip(np.searchsorted(self.t, d, side="right") - 1, 0, n_points - 2)
        hi = lo + 1
        w = (d - self.t[lo]) / (self.t[hi] - self.t[lo])
        rows = np.arange(n_theta)
        return curve[rows, lo] * (1.0 - w) + curve[rows, hi] * w

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
        return self.t[lo] + w * (self.t[hi] - self.t[lo])


def choice_mass(
    predictor: Predictor,
    theta: ArrayLike,
    choices: Sequence[float] | NDArray,
    grid: IntegrationGrid = IntegrationGrid(),
    chunk_size: int = 512,
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
        Reaction-time grid; see :class:`IntegrationGrid` for the tail policy.
    chunk_size
        Number of parameter vectors evaluated per network call.

    Returns
    -------
    ChoiceMass
        Cumulative mass with ``cdf`` of shape ``(n_theta, n_choices, n_points)``.

    Raises
    ------
    ValueError
        If ``predictor.input_width != n_params + 2`` — the usual symptom of
        pairing a parameter set with the wrong LAN.
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
        rows[..., n_params] = t[None, None, :]
        rows[..., n_params + 1] = choice_arr[None, :, None]
        log_density = np.asarray(predictor(rows.reshape(-1, expected_width)))
        density = np.exp(
            log_density.reshape(n_chunk, n_choices, n_points).astype(np.float64)
        )
        cdf[start : start + n_chunk] = cumulative_trapezoid(
            density, t, axis=-1, initial=0.0
        )

    return ChoiceMass(t=t, choices=choice_arr, cdf=cdf)
