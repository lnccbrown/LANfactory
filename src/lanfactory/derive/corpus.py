"""Turn a LAN's per-choice masses into cpn / opn / gonogo training corpora.

:mod:`lanfactory.derive.integrate` integrates a trained LAN's density over
reaction time. This module samples parameter vectors the way ssms samples a
simulated corpus, asks the LAN for the masses, and writes pickles whose layout
is what the trainers already read — so an auxiliary network can be derived
from a production LAN on a laptop in minutes, with the trainer unchanged.

Row layouts
-----------
Every corpus has ``n_params + 1`` input columns, in ssms parameter order:

=========  =====================================  ==============================
type       training row                           label
=========  =====================================  ==============================
cpn        ``[theta..., choice]`` — one row per   ``mass(choice) / total``
           choice code in ``model_config["choices"]``
opn        ``[theta..., deadline]``               ``1 - F(deadline) / total``
gonogo     ``[theta..., deadline]``               ``nogo_before(deadline) / total``
                                                  ``+ (1 - F(deadline) / total)``
=========  =====================================  ==============================

with ``total`` the LAN's mass on ``[t_min, max_t]`` summed over choices,
``F(d) = mass_before(d)`` summed over choices and ``nogo_before(d)`` the
mass before ``d`` of every choice but the largest code (ssms' own ``nogo_p``:
a trial is *nogo* when its choice is anything but the largest choice code,
or when it is omitted).

Labels are renormalised by the network's own total
---------------------------------------------------
The LAN's total mass is not one: it carries the network's scale error (on the
Hub ddm LAN, median ``|total − 1|`` 0.0037 but up to 0.22 in the corners of
the box; see :func:`.integrate.survey`). Against simulation that error is
mostly *scale* — dividing by the total takes the cpn error from mean 0.035 /
max 0.17 to 0.004 / 0.033 and the opn error from 0.026 / 0.22 to 0.012 /
0.18 — so every label is the LAN's mass as a fraction of its own total, and
the total is recorded in every pickle (``generator_config["derive_stats"]``)
and in the manifest, never hidden. The ratios lie in ``[0, 1]`` up to
rounding; the float32 clip in :func:`_labels` is kept as a safety net, and
the omission term is formed once so that the decomposition
``gonogo_label == nogo_before(deadline) / total + opn_label`` holds row by
row in the written corpus. A parameter vector with zero total is rejected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler

from lanfactory import __version__

from .integrate import (
    ChoiceMass,
    IntegrationGrid,
    OnsetGrid,
    choice_mass,
    load_onnx_predictor,
    survey,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AUX_CATEGORY",
    "DERIVATION_METHOD",
    "MANIFEST_NAME",
    "NETWORK_TYPES",
    "SourceLAN",
    "cpn_labels",
    "derive_aux_corpus",
    "gonogo_labels",
    "grid_description",
    "opn_labels",
    "sample_deadlines",
    "sample_theta",
    "simulate_labels",
]

NETWORK_TYPES: tuple[str, ...] = ("cpn", "opn", "gonogo")
"""Auxiliary network types a corpus can be derived for."""

AUX_CATEGORY: dict[str, str] = {"cpn": "choice", "opn": "omission", "gonogo": "nogo"}
"""What each network type's label is the probability of."""

DERIVATION_METHOD = "derived-from-lan"
"""Value of the ``derivation_method`` provenance key."""

MANIFEST_NAME = "derive_manifest.json"
"""Sidecar written next to the pickles by :func:`derive_aux_corpus`."""

_RUN_UUID = re.compile(r"^[0-9a-f]{32}$")


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


class _OrderedUniformSampler(UniformParameterSampler):
    """ssms' uniform sampler drawing the parameters in ``param_space`` order.

    ssms orders the draws by a topological sort over ``set`` objects, so the
    column that consumes each slice of the generator's stream — and with it
    every sampled theta — changes with ``PYTHONHASHSEED`` from one process to
    the next. Insertion order of ``param_space`` (the model's parameter
    order) is a valid topological order once each parameter is placed after
    the parameters its bounds name, and it is the same in every process.
    """

    def _topological_sort(self) -> list[str]:
        order: list[str] = []
        pending = list(self.param_space)
        while pending:
            for name in pending:
                needs = {b for b in self.param_space[name] if isinstance(b, str)}
                if needs.issubset(order):
                    order.append(name)
                    pending.remove(name)
                    break
            else:
                raise ValueError(
                    f"circular parameter dependency among {pending} in param_space"
                )
        return order


def sample_theta(
    model_config: dict, n: int, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Sample parameter vectors exactly as ssms samples a training corpus.

    Uses ssms' ``UniformParameterSampler`` on ``model_config["param_bounds_dict"]``
    with the model's sampling transforms (``ModelConfigBuilder
    .get_sampling_transforms``), which is what ssms' simulation pipeline
    builds — except that the parameters are drawn in ``model_config["params"]``
    order rather than ssms' hash-dependent order, so the same ``rng`` state
    gives the same thetas in every process (ssms' own order varies with
    ``PYTHONHASHSEED``). Columns follow ``model_config["params"]``.

    Parameters
    ----------
    model_config
        An ssms model config (``ModelConfigBuilder.from_model(...)``).
    n
        Number of parameter vectors.
    rng
        Generator handed to the sampler.

    Returns
    -------
    NDArray[np.float32]
        ``(n, n_params)``; float32 like ssms' own samples.
    """
    bounds = model_config["param_bounds_dict"]
    params = list(model_config["params"])
    param_space = {name: bounds[name] for name in params if name in bounds}
    param_space.update({name: b for name, b in bounds.items() if name not in params})
    sampler = _OrderedUniformSampler(
        param_space=param_space,
        constraints=ModelConfigBuilder.get_sampling_transforms(model_config),
    )
    samples = sampler.sample(n_samples=n, rng=rng)
    columns = [np.asarray(samples[name]).reshape(n) for name in params]
    return np.column_stack(columns).astype(np.float32)


def sample_deadlines(
    mass: ChoiceMass,
    rng: np.random.Generator,
    deadline_quantile_frac: float = 0.7,
    bounds: tuple[float, float] = (0.001, 10.0),
) -> NDArray[np.float64]:
    """Draw one training deadline per parameter vector.

    A fraction ``1 - deadline_quantile_frac`` of the deadlines is uniform on
    ``bounds`` (ssms' deadline box, so the corpus covers it); the rest are the
    LAN's own reaction-time quantiles under each theta —
    :meth:`ChoiceMass.quantile` at ``u ~ U(0.05, 0.95)`` — so the deadlines
    concentrate where the omission probability is informative. Everything is
    clipped to ``bounds``.

    Parameters
    ----------
    mass
        Masses for the thetas the deadlines are for; one deadline per row.
    rng
        Generator for both shares.
    deadline_quantile_frac
        Share drawn from the LAN's quantiles, in ``[0, 1]``.
    bounds
        ``(lower, upper)`` deadline bounds; ssms' default is ``(0.001, 10)``.

    Returns
    -------
    NDArray[np.float64]
        ``(n_theta,)`` deadlines in seconds.
    """
    if not 0.0 <= deadline_quantile_frac <= 1.0:
        raise ValueError(
            f"deadline_quantile_frac must lie in [0, 1], got {deadline_quantile_frac}"
        )
    lower, upper = bounds
    if not lower < upper:
        raise ValueError(f"bounds must satisfy lower < upper, got {bounds}")
    n_theta = mass.cdf.shape[0]
    from_quantile = rng.random(n_theta) < deadline_quantile_frac
    uniform = rng.uniform(lower, upper, size=n_theta)
    quantile = mass.quantile(rng.uniform(0.05, 0.95, size=n_theta))
    return np.clip(np.where(from_quantile, quantile, uniform), lower, upper)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def _as_theta(theta: ArrayLike) -> NDArray[np.float32]:
    theta_arr = np.atleast_2d(np.asarray(theta, dtype=np.float32))
    if theta_arr.ndim != 2:
        raise ValueError(f"theta must be (n_theta, n_params), got {theta_arr.shape}")
    return theta_arr


def _check_rows(mass: ChoiceMass, theta: NDArray) -> None:
    if mass.cdf.shape[0] != theta.shape[0]:
        raise ValueError(
            f"mass has {mass.cdf.shape[0]} parameter vectors but theta has "
            f"{theta.shape[0]}"
        )


def _labels(values: NDArray) -> NDArray[np.float32]:
    """``(n_rows, 1)`` float32 probabilities, clipped to ``[0, 1]``."""
    return np.clip(values, 0.0, 1.0).astype(np.float32).reshape(-1, 1)


def _total(mass: ChoiceMass) -> NDArray[np.float64]:
    """The per-theta total the labels are renormalised by; must be positive."""
    total = mass.total
    if np.any(total <= 0.0):
        bad = np.flatnonzero(total <= 0.0)
        raise ValueError(
            f"total mass must be > 0 to renormalise labels; parameter vectors "
            f"{bad.tolist()} have total {total[bad].tolist()}"
        )
    return total


def _omission(mass: ChoiceMass, deadline: NDArray[np.float64]) -> NDArray[np.float64]:
    """``P(no response before deadline)`` as a float64 probability.

    ``1 - F(deadline) / total`` with ``F`` the mass before the deadline
    summed over choices, clipped to ``[0, 1]`` once so that the same term
    enters :func:`opn_labels` and :func:`gonogo_labels`.
    """
    return np.clip(1.0 - mass.mass_before(deadline) / _total(mass), 0.0, 1.0)


def _nogo_before(
    mass: ChoiceMass, deadline: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Mass of every non-maximal choice before ``deadline`` (ssms' nogo set)."""
    go_choice = mass.choices.max()
    return sum(
        (mass.mass_before(deadline, c) for c in mass.choices if c != go_choice),
        np.zeros(mass.cdf.shape[0]),
    )


def _deadline_rows(
    theta: NDArray[np.float32], deadlines: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Rows ``[theta..., deadline]`` and the deadlines as the network sees them."""
    d32 = np.broadcast_to(
        np.asarray(deadlines, dtype=np.float32), (theta.shape[0],)
    ).astype(np.float32)
    data = np.concatenate([theta, d32[:, None]], axis=1)
    # Labels are computed at the float32-rounded deadline the row carries, so
    # row and label describe exactly the same deadline.
    return data, d32.astype(np.float64)


def cpn_labels(
    mass: ChoiceMass, theta: ArrayLike, choices: Sequence[float] | NDArray
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """CPN rows ``[theta..., choice]`` for every choice, labelled ``mass(choice) / total``.

    Rows are theta-major: all choices of ``theta[0]``, then of ``theta[1]``, …
    The labels of one theta sum to one.

    Parameters
    ----------
    mass
        Masses for ``theta``.
    theta
        ``(n_theta, n_params)``.
    choices
        Choice codes to emit rows for, e.g. ``model_config["choices"]``.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        ``data`` of shape ``(n_theta * n_choices, n_params + 1)`` and
        ``labels`` of shape ``(n_theta * n_choices, 1)``.
    """
    theta_arr = _as_theta(theta)
    _check_rows(mass, theta_arr)
    choice_arr = np.asarray(choices, dtype=np.float32).reshape(-1)
    n_theta, n_choices = theta_arr.shape[0], choice_arr.shape[0]
    data = np.concatenate(
        [
            np.repeat(theta_arr, n_choices, axis=0),
            np.tile(choice_arr, n_theta)[:, None],
        ],
        axis=1,
    )
    # Look the masses up by the original codes (choice_arr is the float32 the
    # row carries); stacked along axis 1 so the flattened labels are theta-major.
    per_choice = np.stack(
        [mass.mass(float(c)) for c in np.asarray(choices).reshape(-1)], axis=1
    )
    return data, _labels((per_choice / _total(mass)[:, None]).reshape(-1))


def opn_labels(
    mass: ChoiceMass, theta: ArrayLike, deadlines: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """OPN rows ``[theta..., deadline]`` labelled ``1 - F(deadline) / total``.

    The label is the probability of *not* responding before the deadline —
    the omission probability ssms marks with ``rt == -999`` — as a fraction
    of the LAN's own total mass.

    Parameters
    ----------
    mass
        Masses for ``theta``.
    theta
        ``(n_theta, n_params)``.
    deadlines
        Scalar or ``(n_theta,)`` deadlines in seconds.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        ``data`` of shape ``(n_theta, n_params + 1)`` and ``labels`` of shape
        ``(n_theta, 1)``.
    """
    theta_arr = _as_theta(theta)
    _check_rows(mass, theta_arr)
    data, d = _deadline_rows(theta_arr, deadlines)
    return data, _labels(_omission(mass, d))


def gonogo_labels(
    mass: ChoiceMass, theta: ArrayLike, deadlines: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Go/no-go rows ``[theta..., deadline]`` labelled with ssms' ``nogo_p``.

    ssms counts a trial as *nogo* when its choice is not the largest choice
    code **or** it is omitted, so the label is the mass of every non-maximal
    choice before the deadline plus the omission probability, both as a
    fraction of the LAN's own total: ``gonogo == nogo_before / total + opn``
    row by row.

    Parameters
    ----------
    mass
        Masses for ``theta``.
    theta
        ``(n_theta, n_params)``.
    deadlines
        Scalar or ``(n_theta,)`` deadlines in seconds.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        ``data`` of shape ``(n_theta, n_params + 1)`` and ``labels`` of shape
        ``(n_theta, 1)``.
    """
    theta_arr = _as_theta(theta)
    _check_rows(mass, theta_arr)
    data, d = _deadline_rows(theta_arr, deadlines)
    return data, _labels(_nogo_before(mass, d) / _total(mass) + _omission(mass, d))


# --------------------------------------------------------------------------
# Simulation fallback
# --------------------------------------------------------------------------


def simulate_labels(
    model: str,
    theta: ArrayLike,
    network_type: str,
    deadlines: ArrayLike | None,
    n_sim: int,
    rng: np.random.Generator,
    *,
    max_t: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Labels from ssms for parameter vectors the LAN cannot be trusted on.

    The corpus falls back to this for every theta whose total mass lies
    outside the fallback window. The labels estimate the same quantities the
    renormalised LAN labels do:

    * ``cpn``: ``mean(choices == c & rts < max_t) / mean(rts < max_t)`` per
      choice — the choice frequency conditional on responding within
      ``max_t``, which is what ``mass(c) / total`` estimates (ssms does not
      censor a base model at ``max_t``; an un-terminated trial comes back at
      ``rt ≈ max_t + t`` with a choice).
    * ``opn``: ``mean(rts == -999)`` under the ``{model}_deadline`` simulator
      with the row's deadline appended to theta — the omission probability.
    * ``gonogo``: ``mean(choices != max(choices) | rts == -999)`` under the
      same simulator — ssms' ``nogo_p``.

    Parameters
    ----------
    model
        ssms base model name, e.g. ``"ddm"``; ``opn`` / ``gonogo`` simulate
        ``f"{model}_deadline"``.
    theta
        ``(n_theta, n_params)`` parameter vectors of the base model.
    network_type
        One of :data:`NETWORK_TYPES`.
    deadlines
        Scalar or ``(n_theta,)`` deadlines in seconds; required for ``opn``
        and ``gonogo``, ignored for ``cpn``.
    n_sim
        Trials simulated per parameter vector.
    rng
        Generator the simulator seeds are drawn from (one for the base
        model, one for the deadline model), so a corpus file stays a
        function of its own seed.
    max_t
        The base model's ``max_t``; the cpn labels condition on ``rts <
        max_t`` and ``past_max_t`` counts the trials at or beyond it.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``labels`` of shape ``(n_theta, n_choices)`` for ``cpn`` (choice
        order of the model config, so ``labels.reshape(-1)`` is theta-major
        like :func:`cpn_labels`) and ``(n_theta, 1)`` otherwise, and
        ``past_max_t`` of shape ``(n_theta,)``: the share of base-model
        trials with ``rt >= max_t``, the part of the truth no LAN trained
        on ``[0, max_t]`` can see.

    Raises
    ------
    ValueError
        For an unknown ``network_type``, ``n_sim < 1``, empty ``theta``,
        missing deadlines for ``opn`` / ``gonogo``, or a parameter vector
        under which no trial responds before ``max_t``.
    """
    _check_network_type(network_type)
    theta_arr = _as_theta(theta)
    n_theta = theta_arr.shape[0]
    if n_theta == 0:
        raise ValueError("theta must hold at least one parameter vector")
    if n_sim < 1:
        raise ValueError(f"n_sim must be >= 1, got {n_sim}")
    choices = np.asarray(ModelConfigBuilder.from_model(model)["choices"])
    seed_base, seed_deadline = (
        int(s) for s in rng.integers(np.iinfo(np.int32).max, size=2)
    )

    base = simulator(
        theta=theta_arr,
        model=model,
        n_samples=n_sim,
        max_t=max_t,
        random_state=seed_base,
    )
    rts = np.asarray(base["rts"]).reshape(n_sim, n_theta)
    responded = rts < max_t
    past_max_t = 1.0 - responded.mean(axis=0)

    if network_type == "cpn":
        chosen = np.asarray(base["choices"]).reshape(n_sim, n_theta)
        n_responded = responded.sum(axis=0)
        if np.any(n_responded == 0):
            bad = np.flatnonzero(n_responded == 0).tolist()
            raise ValueError(
                f"no simulated trial responds before max_t={max_t} for parameter "
                f"vectors {bad}; the conditional choice probability is undefined"
            )
        labels = np.stack(
            [((chosen == c) & responded).sum(axis=0) / n_responded for c in choices],
            axis=1,
        )
        return labels, past_max_t

    if deadlines is None:
        raise ValueError(f"{network_type} labels need deadlines")
    d = np.broadcast_to(np.asarray(deadlines, dtype=np.float32), (n_theta,))
    with_deadline = simulator(
        theta=np.concatenate([theta_arr, d[:, None]], axis=1),
        model=f"{model}_deadline",
        n_samples=n_sim,
        max_t=max_t,
        random_state=seed_deadline,
    )
    omitted = np.asarray(with_deadline["rts"]).reshape(n_sim, n_theta) == -999.0
    if network_type == "opn":
        labels = omitted.mean(axis=0)
    else:
        chosen = np.asarray(with_deadline["choices"]).reshape(n_sim, n_theta)
        labels = ((chosen != choices.max()) | omitted).mean(axis=0)
    return labels[:, None], past_max_t


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _run_uuid_from_name(path: Path) -> str | None:
    """The 32-hex ``uuid1().hex`` the trainers put in artifact names, if any.

    torch names ``{model}_{type}_{run_uuid}_model.onnx``, jax
    ``{run_uuid}_{type}_{model}__model.onnx`` (double underscore); a bare
    ``ddm.onnx`` has none.
    """
    for token in path.stem.split("_"):
        if _RUN_UUID.match(token):
            return token
    return None


@dataclass(frozen=True)
class SourceLAN:
    """Identity of the LAN a corpus was derived from.

    Build with :meth:`from_onnx`. Attributes are what a downstream consumer
    needs to find the exact network again: the file hash always, the
    trainer's run uuid and the Hub coordinates when known.

    Attributes
    ----------
    path : Path
        The ONNX artifact the masses came from.
    sha256 : str
        Hex digest of the file, computed by streaming it.
    run_uuid : str | None
        The training run's uuid, parsed from the trainer's filename
        conventions when present.
    hf_repo : str | None
        Hub repository the artifact was downloaded from, if any.
    hf_revision : str | None
        Hub revision (commit, tag, or branch) of that download, if any.
    """

    path: Path
    sha256: str
    run_uuid: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None

    @classmethod
    def from_onnx(
        cls,
        path: str | Path,
        *,
        run_uuid: str | None = None,
        hf_repo: str | None = None,
        hf_revision: str | None = None,
    ) -> SourceLAN:
        """Hash an ONNX file and record where it came from.

        Parameters
        ----------
        path
            The ONNX artifact.
        run_uuid
            Training run uuid; parsed from the filename when omitted.
        hf_repo
            Hub repository, if the artifact was downloaded.
        hf_revision
            Hub revision of that download.

        Returns
        -------
        SourceLAN
        """
        path = Path(path)
        return cls(
            path=path,
            sha256=_sha256(path),
            run_uuid=run_uuid if run_uuid is not None else _run_uuid_from_name(path),
            hf_repo=hf_repo,
            hf_revision=hf_revision,
        )

    def provenance(self, network_type: str, grid: IntegrationGrid | OnsetGrid) -> dict:
        """The flat provenance record other tools read by key.

        Parameters
        ----------
        network_type
            One of :data:`NETWORK_TYPES`; selects ``aux_category``.
        grid
            The integration grid the masses were computed on.

        Returns
        -------
        dict
            Exactly the keys ``derivation_method``, ``aux_category``,
            ``source_lan_run_uuid``, ``source_lan_sha256``,
            ``source_lan_hf_commit``, ``source_lan_run_id``,
            ``integration_grid``, ``integration_max_t``. ``source_lan_run_id``
            (an MLflow run id) is ``None`` here and is filled in by the caller
            that knows it; ``integration_grid`` is the number of grid points
            per parameter vector (see :func:`grid_description`).
        """
        _check_network_type(network_type)
        return {
            "derivation_method": DERIVATION_METHOD,
            "aux_category": AUX_CATEGORY[network_type],
            "source_lan_run_uuid": self.run_uuid,
            "source_lan_sha256": self.sha256,
            "source_lan_hf_commit": self.hf_revision,
            "source_lan_run_id": None,
            "integration_grid": grid.n_points,
            "integration_max_t": grid.max_t,
        }


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def _check_network_type(network_type: str) -> None:
    if network_type not in NETWORK_TYPES:
        raise ValueError(
            f"network_type must be one of {list(NETWORK_TYPES)}, got {network_type!r}"
        )


def _plain(config: dict) -> dict:
    """A stdlib-picklable copy of an ssms model config.

    Callables (simulator, boundary) are dropped; anything else stdlib pickle
    rejects is kept as its ``repr``. This deliberately differs from the
    trainers' ``_picklable_copy`` (which ``repr``s callables too): a derived
    corpus is written once and read many times, and a ``repr`` of a simulator
    function carries no information a consumer could use, whereas dropping it
    keeps the config a plain description of the model. Kept separate so
    ``lanfactory.derive`` does not import torch.
    """
    out: dict = {}
    for key, value in config.items():
        if callable(value):
            continue
        try:
            pickle.dumps(value)
        except Exception:  # noqa: BLE001 - keep provenance, drop the object
            out[key] = repr(value)
        else:
            out[key] = value
    return out


def grid_description(
    grid: IntegrationGrid | OnsetGrid, onset_param: str | None
) -> dict:
    """The grid as the pickles and the manifest record it.

    Parameters
    ----------
    grid
        The integration grid.
    onset_param
        The model's onset parameter when it has one — the :class:`OnsetGrid`
        is built around it, and the leak statistic is measured at it on that
        grid; ``None`` when the model has none.

    Returns
    -------
    dict
        ``kind`` (``"onset"`` / ``"uniform"``), ``onset_param``, ``n_points``
        (points per parameter vector) and the grid's own fields
        (``max_t``, ``t_min`` and, for an onset grid, its segment sizes).
    """
    return {
        "kind": "onset" if isinstance(grid, OnsetGrid) else "uniform",
        "onset_param": onset_param,
        "n_points": grid.n_points,
        **asdict(grid),
    }


def _resolve_grid(
    grid: IntegrationGrid | OnsetGrid | None,
    onset_param: str | None,
    params: Sequence[str],
    model: str,
) -> tuple[IntegrationGrid | OnsetGrid, int | None]:
    """The grid to integrate on and the column of the onset parameter.

    ``None`` picks :class:`OnsetGrid` when ``onset_param`` names a parameter
    of the model and falls back to a uniform :class:`IntegrationGrid` (with
    a warning) otherwise; an explicit :class:`OnsetGrid` requires the onset
    parameter. The onset column is returned whenever the model has one (the
    grid record names it on either grid); the leak below it is only measured
    on an :class:`OnsetGrid`, since a uniform grid's leak is mostly its own
    quadrature error.
    """
    onset_idx = (
        list(params).index(onset_param)
        if onset_param is not None and onset_param in params
        else None
    )
    if grid is None:
        if onset_idx is not None:
            return OnsetGrid(), onset_idx
        if onset_param is not None:
            logger.warning(
                "model %r has no parameter %r; integrating on the uniform %r",
                model,
                onset_param,
                IntegrationGrid(),
            )
        return IntegrationGrid(), onset_idx
    if isinstance(grid, OnsetGrid) and onset_idx is None:
        raise ValueError(
            f"an OnsetGrid needs onset_param to name a parameter of {model!r} "
            f"({list(params)}), got onset_param={onset_param!r}"
        )
    return grid, onset_idx


def _derive_stats(
    total: NDArray[np.float64],
    flagged: NDArray[np.bool_],
    past_max_t: NDArray[np.float64] | None,
    leak: NDArray[np.float64] | None,
) -> dict[str, float | None]:
    """The flat ``derive_stats`` record, for one file or a whole corpus.

    ``past_max_t`` is the per-fallback-theta share of base-model trials at or
    beyond ``max_t`` (``None`` or empty when nothing fell back); ``leak`` the
    per-theta mass below the onset (``None`` without an onset grid).
    """
    past = None if past_max_t is None or past_max_t.size == 0 else past_max_t
    return {
        "derive_total_mass_mean": float(total.mean()),
        "derive_total_mass_min": float(total.min()),
        "derive_total_mass_max": float(total.max()),
        "derive_fallback_frac": float(flagged.mean()),
        "derive_sim_past_max_t_max": None if past is None else float(past.max()),
        "derive_leak_below_onset_p99": (
            None if leak is None else float(np.percentile(leak, 99))
        ),
    }


def _flag_totals(
    total: NDArray[np.float64], window: tuple[float, float] | None
) -> NDArray[np.bool_]:
    """Which parameter vectors fall back to simulation."""
    if window is None:
        return np.zeros(total.shape[0], dtype=bool)
    lo, hi = window
    return (total < lo) | (total > hi)


def derive_aux_corpus(
    onnx_path: str | Path,
    model: str,
    network_type: str,
    out_folder: str | Path,
    *,
    n_files: int = 100,
    n_theta_per_file: int = 4096,
    onset_param: str | None = "t",
    grid: IntegrationGrid | OnsetGrid | None = None,
    deadline_quantile_frac: float = 0.7,
    fallback_window: tuple[float, float] | None = (0.98, 1.03),
    fallback_n_sim: int = 20_000,
    survey_n_theta: int = 20_000,
    seed: int = 0,
    source: SourceLAN | None = None,
) -> list[Path]:
    """Write a cpn / opn / gonogo training corpus derived from a LAN.

    Each file holds ``n_theta_per_file`` parameter vectors sampled as ssms
    would (:func:`sample_theta`), their masses under the LAN, and the rows and
    labels of :func:`cpn_labels`, :func:`opn_labels` or :func:`gonogo_labels`.
    The pickles carry ``{type}_data``, ``{type}_labels``, ``generator_config``
    and ``model_config`` — the layout ``DatasetTorch`` and the training CLIs
    read — so the result trains with the trainers unchanged. A
    :data:`MANIFEST_NAME` sidecar lists the files and the provenance.

    Labels are the LAN's masses renormalised by its own total (see the module
    notes). A parameter vector whose total lies outside ``fallback_window`` is
    one the LAN gets wrong beyond a scale error, so its labels come from
    :func:`simulate_labels` on ssms instead; the share that fell back and the
    simulation's own blind spot (trials past ``max_t``) are recorded beside
    the total-mass statistics in every pickle's
    ``generator_config["derive_stats"]`` and in the manifest.

    The manifest also carries :func:`survey` of the LAN over its whole
    training box (``lan_survey``, ``survey_n_theta`` draws), so every derived
    corpus records the mass statistics of the network it came from.

    File ``i`` is generated from ``np.random.default_rng([seed, i])``, so a
    file's content depends on ``seed`` and its index only — the simulator
    seeds of the fallback are drawn from the same generator.

    Parameters
    ----------
    onnx_path
        The LAN, a ``(1, n_params + 2)`` ONNX graph.
    model
        ssms model name the LAN was trained for, e.g. ``"ddm"``.
    network_type
        One of :data:`NETWORK_TYPES`.
    out_folder
        Destination directory; created if needed.
    n_files
        Number of pickles; at least 2, since the trainers split files into
        train and validation sets.
    n_theta_per_file
        Parameter vectors per file. Rows per file are this for opn / gonogo
        and this times the number of choices for cpn; the trainers require
        the batch size to divide the row count.
    onset_param
        Name of the model's non-decision-time parameter. Its column is the
        onset the :class:`OnsetGrid` is refined around and the edge the leak
        statistic is measured at (on that grid only). ``None``, or a name the
        model lacks, means no onset grid.
    grid
        Integration grid. ``None`` (the default) is an :class:`OnsetGrid`
        when ``onset_param`` names a parameter of ``model`` and a uniform
        :class:`IntegrationGrid` otherwise, with a logged warning — the
        uniform grid under-resolves a LAN's onset (see the grid policy in
        :mod:`.integrate`). An explicit :class:`OnsetGrid` requires
        ``onset_param``; an explicit :class:`IntegrationGrid` is used as is.
        The grid of record is written to ``generator_config["derive"]``
        (``grid`` = ``"onset"`` / ``"uniform"`` and ``grid_config``) and to
        the manifest (``grid``).
    deadline_quantile_frac
        Passed to :func:`sample_deadlines` (opn / gonogo only).
    fallback_window
        ``(lo, hi)``: a theta whose total mass is not in the open interval
        is labelled by simulation. The default flags 3.7 % of the ddm box
        on the Hub LAN and bounds the unflagged error against simulation at
        0.023 (cpn) / 0.033 (opn). ``None`` disables the fallback.
    fallback_n_sim
        Trials simulated per fallback theta.
    survey_n_theta
        Parameter vectors :func:`survey` draws for the manifest's
        ``lan_survey`` (seeded with ``seed``; about five seconds per 20 000
        on the Hub ddm LAN).
    seed
        Base seed.
    source
        Provenance of the LAN; :meth:`SourceLAN.from_onnx` on ``onnx_path``
        when omitted.

    Returns
    -------
    list[Path]
        The pickles written, in file order.

    Raises
    ------
    ValueError
        For an unknown ``network_type``, ``n_files < 2``,
        ``n_theta_per_file < 1``, a ``fallback_window`` that is not
        ``0 < lo < hi``, ``fallback_n_sim < 1``, ``survey_n_theta < 1``,
        an :class:`OnsetGrid`
        without an onset parameter, a LAN whose input width is not
        ``n_params + 2`` for ``model``, or a theta with zero total mass.
    """
    _check_network_type(network_type)
    if n_files < 2:
        raise ValueError(
            f"n_files must be >= 2 (the trainers split files into train and "
            f"validation sets), got {n_files}"
        )
    if n_theta_per_file < 1:
        raise ValueError(f"n_theta_per_file must be >= 1, got {n_theta_per_file}")
    if fallback_window is not None:
        lo, hi = (float(x) for x in fallback_window)
        if not 0.0 < lo < hi:
            raise ValueError(
                f"fallback_window must satisfy 0 < lo < hi, got {fallback_window}"
            )
        fallback_window = (lo, hi)
    if fallback_n_sim < 1:
        raise ValueError(f"fallback_n_sim must be >= 1, got {fallback_n_sim}")
    if survey_n_theta < 1:
        raise ValueError(f"survey_n_theta must be >= 1, got {survey_n_theta}")

    onnx_path = Path(onnx_path)
    out_folder = Path(out_folder)
    base_config = ModelConfigBuilder.from_model(model)
    params = list(base_config["params"])
    choices = list(base_config["choices"])
    n_params = len(params)
    grid, onset_idx = _resolve_grid(grid, onset_param, params, model)
    grid_record = grid_description(grid, onset_param if onset_idx is not None else None)

    predictor = load_onnx_predictor(onnx_path)
    if predictor.input_width != n_params + 2:
        raise ValueError(
            f"{onnx_path} takes rows of width {predictor.input_width}, but model "
            f"{model!r} has {n_params} parameters so its LAN takes rows of width "
            f"{n_params + 2} ([theta..., rt, choice]); is this the right LAN for "
            f"{model!r}?"
        )

    if network_type == "cpn":
        model_config = _plain(base_config)
        input_columns = params + ["choice"]
        deadline_bounds: tuple[float, float] | None = None
    else:
        deadline_config = ModelConfigBuilder.with_deadline(base_config)
        model_config = _plain(deadline_config)
        input_columns = list(deadline_config["params"])
        lo, hi = deadline_config["param_bounds_dict"]["deadline"]
        deadline_bounds = (float(lo), float(hi))
    model_config["input_columns"] = input_columns

    if source is None:
        source = SourceLAN.from_onnx(onnx_path)
    provenance = source.provenance(network_type, grid)
    derive_settings = {
        "grid": grid_record["kind"],
        "grid_config": grid_record,
        "integration_grid": grid.n_points,
        "integration_max_t": grid.max_t,
        "t_min": grid.t_min,
        "deadline_quantile_frac": deadline_quantile_frac,
        "deadline_bounds": deadline_bounds,
        "fallback_window": fallback_window,
        "fallback_n_sim": fallback_n_sim,
        "seed": seed,
    }

    # The whole-box statistics of the source LAN, on the grid of record; a
    # uniform grid measures no leak (survey pairs the onset with the grid).
    lan_survey = survey(
        predictor,
        base_config["param_bounds_dict"],
        params,
        choices,
        n_theta=survey_n_theta,
        seed=seed,
        onset_param=onset_param if isinstance(grid, OnsetGrid) else None,
        grid=grid,
    )

    out_folder.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    file_records: list[dict] = []
    totals: list[NDArray[np.float64]] = []
    flags: list[NDArray[np.bool_]] = []
    pasts: list[NDArray[np.float64]] = []
    leaks: list[NDArray[np.float64]] = []
    for i in range(n_files):
        rng = np.random.default_rng([seed, i])
        theta = sample_theta(base_config, n_theta_per_file, rng)
        onset = theta[:, onset_idx] if isinstance(grid, OnsetGrid) else None
        mass = choice_mass(predictor, theta, choices, grid=grid, onset=onset)
        deadlines: NDArray[np.float64] | None = None
        if network_type == "cpn":
            data, labels = cpn_labels(mass, theta, choices)
        else:
            assert deadline_bounds is not None
            deadlines = sample_deadlines(
                mass, rng, deadline_quantile_frac, bounds=deadline_bounds
            )
            build = opn_labels if network_type == "opn" else gonogo_labels
            data, labels = build(mass, theta, deadlines)

        flagged = _flag_totals(mass.total, fallback_window)
        past_max_t = None
        if flagged.any():
            simulated, past_max_t = simulate_labels(
                model,
                theta[flagged],
                network_type,
                None if deadlines is None else data[flagged, -1],
                fallback_n_sim,
                rng,
                max_t=grid.max_t,
            )
            if network_type == "cpn":
                n_choices = len(choices)
                rows = (
                    np.flatnonzero(flagged)[:, None] * n_choices + np.arange(n_choices)
                ).reshape(-1)
            else:
                rows = np.flatnonzero(flagged)
            labels[rows] = _labels(simulated.reshape(-1))
        # The leak is a property of the onset grid; a uniform grid's leak is
        # mostly quadrature error, and the survey records none there either.
        leak = None if onset is None else mass.leak_below(onset)

        stats = _derive_stats(mass.total, flagged, past_max_t, leak)
        generator_config = {
            "generator_approach": "derived",
            "model": model,
            "network_type": network_type,
            "n_files": n_files,
            "n_theta_per_file": n_theta_per_file,
            "file_index": i,
            "derive": derive_settings,
            "source": provenance,
            "derive_stats": stats,
        }
        path = out_folder / f"training_data_{i:05d}.pickle"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    f"{network_type}_data": data,
                    f"{network_type}_labels": labels,
                    "generator_config": generator_config,
                    "model_config": model_config,
                },
                f,
                protocol=4,
            )
        files.append(path)
        file_records.append({"file": path.name, "n_rows": int(data.shape[0]), **stats})
        totals.append(mass.total)
        flags.append(flagged)
        if past_max_t is not None:
            pasts.append(past_max_t)
        if leak is not None:
            leaks.append(leak)

    manifest = {
        "lanfactory_version": __version__,
        "model": model,
        "network_type": network_type,
        "aux_category": AUX_CATEGORY[network_type],
        "n_files": n_files,
        "n_theta_per_file": n_theta_per_file,
        "n_rows_per_file": int(data.shape[0]),
        "input_columns": input_columns,
        "choices": choices,
        "seed": seed,
        "grid": grid_record,
        "deadline_quantile_frac": deadline_quantile_frac,
        "deadline_bounds": deadline_bounds,
        "fallback_window": fallback_window,
        "fallback_n_sim": fallback_n_sim,
        "survey_n_theta": survey_n_theta,
        "source": {
            **{
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(source).items()
            },
            **provenance,
        },
        "derive_stats": _derive_stats(
            np.concatenate(totals),
            np.concatenate(flags),
            np.concatenate(pasts) if pasts else None,
            np.concatenate(leaks) if leaks else None,
        ),
        "lan_survey": lan_survey,
        "files": file_records,
    }
    with open(out_folder / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return files
