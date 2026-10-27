"""Turn a LAN's per-choice masses into cpn / opn training corpora.

Row layouts
-----------
Every corpus has ``n_params + 1`` input columns, in ssms parameter order:

=========  =====================================  ==============================
type       training row                           label
=========  =====================================  ==============================
cpn        ``[theta..., choice]`` — one row per   ``mass(choice) / total``
           choice code in ``model_config["choices"]``
opn        ``[theta..., deadline]``               ``1 - F(deadline) / total``
=========  =====================================  ==============================

with ``total`` the LAN's mass on ``[t_min, max_t]`` summed over choices and
``F(d) = mass_before(d)`` summed over choices.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler

from lanfactory import __version__

from .integrate import (
    ChoiceMass,
    OnsetGrid,
    choice_mass,
    load_onnx_predictor,
)

NETWORK_TYPES: tuple[str, ...] = ("cpn", "opn")
"""Auxiliary network types a corpus can be derived for."""

AUX_CATEGORY: dict[str, str] = {"cpn": "choice", "opn": "omission"}
"""What each network type's label is the probability of."""

DERIVATION_METHOD = "derived-from-lan"
"""Value of the ``derivation_method`` provenance key."""

MANIFEST_NAME = "derive_manifest.json"
"""Sidecar written next to the pickles by :func:`derive_aux_corpus`."""

_RUN_UUID = re.compile(r"^[0-9a-f]{32}$")


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample_theta(
    model_config: dict, n: int, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Sample parameter vectors exactly as ssms samples a training corpus.

    Uses ssms' ``UniformParameterSampler`` on ``model_config["param_bounds_dict"]``
    with the model's sampling transforms (``ModelConfigBuilder
    .get_sampling_transforms``), which is what ssms' simulation pipeline
    builds. The bounds are declared in ``model_config["params"]`` order, which
    ssms draws in, so the same ``rng`` state gives the same thetas in every
    process. Columns follow ``model_config["params"]``.

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
    sampler = UniformParameterSampler(
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


def _labels(values: NDArray) -> NDArray[np.float32]:
    """``(n_rows, 1)`` float32 probabilities, clipped to ``[0, 1]``."""
    return np.clip(values, 0.0, 1.0).astype(np.float32).reshape(-1, 1)


def _total(mass: ChoiceMass, flagged: NDArray[np.bool_] | None) -> NDArray[np.float64]:
    """The per-theta divisor of the labels: the LAN's total, 1 where ``flagged``.

    Flagged rows are relabelled by simulation, so only the others need a
    positive total; a NaN total is refused like a zero one.
    """
    total = mass.total
    trusted = np.ones(total.shape, dtype=bool) if flagged is None else ~flagged
    bad = np.flatnonzero(trusted & ~(total > 0.0))
    if bad.size:
        raise ValueError(
            f"total mass must be > 0 to renormalise labels; parameter vectors "
            f"{bad.tolist()} have total {total[bad].tolist()}"
        )
    return np.where(trusted, total, 1.0)


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
    mass: ChoiceMass,
    theta: NDArray[np.float32],
    choices: Sequence[float] | NDArray,
    flagged: NDArray[np.bool_] | None = None,
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
    flagged
        ``(n_theta,)``: thetas that fall back to simulation, whose labels are
        placeholders for the caller to overwrite; ``None`` flags none.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        ``data`` of shape ``(n_theta * n_choices, n_params + 1)`` and
        ``labels`` of shape ``(n_theta * n_choices, 1)``.
    """
    choice_arr = np.asarray(choices, dtype=np.float32).reshape(-1)
    n_theta, n_choices = theta.shape[0], choice_arr.shape[0]
    data = np.concatenate(
        [
            np.repeat(theta, n_choices, axis=0),
            np.tile(choice_arr, n_theta)[:, None],
        ],
        axis=1,
    )
    # Look the masses up by the original codes (choice_arr is the float32 the
    # row carries); stacked along axis 1 so the flattened labels are theta-major.
    per_choice = np.stack(
        [mass.mass(float(c)) for c in np.asarray(choices).reshape(-1)], axis=1
    )
    total = _total(mass, flagged)
    return data, _labels((per_choice / total[:, None]).reshape(-1))


def opn_labels(
    mass: ChoiceMass,
    theta: NDArray[np.float32],
    deadlines: ArrayLike,
    flagged: NDArray[np.bool_] | None = None,
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
    flagged
        As for :func:`cpn_labels`.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        ``data`` of shape ``(n_theta, n_params + 1)`` and ``labels`` of shape
        ``(n_theta, 1)``.
    """
    data, d = _deadline_rows(theta, deadlines)
    return data, _labels(1.0 - mass.mass_before(d) / _total(mass, flagged))


# --------------------------------------------------------------------------
# Simulation fallback
# --------------------------------------------------------------------------


def simulate_labels(
    model: str,
    theta: NDArray[np.float32],
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

    * ``cpn``: ``mean(choices == c & responded) / mean(responded)`` per
      choice, ``responded = (rts >= 0) & (rts < max_t)`` — the choice
      frequency conditional on responding within ``max_t``, which is what
      ``mass(c) / total`` estimates. A trial still running at ``max_t``
      comes back either at ``rt ≈ max_t + t`` with a choice (ddm) or
      censored as ``rt == -999`` (racing_diffusion_3, poisson_race, addm);
      both count as past ``max_t``.
    * ``opn``: ``mean(rts == -999)`` under the ``{model}_deadline`` simulator
      with the row's deadline appended to theta — the omission probability.

    Parameters
    ----------
    model
        ssms base model name, e.g. ``"ddm"``; ``opn`` simulates
        ``f"{model}_deadline"``.
    theta
        ``(n_theta, n_params)`` parameter vectors of the base model.
    network_type
        One of :data:`NETWORK_TYPES`.
    deadlines
        Scalar or ``(n_theta,)`` deadlines in seconds; required for ``opn``,
        ignored for ``cpn``.
    n_sim
        Trials simulated per parameter vector.
    rng
        Generator the simulator seeds are drawn from (one for the base
        model, one for the deadline model), so a corpus file stays a
        function of its own seed.
    max_t
        The base model's ``max_t``; the cpn labels condition on responding
        before it and ``past_max_t`` counts the trials that do not.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``labels`` of shape ``(n_theta, n_choices)`` for ``cpn`` (choice
        order of the model config, so ``labels.reshape(-1)`` is theta-major
        like :func:`cpn_labels`) and ``(n_theta, 1)`` otherwise, and
        ``past_max_t`` of shape ``(n_theta,)``: the share of base-model
        trials that do not respond before ``max_t``, the part of the truth no
        LAN trained on ``[0, max_t]`` can see.

    Raises
    ------
    ValueError
        For an unknown ``network_type``, ``n_sim < 1``, empty ``theta``,
        missing deadlines for ``opn``, or a parameter vector
        under which no trial responds before ``max_t``.
    """
    _check_network_type(network_type)
    n_theta = theta.shape[0]
    if n_theta == 0:
        raise ValueError("theta must hold at least one parameter vector")
    if n_sim < 1:
        raise ValueError(f"n_sim must be >= 1, got {n_sim}")
    choices = np.asarray(ModelConfigBuilder.from_model(model)["choices"])
    seed_base, seed_deadline = (
        int(s) for s in rng.integers(np.iinfo(np.int32).max, size=2)
    )

    base = simulator(
        theta=theta,
        model=model,
        n_samples=n_sim,
        max_t=max_t,
        random_state=seed_base,
    )
    rts = np.asarray(base["rts"]).reshape(n_sim, n_theta)
    responded = (rts >= 0) & (rts < max_t)
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
        theta=np.concatenate([theta, d[:, None]], axis=1),
        model=f"{model}_deadline",
        n_samples=n_sim,
        max_t=max_t,
        random_state=seed_deadline,
    )
    omitted = np.asarray(with_deadline["rts"]).reshape(n_sim, n_theta) == -999.0
    return omitted.mean(axis=0)[:, None], past_max_t


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


def _provenance(
    onnx_path: Path,
    network_type: str,
    grid: OnsetGrid,
    run_uuid: str | None,
    hf_revision: str | None,
) -> dict:
    """The flat provenance record downstream tools read by key.

    ``source_lan_run_uuid`` is ``run_uuid``, parsed from the trainer's file
    name when ``None``; ``integration_grid`` is the number of grid points per
    parameter vector.
    """
    return {
        "derivation_method": DERIVATION_METHOD,
        "aux_category": AUX_CATEGORY[network_type],
        "source_lan_run_uuid": (
            run_uuid if run_uuid is not None else _run_uuid_from_name(onnx_path)
        ),
        "source_lan_sha256": _sha256(onnx_path),
        "source_lan_hf_commit": hf_revision,
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
        except Exception:  # keep provenance, drop the object
            out[key] = repr(value)
        else:
            out[key] = value
    return out


def _derive_stats(
    total: NDArray[np.float64],
    flagged: NDArray[np.bool_],
    past_max_t: NDArray[np.float64] | None,
    leak: NDArray[np.float64],
) -> dict[str, float | None]:
    """The flat ``derive_stats`` record, for one file or a whole corpus.

    ``past_max_t`` is the per-fallback-theta share of base-model trials at or
    beyond ``max_t`` (``None`` when nothing fell back); ``leak`` the
    per-theta mass below the onset. A NaN total, which always falls back, is
    left out of the mass statistics so the record stays valid JSON.
    """
    past = None if past_max_t is None else float(past_max_t.max())
    return {
        "derive_total_mass_mean": float(np.nanmean(total)),
        "derive_total_mass_min": float(np.nanmin(total)),
        "derive_total_mass_max": float(np.nanmax(total)),
        "derive_fallback_frac": float(flagged.mean()),
        "derive_sim_past_max_t_max": past,
        "derive_leak_below_onset_p99": float(np.nanpercentile(leak, 99)),
    }


def _flag_totals(
    total: NDArray[np.float64], window: tuple[float, float] | None
) -> NDArray[np.bool_]:
    """Which parameter vectors fall back to simulation; a NaN total does."""
    if window is None:
        return np.zeros(total.shape[0], dtype=bool)
    lo, hi = window
    return ~((total >= lo) & (total <= hi))


def derive_aux_corpus(
    onnx_path: str | Path,
    model: str,
    network_type: str,
    out_folder: str | Path,
    *,
    n_files: int = 100,
    n_theta_per_file: int = 4096,
    grid: OnsetGrid = OnsetGrid(),  # noqa: B008 - frozen, so never mutated
    deadline_quantile_frac: float = 0.7,
    fallback_window: tuple[float, float] | None = (0.98, 1.03),
    fallback_n_sim: int = 20_000,
    seed: int = 0,
    source_run_uuid: str | None = None,
    source_hf_repo: str | None = None,
    source_hf_revision: str | None = None,
) -> list[Path]:
    """Write a cpn / opn training corpus derived from a LAN.

    Each file holds ``n_theta_per_file`` parameter vectors sampled as ssms
    would (:func:`sample_theta`), their masses under the LAN, and the rows and
    labels of :func:`cpn_labels` or :func:`opn_labels`.
    The pickles carry ``{type}_data``, ``{type}_labels``, ``generator_config``
    and ``model_config`` — the layout ``DatasetTorch`` and the training CLIs
    read — so the result trains with the trainers unchanged. A
    :data:`MANIFEST_NAME` sidecar lists the files and the provenance.

    Labels are the LAN's masses renormalised by its own total. A parameter
    vector whose total lies outside ``fallback_window`` is one the LAN gets
    wrong beyond a scale error, so its labels come from
    :func:`simulate_labels` on ssms instead; the share that fell back and the
    simulation's own blind spot (trials past ``max_t``) are recorded beside
    the total-mass statistics in every pickle's
    ``generator_config["derive_stats"]`` and in the manifest.

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
        Parameter vectors per file. Rows per file are this for opn
        and this times the number of choices for cpn; the trainers require
        the batch size to divide the row count.
    grid
        Integration grid, refined per parameter vector around the model's
        non-decision time ``t``, the edge the leak statistic is measured at.
        It is recorded as ``grid`` in ``generator_config["derive"]`` and in
        the manifest.
    deadline_quantile_frac
        Passed to :func:`sample_deadlines` (opn only).
    fallback_window
        ``(lo, hi)``: a theta whose total mass lies outside ``[lo, hi]`` (the
        endpoints are trusted), or is NaN, is labelled by simulation. The
        default flags 3.7 % of the ddm box on the Hub LAN and bounds the
        unflagged error against simulation at 0.023 (cpn) / 0.033 (opn).
        ``None`` disables the fallback.
    fallback_n_sim
        Trials simulated per fallback theta.
    seed
        Base seed.
    source_run_uuid
        Training run uuid of the LAN; parsed from the trainer's file name
        conventions when omitted.
    source_hf_repo
        Hub repository the LAN was downloaded from, if any.
    source_hf_revision
        Hub revision (commit, tag, or branch) of that download, if any;
        recorded as ``source_lan_hf_commit``.

    Returns
    -------
    list[Path]
        The pickles written, in file order.

    Raises
    ------
    ValueError
        For an unknown ``network_type``, ``n_files < 2``,
        ``n_theta_per_file < 1``, a ``fallback_window`` that is not
        ``0 < lo < hi``, ``fallback_n_sim < 1``, a ``deadline_quantile_frac``
        outside ``[0, 1]``, a model without a non-decision time ``t``, a LAN whose input width is
        not ``n_params + 2`` for ``model``, or, with ``fallback_window=None``,
        a theta whose total mass is not positive.
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
    if not 0.0 <= deadline_quantile_frac <= 1.0:
        raise ValueError(
            f"deadline_quantile_frac must lie in [0, 1], got {deadline_quantile_frac}"
        )

    onnx_path = Path(onnx_path)
    out_folder = Path(out_folder)
    base_config = ModelConfigBuilder.from_model(model)
    params = list(base_config["params"])
    choices = list(base_config["choices"])
    n_params = len(params)
    if "t" not in params:
        raise ValueError(
            f"model {model!r} has no non-decision time 't' (params {params}); "
            "the integration grid is refined around it"
        )
    onset_idx = params.index("t")
    grid_record = {"onset_param": "t", "n_points": grid.n_points, **asdict(grid)}

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

    provenance = _provenance(
        onnx_path, network_type, grid, source_run_uuid, source_hf_revision
    )
    derive_settings = {
        "grid": grid_record,
        "deadline_quantile_frac": deadline_quantile_frac,
        "deadline_bounds": deadline_bounds,
        "fallback_window": fallback_window,
        "fallback_n_sim": fallback_n_sim,
        "seed": seed,
    }

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
        onset = theta[:, onset_idx]
        mass = choice_mass(predictor, theta, choices, grid=grid, onset=onset)
        # Flag before renormalising: a theta outside the window (a zero or NaN
        # total among them) gets placeholder labels, simulated below.
        flagged = _flag_totals(mass.total, fallback_window)
        deadlines: NDArray[np.float64] | None = None
        if network_type == "cpn":
            data, labels = cpn_labels(mass, theta, choices, flagged)
        else:
            assert deadline_bounds is not None
            deadlines = sample_deadlines(
                mass, rng, deadline_quantile_frac, bounds=deadline_bounds
            )
            data, labels = opn_labels(mass, theta, deadlines, flagged)

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
        leak = mass.mass_before(onset)

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
        "source": {"path": str(onnx_path), "hf_repo": source_hf_repo, **provenance},
        "derive_stats": _derive_stats(
            np.concatenate(totals),
            np.concatenate(flags),
            np.concatenate(pasts) if pasts else None,
            np.concatenate(leaks),
        ),
        "files": file_records,
    }
    with open(out_folder / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return files
