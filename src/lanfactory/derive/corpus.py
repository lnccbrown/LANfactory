"""Turn a LAN's integrated masses into cpn / opn training corpora.

Row layouts
-----------
Every corpus has ``n_params + 1`` input columns, in ssms parameter order:

=========  =====================================  ==============================
type       training row                           label
=========  =====================================  ==============================
cpn        ``[theta..., choice]`` — one row per   ``by_choice / total``
           choice code in ``model_config["choices"]``
opn        ``[theta..., deadline]``               ``1 - F(deadline) / total``
=========  =====================================  ==============================

with ``by_choice``, ``total`` and ``F = mass_before`` from
:class:`~lanfactory.derive.ChoiceMass`.
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

from .integrate import ChoiceMass, OnsetGrid, choice_mass, load_onnx_predictor

NETWORK_TYPES: tuple[str, ...] = ("cpn", "opn")
"""Auxiliary network types a corpus can be derived for."""

AUX_CATEGORY: dict[str, str] = {"cpn": "choice", "opn": "omission"}
"""What each network type's label is the probability of."""

DERIVATION_METHOD = "derived-from-lan"
"""Value of the ``derivation_method`` provenance key."""

MANIFEST_NAME = "derive_manifest.json"
"""Sidecar written next to the pickles by :func:`derive_aux_corpus`."""

GRID = OnsetGrid()
"""The integration grid of every corpus, refined around the model's ``t``."""

FALLBACK_WINDOW = (0.98, 1.03)
"""A theta whose LAN total lies outside ``[lo, hi]``, or is NaN, is labelled by
simulation; the network types page gives the measurements behind it."""

DEADLINE_QUANTILE_FRAC = 0.7
"""Share of opn deadlines drawn from the LAN's own reaction-time quantiles."""

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
    sampler = UniformParameterSampler(
        param_space={name: bounds[name] for name in params},
        constraints=ModelConfigBuilder.get_sampling_transforms(model_config),
    )
    samples = sampler.sample(n_samples=n, rng=rng)
    columns = [np.asarray(samples[name]).reshape(n) for name in params]
    return np.column_stack(columns).astype(np.float32)


def sample_deadlines(
    mass: ChoiceMass,
    rng: np.random.Generator,
    quantile_frac: float,
    bounds: tuple[float, float],
) -> NDArray[np.float64]:
    """Draw one training deadline per parameter vector.

    A fraction ``1 - quantile_frac`` of the deadlines is uniform on ``bounds``
    (ssms' deadline box, so the corpus covers it); the rest are the LAN's own
    reaction-time quantiles under each theta — :meth:`ChoiceMass.quantile` at
    ``u ~ U(0.05, 0.95)`` — so the deadlines concentrate where the omission
    probability is informative. A theta whose LAN has no mass to invert (zero
    or NaN total) draws from the uniform share. Everything is clipped to
    ``bounds``.

    Parameters
    ----------
    mass
        Masses for the thetas the deadlines are for; one deadline per row.
    rng
        Generator for both shares.
    quantile_frac
        Share drawn from the LAN's quantiles, in ``[0, 1]``.
    bounds
        ``(lower, upper)`` deadline bounds.

    Returns
    -------
    NDArray[np.float64]
        ``(n_theta,)`` deadlines in seconds.
    """
    lower, upper = bounds
    n_theta = mass.cdf.shape[0]
    from_quantile = rng.random(n_theta) < quantile_frac
    uniform = rng.uniform(lower, upper, size=n_theta)
    quantile = mass.quantile(rng.uniform(0.05, 0.95, size=n_theta))
    from_quantile &= ~np.isnan(quantile)
    return np.clip(np.where(from_quantile, quantile, uniform), lower, upper)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def _labels(values: NDArray) -> NDArray[np.float32]:
    """``(n_rows, 1)`` float32 probabilities, clipped to ``[0, 1]``."""
    return np.clip(values, 0.0, 1.0).astype(np.float32).reshape(-1, 1)


def cpn_labels(
    mass: ChoiceMass,
    theta: NDArray[np.float32],
    choices: Sequence[float] | NDArray,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """CPN rows ``[theta..., choice]`` for every choice, labelled ``by_choice / total``.

    Rows are theta-major: all choices of ``theta[0]``, then of ``theta[1]``, …
    The labels of one theta sum to one; a zero or NaN total gives NaN labels.

    Parameters
    ----------
    mass
        Masses for ``theta``, integrated over ``choices``.
    theta
        ``(n_theta, n_params)``.
    choices
        The choice codes ``mass`` was integrated over, in the same order.

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
    return data, _labels(mass.by_choice / mass.total[:, None])


def opn_labels(
    mass: ChoiceMass,
    theta: NDArray[np.float32],
    deadlines: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """OPN rows ``[theta..., deadline]`` labelled ``1 - F(deadline) / total``.

    The label is the probability of *not* responding before the deadline —
    the omission probability ssms marks with ``rt == -999`` — as a fraction
    of the LAN's own total mass; a zero or NaN total gives a NaN label.

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
    d32 = np.broadcast_to(np.asarray(deadlines, dtype=np.float32), (theta.shape[0],))
    data = np.column_stack([theta, d32])
    # The label is computed at the float32 deadline the row carries, so row
    # and label describe exactly the same deadline.
    d = d32.astype(np.float64)
    return data, _labels(1.0 - mass.mass_before(d) / mass.total)


# --------------------------------------------------------------------------
# Simulation fallback
# --------------------------------------------------------------------------


def simulate_labels(
    model: str,
    theta: NDArray[np.float32],
    deadlines: ArrayLike | None,
    n_sim: int,
    rng: np.random.Generator,
    *,
    max_t: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Labels from ssms for parameter vectors the LAN cannot be trusted on.

    The corpus falls back to this for every theta whose total mass lies
    outside :data:`FALLBACK_WINDOW`. One base-model run per theta gives
    either label:

    * ``cpn`` (``deadlines is None``): ``mean(choices == c & responded) /
      mean(responded)`` per choice, ``responded = (rts >= 0) & (rts <
      max_t)`` — the choice frequency conditional on responding within
      ``max_t``, which is what ``by_choice / total`` estimates. A trial still
      running at ``max_t`` comes back either at ``rt ≈ max_t + t`` with a
      choice (ddm) or censored as ``rt == -999`` (racing_diffusion_3,
      poisson_race, addm); both count as past ``max_t``.
    * ``opn``: the share of trials that do not respond before the row's
      deadline, what ssms' ``{model}_deadline`` simulators mark
      ``rt == -999``. Unlike the renormalised LAN label it is not conditional
      on responding before ``max_t``.

    Parameters
    ----------
    model
        ssms base model name, e.g. ``"ddm"``.
    theta
        ``(n_theta, n_params)`` parameter vectors of the base model.
    deadlines
        Scalar or ``(n_theta,)`` deadlines in seconds for ``opn``; ``None``
        for ``cpn``.
    n_sim
        Trials simulated per parameter vector.
    rng
        Generator the simulator seed is drawn from, so a corpus file stays a
        function of its own seed.
    max_t
        The base model's ``max_t``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``labels`` of shape ``(n_theta, n_choices)`` for ``cpn`` (choice
        order of the model config, so ``labels.reshape(-1)`` is theta-major
        like :func:`cpn_labels`) and ``(n_theta, 1)`` for ``opn``, and
        ``past_max_t`` of shape ``(n_theta,)``: the share of trials that do
        not respond before ``max_t``, the part of the truth no LAN trained on
        ``[0, max_t]`` can see.

    Raises
    ------
    ValueError
        For ``cpn``, if no trial responds before ``max_t`` under some
        parameter vector.
    """
    n_theta = theta.shape[0]
    out = simulator(
        theta=theta,
        model=model,
        n_samples=n_sim,
        max_t=max_t,
        random_state=int(rng.integers(np.iinfo(np.int32).max)),
    )
    rts = np.asarray(out["rts"]).reshape(n_sim, n_theta)
    responded = (rts >= 0) & (rts < max_t)
    past_max_t = 1.0 - responded.mean(axis=0)

    if deadlines is not None:
        d = np.broadcast_to(np.asarray(deadlines, dtype=np.float32), (n_theta,))
        before = (rts >= 0) & (rts < d)
        return 1.0 - before.mean(axis=0)[:, None], past_max_t

    chosen = np.asarray(out["choices"]).reshape(n_sim, n_theta)
    n_responded = responded.sum(axis=0)
    if np.any(n_responded == 0):
        bad = np.flatnonzero(n_responded == 0).tolist()
        raise ValueError(
            f"no simulated trial responds before max_t={max_t} for parameter "
            f"vectors {bad}; the conditional choice probability is undefined"
        )
    choices = ModelConfigBuilder.from_model(model)["choices"]
    labels = np.stack(
        [((chosen == c) & responded).sum(axis=0) / n_responded for c in choices],
        axis=1,
    )
    return labels, past_max_t


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


def _plain(config: dict) -> dict:
    """An ssms model config without callables, and with anything else stdlib
    pickle rejects kept as its ``repr``.
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
    """The corpus-wide ``derive_stats`` record.

    ``past_max_t`` is the per-fallback-theta share of simulated trials at or
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


def derive_aux_corpus(
    onnx_path: str | Path,
    model: str,
    network_type: str,
    out_folder: str | Path,
    *,
    n_files: int = 100,
    n_theta_per_file: int = 4096,
    fallback_n_sim: int = 20_000,
    seed: int = 0,
    source_run_uuid: str | None = None,
    source_hf_revision: str | None = None,
) -> list[Path]:
    """Write a cpn / opn training corpus derived from a LAN.

    Each file holds ``n_theta_per_file`` parameter vectors sampled as ssms
    would (:func:`sample_theta`), their masses under the LAN on :data:`GRID`,
    and the rows and labels of :func:`cpn_labels` or :func:`opn_labels`. The
    pickles carry ``{type}_data``, ``{type}_labels``, ``generator_config``
    and ``model_config`` — the layout ``DatasetTorch`` and the training CLIs
    read — so the result trains with the trainers unchanged.

    Labels are the LAN's masses renormalised by its own total. A parameter
    vector whose total lies outside :data:`FALLBACK_WINDOW` is one the LAN
    gets wrong beyond a scale error, so its labels come from
    :func:`simulate_labels` on ssms instead. opn deadlines are drawn by
    :func:`sample_deadlines` with :data:`DEADLINE_QUANTILE_FRAC` on ssms'
    deadline box.

    Every pickle's ``generator_config`` is one corpus record (model, type,
    sizes, settings under ``derive``, provenance under ``source``) plus its
    ``file_index``. The :data:`MANIFEST_NAME` sidecar is that record plus the
    corpus-wide ``derive_stats`` (total-mass statistics, the share that fell
    back, and the simulation's blind spot past ``max_t``) and the file names.

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
        Destination directory; created when the first file is written.
    n_files
        Number of pickles; at least 2, since the trainers split files into
        train and validation sets.
    n_theta_per_file
        Parameter vectors per file. Rows per file are this for opn
        and this times the number of choices for cpn; the trainers require
        the batch size to divide the row count.
    fallback_n_sim
        Trials simulated per fallback theta.
    seed
        Base seed.
    source_run_uuid
        Training run uuid of the LAN; parsed from the trainer's file name
        conventions when omitted.
    source_hf_revision
        Hub revision (commit, tag, or branch) the LAN was downloaded at, if
        any; recorded as ``source_lan_hf_commit``.

    Returns
    -------
    list[Path]
        The pickles written, in file order.

    Raises
    ------
    ValueError
        For an unknown ``network_type``, ``n_files < 2``,
        ``n_theta_per_file < 1``, ``fallback_n_sim < 1``, a model without a
        non-decision time ``t``, or a LAN whose input width is not
        ``n_params + 2`` for ``model``.
    """
    if network_type not in NETWORK_TYPES:
        raise ValueError(
            f"network_type must be one of {list(NETWORK_TYPES)}, got {network_type!r}"
        )
    if n_files < 2:
        raise ValueError(
            f"n_files must be >= 2 (the trainers split files into train and "
            f"validation sets), got {n_files}"
        )
    if n_theta_per_file < 1:
        raise ValueError(f"n_theta_per_file must be >= 1, got {n_theta_per_file}")
    if fallback_n_sim < 1:
        raise ValueError(f"fallback_n_sim must be >= 1, got {fallback_n_sim}")

    onnx_path = Path(onnx_path)
    out_folder = Path(out_folder)
    base_config = ModelConfigBuilder.from_model(model)
    params = list(base_config["params"])
    choices = list(base_config["choices"])
    if "t" not in params:
        raise ValueError(
            f"model {model!r} has no non-decision time 't' (params {params}); "
            "the integration grid is refined around it"
        )
    onset_idx = params.index("t")
    predictor = load_onnx_predictor(onnx_path)

    if network_type == "cpn":
        model_config = _plain(base_config)
        input_columns = params + ["choice"]
    else:
        deadline_config = ModelConfigBuilder.with_deadline(base_config)
        model_config = _plain(deadline_config)
        input_columns = list(deadline_config["params"])
        deadline_bounds = deadline_config["param_bounds_dict"]["deadline"]
    model_config["input_columns"] = input_columns

    record = {
        "generator_approach": "derived",
        "model": model,
        "network_type": network_type,
        "n_files": n_files,
        "n_theta_per_file": n_theta_per_file,
        "derive": {
            "grid": asdict(GRID),
            "deadline_quantile_frac": DEADLINE_QUANTILE_FRAC,
            "fallback_window": FALLBACK_WINDOW,
            "fallback_n_sim": fallback_n_sim,
            "seed": seed,
        },
        "source": _provenance(
            onnx_path, network_type, GRID, source_run_uuid, source_hf_revision
        ),
    }

    lo, hi = FALLBACK_WINDOW
    files: list[Path] = []
    totals: list[NDArray[np.float64]] = []
    flags: list[NDArray[np.bool_]] = []
    pasts: list[NDArray[np.float64]] = []
    leaks: list[NDArray[np.float64]] = []
    for i in range(n_files):
        rng = np.random.default_rng([seed, i])
        theta = sample_theta(base_config, n_theta_per_file, rng)
        onset = theta[:, onset_idx]
        mass = choice_mass(predictor, theta, choices, onset, GRID)
        # Outside the window, a zero or NaN total among them, the LAN is not
        # trusted: those rows get placeholder labels, simulated below.
        flagged = ~((mass.total >= lo) & (mass.total <= hi))
        with np.errstate(divide="ignore", invalid="ignore"):
            if network_type == "cpn":
                data, labels = cpn_labels(mass, theta, choices)
            else:
                deadlines = sample_deadlines(
                    mass, rng, DEADLINE_QUANTILE_FRAC, deadline_bounds
                )
                data, labels = opn_labels(mass, theta, deadlines)

        if flagged.any():
            simulated, past_max_t = simulate_labels(
                model,
                theta[flagged],
                None if network_type == "cpn" else data[flagged, -1],
                fallback_n_sim,
                rng,
                max_t=GRID.max_t,
            )
            # A view: the rows of one theta are contiguous (theta-major).
            labels.reshape(n_theta_per_file, -1)[flagged] = np.clip(simulated, 0, 1)
            pasts.append(past_max_t)

        path = out_folder / f"training_data_{i:05d}.pickle"
        out_folder.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    f"{network_type}_data": data,
                    f"{network_type}_labels": labels,
                    "generator_config": {**record, "file_index": i},
                    "model_config": model_config,
                },
                f,
                protocol=4,
            )
        files.append(path)
        totals.append(mass.total)
        flags.append(flagged)
        leaks.append(mass.mass_before(onset))

    manifest = {
        "lanfactory_version": __version__,
        **record,
        "derive_stats": _derive_stats(
            np.concatenate(totals),
            np.concatenate(flags),
            np.concatenate(pasts) if pasts else None,
            np.concatenate(leaks),
        ),
        "files": [path.name for path in files],
    }
    with open(out_folder / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return files
