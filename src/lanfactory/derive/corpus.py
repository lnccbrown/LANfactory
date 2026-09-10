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
cpn        ``[theta..., choice]`` — one row per   ``mass(choice)``
           choice code in ``model_config["choices"]``
opn        ``[theta..., deadline]``               ``1 - mass_before(deadline)``
gonogo     ``[theta..., deadline]``               ``mass_before(deadline, nogo)``
                                                  ``+ (1 - mass_before(deadline))``
=========  =====================================  ==============================

The gonogo label follows ssms' own ``nogo_p``: a trial is *nogo* when its
choice is anything but the largest choice code, or when it is omitted.

Labels are probabilities for a BCE-with-logits consumer and are clipped to
``[0, 1]``: the LAN's approximation error can put its total mass a few
thousandths above one. The omission term ``1 - mass_before(deadline)`` is
clipped *once*, before it enters either label, so the decomposition
``gonogo_label == mass_before(deadline, nogo) + opn_label`` holds in the
written corpus. Clipping is not renormalisation — the per-corpus total-mass
statistics recorded in every pickle and in the manifest keep the deficit (or
excess) visible; see the tail policy in :mod:`.integrate`.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ssms.config import ModelConfigBuilder
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler

from lanfactory import __version__

from .integrate import ChoiceMass, IntegrationGrid, choice_mass, load_onnx_predictor

__all__ = [
    "AUX_CATEGORY",
    "DERIVATION_METHOD",
    "MANIFEST_NAME",
    "NETWORK_TYPES",
    "SourceLAN",
    "cpn_labels",
    "derive_aux_corpus",
    "gonogo_labels",
    "opn_labels",
    "sample_deadlines",
    "sample_theta",
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


def sample_theta(
    model_config: dict, n: int, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Sample parameter vectors exactly as ssms samples a training corpus.

    Uses ssms' ``UniformParameterSampler`` on ``model_config["param_bounds_dict"]``
    with the model's sampling transforms (``ModelConfigBuilder
    .get_sampling_transforms``), which is what ssms' simulation pipeline
    builds. Columns follow ``model_config["params"]``.

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
    sampler = UniformParameterSampler(
        param_space=model_config["param_bounds_dict"],
        constraints=ModelConfigBuilder.get_sampling_transforms(model_config),
    )
    samples = sampler.sample(n_samples=n, rng=rng)
    columns = [np.asarray(samples[name]).reshape(n) for name in model_config["params"]]
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


def _omission(mass: ChoiceMass, deadline: NDArray[np.float64]) -> NDArray[np.float64]:
    """``P(no response before deadline)`` as a float64 probability.

    ``1 - mass_before(deadline)`` summed over choices, clipped to ``[0, 1]``
    once so that the same term enters :func:`opn_labels` and
    :func:`gonogo_labels` (a LAN whose total exceeds one would otherwise give
    a negative omission mass).
    """
    return np.clip(1.0 - mass.mass_before(deadline), 0.0, 1.0)


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
    """CPN rows ``[theta..., choice]`` for every choice, labelled ``mass(choice)``.

    Rows are theta-major: all choices of ``theta[0]``, then of ``theta[1]``, …

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
    return data, _labels(per_choice.reshape(-1))


def opn_labels(
    mass: ChoiceMass, theta: ArrayLike, deadlines: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """OPN rows ``[theta..., deadline]`` labelled ``1 - mass_before(deadline)``.

    The label is the probability of *not* responding before the deadline,
    summed over choices — the omission probability ssms marks with
    ``rt == -999``.

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
    choice before the deadline plus the omission probability.

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
    return data, _labels(_nogo_before(mass, d) + _omission(mass, d))


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

    def provenance(self, network_type: str, grid: IntegrationGrid) -> dict:
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
            that knows it.
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


def _mass_stats(total: NDArray[np.float64]) -> dict[str, float]:
    return {
        "derive_total_mass_mean": float(total.mean()),
        "derive_total_mass_min": float(total.min()),
        "derive_total_mass_max": float(total.max()),
    }


def derive_aux_corpus(
    onnx_path: str | Path,
    model: str,
    network_type: str,
    out_folder: str | Path,
    *,
    n_files: int = 100,
    n_theta_per_file: int = 4096,
    grid: IntegrationGrid = IntegrationGrid(),
    deadline_quantile_frac: float = 0.7,
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

    File ``i`` is generated from ``np.random.default_rng([seed, i])``, so a
    file's content depends on ``seed`` and its index only.

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
    grid
        Integration grid; see :class:`IntegrationGrid` for the tail policy.
    deadline_quantile_frac
        Passed to :func:`sample_deadlines` (opn / gonogo only).
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
        ``n_theta_per_file < 1``, or a LAN whose input width is not
        ``n_params + 2`` for ``model``.
    """
    _check_network_type(network_type)
    if n_files < 2:
        raise ValueError(
            f"n_files must be >= 2 (the trainers split files into train and "
            f"validation sets), got {n_files}"
        )
    if n_theta_per_file < 1:
        raise ValueError(f"n_theta_per_file must be >= 1, got {n_theta_per_file}")

    onnx_path = Path(onnx_path)
    out_folder = Path(out_folder)
    base_config = ModelConfigBuilder.from_model(model)
    params = list(base_config["params"])
    choices = list(base_config["choices"])
    n_params = len(params)

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
        "integration_grid": grid.n_points,
        "integration_max_t": grid.max_t,
        "t_min": grid.t_min,
        "deadline_quantile_frac": deadline_quantile_frac,
        "deadline_bounds": deadline_bounds,
        "seed": seed,
    }

    out_folder.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    file_records: list[dict] = []
    totals: list[NDArray[np.float64]] = []
    for i in range(n_files):
        rng = np.random.default_rng([seed, i])
        theta = sample_theta(base_config, n_theta_per_file, rng)
        mass = choice_mass(predictor, theta, choices, grid=grid)
        if network_type == "cpn":
            data, labels = cpn_labels(mass, theta, choices)
        else:
            assert deadline_bounds is not None
            deadlines = sample_deadlines(
                mass, rng, deadline_quantile_frac, bounds=deadline_bounds
            )
            build = opn_labels if network_type == "opn" else gonogo_labels
            data, labels = build(mass, theta, deadlines)

        stats = _mass_stats(mass.total)
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
        "grid": {"n_points": grid.n_points, "max_t": grid.max_t, "t_min": grid.t_min},
        "deadline_quantile_frac": deadline_quantile_frac,
        "deadline_bounds": deadline_bounds,
        "source": {
            **{
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(source).items()
            },
            **provenance,
        },
        "derive_stats": _mass_stats(np.concatenate(totals)),
        "files": file_records,
    }
    with open(out_folder / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return files
