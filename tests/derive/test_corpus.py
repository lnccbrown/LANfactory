"""Tests for lanfactory.derive.corpus against the production ddm LAN.

The fixture ``tests/fixtures/onnx/ddm.onnx`` is franklab/HSSM's root
``ddm.onnx`` (see ``tests/fixtures/onnx/PROVENANCE.md``). The identity tests
compare the masses integrated from it with ssms simulations of the same
parameters; the layout tests check that the derived corpora are what the
trainers read. Seeded and fast.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder

from lanfactory.derive import (
    AUX_CATEGORY,
    DERIVATION_METHOD,
    MANIFEST_NAME,
    NETWORK_TYPES,
    ChoiceMass,
    OnsetGrid,
    choice_mass,
    derive_aux_corpus,
    load_onnx_predictor,
)
from lanfactory.derive import corpus as corpus_module
from lanfactory.derive.corpus import (
    _provenance,
    _sha256,
    cpn_labels,
    opn_labels,
    sample_deadlines,
    sample_theta,
    simulate_labels,
)
from lanfactory.trainers.torch_mlp import DatasetTorch
from tests._onnx_utils import DDM_ONNX, DDM_SHA256, export_tiny_torch_lan

DEADLINE_BOUNDS = [0.001, 10.0]  # ssms' DEADLINE_PARAM_CONFIG, as the manifest lists it

CHOICES = [-1, 1]
MAX_T = 20.0
FALLBACK_WINDOW = (0.98, 1.03)
# Five parameter vectors well inside the ddm training box
# (v: [-3, 3], a: [0.3, 2.5], z: [0.1, 0.9], t: [0, 2]) whose total mass
# under the fixture LAN lies inside the fallback window, so the identity
# tests exercise the renormalised LAN labels, not the simulation fallback.
THETA = np.array(
    [
        [0.5, 1.2, 0.5, 0.3],
        [-1.5, 0.8, 0.6, 0.8],
        [2.0, 2.0, 0.4, 1.0],
        [1.0, 1.5, 0.5, 0.5],
        [-0.8, 1.8, 0.3, 0.6],
    ],
    dtype=np.float32,
)
# One deadline per theta, chosen so the omission probability is informative
# (roughly 30-60 %) rather than pinned at 0 or 1.
DEADLINES = np.array([1.5, 1.3, 2.0, 1.5, 2.0])
N_SIM = 100_000
# The provenance keys LAN_pipeline_minimal's publish step requires
# (AUX_PROVENANCE_KEYS in publish/publish_network.py), copied, not imported.
AUX_PROVENANCE_KEYS = (
    "derivation_method",
    "aux_category",
    "source_lan_run_uuid",
    "source_lan_sha256",
    "source_lan_hf_commit",
    "integration_grid",
    "integration_max_t",
)
STATS_KEYS = {
    "derive_total_mass_mean",
    "derive_total_mass_min",
    "derive_total_mass_max",
    "derive_fallback_frac",
    "derive_sim_past_max_t_max",
    "derive_leak_below_onset_p99",
}
# The grid of record, as the pickles and the manifest carry it.
GRID_RECORD = {"onset_param": "t", "n_points": 1160, **asdict(OnsetGrid())}
# Trials per fallback theta in the corpus tests: small enough to keep them fast.
FALLBACK_N_SIM = 500


@pytest.fixture(scope="module")
def ddm_mass() -> ChoiceMass:
    """Masses of THETA on the grid of record, the onset grid around ``t``."""
    return choice_mass(
        load_onnx_predictor(DDM_ONNX),
        THETA,
        CHOICES,
        grid=OnsetGrid(),
        onset=THETA[:, 3],
    )


def _simulate(model: str, theta: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    out = simulator(
        theta=theta, model=model, n_samples=N_SIM, max_t=20, random_state=seed
    )
    return out["rts"].ravel(), out["choices"].ravel()


# --------------------------------------------------------------------------
# 1. cpn identity: mass(c) vs simulated choice frequency, for every choice
# --------------------------------------------------------------------------


def test_fixture_is_the_production_ddm_lan():
    """The on-disk file is the pinned production network."""
    assert _sha256(DDM_ONNX) == DDM_SHA256
    assert load_onnx_predictor(DDM_ONNX).input_width == 6


@pytest.mark.parametrize("choice", CHOICES)
def test_cpn_label_matches_simulated_choice_frequency(ddm_mass, choice):
    """One label per choice code, no category assumed.

    The label is ``mass(c) / total``, which estimates the choice frequency
    *conditional on responding within max_t*: ``mean(choices == c & responded)
    / mean(responded)`` with ``responded = (rts >= 0) & (rts < max_t)``. ddm
    returns a trial still running at ``max_t`` at ``rt ≈ max_t + t`` with a
    choice, so the unconditional ``choice_p`` a simulated corpus carries is
    printed beside it; for these thetas no trial reaches 20 s and the two
    coincide. Inside the fallback window the renormalised label is within
    0.01 of simulation (0.02 before renormalisation).
    """
    _, labels = cpn_labels(ddm_mass, THETA, CHOICES)
    j = CHOICES.index(choice)
    for i, theta in enumerate(THETA):
        rts, choices = _simulate("ddm", theta, seed=i)
        responded = (rts >= 0) & (rts < MAX_T)
        conditional = np.mean((choices == choice) & responded) / responded.mean()
        label = labels[i * len(CHOICES) + j, 0]
        print(
            f"theta={theta.tolist()} choice={choice}: label={label:.4f} "
            f"sim(rt<20)={conditional:.4f} choice_p={np.mean(choices == choice):.4f} "
            f"total={ddm_mass.total[i]:.4f}"
        )
        assert label == pytest.approx(conditional, abs=0.01)


def test_identity_thetas_sit_inside_the_fallback_window(ddm_mass):
    """The identity tests measure the LAN, so its totals must not be flagged."""
    lo, hi = FALLBACK_WINDOW
    assert np.all((ddm_mass.total > lo) & (ddm_mass.total < hi)), ddm_mass.total


# --------------------------------------------------------------------------
# 2. opn identity vs ddm_deadline simulations
# --------------------------------------------------------------------------


def test_opn_matches_ddm_deadline_simulations(ddm_mass):
    _, opn = opn_labels(ddm_mass, THETA, DEADLINES)
    for i, (theta, d) in enumerate(zip(THETA, DEADLINES, strict=True)):
        rts, _ = _simulate("ddm_deadline", np.append(theta, d), seed=10 + i)
        # ssms marks an omission in rts (-999.0).
        omission_p = np.mean(rts == -999.0)
        print(f"theta={theta.tolist()} d={d}: opn={opn[i, 0]:.4f} sim={omission_p:.4f}")
        assert 0.2 < omission_p < 0.7, "deadline should leave omissions informative"
        assert opn[i, 0] == pytest.approx(omission_p, abs=0.02)


def _synthetic_mass(per_choice_total: list[float], n_points: int = 50) -> ChoiceMass:
    """A ChoiceMass whose per-choice cdf ramps linearly to the given totals."""
    t = np.linspace(1e-4, 20.0, n_points)[None, :]
    ramp = np.linspace(0.0, 1.0, n_points)
    cdf = np.stack([np.stack([total * ramp for total in per_choice_total])])
    choices = np.array(CHOICES if len(per_choice_total) == 2 else [0, 1, 2])
    return ChoiceMass(t=t, choices=choices, cdf=cdf)


def test_labels_are_renormalised_by_the_total():
    """Every label is a fraction of the LAN's own total; a zero total is refused.

    With a total of 1.05 the raw masses would give a negative omission and,
    for ``[1.05, 0.02]``, a choice mass above one; renormalised they are
    probabilities without needing the clip.
    """
    theta = np.zeros((1, 2), dtype=np.float32)
    mass = _synthetic_mass([0.6, 0.45])
    assert mass.total[0] == pytest.approx(1.05)
    _, cpn = cpn_labels(mass, theta, CHOICES)
    np.testing.assert_allclose(cpn[:, 0], [0.6 / 1.05, 0.45 / 1.05], rtol=1e-6)
    assert cpn[:, 0].sum() == pytest.approx(1.0, abs=1e-6)
    _, opn = opn_labels(mass, theta, 20.0)
    assert opn[0, 0] == pytest.approx(0.0, abs=1e-7)  # F(max_t) == total
    # Part-way down the ramp the omission is the un-reached share of the total.
    _, opn_half = opn_labels(mass, theta, mass.t[0, 24])
    assert opn_half[0, 0] == pytest.approx(1.0 - 24 / 49, abs=1e-6)

    over = _synthetic_mass([1.05, 0.02])
    _, cpn = cpn_labels(over, theta, CHOICES)
    np.testing.assert_allclose(cpn[:, 0], [1.05 / 1.07, 0.02 / 1.07], rtol=1e-6)
    assert np.all((cpn >= 0.0) & (cpn <= 1.0))

    empty = _synthetic_mass([0.0, 0.0])
    for build, extra in ((cpn_labels, CHOICES), (opn_labels, 1.0)):
        with pytest.raises(ValueError, match=r"total mass must be > 0.*\[0\]"):
            build(empty, theta, extra)


def test_cpn_emits_a_row_per_choice_code():
    """Three choices: a row per code, in code order, labelled with its share."""
    mass = _synthetic_mass([0.2, 0.3, 0.4])  # total 0.9; labels are shares of it
    theta = np.zeros((1, 2), dtype=np.float32)
    data, cpn = cpn_labels(mass, theta, mass.choices)
    np.testing.assert_array_equal(data[:, -1], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(cpn[:, 0], np.array([0.2, 0.3, 0.4]) / 0.9, rtol=1e-6)


def test_deadline_labels_use_the_float32_deadline_the_row_carries():
    """Row and label describe the same (float32) deadline.

    A steep synthetic cdf makes the float32 rounding of 1.0005 visible: at the
    float64 deadline the mass before it is exactly 0.5.
    """
    t = np.array([[1e-4, 1.0, 1.001, 20.0]])
    cdf = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    mass = ChoiceMass(t=t, choices=np.array(CHOICES), cdf=cdf)
    theta = np.zeros((1, 2), dtype=np.float32)
    data, opn = opn_labels(mass, theta, 1.0005)
    d32 = float(np.float32(1.0005))
    assert data[0, -1] == np.float32(1.0005)
    assert abs(d32 - 1.0005) > 1e-8, "1.0005 must not be exact in float32"
    assert opn[0, 0] == pytest.approx(1.0 - (d32 - 1.0) / 0.001, abs=1e-7)
    assert abs(opn[0, 0] - 0.5) > 1e-6  # the float64 deadline would give 0.5


# --------------------------------------------------------------------------
# 3. Row layouts
# --------------------------------------------------------------------------


def test_cpn_rows_are_theta_major_over_every_choice(ddm_mass):
    data, labels = cpn_labels(ddm_mass, THETA, CHOICES)
    assert data.shape == (len(THETA) * len(CHOICES), THETA.shape[1] + 1)
    assert labels.shape == (len(THETA) * len(CHOICES), 1)
    assert data.dtype == labels.dtype == np.float32
    for i, theta in enumerate(THETA):
        for j, c in enumerate(CHOICES):
            row = i * len(CHOICES) + j
            np.testing.assert_array_equal(data[row, :-1], theta)
            assert data[row, -1] == c
            expected = ddm_mass.mass(c)[i] / ddm_mass.total[i]
            assert labels[row, 0] == pytest.approx(expected, abs=1e-7)
        assert labels[i * len(CHOICES) : (i + 1) * len(CHOICES), 0].sum() == (
            pytest.approx(1.0, abs=1e-6)
        )


def test_deadline_rows_put_the_deadline_last(ddm_mass):
    data, labels = opn_labels(ddm_mass, THETA, DEADLINES)
    assert data.shape == (len(THETA), THETA.shape[1] + 1)
    assert labels.shape == (len(THETA), 1)
    assert data.dtype == labels.dtype == np.float32
    np.testing.assert_array_equal(data[:, :-1], THETA)
    np.testing.assert_array_equal(data[:, -1], DEADLINES.astype(np.float32))


# --------------------------------------------------------------------------
# 4. Sampling
# --------------------------------------------------------------------------


def test_sample_theta_matches_ssms_bounds_and_order():
    config = ModelConfigBuilder.from_model("ddm")
    theta = sample_theta(config, 5000, np.random.default_rng(0))
    assert theta.shape == (5000, 4) and theta.dtype == np.float32
    lower, upper = np.array(config["param_bounds"])
    assert np.all(theta >= lower) and np.all(theta <= upper)
    # Fills the box (uniform), column order is config["params"].
    assert np.all(theta.min(axis=0) < lower + 0.05 * (upper - lower))
    assert np.all(theta.max(axis=0) > upper - 0.05 * (upper - lower))
    np.testing.assert_array_equal(
        theta, sample_theta(config, 5000, np.random.default_rng(0))
    )


@pytest.fixture(scope="module")
def box_mass() -> ChoiceMass:
    config = ModelConfigBuilder.from_model("ddm")
    theta = sample_theta(config, 2000, np.random.default_rng(1))
    return choice_mass(load_onnx_predictor(DDM_ONNX), theta, CHOICES, onset=theta[:, 3])


def test_sample_deadlines_covers_the_box_and_concentrates(box_mass):
    bounds = (0.001, 10.0)
    rng = np.random.default_rng(2)
    d = sample_deadlines(box_mass, rng, deadline_quantile_frac=0.7, bounds=bounds)
    assert d.shape == (2000,)
    assert np.all((d >= bounds[0]) & (d <= bounds[1]))
    # The uniform share reaches both edges of the box ...
    assert d.min() < 0.1 and d.max() > 9.5
    # ... while most deadlines sit where the LAN puts its reaction times.
    lo, hi = box_mass.quantile(0.05), box_mass.quantile(0.95)
    inside = (d >= lo) & (d <= hi)
    assert 0.6 < inside.mean() < 0.85

    only_quantiles = sample_deadlines(box_mass, rng, 1.0, bounds)
    assert np.all((only_quantiles >= lo - 1e-9) & (only_quantiles <= hi + 1e-9))
    only_uniform = sample_deadlines(box_mass, rng, 0.0, bounds)
    assert only_uniform.max() > 9.0 and np.mean(only_uniform > 5.0) > 0.4

    np.testing.assert_array_equal(
        sample_deadlines(box_mass, np.random.default_rng(3)),
        sample_deadlines(box_mass, np.random.default_rng(3)),
    )
    with pytest.raises(ValueError, match="bounds"):
        sample_deadlines(box_mass, rng, bounds=(2.0, 1.0))


# --------------------------------------------------------------------------
# 5. Provenance
# --------------------------------------------------------------------------


def test_provenance_hashes_and_parses_run_uuid(tmp_path):
    record = _provenance(DDM_ONNX, "opn", OnsetGrid(max_t=15.0), None, "abc")
    assert tuple(record) == AUX_PROVENANCE_KEYS
    assert record["derivation_method"] == DERIVATION_METHOD == "derived-from-lan"
    assert record["aux_category"] == "omission"
    assert record["source_lan_run_uuid"] is None  # a bare ddm.onnx has no uuid
    assert record["source_lan_sha256"] == DDM_SHA256
    assert record["source_lan_hf_commit"] == "abc"
    assert record["integration_grid"] == 1160
    assert record["integration_max_t"] == 15.0
    assert AUX_CATEGORY == {"cpn": "choice", "opn": "omission"}

    run_uuid = "56d99936415e11f0a2bf3cecefb6d5ee"
    torch_style = tmp_path / f"ddm_lan_{run_uuid}_model.onnx"
    jax_style = tmp_path / f"{run_uuid}_lan_ddm__model.onnx"  # jaxtrain's spelling
    for path in (torch_style, jax_style):
        shutil.copy(DDM_ONNX, path)
        parsed = _provenance(path, "cpn", OnsetGrid(), None, None)
        assert parsed["source_lan_run_uuid"] == run_uuid
    explicit = _provenance(torch_style, "cpn", OnsetGrid(), "explicit", None)
    assert explicit["source_lan_run_uuid"] == "explicit"


# --------------------------------------------------------------------------
# 6. Corpus layout, read back through DatasetTorch
# --------------------------------------------------------------------------

N_THETA = 64


@pytest.fixture(scope="module", params=NETWORK_TYPES)
def derived(request, tmp_path_factory) -> tuple[str, Path, list[Path]]:
    network_type = request.param
    out = tmp_path_factory.mktemp("derived") / network_type
    files = derive_aux_corpus(
        DDM_ONNX,
        "ddm",
        network_type,
        out,
        n_files=2,
        n_theta_per_file=N_THETA,
        fallback_n_sim=FALLBACK_N_SIM,
    )
    return network_type, out, files


def test_dataset_torch_loads_a_derived_corpus(derived):
    network_type, _, files = derived
    n_rows = N_THETA * (len(CHOICES) if network_type == "cpn" else 1)
    assert len(files) == 2 and all(f.suffix == ".pickle" for f in files)

    dataset = DatasetTorch(
        file_ids=files,
        batch_size=32,
        features_key=f"{network_type}_data",
        label_key=f"{network_type}_labels",
    )
    assert dataset.input_dim == 4 + 1
    assert dataset.file_shape_dict == {"inputs": (n_rows, 5), "labels": (n_rows, 1)}
    x, y = dataset[0]
    assert x.shape == (32, 5) and y.shape == (32, 1)
    assert x.dtype == y.dtype == np.float32
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert isinstance(dataset.data_generator_config, dict)
    assert isinstance(dataset.data_model_config, dict)


def test_pickle_carries_the_contract_keys(derived):
    network_type, _, files = derived
    with open(files[1], "rb") as f:
        content = pickle.load(f)
    assert set(content) == {
        f"{network_type}_data",
        f"{network_type}_labels",
        "generator_config",
        "model_config",
    }
    generator_config = content["generator_config"]
    assert generator_config["generator_approach"] == "derived"
    assert generator_config["model"] == "ddm"
    assert generator_config["network_type"] == network_type
    assert generator_config["n_files"] == 2
    assert generator_config["n_theta_per_file"] == N_THETA
    assert generator_config["file_index"] == 1
    assert tuple(generator_config["source"]) == AUX_PROVENANCE_KEYS
    assert generator_config["source"]["aux_category"] == AUX_CATEGORY[network_type]
    assert generator_config["source"]["source_lan_sha256"] == DDM_SHA256
    assert generator_config["source"]["source_lan_run_uuid"] is None
    assert set(generator_config["derive_stats"]) == STATS_KEYS
    # The stats are total-mass stats of this file's thetas: near one, ordered.
    stats = generator_config["derive_stats"]
    assert 0.9 < stats["derive_total_mass_mean"] < 1.1
    assert stats["derive_total_mass_min"] <= stats["derive_total_mass_mean"]
    assert stats["derive_total_mass_mean"] <= stats["derive_total_mass_max"]
    assert 0.0 <= stats["derive_fallback_frac"] <= 1.0
    if stats["derive_fallback_frac"] > 0.0:
        assert 0.0 <= stats["derive_sim_past_max_t_max"] <= 1.0
    else:
        assert stats["derive_sim_past_max_t_max"] is None
    assert 0.0 <= stats["derive_leak_below_onset_p99"] < 0.2
    derive = generator_config["derive"]
    assert derive["fallback_window"] == (0.98, 1.03)
    assert derive["fallback_n_sim"] == FALLBACK_N_SIM
    assert derive["grid"] == GRID_RECORD
    assert derive["deadline_quantile_frac"] == 0.7
    assert derive["seed"] == 0
    if network_type == "cpn":
        assert derive["deadline_bounds"] is None
    else:
        assert derive["deadline_bounds"] == tuple(DEADLINE_BOUNDS)

    model_config = content["model_config"]
    assert not any(callable(v) for v in model_config.values())
    if network_type == "cpn":
        assert model_config["name"] == "ddm"
        assert model_config["input_columns"] == ["v", "a", "z", "t", "choice"]
        assert model_config["params"] == ["v", "a", "z", "t"]
    else:
        assert model_config["name"] == "ddm_deadline"
        assert model_config["input_columns"] == ["v", "a", "z", "t", "deadline"]
        assert model_config["params"] == ["v", "a", "z", "t", "deadline"]
        assert model_config["param_bounds"][0][-1] == 0.001
        assert model_config["param_bounds"][1][-1] == 10.0
        assert model_config["param_bounds_dict"]["deadline"] == (0.001, 10.0)
    assert model_config["choices"] == CHOICES


def test_manifest_lists_the_files(derived):
    network_type, out, files = derived
    n_rows = N_THETA * (len(CHOICES) if network_type == "cpn" else 1)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert [entry["file"] for entry in manifest["files"]] == [f.name for f in files]
    assert manifest["model"] == "ddm"
    assert manifest["network_type"] == network_type
    assert manifest["aux_category"] == AUX_CATEGORY[network_type]
    assert manifest["n_files"] == 2 and manifest["n_theta_per_file"] == N_THETA
    assert manifest["n_rows_per_file"] == n_rows
    last = "choice" if network_type == "cpn" else "deadline"
    assert manifest["input_columns"] == ["v", "a", "z", "t", last]
    assert manifest["choices"] == CHOICES
    assert set(manifest["derive_stats"]) == STATS_KEYS
    assert 0.9 < manifest["derive_stats"]["derive_total_mass_mean"] < 1.1
    assert manifest["fallback_window"] == [0.98, 1.03]  # JSON has no tuples
    assert manifest["fallback_n_sim"] == FALLBACK_N_SIM
    # Corpus-level stats aggregate the files: the fraction is the mean over
    # thetas (equal-sized files), the past-max_t share the max over files.
    per_file = manifest["files"]
    assert manifest["derive_stats"]["derive_fallback_frac"] == pytest.approx(
        np.mean([entry["derive_fallback_frac"] for entry in per_file])
    )
    pasts = [
        entry["derive_sim_past_max_t_max"]
        for entry in per_file
        if entry["derive_sim_past_max_t_max"] is not None
    ]
    assert manifest["derive_stats"]["derive_sim_past_max_t_max"] == (
        max(pasts) if pasts else None
    )
    # The pickles' provenance plus where the LAN came from, each key once.
    assert tuple(manifest["source"]) == ("path", "hf_repo", *AUX_PROVENANCE_KEYS)
    assert manifest["source"]["path"] == str(DDM_ONNX)
    assert manifest["source"]["hf_repo"] is None
    assert manifest["grid"] == GRID_RECORD
    assert manifest["deadline_quantile_frac"] == 0.7
    expected_bounds = None if network_type == "cpn" else DEADLINE_BOUNDS
    assert manifest["deadline_bounds"] == expected_bounds
    assert manifest["seed"] == 0
    assert manifest["lanfactory_version"]
    per_file_min = min(entry["derive_total_mass_min"] for entry in manifest["files"])
    assert manifest["derive_stats"]["derive_total_mass_min"] == per_file_min
    for entry in manifest["files"]:
        assert STATS_KEYS <= set(entry)
        assert entry["n_rows"] == n_rows


def _manifest(folder: Path) -> dict:
    return json.loads((folder / MANIFEST_NAME).read_text())


def _arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        content = pickle.load(f)
    return content["opn_data"], content["opn_labels"]


def test_corpus_is_reproducible_per_file(tmp_path):
    kwargs = {"n_files": 2, "n_theta_per_file": 8, "fallback_n_sim": FALLBACK_N_SIM}
    a = derive_aux_corpus(DDM_ONNX, "ddm", "opn", tmp_path / "a", **kwargs)
    # Same arguments: byte-identical files.
    again = derive_aux_corpus(DDM_ONNX, "ddm", "opn", tmp_path / "again", **kwargs)
    for x, y in zip(a, again, strict=True):
        assert x.read_bytes() == y.read_bytes()
    # File i depends on (seed, i) only: adding files leaves earlier ones alone.
    b = derive_aux_corpus(
        DDM_ONNX,
        "ddm",
        "opn",
        tmp_path / "b",
        n_files=3,
        n_theta_per_file=8,
        fallback_n_sim=FALLBACK_N_SIM,
    )
    for x, y in zip(a, b[:2], strict=True):
        for got, expected in zip(_arrays(x), _arrays(y), strict=True):
            np.testing.assert_array_equal(got, expected)
    c = derive_aux_corpus(DDM_ONNX, "ddm", "opn", tmp_path / "c", seed=1, **kwargs)
    assert not np.array_equal(_arrays(a[0])[0], _arrays(c[0])[0])


def test_corpus_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="n_files must be >= 2"):
        derive_aux_corpus(DDM_ONNX, "ddm", "cpn", tmp_path, n_files=1)
    with pytest.raises(ValueError, match="network_type must be one of"):
        derive_aux_corpus(DDM_ONNX, "ddm", "lan", tmp_path)
    with pytest.raises(ValueError, match="n_theta_per_file"):
        derive_aux_corpus(DDM_ONNX, "ddm", "cpn", tmp_path, n_theta_per_file=0)
    # Checked for cpn too, so a manifest never records an impossible fraction.
    with pytest.raises(ValueError, match="deadline_quantile_frac"):
        derive_aux_corpus(
            DDM_ONNX, "ddm", "cpn", tmp_path / "frac", deadline_quantile_frac=1.5
        )
    assert not (tmp_path / "frac").exists()
    # A LAN of the wrong width: angle has 5 parameters, so its LAN takes 7.
    _, wrong = export_tiny_torch_lan(tmp_path, 7)
    with pytest.raises(ValueError, match=r"width 7.*width 6.*right LAN"):
        derive_aux_corpus(wrong, "ddm", "cpn", tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    # lba2 has four parameters, like ddm, but no onset to refine the grid around.
    with pytest.raises(ValueError, match="no non-decision time 't'"):
        derive_aux_corpus(DDM_ONNX, "lba2", "cpn", tmp_path / "lba2")
    assert not (tmp_path / "lba2").exists()


# --------------------------------------------------------------------------
# 7. Simulation fallback
# --------------------------------------------------------------------------


def test_simulate_labels_estimate_what_the_lan_labels_do():
    """cpn shares sum to one, opn is the deadline model's omission rate, and a
    trial ssms censors at ``max_t`` (``rt == -999``) is past ``max_t``, not a
    response (racing_diffusion_3 censors 12 % of trials here).
    """
    rng = np.random.default_rng(5)
    cpn, _ = simulate_labels("ddm", THETA, "cpn", None, 2000, rng)
    np.testing.assert_allclose(cpn.sum(axis=1), 1.0)
    opn, _ = simulate_labels("ddm", THETA, "opn", DEADLINES, 20_000, rng)
    assert np.all((0.0 < opn) & (opn < 1.0))
    for i, (theta, d) in enumerate(zip(THETA, DEADLINES, strict=True)):
        rts, _ = _simulate("ddm_deadline", np.append(theta, d), seed=10 + i)
        assert opn[i, 0] == pytest.approx(np.mean(rts == -999.0), abs=0.02)

    race = np.array([[0.0, 0.0, 0.0, 0.0, 3.0, 0.0]], dtype=np.float32)
    labels, past = simulate_labels("racing_diffusion_3", race, "cpn", None, 4000, rng)
    assert labels.sum() == pytest.approx(1.0)
    assert past[0] == pytest.approx(0.12, abs=0.03)

    theta, n_sim = THETA[:2], 2000
    with pytest.raises(ValueError, match="need deadlines"):
        simulate_labels("ddm", theta, "opn", None, n_sim, np.random.default_rng(0))
    with pytest.raises(ValueError, match="network_type"):
        simulate_labels("ddm", theta, "lan", None, n_sim, np.random.default_rng(0))
    with pytest.raises(ValueError, match="n_sim"):
        simulate_labels("ddm", theta, "cpn", None, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one"):
        simulate_labels("ddm", theta[:0], "cpn", None, n_sim, np.random.default_rng(0))


class _ScaledPredictor:
    """Multiplies the LAN's density by ``factor`` (0 or NaN too) for one theta."""

    def __init__(self, inner, planted: np.ndarray, factor: float) -> None:
        self.inner = inner
        self.planted = planted.astype(np.float32)
        with np.errstate(divide="ignore"):  # log(0) = -inf is the point
            self.log_factor = np.log(factor)

    @property
    def input_width(self) -> int:
        return self.inner.input_width

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        n = self.planted.shape[0]
        hit = np.all(batch[:, :n] == self.planted, axis=1)
        return np.asarray(self.inner(batch)) + np.where(hit, self.log_factor, 0.0)


def _planted_theta(file_index: int, n_theta: int, row: int) -> np.ndarray:
    """The theta ``derive_aux_corpus`` samples at ``row`` of ``file_index``."""
    config = ModelConfigBuilder.from_model("ddm")
    return sample_theta(config, n_theta, np.random.default_rng([0, file_index]))[row]


@pytest.mark.parametrize("network_type", NETWORK_TYPES)
def test_thetas_outside_the_window_are_labelled_by_simulation(
    tmp_path, monkeypatch, network_type
):
    """Exactly the flagged rows go to ``simulate_labels``; the rest stay LAN.

    The density of one sampled theta is scaled by 1.3 (total ≈ 1.3, outside
    the window) through a wrapped predictor; ``simulate_labels`` is replaced
    by a sentinel that records what it was asked for and returns 0.5. Any
    theta the real LAN already flags is expected to fall back too, so the
    expected set is recomputed from the same scaled predictor.
    """
    n_theta, planted_row = 32, 3
    planted = _planted_theta(0, n_theta, planted_row)
    real = load_onnx_predictor(DDM_ONNX)
    scaled = _ScaledPredictor(real, planted, 1.3)
    monkeypatch.setattr(corpus_module, "load_onnx_predictor", lambda path: scaled)

    calls: list[dict] = []

    def sentinel(model, theta, kind, deadlines, n_sim, rng, *, max_t):
        calls.append(
            {
                "model": model,
                "theta": np.array(theta),
                "kind": kind,
                "d": deadlines,
                "n": n_sim,
            }
        )
        n_labels = len(CHOICES) if kind == "cpn" else 1
        return np.full((theta.shape[0], n_labels), 0.5), np.full(theta.shape[0], 0.01)

    monkeypatch.setattr(corpus_module, "simulate_labels", sentinel)

    out = tmp_path / network_type
    files = derive_aux_corpus(
        DDM_ONNX,
        "ddm",
        network_type,
        out,
        n_files=2,
        n_theta_per_file=n_theta,
        fallback_n_sim=123,
    )

    config = ModelConfigBuilder.from_model("ddm")
    expected_calls = 0
    for i, path in enumerate(files):
        theta = sample_theta(config, n_theta, np.random.default_rng([0, i]))
        mass = choice_mass(scaled, theta, CHOICES, grid=OnsetGrid(), onset=theta[:, 3])
        flagged = (mass.total < 0.98) | (mass.total > 1.03)
        if i == 0:
            assert flagged[planted_row] and mass.total[planted_row] > 1.2
        with open(path, "rb") as f:
            content = pickle.load(f)
        labels = content[f"{network_type}_labels"][:, 0]
        stats = content["generator_config"]["derive_stats"]
        assert stats["derive_fallback_frac"] == pytest.approx(flagged.mean())
        if not flagged.any():
            assert stats["derive_sim_past_max_t_max"] is None
            continue
        assert stats["derive_sim_past_max_t_max"] == pytest.approx(0.01)
        call = calls[expected_calls]
        expected_calls += 1
        assert call["model"] == "ddm" and call["kind"] == network_type
        assert call["n"] == 123
        np.testing.assert_array_equal(call["theta"], theta[flagged])
        data = content[f"{network_type}_data"]
        if network_type == "cpn":
            rows = np.repeat(flagged, len(CHOICES))
            assert call["d"] is None
        else:
            rows = flagged
            np.testing.assert_array_equal(call["d"], data[flagged, -1])
        assert np.all(labels[rows] == 0.5)
        assert not np.any(labels[~rows] == 0.5)
        if network_type == "cpn":  # the unflagged rows keep the LAN's shares
            per_theta = labels.reshape(n_theta, len(CHOICES))[~flagged]
            np.testing.assert_allclose(per_theta.sum(axis=1), 1.0, atol=1e-6)
    assert len(calls) == expected_calls >= 1
    manifest = _manifest(out)
    assert manifest["derive_stats"]["derive_sim_past_max_t_max"] == pytest.approx(0.01)
    assert manifest["derive_stats"]["derive_fallback_frac"] > 0.0


@pytest.mark.parametrize("network_type", NETWORK_TYPES)
def test_zero_and_nan_totals_fall_back_instead_of_failing(
    tmp_path, monkeypatch, network_type
):
    """A theta whose LAN total is 0 or NaN is simulated, not refused or NaN.

    The window is applied before the labels are renormalised, and a NaN total
    lies outside it; ``simulate_labels`` is a sentinel returning 0.5.
    """
    zero, nan = _planted_theta(0, 8, 0), _planted_theta(0, 8, 1)
    real = load_onnx_predictor(DDM_ONNX)
    broken = _ScaledPredictor(_ScaledPredictor(real, zero, 0.0), nan, np.nan)
    monkeypatch.setattr(corpus_module, "load_onnx_predictor", lambda path: broken)

    def sentinel(model, theta, kind, deadlines, n_sim, rng, *, max_t):
        n_labels = len(CHOICES) if kind == "cpn" else 1
        return np.full((len(theta), n_labels), 0.5), np.zeros(len(theta))

    monkeypatch.setattr(corpus_module, "simulate_labels", sentinel)
    files = derive_aux_corpus(
        DDM_ONNX, "ddm", network_type, tmp_path, n_files=2, n_theta_per_file=8
    )
    with open(files[0], "rb") as f:
        content = pickle.load(f)
    labels = content[f"{network_type}_labels"][:, 0]
    rows_per_theta = len(CHOICES) if network_type == "cpn" else 1
    assert np.all(labels[: 2 * rows_per_theta] == 0.5)  # thetas 0 and 1
    assert not np.isnan(labels).any()
    assert not np.isnan(content[f"{network_type}_data"]).any()  # opn: deadlines
    stats = _manifest(tmp_path)["derive_stats"]
    assert all(np.isfinite(v) for v in stats.values() if v is not None)


def test_no_fallback_never_simulates(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("simulate_labels must not be called")

    monkeypatch.setattr(corpus_module, "simulate_labels", refuse)
    files = derive_aux_corpus(
        DDM_ONNX,
        "ddm",
        "cpn",
        tmp_path,
        n_files=2,
        n_theta_per_file=256,
        fallback_window=None,
    )
    for path in files:
        with open(path, "rb") as f:
            generator_config = pickle.load(f)["generator_config"]
        assert generator_config["derive"]["fallback_window"] is None
        assert generator_config["derive_stats"]["derive_fallback_frac"] == 0.0
        assert generator_config["derive_stats"]["derive_sim_past_max_t_max"] is None
    manifest = _manifest(tmp_path)
    assert manifest["fallback_window"] is None
    assert manifest["derive_stats"]["derive_fallback_frac"] == 0.0
    assert manifest["derive_stats"]["derive_sim_past_max_t_max"] is None
    # 256 thetas from the box: some totals are outside the default window,
    # so the fallback would have been exercised had it been on.
    assert manifest["derive_stats"]["derive_total_mass_max"] > 1.03


def test_fallback_arguments_are_validated(tmp_path):
    kwargs = {"n_files": 2, "n_theta_per_file": 4}
    with pytest.raises(ValueError, match="fallback_window"):
        derive_aux_corpus(
            DDM_ONNX, "ddm", "cpn", tmp_path, fallback_window=(1.03, 0.98), **kwargs
        )
    with pytest.raises(ValueError, match="fallback_window"):
        derive_aux_corpus(
            DDM_ONNX, "ddm", "cpn", tmp_path, fallback_window=(0.0, 1.03), **kwargs
        )
    with pytest.raises(ValueError, match="fallback_n_sim"):
        derive_aux_corpus(DDM_ONNX, "ddm", "cpn", tmp_path, fallback_n_sim=0, **kwargs)
    assert not list(tmp_path.glob("*.pickle"))
