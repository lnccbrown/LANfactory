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
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder

from lanfactory.derive import (
    AUX_CATEGORY,
    DERIVATION_METHOD,
    MANIFEST_NAME,
    NETWORK_TYPES,
    ChoiceMass,
    IntegrationGrid,
    OnsetGrid,
    SourceLAN,
    choice_mass,
    cpn_labels,
    derive_aux_corpus,
    gonogo_labels,
    grid_description,
    load_onnx_predictor,
    opn_labels,
    sample_deadlines,
    sample_theta,
)
from lanfactory.derive.corpus import _nogo_before, _omission
from lanfactory.trainers.torch_mlp import DatasetTorch
from tests._onnx_utils import export_tiny_torch_lan
from tests.derive.conftest import DDM_ONNX

DDM_SHA256 = "09f685c18d3bdbd9b54fa89bed3b5bc0e2c76566e7ed0ae5e24df2c9b04f1b0e"
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
PROVENANCE_KEYS = {
    "derivation_method",
    "aux_category",
    "source_lan_run_uuid",
    "source_lan_sha256",
    "source_lan_hf_commit",
    "source_lan_run_id",
    "integration_grid",
    "integration_max_t",
}
STATS_KEYS = {
    "derive_total_mass_mean",
    "derive_total_mass_min",
    "derive_total_mass_max",
}


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


def test_fixture_is_the_production_ddm_lan(ddm_provenance):
    """The on-disk file, the in-code constant and PROVENANCE.md agree."""
    assert SourceLAN.from_onnx(DDM_ONNX).sha256 == DDM_SHA256
    assert ddm_provenance["sha256"] == DDM_SHA256
    assert re.fullmatch(r"[0-9a-f]{40}", ddm_provenance["Hub commit"])
    assert ddm_provenance["Source"].endswith("franklab/HSSM/blob/main/ddm.onnx")
    assert load_onnx_predictor(DDM_ONNX).input_width == 6


@pytest.mark.parametrize("choice", CHOICES)
def test_cpn_label_matches_simulated_choice_frequency(ddm_mass, choice):
    """Contract B: one label per choice code, no category assumed.

    The label is ``mass(c) / total``, which estimates the choice frequency
    *conditional on responding within max_t*: ``mean(choices == c & rts <
    max_t) / mean(rts < max_t)``. ssms does not censor a base model at
    ``max_t`` (an un-terminated trial comes back at ``rt ≈ max_t + t`` with a
    choice), so the unconditional ``choice_p`` a simulated corpus carries is
    printed beside it; for these thetas no trial reaches 20 s and the two
    coincide. Inside the fallback window the renormalised label is within
    0.01 of simulation (0.02 before renormalisation).
    """
    _, labels = cpn_labels(ddm_mass, THETA, CHOICES)
    j = CHOICES.index(choice)
    for i, theta in enumerate(THETA):
        rts, choices = _simulate("ddm", theta, seed=i)
        responded = rts < MAX_T
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
# 2. opn / gonogo identity vs ddm_deadline simulations
# --------------------------------------------------------------------------


def test_opn_and_gonogo_match_ddm_deadline_simulations(ddm_mass):
    _, opn = opn_labels(ddm_mass, THETA, DEADLINES)
    _, gonogo = gonogo_labels(ddm_mass, THETA, DEADLINES)
    for i, (theta, d) in enumerate(zip(THETA, DEADLINES, strict=True)):
        rts, choices = _simulate("ddm_deadline", np.append(theta, d), seed=10 + i)
        # ssms marks an omission in rts (-999.0) and leaves choices alone.
        omitted = rts == -999.0
        omission_p = omitted.mean()
        nogo_p = ((choices != max(CHOICES)) | omitted).mean()  # ssms' nogo_p
        print(
            f"theta={theta.tolist()} d={d}: opn={opn[i, 0]:.4f} sim={omission_p:.4f}"
            f" | gonogo={gonogo[i, 0]:.4f} sim={nogo_p:.4f}"
        )
        assert 0.2 < omission_p < 0.7, "deadline should leave omissions informative"
        assert opn[i, 0] == pytest.approx(omission_p, abs=0.02)
        assert gonogo[i, 0] == pytest.approx(nogo_p, abs=0.02)


def test_gonogo_decomposes_into_nogo_mass_plus_opn(ddm_mass):
    """gonogo = mass_before(d, c_nogo) / total + opn, c_nogo every non-max choice.

    The float64 terms the labels are cast from satisfy the identity to 1e-12
    against ``ChoiceMass`` directly (ddm has one nogo choice, ``-1``); the
    emitted float32 labels satisfy it to float32 precision.
    """
    # Labels are computed at the float32-rounded deadline the row carries.
    d = DEADLINES.astype(np.float32).astype(np.float64)
    total = ddm_mass.total
    nogo_before = ddm_mass.mass_before(d, -1) / total
    omission = 1.0 - ddm_mass.mass_before(d) / total
    assert np.all((0.0 < omission) & (omission < 1.0)), "clip must be a no-op here"
    np.testing.assert_allclose(
        _nogo_before(ddm_mass, d) / total, nogo_before, atol=1e-12
    )
    np.testing.assert_allclose(_omission(ddm_mass, d), omission, atol=1e-12)

    _, opn = opn_labels(ddm_mass, THETA, DEADLINES)
    _, gonogo = gonogo_labels(ddm_mass, THETA, DEADLINES)
    np.testing.assert_allclose(opn[:, 0], omission, rtol=0, atol=1e-7)
    np.testing.assert_allclose(gonogo[:, 0], nogo_before + omission, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        gonogo[:, 0].astype(np.float64),
        nogo_before + opn[:, 0].astype(np.float64),
        rtol=0,
        atol=1e-6,  # float32 labels
    )


def _synthetic_mass(per_choice_total: list[float], n_points: int = 50) -> ChoiceMass:
    """A ChoiceMass whose per-choice cdf ramps linearly to the given totals."""
    t = np.linspace(1e-4, 20.0, n_points)
    ramp = np.linspace(0.0, 1.0, n_points)
    cdf = np.stack([np.stack([total * ramp for total in per_choice_total])])
    choices = np.array(CHOICES if len(per_choice_total) == 2 else [0, 1, 2])
    return ChoiceMass(t=t, choices=choices, cdf=cdf)


def test_labels_are_renormalised_by_the_total():
    """Every label is a fraction of the LAN's own total; a zero total is refused.

    With a total of 1.05 the raw masses would give a negative omission and,
    for ``[1.05, 0.02]``, a choice mass above one; renormalised they are
    probabilities without needing the clip, and the decomposition
    ``gonogo == nogo_before / total + opn`` holds exactly.
    """
    theta = np.zeros((1, 2), dtype=np.float32)
    mass = _synthetic_mass([0.6, 0.45])
    assert mass.total[0] == pytest.approx(1.05)
    _, cpn = cpn_labels(mass, theta, CHOICES)
    np.testing.assert_allclose(cpn[:, 0], [0.6 / 1.05, 0.45 / 1.05], rtol=1e-6)
    assert cpn[:, 0].sum() == pytest.approx(1.0, abs=1e-6)
    _, opn = opn_labels(mass, theta, 20.0)
    assert opn[0, 0] == pytest.approx(0.0, abs=1e-7)  # F(max_t) == total
    _, gonogo = gonogo_labels(mass, theta, 20.0)
    assert gonogo[0, 0] == pytest.approx(0.6 / 1.05)
    assert gonogo[0, 0] == pytest.approx(mass.mass(-1)[0] / 1.05 + opn[0, 0])
    # Part-way down the ramp the omission is the un-reached share of the total.
    _, opn_half = opn_labels(mass, theta, mass.t[24])
    assert opn_half[0, 0] == pytest.approx(1.0 - 24 / 49, abs=1e-6)

    over = _synthetic_mass([1.05, 0.02])
    _, cpn = cpn_labels(over, theta, CHOICES)
    np.testing.assert_allclose(cpn[:, 0], [1.05 / 1.07, 0.02 / 1.07], rtol=1e-6)
    _, gonogo = gonogo_labels(over, theta, 20.0)
    assert gonogo[0, 0] == pytest.approx(1.05 / 1.07)
    assert np.all((gonogo >= 0.0) & (gonogo <= 1.0) & (cpn >= 0.0) & (cpn <= 1.0))

    empty = _synthetic_mass([0.0, 0.0])
    for build, extra in (
        (cpn_labels, CHOICES),
        (opn_labels, 1.0),
        (gonogo_labels, 1.0),
    ):
        with pytest.raises(ValueError, match=r"total mass must be > 0.*\[0\]"):
            build(empty, theta, extra)


def test_gonogo_sums_every_non_maximal_choice():
    """With three choices the nogo set is {0, 1}, not just the smallest code."""
    mass = _synthetic_mass([0.2, 0.3, 0.4])  # total 0.9; labels are shares of it
    theta = np.zeros((1, 2), dtype=np.float32)
    data, gonogo = gonogo_labels(mass, theta, 20.0)
    assert gonogo[0, 0] == pytest.approx(0.5 / 0.9)  # min-only would give 0.2 / 0.9
    _, opn = opn_labels(mass, theta, 20.0)
    assert opn[0, 0] == pytest.approx(0.0, abs=1e-7)
    # Part-way down the ramp every choice term scales, the omission fills up.
    frac = 24 / 49  # ramp value at grid node 24 of 50
    _, part = gonogo_labels(mass, theta, mass.t[24])
    assert part[0, 0] == pytest.approx(frac * 0.5 / 0.9 + (1.0 - frac))
    # cpn emits a row per code, in code order, labelled with that code's share.
    data, cpn = cpn_labels(mass, theta, mass.choices)
    np.testing.assert_array_equal(data[:, -1], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(cpn[:, 0], np.array([0.2, 0.3, 0.4]) / 0.9, rtol=1e-6)


def test_deadline_labels_use_the_float32_deadline_the_row_carries():
    """Row and label describe the same (float32) deadline.

    A steep synthetic cdf makes the float32 rounding of 1.0005 visible: at the
    float64 deadline the mass before it is exactly 0.5.
    """
    t = np.array([1e-4, 1.0, 1.001, 20.0])
    cdf = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    mass = ChoiceMass(t=t, choices=np.array(CHOICES), cdf=cdf)
    theta = np.zeros((1, 2), dtype=np.float32)
    data, opn = opn_labels(mass, theta, 1.0005)
    d32 = float(np.float32(1.0005))
    assert data[0, -1] == np.float32(1.0005)
    assert abs(d32 - 1.0005) > 1e-8, "1.0005 must not be exact in float32"
    assert opn[0, 0] == pytest.approx(1.0 - (d32 - 1.0) / 0.001, abs=1e-7)
    assert abs(opn[0, 0] - 0.5) > 1e-6  # the float64 deadline would give 0.5
    _, gonogo = gonogo_labels(mass, theta, 1.0005)
    assert gonogo[0, 0] == pytest.approx((d32 - 1.0) / 0.001 + opn[0, 0], abs=1e-7)


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
    for build in (opn_labels, gonogo_labels):
        data, labels = build(ddm_mass, THETA, DEADLINES)
        assert data.shape == (len(THETA), THETA.shape[1] + 1)
        assert labels.shape == (len(THETA), 1)
        assert data.dtype == labels.dtype == np.float32
        np.testing.assert_array_equal(data[:, :-1], THETA)
        np.testing.assert_array_equal(data[:, -1], DEADLINES.astype(np.float32))


@pytest.mark.parametrize(
    "build,extra",
    [(cpn_labels, CHOICES), (opn_labels, 1.0), (gonogo_labels, 1.0)],
    ids=["cpn", "opn", "gonogo"],
)
def test_label_functions_reject_bad_theta(ddm_mass, build, extra):
    with pytest.raises(ValueError, match="parameter vectors"):
        build(ddm_mass, THETA[:2], extra)
    with pytest.raises(ValueError, match="theta must be"):
        build(ddm_mass, THETA[:, :, None], extra)


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
    return choice_mass(load_onnx_predictor(DDM_ONNX), theta, CHOICES)


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
    with pytest.raises(ValueError, match="deadline_quantile_frac"):
        sample_deadlines(box_mass, rng, 1.5)
    with pytest.raises(ValueError, match="bounds"):
        sample_deadlines(box_mass, rng, bounds=(2.0, 1.0))


# --------------------------------------------------------------------------
# 5. SourceLAN
# --------------------------------------------------------------------------


def test_source_lan_hashes_and_parses_run_uuid(tmp_path):
    bare = SourceLAN.from_onnx(DDM_ONNX)
    assert bare.sha256 == DDM_SHA256 and bare.run_uuid is None
    assert bare.hf_repo is None and bare.hf_revision is None

    run_uuid = "56d99936415e11f0a2bf3cecefb6d5ee"
    torch_style = tmp_path / f"ddm_lan_{run_uuid}_model.onnx"
    jax_style = tmp_path / f"{run_uuid}_lan_ddm__model.onnx"  # jaxtrain's spelling
    shutil.copy(DDM_ONNX, torch_style)
    shutil.copy(DDM_ONNX, jax_style)
    assert SourceLAN.from_onnx(torch_style).run_uuid == run_uuid
    assert SourceLAN.from_onnx(jax_style).run_uuid == run_uuid
    assert SourceLAN.from_onnx(torch_style, run_uuid="explicit").run_uuid == "explicit"
    # The parsed uuid reaches the provenance record under its contract key.
    parsed = SourceLAN.from_onnx(torch_style).provenance("cpn", IntegrationGrid())
    assert parsed["source_lan_run_uuid"] == run_uuid
    assert parsed["aux_category"] == "choice"

    hub = SourceLAN.from_onnx(DDM_ONNX, hf_repo="franklab/HSSM", hf_revision="abc")
    record = hub.provenance("opn", IntegrationGrid(n_points=500, max_t=15.0))
    assert set(record) == PROVENANCE_KEYS
    assert record["derivation_method"] == DERIVATION_METHOD == "derived-from-lan"
    assert record["aux_category"] == "omission"
    assert record["source_lan_run_uuid"] is None  # a bare ddm.onnx has no uuid
    assert record["source_lan_sha256"] == DDM_SHA256
    assert record["source_lan_hf_commit"] == "abc"
    assert record["source_lan_run_id"] is None
    assert record["integration_grid"] == 500
    assert record["integration_max_t"] == 15.0
    assert AUX_CATEGORY == {"cpn": "choice", "opn": "omission", "gonogo": "nogo"}
    with pytest.raises(ValueError, match="network_type"):
        hub.provenance("lan", IntegrationGrid())


# --------------------------------------------------------------------------
# 6. Corpus layout, read back through DatasetTorch
# --------------------------------------------------------------------------

N_THETA = 64


@pytest.fixture(scope="module", params=NETWORK_TYPES)
def derived(request, tmp_path_factory) -> tuple[str, Path, list[Path]]:
    network_type = request.param
    out = tmp_path_factory.mktemp("derived") / network_type
    files = derive_aux_corpus(
        DDM_ONNX, "ddm", network_type, out, n_files=2, n_theta_per_file=N_THETA
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
    assert set(generator_config["source"]) == PROVENANCE_KEYS
    assert generator_config["source"]["aux_category"] == AUX_CATEGORY[network_type]
    assert generator_config["source"]["source_lan_sha256"] == DDM_SHA256
    assert generator_config["source"]["source_lan_run_uuid"] is None
    assert set(generator_config["derive_stats"]) == STATS_KEYS
    # The stats are total-mass stats of this file's thetas: near one, ordered.
    stats = generator_config["derive_stats"]
    assert 0.9 < stats["derive_total_mass_mean"] < 1.1
    assert stats["derive_total_mass_min"] <= stats["derive_total_mass_mean"]
    assert stats["derive_total_mass_mean"] <= stats["derive_total_mass_max"]
    derive = generator_config["derive"]
    assert derive["grid"] == "onset"
    assert derive["grid_config"] == grid_description(OnsetGrid(), "t")
    assert derive["integration_grid"] == OnsetGrid().n_points == 1160
    assert derive["integration_max_t"] == 20.0
    assert derive["t_min"] == 1e-4
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
    assert PROVENANCE_KEYS <= set(manifest["source"])
    assert manifest["source"]["path"] == str(DDM_ONNX)
    assert manifest["source"]["sha256"] == manifest["source"]["source_lan_sha256"]
    assert manifest["source"]["run_uuid"] is None
    assert manifest["grid"] == grid_description(OnsetGrid(), "t")
    assert manifest["grid"]["kind"] == "onset" and manifest["grid"]["n_points"] == 1160
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


def test_grid_defaults_to_the_onset_grid_and_falls_back_with_a_warning(
    tmp_path, caplog
):
    kwargs = dict(n_files=2, n_theta_per_file=4)
    derive_aux_corpus(DDM_ONNX, "ddm", "cpn", tmp_path / "default", **kwargs)
    assert _manifest(tmp_path / "default")["grid"] == grid_description(OnsetGrid(), "t")

    with caplog.at_level("WARNING", logger="lanfactory.derive.corpus"):
        derive_aux_corpus(
            DDM_ONNX, "ddm", "cpn", tmp_path / "none", onset_param=None, **kwargs
        )
    assert not caplog.records  # asked for no onset grid: nothing to warn about
    assert _manifest(tmp_path / "none")["grid"] == grid_description(
        IntegrationGrid(), None
    )

    with caplog.at_level("WARNING", logger="lanfactory.derive.corpus"):
        derive_aux_corpus(
            DDM_ONNX, "ddm", "cpn", tmp_path / "ndt", onset_param="ndt", **kwargs
        )
    assert any("no parameter 'ndt'" in r.getMessage() for r in caplog.records)
    grid = _manifest(tmp_path / "ndt")["grid"]
    assert grid["kind"] == "uniform" and grid["onset_param"] is None

    # An explicit uniform grid is used as is; the onset column is still known.
    derive_aux_corpus(
        DDM_ONNX,
        "ddm",
        "cpn",
        tmp_path / "explicit",
        grid=IntegrationGrid(n_points=300),
        **kwargs,
    )
    grid = _manifest(tmp_path / "explicit")["grid"]
    assert grid == grid_description(IntegrationGrid(n_points=300), "t")

    with pytest.raises(ValueError, match="OnsetGrid needs onset_param"):
        derive_aux_corpus(
            DDM_ONNX,
            "ddm",
            "cpn",
            tmp_path / "bad",
            onset_param=None,
            grid=OnsetGrid(),
            **kwargs,
        )
    assert not (tmp_path / "bad").exists()


def _arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        content = pickle.load(f)
    return content["opn_data"], content["opn_labels"]


def test_corpus_is_reproducible_per_file(tmp_path):
    kwargs = dict(n_files=2, n_theta_per_file=8)
    a = derive_aux_corpus(DDM_ONNX, "ddm", "opn", tmp_path / "a", **kwargs)
    # Same arguments: byte-identical files.
    again = derive_aux_corpus(DDM_ONNX, "ddm", "opn", tmp_path / "again", **kwargs)
    for x, y in zip(a, again, strict=True):
        assert x.read_bytes() == y.read_bytes()
    # File i depends on (seed, i) only: adding files leaves earlier ones alone.
    b = derive_aux_corpus(
        DDM_ONNX, "ddm", "opn", tmp_path / "b", n_files=3, n_theta_per_file=8
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
    # A LAN of the wrong width: angle has 5 parameters, so its LAN takes 7.
    _, wrong = export_tiny_torch_lan(tmp_path, 7)
    with pytest.raises(ValueError, match=r"width 7.*width 6.*right LAN"):
        derive_aux_corpus(wrong, "ddm", "cpn", tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


# --------------------------------------------------------------------------
# 7. The trainer's own CLI accepts the corpus (dry run: no training)
# --------------------------------------------------------------------------


def test_torchtrain_dry_run_accepts_a_derived_corpus(tmp_path):
    """``torchtrain --dry-run`` builds its dataloaders from a derived folder.

    A full training round trip is what the trainer E2E tests cover; this test
    only checks that ``torchtrain``'s file discovery, key lookup, batch-size
    check and first batch accept a derived folder.
    """
    corpus = tmp_path / "corpus"
    derive_aux_corpus(DDM_ONNX, "ddm", "cpn", corpus, n_files=2, n_theta_per_file=64)
    config = {
        "NETWORK_TYPE": "cpn",
        "CPU_BATCH_SIZE": 64,  # divides 64 thetas x 2 choices = 128 rows
        "GPU_BATCH_SIZE": 64,
        "GENERATOR_APPROACH": "derived",
        "OPTIMIZER_": "adam",
        "N_EPOCHS": 1,
        "TRAINING_DATA_FOLDER": str(corpus),
        "MODEL": "ddm",
        "SHUFFLE": True,
        "LAYER_SIZES": [[16, 16, 1]],
        "ACTIVATIONS": [["tanh", "tanh"]],
        "WEIGHT_DECAY": 0.0,
        "TRAIN_VAL_SPLIT": 0.5,
        "N_TRAINING_FILES": 10000,
        "LABELS_LOWER_BOUND": "None",
        "LEARNING_RATE": 0.001,
        "LR_SCHEDULER": "reduce_on_plateau",
        "LR_SCHEDULER_PARAMS": {"factor": 0.1, "patience": 2},
    }
    config_path = tmp_path / "cpn.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            "torchtrain",
            "--config-path",
            str(config_path),
            "--training-data-folder",
            str(corpus),
            "--networks-path-base",
            str(tmp_path / "networks"),
            "--dry-run",
            "--log-level",
            "INFO",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN: DataLoader validation successful" in result.stderr
    assert "features=torch.Size([64, 5])" in result.stderr
