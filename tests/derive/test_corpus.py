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
    SourceLAN,
    choice_mass,
    cpn_labels,
    derive_aux_corpus,
    gonogo_labels,
    load_onnx_predictor,
    opn_labels,
    sample_deadlines,
    sample_theta,
)
from lanfactory.trainers.torch_mlp import DatasetTorch
from tests._onnx_utils import export_tiny_torch_lan

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "onnx"
DDM_ONNX = FIXTURE_DIR / "ddm.onnx"
DDM_SHA256 = "09f685c18d3bdbd9b54fa89bed3b5bc0e2c76566e7ed0ae5e24df2c9b04f1b0e"

CHOICES = [-1, 1]
# Three parameter vectors well inside the ddm training box
# (v: [-3, 3], a: [0.3, 2.5], z: [0.1, 0.9], t: [0, 2]).
THETA = np.array(
    [
        [0.5, 1.2, 0.5, 0.3],
        [-1.5, 0.8, 0.6, 0.8],
        [2.0, 2.0, 0.4, 1.0],
    ],
    dtype=np.float32,
)
# One deadline per theta, chosen so the omission probability is informative
# (roughly 30-60 %) rather than pinned at 0 or 1.
DEADLINES = np.array([1.5, 1.3, 2.0])
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
    return choice_mass(load_onnx_predictor(DDM_ONNX), THETA, CHOICES)


def _simulate(model: str, theta: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    out = simulator(
        theta=theta, model=model, n_samples=N_SIM, max_t=20, random_state=seed
    )
    return out["rts"].ravel(), out["choices"].ravel()


# --------------------------------------------------------------------------
# 1. cpn identity: mass(c) vs simulated choice frequency, for every choice
# --------------------------------------------------------------------------


def test_fixture_is_the_production_ddm_lan():
    assert SourceLAN.from_onnx(DDM_ONNX).sha256 == DDM_SHA256
    assert load_onnx_predictor(DDM_ONNX).input_width == 6


@pytest.mark.parametrize("choice", CHOICES)
def test_cpn_mass_matches_simulated_choice_frequency(ddm_mass, choice):
    """Contract B: one label per choice code, no category assumed.

    The assertion compares with ``mean(choices == c)`` — ssms' own
    ``choice_p``, which is what a simulated CPN corpus is labelled with. ssms
    does not censor a base model at ``max_t``, so that frequency also counts
    trials finishing past 20 s, while the LAN mass stops at 20 s; the
    censored frequency and the total mass are printed so the shortfall is
    visible. For these thetas no trial reaches 20 s and the two coincide.
    """
    for i, theta in enumerate(THETA):
        rts, choices = _simulate("ddm", theta, seed=i)
        frequency = np.mean(choices == choice)
        censored = np.mean((choices == choice) & (rts < 20.0))
        mass = ddm_mass.mass(choice)[i]
        print(
            f"theta={theta.tolist()} choice={choice}: mass={mass:.4f} "
            f"sim={frequency:.4f} sim(rt<20)={censored:.4f} "
            f"total={ddm_mass.total[i]:.4f}"
        )
        assert mass == pytest.approx(frequency, abs=0.02)


def test_total_mass_is_close_to_one_inside_the_box(ddm_mass):
    # The LAN approximates a normalised density; not renormalised here.
    assert np.all(np.abs(ddm_mass.total - 1.0) < 0.02)


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
    """gonogo = mass_before(d, c_nogo) + opn, with c_nogo every non-max choice."""
    # Labels are computed at the float32-rounded deadline the row carries.
    d = DEADLINES.astype(np.float32).astype(np.float64)
    nogo_before = ddm_mass.mass_before(d, -1)
    omission = 1.0 - ddm_mass.mass_before(d)
    # In float64 (the quantities the labels are cast from) the identity is exact.
    total_nogo = nogo_before + omission
    np.testing.assert_allclose(
        total_nogo,
        ddm_mass.mass_before(d, -1) + 1.0 - ddm_mass.mass_before(d),
        rtol=0,
        atol=1e-12,
    )
    _, opn = opn_labels(ddm_mass, THETA, DEADLINES)
    _, gonogo = gonogo_labels(ddm_mass, THETA, DEADLINES)
    np.testing.assert_allclose(opn[:, 0], omission, rtol=0, atol=1e-7)
    np.testing.assert_allclose(gonogo[:, 0], total_nogo, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        gonogo[:, 0].astype(np.float64),
        nogo_before + opn[:, 0].astype(np.float64),
        rtol=0,
        atol=1e-6,  # float32 labels
    )


def test_labels_are_clipped_probabilities():
    """The LAN's total can exceed one; labels are clipped, never renormalised."""
    t = np.linspace(1e-4, 20.0, 50)
    ramp = np.linspace(0.0, 1.0, 50)
    cdf = np.stack([np.stack([0.6 * ramp, 0.45 * ramp])])  # total 1.05
    mass = ChoiceMass(t=t, choices=np.array(CHOICES), cdf=cdf)
    theta = np.zeros((1, 2), dtype=np.float32)
    assert mass.total[0] == pytest.approx(1.05)
    _, opn = opn_labels(mass, theta, 20.0)
    assert opn[0, 0] == 0.0  # 1 - 1.05 clipped
    _, cpn = cpn_labels(mass, theta, CHOICES)
    np.testing.assert_allclose(cpn[:, 0], [0.6, 0.45], rtol=1e-6)
    # gonogo adds the *raw* omission term (-0.05) before clipping the sum, so
    # the decomposition gonogo = nogo mass + (1 - mass_before) stays exact.
    _, gonogo = gonogo_labels(mass, theta, 20.0)
    assert gonogo[0, 0] == pytest.approx(0.55)


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
            expected = min(ddm_mass.mass(c)[i], 1.0)  # clipped probability
            assert labels[row, 0] == pytest.approx(expected, abs=1e-7)


def test_deadline_rows_put_the_deadline_last(ddm_mass):
    for build in (opn_labels, gonogo_labels):
        data, labels = build(ddm_mass, THETA, DEADLINES)
        assert data.shape == (len(THETA), THETA.shape[1] + 1)
        assert labels.shape == (len(THETA), 1)
        assert data.dtype == labels.dtype == np.float32
        np.testing.assert_array_equal(data[:, :-1], THETA)
        np.testing.assert_array_equal(data[:, -1], DEADLINES.astype(np.float32))


def test_label_functions_reject_mismatched_theta(ddm_mass):
    with pytest.raises(ValueError, match="parameter vectors"):
        opn_labels(ddm_mass, THETA[:2], 1.0)


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
    jax_style = tmp_path / f"{run_uuid}_lan_ddm_model.onnx"
    shutil.copy(DDM_ONNX, torch_style)
    shutil.copy(DDM_ONNX, jax_style)
    assert SourceLAN.from_onnx(torch_style).run_uuid == run_uuid
    assert SourceLAN.from_onnx(jax_style).run_uuid == run_uuid
    assert SourceLAN.from_onnx(torch_style, run_uuid="explicit").run_uuid == "explicit"

    hub = SourceLAN.from_onnx(DDM_ONNX, hf_repo="franklab/HSSM", hf_revision="abc")
    record = hub.provenance("opn", IntegrationGrid(n_points=500, max_t=15.0))
    assert set(record) == PROVENANCE_KEYS
    assert record["derivation_method"] == DERIVATION_METHOD == "derived-from-lan"
    assert record["aux_category"] == "omission"
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
    with open(files[0], "rb") as f:
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
    assert set(generator_config["source"]) == PROVENANCE_KEYS
    assert generator_config["source"]["aux_category"] == AUX_CATEGORY[network_type]
    assert generator_config["source"]["source_lan_sha256"] == DDM_SHA256
    assert set(generator_config["derive_stats"]) == STATS_KEYS
    assert generator_config["derive"]["integration_grid"] == 1000
    assert generator_config["derive"]["integration_max_t"] == 20.0
    assert generator_config["derive"]["seed"] == 0

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
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert [entry["file"] for entry in manifest["files"]] == [f.name for f in files]
    assert manifest["network_type"] == network_type
    assert manifest["n_files"] == 2 and manifest["n_theta_per_file"] == N_THETA
    assert set(manifest["derive_stats"]) == STATS_KEYS
    assert PROVENANCE_KEYS <= set(manifest["source"])
    assert manifest["grid"] == {"n_points": 1000, "max_t": 20.0, "t_min": 1e-4}
    assert manifest["seed"] == 0
    assert manifest["lanfactory_version"]
    for entry in manifest["files"]:
        assert STATS_KEYS <= set(entry)
        assert entry["n_rows"] == manifest["n_rows_per_file"]


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

    The full one-epoch round trip is not in the unit suite (it needs the
    trainers' pandas<3 overlay in this environment; L3 owns the committed
    round trip). The dry run exercises the real file discovery, key lookup,
    batch-size check and a first batch.
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
