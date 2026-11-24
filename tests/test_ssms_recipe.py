"""Regressions for the ssms recipe's data and validation boundaries."""

from copy import deepcopy

import numpy as np
import pytest
from ssms.config import ModelConfigBuilder, get_lan_kde_config

from lanfactory.recipes.from_ssms import LANArtifact, batch_divisor, train_lan_for_model
from lanfactory.recipes.guards import (
    GuardedSimulationPipeline,
    normalize_model_config,
    simulation_issues,
)
from lanfactory.recipes.validation import gate_density, hssm_config


def fixture_sim(rts=None, choices=None):
    return {
        "rts": np.array([0.3, 0.5, 0.7, 0.4, 0.6, 0.8] if rts is None else rts),
        "choices": np.array([0, 0, 0, 1, 1, 1] if choices is None else choices),
        "metadata": {"possible_choices": [0, 1], "t": 0.1, "max_t": 20.0},
    }


def pipeline(filters=None):
    obj = GuardedSimulationPipeline.__new__(GuardedSimulationPipeline)
    obj.generator_config = get_lan_kde_config()
    if filters is not None:
        obj.generator_config["pipeline"]["simulation_filters"] = filters
    return obj


@pytest.mark.parametrize(
    "sim",
    [
        fixture_sim([20.1] * 6),
        fixture_sim(choices=[0] * 6),
        fixture_sim([0.4] * 3 + [0.6] * 3),
        fixture_sim([-0.999, 0.5, 0.7, 0.4, 0.6, 0.8]),
        fixture_sim([float("nan"), 0.5, 0.7, 0.4, 0.6, 0.8]),
    ],
)
def test_canonical_filters_reject_degenerate_simulations(sim):
    assert not pipeline()._is_valid_simulation(sim)[0]


@pytest.mark.parametrize(
    "filters",
    [
        {"mode": 0.3},
        {"std": np.std([0.3, 0.5, 0.7])},
        {"choice_cnt": 3},
        {"mean_rt": 0.1},
        {"mode_cnt_rel": 0.2},
    ],
)
def test_canonical_key_and_strict_thresholds(filters):
    assert not pipeline(filters)._is_valid_simulation(fixture_sim())[0]


def test_valid_simulation_and_guards_are_local():
    assert pipeline()._is_valid_simulation(fixture_sim())[0]
    from ssms.dataset_generators.pipelines.simulation_pipeline import SimulationPipeline

    assert (
        SimulationPipeline._is_valid_simulation
        is not GuardedSimulationPipeline._is_valid_simulation
    )


@pytest.mark.parametrize("rt", [-999.0, 0.0, np.inf, 20.1, 21.0])
def test_horizon_policy_rejects_censor_clamp_and_ignored_horizon(rt):
    issues = simulation_issues(fixture_sim([rt] * 6), 20.0)
    assert issues["invalid_rt"] + issues["at_or_past_horizon"] > 0


def test_registered_bounds_are_normalized_without_mutation(monkeypatch):
    from ssms.config import get_model_registry, register_model_config

    config = ModelConfigBuilder.from_model("angle")
    config["name"] = "recipe_runtime_angle"
    config.pop("param_bounds_dict", None)
    before = deepcopy(config)
    registry = get_model_registry()
    monkeypatch.setattr(registry, "_configs", dict(registry._configs))
    register_model_config(config["name"], config)
    runtime = ModelConfigBuilder.from_model(config["name"])
    normalized = normalize_model_config(runtime)
    assert list(normalized["param_bounds_dict"]) == config["params"]
    assert "param_bounds_dict" not in config
    assert config == before


def test_bounds_from_raw_source_override_stale_derived_bounds():
    cfg = ModelConfigBuilder.from_model("ddm")
    cfg["param_bounds"][0][0], cfg["param_bounds"][1][0] = -1.0, 1.0
    normalized = normalize_model_config(cfg)
    assert normalized["param_bounds_dict"]["v"] == (-1.0, 1.0)


@pytest.mark.parametrize(
    "change",
    [
        {"param_bounds": [[0.0], [1.0]]},
        {"param_bounds": {"v": (-1.0, 1.0)}},
        {"param_bounds": [[-1.0, 0.3, 0.1, 0.0], [float("inf"), 3.0, 0.9, 2.0]]},
        {"param_bounds": [[-1.0, 0.3, 0.1, 0.0], [-1.0, 3.0, 0.9, 2.0]]},
    ],
)
def test_malformed_bounds_fail_before_training(change):
    with pytest.raises(ValueError):
        normalize_model_config(ModelConfigBuilder.from_model("ddm") | change)


def test_fixed_parameters_are_fixed_hssm_priors():
    cfg = normalize_model_config(
        ModelConfigBuilder.from_model("ddm"), {"a": 1.0, "z": 0.5, "t": 0.2}
    )
    out = hssm_config(cfg)
    assert out["list_params"] == cfg["params"]
    assert out["bounds"] == {"v": [-3.0, 3.0]}
    assert out["default_priors"] == {"a": 1.0, "z": 0.5, "t": 0.2}


def test_bad_fixed_value_and_all_fixed_are_rejected():
    cfg = ModelConfigBuilder.from_model("ddm")
    with pytest.raises(ValueError, match="outside"):
        normalize_model_config(cfg, {"v": 100})
    with pytest.raises(ValueError, match="at least one"):
        normalize_model_config(cfg, dict(zip(cfg["params"], cfg["default_params"])))


def test_batch_divisor():
    with pytest.warns(UserWarning, match="using 250"):
        assert batch_divisor(1000, 256) == 250
    assert batch_divisor(4096, 256) == 256
    with pytest.raises(ValueError):
        batch_divisor(0, 256)


def test_production_needs_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError, match="production requires"):
        train_lan_for_model("ddm", tmp_path, preset="production")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"preset": "unknown"}, "unknown preset"),
        ({"seed": -1}, "nonnegative"),
        ({"generator_config": {"missing": {}}}, "generator override"),
        ({"generator_config": {"pipeline": 1}}, "generator override"),
        ({"estimator": "unknown"}, "estimator must"),
        (
            {
                "estimator": "pyddm",
                "model_config": ModelConfigBuilder.from_model("race_3"),
            },
            "name must agree",
        ),
    ],
)
def test_invalid_recipe_requests_fail_without_artifacts(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        train_lan_for_model("ddm", tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_pyddm_rejects_multi_accumulator_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="single-particle"):
        train_lan_for_model("race_3", tmp_path, estimator="pyddm")
    assert not list(tmp_path.iterdir())


def test_recipe_never_reuses_a_stale_output_certificate(tmp_path):
    certificate = tmp_path / "model_config.json"
    certificate.write_text("prior run")
    with pytest.raises(FileExistsError, match="never reuse"):
        train_lan_for_model("ddm", tmp_path)
    assert certificate.read_text() == "prior run"


def test_skipped_or_missing_gate_cannot_certify(tmp_path):
    passing = {g: {"passed": True} for g in ("G1", "G2", "G3", "G4", "G5")}
    artifact = LANArtifact(tmp_path / "net.onnx", {}, passing, {}, tmp_path)
    assert artifact.validated
    passing["G5"]["skipped"] = True
    assert not artifact.validated
    del passing["G5"]
    assert not artifact.validated


def test_constant_bad_density_cannot_pass(monkeypatch):
    from lanfactory.recipes import validation

    monkeypatch.setattr(validation, "simulate", lambda *a, **kw: fixture_sim())

    class KDE:
        def evaluate(self, rt, choice):
            return np.full(len(rt), -np.log(2 * 20))

    monkeypatch.setattr(validation, "make_kde", lambda x: KDE())
    cfg = {"choices": [0, 1]}
    check = gate_density(
        lambda rows: np.zeros(len(rows)),
        cfg,
        [np.array([1.0])],
        n_sim=6,
        seed=1,
        max_t=20.0,
        delta_t=0.001,
    )
    assert check["passed"] is False


def test_nonfinite_reference_cannot_pass(monkeypatch):
    from lanfactory.recipes import validation

    monkeypatch.setattr(validation, "simulate", lambda *a, **kw: fixture_sim())

    class KDE:
        def evaluate(self, rt, choice):
            return np.full(len(rt), np.nan)

    monkeypatch.setattr(validation, "make_kde", lambda x: KDE())
    check = gate_density(
        lambda rows: np.zeros(len(rows)),
        {"choices": [0, 1]},
        [np.array([1.0])],
        n_sim=6,
        seed=1,
        max_t=20.0,
        delta_t=0.001,
    )
    assert check["passed"] is False
    assert "nonfinite reference" in check["error"]


def test_nonfinite_failed_metrics_remain_strict_json(tmp_path):
    import json

    from lanfactory.recipes.from_ssms import _write

    path = tmp_path / "failure.json"
    _write(path, {"passed": False, "error": np.array([np.nan, np.inf])})
    assert json.loads(path.read_text())["error"] == [None, None]


@pytest.mark.slow
def test_toy_recipe_runs_real_runtime_angle_training_without_certificate(tmp_path):
    cfg = ModelConfigBuilder.from_model("angle")
    cfg["name"] = "recipe_toy_angle"
    cfg.pop("param_bounds_dict", None)
    artifact = train_lan_for_model(
        cfg["name"],
        tmp_path,
        model_config=cfg,
        fixed_parameters={"a": 1.0, "z": 0.5, "t": 0.2, "theta": 0.2},
        generator_config={
            "pipeline": {"n_parameter_sets": 4},
            "training": {"n_samples_per_param": 64},
        },
        train_config={"n_epochs": 1},
        validate=False,
    )
    assert artifact.onnx_path.exists()
    assert artifact.hssm_model_config is None
    assert not artifact.validated
    assert not (tmp_path / "model_config.json").exists()
    assert len(artifact.manifest["files"]) == 2
    from lanfactory.recipes.validation import gate_structure

    assert gate_structure(artifact.onnx_path, len(cfg["params"]) + 2)["passed"]


@pytest.mark.slow
def test_real_hssm_validation_rejects_an_undertrained_nonanalytic_model(tmp_path):
    pytest.importorskip("hssm")
    cfg = ModelConfigBuilder.from_model("angle")
    cfg["param_bounds"][0][0] = -0.5
    cfg["param_bounds"][1][0] = 0.5
    artifact = train_lan_for_model(
        "angle",
        tmp_path,
        model_config=cfg,
        fixed_parameters={"a": 1.0, "z": 0.5, "t": 0.2, "theta": 0.2},
        generator_config={
            "pipeline": {"n_parameter_sets": 4},
            "training": {"n_samples_per_param": 64},
        },
        network_config={"layer_sizes": [16, 1], "activations": ["tanh", "linear"]},
        train_config={"n_epochs": 1},
    )
    assert artifact.validation["G1"]["passed"]
    assert artifact.validation["G2"]["passed"]
    assert artifact.validation["G3"]["passed"], artifact.validation["G3"]
    # This is an intentionally undertrained likelihood: good export and
    # finite HSSM logp do not establish a usable density or its sensitivity.
    assert not artifact.validation["G4"]["passed"]
    assert len(artifact.validation["G5"]["points"]) == 5
    assert (
        max(p["autodiff_max_error"] for p in artifact.validation["G5"]["points"]) < 0.01
    )
    assert artifact.hssm_model_config is None
    assert not (tmp_path / "model_config.json").exists()


@pytest.mark.parametrize("structure_passes", [True, False])
def test_certificate_output_boundary_with_injected_gate_verdicts(
    tmp_path, monkeypatch, structure_passes
):
    from lanfactory.recipes import validation

    monkeypatch.setattr(
        validation, "gate_structure", lambda *a: {"passed": structure_passes}
    )
    # Isolate orchestration/output from the independently tested statistical
    # gates. Training and export are real; this is not a likelihood-quality test.
    for name in ("gate_hssm", "gate_density", "gate_gradient"):
        monkeypatch.setattr(validation, name, lambda *a, **kw: {"passed": True})
    artifact = train_lan_for_model(
        "angle",
        tmp_path,
        fixed_parameters={"a": 1.0, "z": 0.5, "t": 0.2, "theta": 0.2},
        generator_config={
            "pipeline": {"n_parameter_sets": 4},
            "training": {"n_samples_per_param": 64},
        },
        train_config={"n_epochs": 1},
    )
    assert artifact.validated is structure_passes
    assert (tmp_path / "model_config.json").exists() is structure_passes
    if structure_passes:
        assert artifact.hssm_model_config["default_priors"]["theta"] == 0.2
    else:
        assert all(artifact.validation[g]["skipped"] for g in ("G2", "G3", "G4", "G5"))
