"""Tests for network inspector API validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanfactory.network_inspectors import api


def _predictor(batch: np.ndarray) -> np.ndarray:
    return np.zeros((batch.shape[0], 1), dtype=np.float32)


def test_kde_vs_lan_rejects_empty_parameter_frame():
    with pytest.raises(ValueError, match="at least one parameter vector"):
        api.kde_vs_lan_likelihoods(pd.DataFrame(), "ddm", _predictor)


def test_kde_vs_lan_uses_model_parameter_order(monkeypatch):
    seen_params = []

    monkeypatch.setattr(api, "make_rt_choice_grid", lambda spec, grid: np.zeros((1, 2)))
    monkeypatch.setattr(
        api,
        "evaluate_network",
        lambda spec, params, grid: seen_params.append(params) or np.zeros(1),
    )
    monkeypatch.setattr(
        api, "simulate_ground_truth", lambda spec, params, n_samples: {}
    )
    monkeypatch.setattr(api, "evaluate_kde", lambda sim_out, grid: np.zeros(1))
    monkeypatch.setattr(api, "plot_kde_vs_lan", lambda grid, results, spec, cfg: None)

    parameter_df = pd.DataFrame(
        [{"extra": 99.0, "t": 0.3, "z": 0.5, "a": 1.5, "v": 0.2}]
    )

    api.kde_vs_lan_likelihoods(parameter_df, "ddm", _predictor, n_reps=1)

    np.testing.assert_array_equal(
        seen_params[0],
        np.array([0.2, 1.5, 0.5, 0.3], dtype=np.float32),
    )


def test_lan_manifold_validates_vary_dict():
    parameter_df = pd.DataFrame([{"v": 0.2, "a": 1.5, "z": 0.5, "t": 0.3}])

    with pytest.raises(ValueError, match="exactly one parameter"):
        api.lan_manifold(parameter_df, {}, "ddm", _predictor)

    with pytest.raises(ValueError, match="exactly one parameter"):
        api.lan_manifold(parameter_df, {"v": [0.1], "a": [1.0]}, "ddm", _predictor)

    with pytest.raises(ValueError, match="at least one value"):
        api.lan_manifold(parameter_df, {"v": []}, "ddm", _predictor)


def test_lan_manifold_normalizes_2d_parameter_array(monkeypatch):
    captured = {}

    def build_manifold_stub(spec, parameters, vary_name, vary_values, grid):
        captured["parameters"] = parameters
        return pd.DataFrame(
            {"rt": [1.0], "choice": [1], "vary": [0.2], "likelihood": [0.5]}
        )

    monkeypatch.setattr(api, "make_manifold_grid", lambda grid: np.zeros((1, 2)))
    monkeypatch.setattr(api, "build_manifold", build_manifold_stub)
    monkeypatch.setattr(
        api, "plot_manifold", lambda manifold, spec, vary_name, cfg: None
    )

    api.lan_manifold(
        np.array([[0.2, 1.5, 0.5, 0.3]], dtype=np.float32),
        {"v": [0.2]},
        "ddm",
        _predictor,
    )

    np.testing.assert_array_equal(
        captured["parameters"],
        np.array([0.2, 1.5, 0.5, 0.3], dtype=np.float32),
    )


def test_lan_manifold_rejects_wrong_parameter_shape():
    with pytest.raises(ValueError, match="Expected one parameter vector"):
        api.lan_manifold(
            np.array([[0.2, 1.5, 0.5]], dtype=np.float32),
            {"v": [0.2]},
            "ddm",
            _predictor,
        )
