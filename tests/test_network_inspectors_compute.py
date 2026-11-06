"""Tests for choice encoding in the network inspector computations."""

import numpy as np
import pytest
import ssms

from lanfactory.network_inspectors.compute import (
    build_manifold,
    encode_choices,
    evaluate_kde,
    make_manifold_grid,
    make_rt_choice_grid,
    simulate_ground_truth,
)
from lanfactory.network_inspectors.config import GridSpec, ModelSpec


def _choice_values(model: str) -> list:
    return sorted(set(encode_choices(_spec(model), _grid(model))[:, 1].tolist()))


def _spec(model: str, predictor=None) -> ModelSpec:
    return ModelSpec.from_model(model, predictor=predictor)


def _grid(model: str) -> np.ndarray:
    return make_rt_choice_grid(_spec(model), GridSpec(n_points_2c=5, n_points_mc=5))


@pytest.mark.parametrize(
    ("model", "expected"),
    [("ddm", [-1, 1]), ("lba2", [0, 1]), ("race_2", [0, 1]), ("race_3", [0, 1, 2])],
)
def test_encode_choices_uses_model_choice_values(model, expected):
    assert _choice_values(model) == expected


def test_encode_choices_keeps_display_grid_and_rts():
    grid = _grid("lba2")

    encoded = encode_choices(_spec("lba2"), grid)

    np.testing.assert_array_equal(encoded[:, 0], grid[:, 0])
    np.testing.assert_array_equal(encoded[grid[:, 1] < 0, 1], 0)
    np.testing.assert_array_equal(encoded[grid[:, 1] > 0, 1], 1)
    assert set(grid[:, 1].tolist()) == {-1, 1}  # input untouched


@pytest.mark.parametrize("model", sorted(ssms.config.model_config))
def test_every_model_spec_has_one_choice_value_per_choice(model):
    spec = _spec(model)

    assert spec.n_choices >= 2
    assert len(set(spec.choices)) == spec.n_choices


def test_model_spec_falls_back_to_nchoices():
    assert _spec("lca_3").choices == [0, 1, 2]


@pytest.mark.parametrize("model", ["ddm", "lba2"])
def test_kde_has_mass_on_both_choices_once_encoded(model):
    spec = _spec(model)
    grid = make_rt_choice_grid(spec, GridSpec(n_points_2c=400, rt_step_2c=0.01))
    params = ssms.config.model_config[model]["default_params"]
    sim = simulate_ground_truth(spec, params, n_samples=2000)

    likelihood = np.exp(evaluate_kde(sim, encode_choices(spec, grid)))

    assert likelihood[grid[:, 1] < 0].max() > 0.01
    assert likelihood[grid[:, 1] > 0].max() > 0.01


def test_build_manifold_feeds_model_choices_but_reports_display_choices():
    seen = []

    def predictor(batch):
        seen.append(batch[:, -1].copy())
        return np.zeros((batch.shape[0], 1))

    spec = _spec("lba2", predictor=predictor)
    grid = make_manifold_grid(GridSpec(n_rt_steps=4))
    base = ssms.config.model_config["lba2"]["default_params"]

    manifold = build_manifold(spec, base, "v0", [0.5, 1.0], grid)

    assert set(np.concatenate(seen).tolist()) == {0, 1}
    assert set(manifold["choice"].tolist()) == {-1, 1}
