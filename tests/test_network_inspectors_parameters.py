"""Tests for network inspector parameter sampling and sweep bounds."""

import numpy as np
import pytest
import ssms

from lanfactory.network_inspectors.parameters import make_parameter_df, sweep_bounds


@pytest.mark.parametrize("model", sorted(ssms.config.model_config))
def test_make_parameter_df_respects_bounds_for_every_model(model):
    config = ssms.config.model_config[model]
    df = make_parameter_df(model=model, n_rows=50, seed=0)

    assert list(df.columns) == list(config["params"])
    for param in config["params"]:
        lower, upper = config["param_bounds_dict"][param]
        lower = df[lower] if isinstance(lower, str) else lower
        upper = df[upper] if isinstance(upper, str) else upper
        assert np.all((df[param] >= lower - 1e-6) & (df[param] <= upper + 1e-6))


def test_sweep_bounds_resolves_dependent_endpoint():
    config = ssms.config.model_config["full_ddm_rv"]
    t_default = config["default_params"][config["params"].index("t")]

    assert sweep_bounds("full_ddm_rv", "st") == pytest.approx((0.001, t_default))
    assert sweep_bounds("full_ddm_rv", "v") == pytest.approx((-3.0, 3.0))


@pytest.mark.parametrize(
    ("model", "larger", "smaller"),
    [("lba2", "b", "A"), ("lba3", "b", "A"), ("lba_angle_3", "a", "z")],
)
def test_make_parameter_df_applies_sampling_constraints(model, larger, smaller):
    df = make_parameter_df(model=model, n_rows=200, seed=0)

    assert np.all(df[larger] >= df[smaller])


def test_sweep_bounds_resolves_against_edited_base_vector():
    assert sweep_bounds("full_ddm_rv", "st", {"t": 1.5}) == pytest.approx((0.001, 1.5))
