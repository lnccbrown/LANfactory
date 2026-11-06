"""Tests for network inspector parameter sampling and sweep bounds."""

import numpy as np
import pandas as pd
import pytest
import ssms

from lanfactory.network_inspectors.parameters import (
    make_parameter_df,
    missing_parameters,
    out_of_bounds,
    sweep_bounds,
)


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


@pytest.mark.parametrize("cleared", [np.nan, None])
def test_missing_parameters_flags_cleared_cells(cleared):
    row = pd.DataFrame([{"v": 0.5, "a": cleared, "z": 0.5, "t": 0.3}])

    assert missing_parameters(row, ["v", "a", "z", "t"]) == ["a"]


def test_missing_parameters_accepts_complete_row():
    row = pd.DataFrame(
        [ssms.config.model_config["ddm"]["default_params"]],
        columns=ssms.config.model_config["ddm"]["params"],
    )

    assert missing_parameters(row, ssms.config.model_config["ddm"]["params"]) == []


def test_out_of_bounds_reports_parameters_outside_training_bounds():
    lower, upper = sweep_bounds("ddm", "a")

    outside = out_of_bounds("ddm", {"v": 0.0, "a": upper + 1.0})

    assert outside == {"a": (lower, upper)}


def test_out_of_bounds_resolves_dependent_bounds_against_values():
    defaults = dict(
        zip(
            ssms.config.model_config["full_ddm_rv"]["params"],
            ssms.config.model_config["full_ddm_rv"]["default_params"],
            strict=True,
        )
    )
    values = {**defaults, "t": 0.3, "st": 0.35}  # st must not exceed t

    assert set(out_of_bounds("full_ddm_rv", values)) == {"st"}
    assert out_of_bounds("full_ddm_rv", {**values, "st": 0.1}) == {}
