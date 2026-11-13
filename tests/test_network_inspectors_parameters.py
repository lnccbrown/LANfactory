"""Tests for network inspector parameter sampling and sweep bounds."""

import numpy as np
import pandas as pd
import pytest
import ssms

from lanfactory.network_inspectors.parameters import (
    make_parameter_df,
    missing_parameters,
    out_of_bounds,
    sweep_bound_problems,
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


def _full_ddm_rv(**overrides):
    config = ssms.config.model_config["full_ddm_rv"]
    values = dict(zip(config["params"], config["default_params"], strict=True))
    return {**values, **overrides}


def test_sweep_bound_problems_checks_dependent_bounds_along_the_sweep():
    base = _full_ddm_rv(t=1.0, st=0.3)

    # st <= t holds at every swept t, although the t default (0.25) is below st.
    assert sweep_bound_problems("full_ddm_rv", "t", base, 0.5, 2.0) == []
    # Sweeping t down to 0.25 crosses below st.
    problems = sweep_bound_problems("full_ddm_rv", "t", base, 0.25, 2.0)
    assert len(problems) == 1
    assert problems[0].startswith("st = 0.3")


def test_sweep_bound_problems_reports_sweep_outside_bounds():
    lower, upper = sweep_bounds("ddm", "a")
    base = dict(
        zip(
            ssms.config.model_config["ddm"]["params"],
            ssms.config.model_config["ddm"]["default_params"],
            strict=True,
        )
    )

    assert sweep_bound_problems("ddm", "a", base, lower, upper) == []
    assert sweep_bound_problems("ddm", "a", base, lower, upper + 1) == [
        f"a sweep {lower:g} to {upper + 1:g} (bounds {lower:g} to {upper:g})"
    ]


def _defaults(model):
    config = ssms.config.model_config[model]
    return dict(zip(config["params"], config["default_params"], strict=True))


def test_sweep_bound_problems_flags_broken_lba_ordering():
    base = {**_defaults("lba2"), "A": 0.3, "b": 0.5}

    assert sweep_bound_problems("lba2", "v0", base, 0.5, 1.0) == []
    # b swept below A at the low end.
    assert sweep_bound_problems("lba2", "b", base, 0.2, 0.9) == [
        "b must exceed A at b = 0.2"
    ]
    # Fixed b already below A.
    flipped = {**base, "A": 0.6}
    assert "b must exceed A" in sweep_bound_problems("lba2", "v0", flipped, 0.5, 1.0)
