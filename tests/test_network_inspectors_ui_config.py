"""Tests for network inspector widget specs."""

from dataclasses import fields

import pytest

from lanfactory.network_inspectors import ui_config
from lanfactory.network_inspectors.ui_config import WidgetSpec

ALL_SPECS = [
    value for value in vars(ui_config).values() if isinstance(value, WidgetSpec)
]


def test_kwargs_omits_unset_fields_and_applies_overrides():
    spec = WidgetSpec(label="Sweep min", help="h", min_value=0)

    assert spec.kwargs() == {"label": "Sweep min", "help": "h", "min_value": 0}
    assert spec.kwargs(value=1.5, min_value=-1) == {
        "label": "Sweep min",
        "help": "h",
        "min_value": -1,
        "value": 1.5,
    }


def test_integer_specs_keep_integer_types():
    # Streamlit picks an int or float widget from these types.
    assert isinstance(ui_config.PARAMETER_SETS.kwargs()["value"], int)
    assert isinstance(ui_config.SEED.kwargs()["max_value"], int)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.label)
def test_specs_have_consistent_ranges(spec):
    assert spec.label and spec.help
    lo, hi, value = spec.min_value, spec.max_value, spec.value
    if lo is not None and hi is not None:
        assert lo < hi
    if value is not None:
        assert lo is None or value >= lo
        assert hi is None or value <= hi


def test_spec_fields_are_streamlit_keyword_names():
    assert {field.name for field in fields(WidgetSpec)} == {
        "label",
        "help",
        "value",
        "min_value",
        "max_value",
        "step",
        "width",
    }
