"""Tests for network inspector Streamlit model discovery helpers."""

from pathlib import Path

import pytest
from lanfactory.network_inspectors.streamlit_app import (
    _available_models,
    _model_directories,
    _resolve_model_paths,
)


def _add_model_artifacts(model_dir: Path, state_name: str, config_name: str) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / state_name).touch()
    (model_dir / config_name).touch()


def test_available_models_finds_flat_and_nested_layouts(tmp_path):
    _add_model_artifacts(
        tmp_path / "angle",
        "angle_state_dict.pt",
        "angle_network_config.pickle",
    )
    _add_model_artifacts(
        tmp_path / "lan" / "ddm",
        "ddm_train_state_dict.pt",
        "ddm_network_config.pickle",
    )

    assert _available_models(str(tmp_path)) == ["angle", "ddm"]


def test_model_directories_ignores_invalid_and_unknown_directories(tmp_path):
    _add_model_artifacts(
        tmp_path / "angle",
        "angle_state_dict.pt",
        "angle_network_config.pickle",
    )
    (tmp_path / "ddm").mkdir()
    (tmp_path / "not_an_ssms_model").mkdir()
    (tmp_path / "not_an_ssms_model" / "state_dict.pt").touch()
    (tmp_path / "not_an_ssms_model" / "network_config.pickle").touch()

    assert _model_directories(str(tmp_path)) == {"angle": tmp_path / "angle"}


def test_resolve_model_paths_finds_nested_artifacts(tmp_path):
    model_dir = tmp_path / "lan" / "ddm"
    _add_model_artifacts(
        model_dir,
        "ddm_train_state_dict.pt",
        "ddm_network_config.pickle",
    )

    state_path, config_path = _resolve_model_paths(str(tmp_path), "ddm")

    assert state_path == model_dir / "ddm_train_state_dict.pt"
    assert config_path == model_dir / "ddm_network_config.pickle"


def test_resolve_model_paths_reports_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        _resolve_model_paths(str(tmp_path), "ddm")
