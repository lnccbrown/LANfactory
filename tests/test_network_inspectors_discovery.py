"""Tests for network inspector model discovery and artifact pairing."""

import os
from pathlib import Path

import pytest

from lanfactory.network_inspectors.discovery import (
    available_models,
    model_directories,
    resolve_model_paths,
)


def _add_model_artifacts(model_dir: Path, state_name: str, config_name: str) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / state_name).touch()
    (model_dir / config_name).touch()


RUN_A = "a" * 32
RUN_B = "b" * 32


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

    assert available_models(str(tmp_path)) == ["angle", "ddm"]


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

    assert model_directories(str(tmp_path)) == {"angle": tmp_path / "angle"}


def test_resolve_model_paths_finds_nested_artifacts(tmp_path):
    model_dir = tmp_path / "lan" / "ddm"
    _add_model_artifacts(
        model_dir,
        "ddm_train_state_dict.pt",
        "ddm_network_config.pickle",
    )

    state_path, config_path = resolve_model_paths(str(tmp_path), "ddm")

    assert state_path == model_dir / "ddm_train_state_dict.pt"
    assert config_path == model_dir / "ddm_network_config.pickle"


def test_resolve_model_paths_reports_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        resolve_model_paths(str(tmp_path), "ddm")


def test_model_directories_skips_non_lan_network_types(tmp_path):
    for network_type in ("cpn", "opn"):
        _add_model_artifacts(
            tmp_path / network_type / "ddm",
            "ddm_state_dict.pt",
            "ddm_network_config.pickle",
        )
    _add_model_artifacts(
        tmp_path / "lan" / "ddm",
        "ddm_train_state_dict.pt",
        "ddm_network_config.pickle",
    )

    assert model_directories(str(tmp_path)) == {"ddm": tmp_path / "lan" / "ddm"}


def test_model_directories_bounds_scan_depth(tmp_path):
    _add_model_artifacts(
        tmp_path / "a" / "b" / "c" / "ddm",
        "ddm_state_dict.pt",
        "ddm_network_config.pickle",
    )

    assert model_directories(str(tmp_path)) == {}


def test_resolve_model_paths_pairs_artifacts_by_run_uuid(tmp_path):
    model_dir = tmp_path / "lan" / "ddm"
    model_dir.mkdir(parents=True)
    # Orphan config from a failed run sorts first and must not be chosen.
    (model_dir / f"{RUN_A}_lan_ddm_network_config.pickle").touch()
    (model_dir / f"{RUN_B}_lan_ddm_network_config.pickle").touch()
    (model_dir / f"ddm_lan_{RUN_B}_train_state_dict.pt").touch()

    state_path, config_path = resolve_model_paths(str(tmp_path), "ddm")

    assert state_path == model_dir / f"ddm_lan_{RUN_B}_train_state_dict.pt"
    assert config_path == model_dir / f"{RUN_B}_lan_ddm_network_config.pickle"


def test_resolve_model_paths_rejects_unmatched_run_uuids(tmp_path):
    model_dir = tmp_path / "ddm"
    _add_model_artifacts(
        model_dir,
        f"ddm_lan_{RUN_A}_train_state_dict.pt",
        f"{RUN_B}_lan_ddm_network_config.pickle",
    )

    with pytest.raises(FileNotFoundError, match="same training run"):
        resolve_model_paths(str(tmp_path), "ddm")


def test_resolve_model_paths_rejects_ambiguous_legacy_artifacts(tmp_path):
    model_dir = tmp_path / "ddm"
    _add_model_artifacts(model_dir, "one_state_dict.pt", "one_network_config.pickle")
    (model_dir / "two_network_config.pickle").touch()

    with pytest.raises(FileNotFoundError, match="same training run"):
        resolve_model_paths(str(tmp_path), "ddm")


def test_model_directories_prefers_pairable_directory(tmp_path):
    # Flat ddm/ sorts before lan/ddm/ but its artifacts come from different runs.
    _add_model_artifacts(
        tmp_path / "ddm",
        f"ddm_lan_{RUN_A}_train_state_dict.pt",
        f"{RUN_B}_lan_ddm_network_config.pickle",
    )
    nested = tmp_path / "lan" / "ddm"
    _add_model_artifacts(
        nested,
        f"ddm_lan_{RUN_A}_train_state_dict.pt",
        f"{RUN_A}_lan_ddm_network_config.pickle",
    )

    assert model_directories(str(tmp_path)) == {"ddm": nested}
    _, config_path = resolve_model_paths(str(tmp_path), "ddm")
    assert config_path == nested / f"{RUN_A}_lan_ddm_network_config.pickle"


@pytest.mark.parametrize("newer", ["flat", "nested"])
def test_model_directories_picks_directory_with_newest_run(tmp_path, newer):
    dirs = {"flat": tmp_path / "ddm", "nested": tmp_path / "lan" / "ddm"}
    for run, model_dir in zip((RUN_A, RUN_B), dirs.values(), strict=True):
        _add_model_artifacts(
            model_dir,
            f"ddm_lan_{run}_train_state_dict.pt",
            f"{run}_lan_ddm_network_config.pickle",
        )
    newest_state = next(dirs[newer].glob("*state_dict*"))
    future = newest_state.stat().st_mtime + 10
    os.utime(newest_state, (future, future))

    assert model_directories(str(tmp_path)) == {"ddm": dirs[newer]}
