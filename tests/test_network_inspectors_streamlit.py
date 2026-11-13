"""Tests for network inspector Streamlit predictor loading."""

import os
from pathlib import Path

from lanfactory.network_inspectors import streamlit_app
from lanfactory.network_inspectors.streamlit_app import (
    _load_predictor,
    _load_predictor_cached,
)


def _add_model_artifacts(model_dir: Path, state_name: str, config_name: str) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / state_name).touch()
    (model_dir / config_name).touch()


RUN_A = "a" * 32
RUN_B = "b" * 32


def test_load_predictor_picks_up_newer_run(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        streamlit_app,
        "get_torch_mlp",
        lambda model_file_path, **_: loaded.append(model_file_path) or len(loaded),
    )
    _load_predictor_cached.clear()
    model_dir = tmp_path / "lan" / "ddm"
    _add_model_artifacts(
        model_dir,
        f"ddm_lan_{RUN_A}_train_state_dict.pt",
        f"{RUN_A}_lan_ddm_network_config.pickle",
    )

    first = _load_predictor(str(tmp_path), "ddm")
    assert _load_predictor(str(tmp_path), "ddm") == first  # cached

    newer_state = model_dir / f"ddm_lan_{RUN_B}_train_state_dict.pt"
    newer_state.touch()
    (model_dir / f"{RUN_B}_lan_ddm_network_config.pickle").touch()
    future = newer_state.stat().st_mtime + 10
    os.utime(newer_state, (future, future))

    assert _load_predictor(str(tmp_path), "ddm") != first
    assert loaded[-1] == str(newer_state)
    _load_predictor_cached.clear()


def test_app_scans_base_directory_once_per_rerun(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from lanfactory.network_inspectors import discovery, loaders

    walks = []
    original = discovery._iter_directories

    def counting(root, max_depth):
        walks.append(root)
        return original(root, max_depth)

    monkeypatch.setattr(discovery, "_iter_directories", counting)
    # AppTest re-executes the script as __main__, which imports get_torch_mlp
    # from the loaders module, so patch it there (not on streamlit_app).
    monkeypatch.setattr(
        loaders, "get_torch_mlp", lambda **_: lambda batch: batch[:, :1]
    )
    _load_predictor_cached.clear()
    _add_model_artifacts(
        tmp_path / "lan" / "ddm",
        f"ddm_lan_{RUN_A}_train_state_dict.pt",
        f"{RUN_A}_lan_ddm_network_config.pickle",
    )

    app = AppTest.from_file(streamlit_app.__file__, default_timeout=60)
    app.session_state["torch_models_base_dir"] = str(tmp_path)
    app.run()

    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert [subheader.value for subheader in app.subheader] == [
        "KDE vs LAN Likelihoods",
        "LAN Manifold",
    ]
    assert len(walks) == 1
    _load_predictor_cached.clear()


def _app_with_fake_ddm(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from lanfactory.network_inspectors import loaders

    monkeypatch.setattr(
        loaders, "get_torch_mlp", lambda **_: lambda batch: batch[:, :1]
    )
    _load_predictor_cached.clear()
    _add_model_artifacts(
        tmp_path / "lan" / "ddm",
        f"ddm_lan_{RUN_A}_train_state_dict.pt",
        f"{RUN_A}_lan_ddm_network_config.pickle",
    )
    app = AppTest.from_file(streamlit_app.__file__, default_timeout=120)
    app.session_state["torch_models_base_dir"] = str(tmp_path)
    return app.run()


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _slider(app, label):
    return next(slider for slider in app.slider if slider.label == label)


def test_manifold_result_survives_widget_changes(tmp_path, monkeypatch):
    app = _app_with_fake_ddm(tmp_path, monkeypatch)

    _button(app, "Run Manifold").click().run()
    assert len(app.get("plotly_chart")) == 1
    assert not app.info

    _slider(app, "Sweep steps").set_value(5).run()

    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert [info.value for info in app.info] == [streamlit_app._STALE_RESULT_NOTE]
    _load_predictor_cached.clear()


def _shows_kde_chart(app):
    return any(caption.value.startswith("KDE vs LAN chart") for caption in app.caption)


def test_kde_plot_options_apply_without_rerunning(tmp_path, monkeypatch):
    app = _app_with_fake_ddm(tmp_path, monkeypatch)
    _slider(app, "Parameter sets").set_value(1)
    _slider(app, "Simulator samples").set_value(_slider(app, "Simulator samples").min)
    _slider(app, "KDE repetitions").set_value(1)
    app.run()

    _button(app, "Run KDE vs LAN").click().run()
    assert not app.exception
    assert _shows_kde_chart(app)

    _slider(app, "KDE alpha").set_value(0.5).run()

    assert _shows_kde_chart(app)
    assert not app.info  # plot options do not invalidate the result
    _load_predictor_cached.clear()
