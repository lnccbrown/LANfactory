import importlib
import sys

import numpy as np
import pandas as pd
import pytest


class DummyLogKDE:
    def __init__(self, out):
        self.out = out

    def kde_eval(self, data):
        return np.zeros(len(data["rts"]), dtype=np.float32)


@pytest.fixture
def two_choice_config():
    return {"choices": [-1, 1], "params": ["v", "a"]}


@pytest.fixture
def three_choice_config():
    return {"choices": [0, 1, 2], "params": ["v", "a"]}


@pytest.fixture
def single_parameter_df():
    return pd.DataFrame([[0.1, 1.0]], columns=["v", "a"])


@pytest.fixture
def network_inspectors_module():
    return importlib.import_module("lanfactory.network_inspectors")


@pytest.mark.xfail(
    reason="lanfactory eagerly imports network_inspectors from package __init__",
    strict=True,
)
def test_import_lanfactory_does_not_eagerly_import_network_inspectors(monkeypatch):
    monkeypatch.delitem(sys.modules, "lanfactory.network_inspectors", raising=False)
    monkeypatch.delitem(sys.modules, "lanfactory", raising=False)

    importlib.import_module("lanfactory")

    assert "lanfactory.network_inspectors" not in sys.modules


@pytest.mark.xfail(
    reason="get_torch_mlp falls through to a late TypeError when LoadTorchMLPInfer is unavailable",
    strict=True,
)
def test_get_torch_mlp_raises_clear_error_when_loader_is_unavailable(
    monkeypatch, network_inspectors_module
):
    monkeypatch.setattr(network_inspectors_module, "LoadTorchMLPInfer", None)

    with pytest.raises(ImportError, match="LoadTorchMLPInfer"):
        network_inspectors_module.get_torch_mlp(
            model_file_path="model.pt",
            network_config={"network_type": "lan"},
            input_dim=4,
        )


@pytest.mark.xfail(
    reason="kde_vs_lan_likelihoods assumes plt.subplots always returns a 2D axes array",
    strict=True,
)
def test_kde_vs_lan_likelihoods_handles_single_subplot_layout(
    monkeypatch, two_choice_config, single_parameter_df, network_inspectors_module
):
    monkeypatch.setattr(
        network_inspectors_module.ModelConfigBuilder,
        "from_model",
        lambda model: two_choice_config,
    )
    monkeypatch.setattr(
        network_inspectors_module, "simulator", lambda **kwargs: {"x": 1}
    )
    monkeypatch.setattr(network_inspectors_module, "LogKDE", DummyLogKDE)
    monkeypatch.setattr(
        network_inspectors_module.sns, "lineplot", lambda *args, **kwargs: None
    )

    network_inspectors_module.kde_vs_lan_likelihoods(
        parameter_df=single_parameter_df,
        model="ddm",
        torch_mlp_predict=lambda batch: np.zeros((batch.shape[0], 1), dtype=np.float32),
        n_reps=1,
        cols=1,
        show=False,
    )


@pytest.mark.xfail(
    reason="kde_vs_lan_likelihoods hardcodes a 4000-row input batch for non-binary choice models",
    strict=True,
)
def test_kde_vs_lan_likelihoods_sizes_input_batch_from_choice_count(
    monkeypatch,
    three_choice_config,
    single_parameter_df,
    network_inspectors_module,
):
    monkeypatch.setattr(
        network_inspectors_module.ModelConfigBuilder,
        "from_model",
        lambda model: three_choice_config,
    )
    monkeypatch.setattr(
        network_inspectors_module, "simulator", lambda **kwargs: {"x": 1}
    )
    monkeypatch.setattr(network_inspectors_module, "LogKDE", DummyLogKDE)
    monkeypatch.setattr(
        network_inspectors_module.sns, "lineplot", lambda *args, **kwargs: None
    )

    network_inspectors_module.kde_vs_lan_likelihoods(
        parameter_df=single_parameter_df,
        model="lca_3",
        torch_mlp_predict=lambda batch: np.zeros((batch.shape[0], 1), dtype=np.float32),
        n_reps=1,
        cols=1,
        show=False,
    )


@pytest.mark.xfail(
    reason="kde_vs_lan_likelihoods ignores the caller-provided font_scale value",
    strict=True,
)
def test_kde_vs_lan_likelihoods_passes_font_scale_argument(
    monkeypatch,
    two_choice_config,
    single_parameter_df,
    network_inspectors_module,
):
    seen = {}
    multi_parameter_df = pd.concat([single_parameter_df] * 4, ignore_index=True)

    monkeypatch.setattr(
        network_inspectors_module.ModelConfigBuilder,
        "from_model",
        lambda model: two_choice_config,
    )
    monkeypatch.setattr(
        network_inspectors_module, "simulator", lambda **kwargs: {"x": 1}
    )
    monkeypatch.setattr(network_inspectors_module, "LogKDE", DummyLogKDE)
    monkeypatch.setattr(
        network_inspectors_module.sns, "lineplot", lambda *args, **kwargs: None
    )

    def fake_set(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(network_inspectors_module.sns, "set", fake_set)

    network_inspectors_module.kde_vs_lan_likelihoods(
        parameter_df=multi_parameter_df,
        model="ddm",
        torch_mlp_predict=lambda batch: np.zeros((batch.shape[0], 1), dtype=np.float32),
        n_reps=1,
        cols=2,
        show=False,
        font_scale=3.25,
    )

    assert seen["font_scale"] == 3.25


@pytest.mark.xfail(
    reason="kde_vs_lan_likelihoods does not validate parameter_df before dereferencing it",
    strict=True,
)
def test_kde_vs_lan_likelihoods_rejects_missing_parameter_df(
    monkeypatch, two_choice_config, network_inspectors_module
):
    monkeypatch.setattr(
        network_inspectors_module.ModelConfigBuilder,
        "from_model",
        lambda model: two_choice_config,
    )

    with pytest.raises(ValueError, match="parameter_df"):
        network_inspectors_module.kde_vs_lan_likelihoods(
            parameter_df=None,
            model="ddm",
            torch_mlp_predict=lambda batch: np.zeros(
                (batch.shape[0], 1), dtype=np.float32
            ),
            show=False,
        )


@pytest.mark.xfail(
    reason="lan_manifold default vary_dict uses a list but the implementation expects an array with .shape",
    strict=True,
)
def test_lan_manifold_accepts_default_vary_dict(
    monkeypatch, single_parameter_df, network_inspectors_module
):
    monkeypatch.setattr(
        network_inspectors_module.ModelConfigBuilder,
        "from_model",
        lambda model: {"choices": [-1, 1], "params": ["v", "a"]},
    )

    network_inspectors_module.lan_manifold(
        parameter_df=single_parameter_df,
        model="ddm",
        torch_mlp_predict=lambda batch: np.zeros((batch.shape[0], 1), dtype=np.float32),
        show=False,
    )


@pytest.mark.xfail(
    reason="lan_manifold does not validate torch_mlp_predict before calling it",
    strict=True,
)
def test_lan_manifold_rejects_missing_predictor(
    monkeypatch, single_parameter_df, network_inspectors_module
):
    monkeypatch.setattr(
        network_inspectors_module.ModelConfigBuilder,
        "from_model",
        lambda model: {"choices": [-1, 1], "params": ["v", "a"]},
    )

    with pytest.raises(ValueError, match="torch_mlp_predict"):
        network_inspectors_module.lan_manifold(
            parameter_df=single_parameter_df,
            vary_dict={"v": np.array([-1.0, 0.0, 1.0], dtype=np.float32)},
            model="ddm",
            torch_mlp_predict=None,
            show=False,
        )
