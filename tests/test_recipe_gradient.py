"""A graph can have correct autodiff while representing the wrong likelihood."""

import numpy as np
import pytest
import torch

from lanfactory.recipes import validation


class Gaussian(torch.nn.Module):
    def __init__(self, correct):
        super().__init__()
        self.correct = correct
        self.projection = torch.nn.Linear(4, 1)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor([[-5.0, 0.0, 1.0, 0.0]]) if correct else torch.zeros(1, 4)
            )
            self.projection.bias.fill_(-2.0 if correct else 0.0)

    def forward(self, x):
        projected = self.projection(x)
        if self.correct:
            return -0.5 * projected**2 - 1.612085714
        return projected


@pytest.mark.parametrize("correct", [False, True])
def test_gradient_fidelity_is_separate_from_autodiff(correct, monkeypatch, tmp_path):
    pytest.importorskip("hssm")
    path = tmp_path / "density.onnx"
    network = Gaussian(correct)
    torch.onnx.export(network, torch.zeros(1, 4), str(path), dynamo=False)

    def simulate(config, theta, *args, **kwargs):
        return {
            "theta": theta,
            "rts": np.array([1.0, 2.0, 3.0] * 2),
            "choices": np.array([0, 0, 0, 1, 1, 1]),
        }

    class KDE:
        def __init__(self, draws):
            self.v = draws["theta"][0]

        def evaluate(self, rt, choices):
            return -0.5 * (rt - (5 * self.v + 2)) ** 2 - 1.612085714

    monkeypatch.setattr(validation, "simulate", simulate)
    monkeypatch.setattr(validation, "make_kde", KDE)
    cfg = {
        "params": ["v", "fixed"],
        "choices": [0, 1],
        "param_bounds_dict": {"v": [-1.0, 1.0], "fixed": [0.2, 0.2]},
    }
    gate = validation.gate_gradient(
        path,
        validation.ONNXPredictor(path),
        cfg,
        [np.array([0.0, 0.2])],
        n_sim=6,
        seed=1,
        max_t=20.0,
        delta_t=0.001,
    )
    assert gate["points"][0]["autodiff_max_error"] < 0.01
    assert gate["passed"] is correct


def test_export_parity_rejects_a_different_network(tmp_path):
    path = tmp_path / "density.onnx"
    network = torch.nn.Linear(4, 1)
    torch.onnx.export(network, torch.zeros(1, 4), str(path), dynamo=False)
    with torch.no_grad():
        network.bias.add_(1)
    gate = validation.gate_parity(
        validation.ONNXPredictor(path), network, np.ones((5, 4))
    )
    assert gate["passed"] is False
