"""Round-trip spike: torch MLP -> ONNX -> {onnxruntime, jaxonnxruntime}.

Validates the toolchain assumptions sbi's exporter (lands in C3) will rely on,
without sbi in the loop. If torch.onnx.export, onnxruntime, or jaxonnxruntime
regress on a vanilla MLP, this test catches it before debugging the real
exporter. Kept as a permanent regression guard per plans/sbi-onnx-integration.md.
"""

from pathlib import Path

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from jaxonnxruntime import call_onnx
from torch import nn


@pytest.mark.flaky(reruns=2)
def test_mlp_three_way_agreement(tmp_path: Path) -> None:
    torch.manual_seed(0)

    input_dim = 6
    hidden_dim = 32

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    ).eval()

    x_input = torch.randn(1, input_dim, dtype=torch.float32)

    with torch.no_grad():
        y_torch = model(x_input).detach().numpy()

    onnx_path = tmp_path / "mlp.onnx"
    dummy_input = torch.randn(1, input_dim, requires_grad=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        dynamo=False,
        opset_version=17,
    )

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    y_ort = sess.run(None, {input_name: x_input.numpy()})[0]

    onnx_model = onnx.load(str(onnx_path))
    model_func, model_weights = call_onnx.call_onnx_model(
        onnx_model, {input_name: np.asarray(x_input.numpy())}
    )
    run_func = jax.tree_util.Partial(model_func, model_weights)
    y_jax = np.asarray(run_func({input_name: x_input.numpy()})[0])

    atol = 1e-5
    assert np.allclose(y_torch, y_ort, atol=atol), (
        f"torch vs onnxruntime mismatch: max |Δ| = {np.abs(y_torch - y_ort).max()}"
    )
    assert np.allclose(y_torch, y_jax, atol=atol), (
        f"torch vs jaxonnxruntime mismatch: max |Δ| = {np.abs(y_torch - y_jax).max()}"
    )
    assert np.allclose(y_ort, y_jax, atol=atol), (
        f"onnxruntime vs jaxonnxruntime mismatch: max |Δ| = {np.abs(y_ort - y_jax).max()}"
    )
