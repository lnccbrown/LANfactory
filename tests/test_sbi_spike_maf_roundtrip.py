"""Round-trip spike: nflows MAF.log_prob -> ONNX -> {onnxruntime, jaxonnxruntime}.

Validates that a non-trivial flow architecture (masked autoregressive flow)
survives the toolchain. Confirms that masked dense layers, log-det-Jacobian
accumulation, and the affine-autoregressive ops translate cleanly into
jaxonnxruntime. Kept as a permanent regression guard per
plans/sbi-onnx-integration.md.
"""

from pathlib import Path

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from jaxonnxruntime import call_onnx, config
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    ReversePermutation,
)
from torch import nn

# Friction discovered in C2: nflows' MAF exports a Reshape whose shape argument
# is a Constant node (not a model initializer). jaxonnxruntime's default strict
# mode rejects this. The flag below tells jaxonnxruntime to treat Constant nodes
# as legitimate static-args during jax.jit — which is correct for our exports
# since these are genuinely constant shapes baked at export time.
#
# Architectural implication: HSSM's onnx2jax.py does NOT set this flag today.
# The real sbi exporter (C3) must either (a) post-process the exported ONNX to
# fold Constant shape nodes into initializers, or (b) we'll need a small patch
# to HSSM. (a) is preferred to keep HSSM untouched per the integration plan.
config.update("jaxort_only_allow_initializers_as_static_args", False)


class _MAFLogProbModule(nn.Module):
    """Wraps a flow so .forward(x) returns log_prob(x) — the thing we export."""

    def __init__(self, flow: Flow) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(x)


def _build_maf(features: int = 4, num_layers: int = 3, hidden: int = 32) -> Flow:
    base_dist = StandardNormal(shape=[features])
    transforms: list = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=features))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=features, hidden_features=hidden
            )
        )
    return Flow(CompositeTransform(transforms), base_dist)


@pytest.mark.flaky(reruns=2)
def test_maf_log_prob_three_way_agreement(tmp_path: Path) -> None:
    torch.manual_seed(0)

    features = 4
    flow = _build_maf(features=features)
    flow.eval()
    module = _MAFLogProbModule(flow).eval()

    x_input = torch.randn(1, features, dtype=torch.float32)

    with torch.no_grad():
        y_torch = module(x_input).detach().numpy()

    onnx_path = tmp_path / "maf.onnx"
    dummy_input = torch.randn(1, features, requires_grad=True)
    torch.onnx.export(
        module,
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
