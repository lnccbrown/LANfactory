"""C5 verification: NRE_A with embedding nets on x exports and round-trips.

Tests two representative embeddings:
  - FCEmbedding (an extra MLP on x): the most common embedding pattern.
  - CNNEmbedding (1D conv stack on x): exercises Conv / MaxPool / etc. in
    the exported ONNX. The C2 / C3 spikes did not touch Conv ops.

Other sbi embeddings (PermutationInvariantEmbedding, ResNetEmbedding1D,
TransformerEmbedding, ...) are out of v1 scope; can be added as follow-up
regression tests if a user needs them.
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jaxonnxruntime import call_onnx, config  # noqa: E402
from sbi.inference import NRE_A  # noqa: E402
from sbi.neural_nets import classifier_nn  # noqa: E402
from sbi.neural_nets.embedding_nets import CNNEmbedding, FCEmbedding  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402
from torch import nn  # noqa: E402

from lanfactory.onnx import transform_sbi_to_onnx  # noqa: E402

config.update("jaxort_only_allow_initializers_as_static_args", False)

# sbi's build_mlp_classifier defaults to nn.LayerNorm between hidden layers, but
# jaxonnxruntime does not implement the LayerNormalization op. Passing
# norm_layer=nn.Identity disables it. Documenting this constraint in C6 docs:
# users training their own NRE classifiers must disable LayerNorm for export.

_THETA_DIM = 2
_X_DIM = 10  # 10-dim flat x — enough to make embedding non-trivial


def _simulator(theta: torch.Tensor) -> torch.Tensor:
    """x | theta: stack of 10 i.i.d. N(theta[:, 0], 1) and N(theta[:, 1], 1).

    Concretely: x is a 10-vector whose first 5 dims are ~ N(theta[0], 1) and
    last 5 dims are ~ N(theta[1], 1). Linear-Gaussian, easy enough for a
    tiny NRE classifier to pick up.
    """
    batch = theta.shape[0]
    first_half = theta[:, 0:1] + torch.randn(batch, 5)
    second_half = theta[:, 1:2] + torch.randn(batch, 5)
    return torch.cat([first_half, second_half], dim=-1)


def _three_way_agreement(
    trained_classifier: torch.nn.Module, onnx_path: Path
) -> None:
    """Shared assertion: torch / onnxruntime / jaxonnxruntime all agree."""
    theta_t = torch.tensor([[0.3, -0.4]], dtype=torch.float32)
    x_t = torch.randn(1, _X_DIM, dtype=torch.float32)
    # Exported graph is rank-1; pass a 1D concatenated vector.
    combined = torch.cat([theta_t, x_t], dim=-1).squeeze(0).numpy()

    with torch.no_grad():
        y_torch = trained_classifier(theta_t, x_t).detach().numpy().flatten()

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    y_ort = sess.run(None, {input_name: combined})[0].flatten()

    onnx_model = onnx.load(str(onnx_path))
    model_func, model_weights = call_onnx.call_onnx_model(
        onnx_model, {input_name: combined}
    )
    run_func = jax.tree_util.Partial(model_func, model_weights)
    y_jax = np.asarray(run_func({input_name: combined})[0]).flatten()

    atol = 1e-5
    assert np.allclose(y_torch, y_ort, atol=atol), (
        f"torch vs onnxruntime: max |Δ| = {np.abs(y_torch - y_ort).max()}"
    )
    assert np.allclose(y_torch, y_jax, atol=atol), (
        f"torch vs jaxonnxruntime: max |Δ| = {np.abs(y_torch - y_jax).max()}"
    )


@pytest.mark.flaky(reruns=2)
def test_nre_with_fc_embedding(tmp_path: Path) -> None:
    """NRE_A + FCEmbedding(x) → ONNX → round-trip."""
    torch.manual_seed(0)
    prior = BoxUniform(
        low=torch.tensor([-3.0, -3.0]),
        high=torch.tensor([3.0, 3.0]),
    )

    embedding_x = FCEmbedding(input_dim=_X_DIM, output_dim=8, num_layers=2)
    classifier_builder = classifier_nn(
        model="mlp",
        embedding_net_x=embedding_x,
        norm_layer=nn.Identity,
    )
    inference = NRE_A(prior=prior, classifier=classifier_builder)

    theta = prior.sample((1000,))
    x = _simulator(theta)
    classifier = inference.append_simulations(theta, x).train(
        training_batch_size=200,
        max_num_epochs=10,
    )
    classifier.eval()

    onnx_path = tmp_path / "nre_fc.onnx"
    transform_sbi_to_onnx(
        classifier,
        str(onnx_path),
        mode="nre",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )
    _three_way_agreement(classifier, onnx_path)


@pytest.mark.flaky(reruns=2)
def test_nre_with_cnn_embedding(tmp_path: Path) -> None:
    """NRE_A + CNNEmbedding(x) → ONNX → round-trip.

    Confirms that Conv / pooling ops survive torch.onnx.export and translate
    cleanly into jaxonnxruntime. x is treated as a length-10 1D signal.
    """
    torch.manual_seed(0)
    prior = BoxUniform(
        low=torch.tensor([-3.0, -3.0]),
        high=torch.tensor([3.0, 3.0]),
    )

    embedding_x = CNNEmbedding(
        input_shape=(_X_DIM,),
        in_channels=1,
        out_channels_per_layer=[4, 4],
        num_conv_layers=2,
        num_linear_layers=1,
        num_linear_units=16,
        output_dim=8,
        kernel_size=3,
        pool_kernel_size=2,
    )
    classifier_builder = classifier_nn(
        model="mlp",
        embedding_net_x=embedding_x,
        norm_layer=nn.Identity,
    )
    inference = NRE_A(prior=prior, classifier=classifier_builder)

    theta = prior.sample((1000,))
    x = _simulator(theta)
    classifier = inference.append_simulations(theta, x).train(
        training_batch_size=200,
        max_num_epochs=10,
    )
    classifier.eval()

    onnx_path = tmp_path / "nre_cnn.onnx"
    transform_sbi_to_onnx(
        classifier,
        str(onnx_path),
        mode="nre",
        example_theta_dim=_THETA_DIM,
        example_x_dim=_X_DIM,
    )
    _three_way_agreement(classifier, onnx_path)
