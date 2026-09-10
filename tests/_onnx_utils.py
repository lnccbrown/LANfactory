"""Shared helpers for the ONNX export tests."""

import pickle
from pathlib import Path

import onnx

TINY_LAN_CONFIG = {
    "layer_sizes": [16, 16, 1],
    "activations": ["tanh", "tanh", "linear"],
    "train_output_type": "logprob",
}


def export_tiny_torch_lan(tmp_dir: Path, input_width: int, seed: int = 0):
    """Build a seeded, untrained TorchMLP and export it via ``transform_to_onnx``.

    Returns ``(net, onnx_path)``: the module in eval mode and the ``(1, D)``
    artifact produced by the real ``transform-onnx`` path (config pickle +
    state dict -> ONNX), so tests exercise the exporter as shipped.
    """
    import torch

    from lanfactory.onnx import transform_to_onnx
    from lanfactory.trainers.torch_mlp import TorchMLP

    torch.manual_seed(seed)
    net = TorchMLP(network_config=TINY_LAN_CONFIG, input_shape=input_width)
    net.eval()

    config_file = tmp_dir / "network_config.pickle"
    state_file = tmp_dir / "state_dict.pt"
    onnx_file = tmp_dir / "lan.onnx"
    with open(config_file, "wb") as f:
        pickle.dump(TINY_LAN_CONFIG, f)
    torch.save(net.state_dict(), state_file)
    transform_to_onnx(str(config_file), str(state_file), input_width, str(onnx_file))
    return net, onnx_file


def max_int64_abs(onnx_model: onnx.ModelProto) -> int:
    """Largest absolute value stored in any int64 tensor in the graph (0 if none)."""
    tensors = list(onnx_model.graph.initializer)
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attr.tensors)
    biggest = 0
    for tensor in tensors:
        if tensor.data_type == onnx.TensorProto.INT64:
            arr = onnx.numpy_helper.to_array(tensor)
            if arr.size:
                # Pure-Python abs over the flattened elements: int() gives
                # arbitrary-precision ints so abs() can't overflow on INT64_MIN
                # (whose true magnitude exceeds INT64_MAX), and this works for
                # 0-d (scalar) tensors too, unlike np.abs(...).max() on object.
                biggest = max(biggest, max(abs(int(v)) for v in arr.flat))
    return biggest
