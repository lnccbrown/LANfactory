"""Shared helpers for the ONNX export tests."""

import numpy as np
import onnx


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
                biggest = max(biggest, int(np.abs(arr).max()))
    return biggest
