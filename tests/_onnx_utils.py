"""Shared helpers for the ONNX export tests."""

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
                # Pure-Python abs over the flattened elements: int() gives
                # arbitrary-precision ints so abs() can't overflow on INT64_MIN
                # (whose true magnitude exceeds INT64_MAX), and this works for
                # 0-d (scalar) tensors too, unlike np.abs(...).max() on object.
                biggest = max(biggest, max(abs(int(v)) for v in arr.flat))
    return biggest
