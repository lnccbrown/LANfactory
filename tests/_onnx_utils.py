"""Shared helpers for the ONNX export tests."""

from pathlib import Path

import onnx

# franklab/HSSM's root ddm.onnx, copied unchanged (see fixtures/onnx/PROVENANCE.md).
DDM_ONNX = Path(__file__).parent / "fixtures" / "onnx" / "ddm.onnx"
DDM_SHA256 = "09f685c18d3bdbd9b54fa89bed3b5bc0e2c76566e7ed0ae5e24df2c9b04f1b0e"
HUB_COMMIT = "01f5d4d0fa9188940ab541a979f933550b29616a"


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
