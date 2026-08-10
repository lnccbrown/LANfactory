"""Tests for the executable single-trial ONNX contract."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from lanfactory.onnx import assert_single_trial_contract


def make_onnx(path, input_dims):
    width = input_dims[-1] if isinstance(input_dims[-1], int) else 6
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(input_dims))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    w = helper.make_tensor(
        "w", TensorProto.FLOAT, [width, 1], np.zeros(width, dtype=np.float32).tolist()
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])], "g", [x], [y], initializer=[w]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_accepts_a_concrete_graph(tmp_path):
    result = assert_single_trial_contract(
        make_onnx(tmp_path / "ok.onnx", (1, 6)), expected_input_width=6
    )
    assert result["input_shape"] == [1, 6]


def test_rejects_a_symbolic_dim(tmp_path):
    # The whole point: jaxonnxruntime bakes shapes at trace time, so a dynamic
    # axis returns wrong numbers rather than failing.
    with pytest.raises(AssertionError, match="symbolic dim"):
        assert_single_trial_contract(make_onnx(tmp_path / "dyn.onnx", ("batch", 6)))


def test_rejects_an_unexpected_input_width(tmp_path):
    with pytest.raises(AssertionError, match="input width"):
        assert_single_trial_contract(
            make_onnx(tmp_path / "w.onnx", (1, 5)), expected_input_width=6
        )


def test_rejects_ops_outside_the_allowed_set(tmp_path):
    # Guards against a pinned 0.x exporter changing its lowering under us.
    with pytest.raises(AssertionError, match="unexpected ops"):
        assert_single_trial_contract(
            make_onnx(tmp_path / "ops.onnx", (1, 6)), allowed_ops={"Gemm", "Tanh"}
        )


def test_rank_is_not_part_of_the_contract(tmp_path):
    """sbi and bayesflow trace rank-1; the MLP exporters trace (1, D). Both are
    valid — only concreteness is required."""
    assert assert_single_trial_contract(make_onnx(tmp_path / "r1.onnx", (6,)))
    assert assert_single_trial_contract(make_onnx(tmp_path / "r2.onnx", (1, 6)))
