"""The single-trial ONNX contract, as a check instead of a paragraph.

Every exporter here produces artifacts that HSSM loads through
``jaxonnxruntime``, which traces against the construction-time dummy and bakes
the resulting shapes into the returned closure. A graph with a dynamic axis
therefore does not fail loudly — it silently returns wrong numbers for any
model with a batch-dependent intermediate. HSSM guards its own door
(``make_jax_func`` raises on symbolic dims), but by then the artifact is
published.

The invariant is exactly one thing: **every input dimension is concrete**.

Rank is *not* part of it, which is the part that keeps getting misremembered.
It follows from how a tracer lowers a dense layer: ``torch.onnx.export`` on a
rank-1 dummy gives rank-agnostic ``MatMul``+``Add`` (sbi, bayesflow), while a
``(1, D)`` dummy — and ``jax2onnx`` always — gives ``Gemm``, whose ONNX spec
*requires* rank 2. Both load in HSSM and run identically once XLA has fused
them. The production networks on franklab/HSSM are ``(1, D)`` Gemm.

Call this from an exporter's tests rather than restating the rules.
"""

from pathlib import Path


def assert_single_trial_contract(
    onnx_path: str | Path,
    expected_input_width: int | None = None,
    allowed_ops: set[str] | None = None,
) -> dict:
    """Raise AssertionError unless the artifact satisfies the contract.

    Parameters
    ----------
    onnx_path
        The exported artifact.
    expected_input_width
        The per-trial input width (the last dimension), when the caller knows
        it. Catches an exporter that silently changed its input layout.
    allowed_ops
        When given, the graph's op types must be a subset. Useful to pin a
        lowering that a pinned 0.x exporter dependency could change under you.

    Returns
    -------
    dict
        ``{"input_shape", "input_width", "ops"}`` for further assertions.
    """
    import onnx
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    for graph_input in model.graph.input:
        for dim in graph_input.type.tensor_type.shape.dim:
            assert dim.HasField("dim_value"), (
                f"symbolic dim {dim.dim_param!r} in input {graph_input.name!r}: "
                "HSSM's make_jax_func rejects dynamic axes at load, and a graph "
                "that slips through returns wrong numbers rather than failing"
            )
            # A dim_value of 0 is set-but-not-concrete: some producers use it
            # for "unknown", and it is a degenerate axis either way.
            assert dim.dim_value > 0, (
                f"zero dim in input {graph_input.name!r}: not a concrete shape"
            )

    # onnxruntime is the arbiter of whether the graph is actually runnable:
    # a rank-1-traced Gemm passes the checker and fails here.
    session = ort.InferenceSession(str(onnx_path))
    input_shape = session.get_inputs()[0].shape
    input_width = int(input_shape[-1])

    if expected_input_width is not None:
        assert input_width == expected_input_width, (
            f"input width {input_width} != expected {expected_input_width}"
        )

    ops = {node.op_type for node in model.graph.node}
    if allowed_ops is not None:
        assert ops <= allowed_ops, f"unexpected ops {sorted(ops - allowed_ops)}"

    return {
        "input_shape": list(input_shape),
        "input_width": input_width,
        "ops": sorted(ops),
    }
