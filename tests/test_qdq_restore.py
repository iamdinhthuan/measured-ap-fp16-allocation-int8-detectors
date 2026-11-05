import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdq_restore import count_qdq, restore  # noqa: E402


def _qdq(prefix, src):
    return [
        helper.make_node("QuantizeLinear", [src, f"{prefix}_s", f"{prefix}_z"], [f"{prefix}_q"], name=f"{prefix}_Q"),
        helper.make_node("DequantizeLinear", [f"{prefix}_q", f"{prefix}_s", f"{prefix}_z"], [f"{prefix}_dq"], name=f"{prefix}_DQ"),
    ]


def _model():
    """x -> QDQ -> /a/Conv(w via QDQ) -> QDQ -> /b/Relu -> y, plus a second consumer of the shared QDQ."""
    w = numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "w")
    inits = [w]
    for p in ("x", "w", "h"):
        inits += [numpy_helper.from_array(np.array(0.1, np.float32), f"{p}_s"),
                  numpy_helper.from_array(np.array(0, np.int8), f"{p}_z")]
    nodes = _qdq("x", "x") + _qdq("w", "w") + [
        helper.make_node("Conv", ["x_dq", "w_dq"], ["h"], name="/a/Conv"),
    ] + _qdq("h", "h") + [
        helper.make_node("Relu", ["h_dq"], ["y"], name="/b/Relu"),
        helper.make_node("Sigmoid", ["h_dq"], ["z"], name="/c/Sigmoid"),
    ]
    vi = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1, 2, 2])  # noqa: E731
    g = helper.make_graph(nodes, "t", [vi("x")], [vi("y"), vi("z")], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def test_restore_rewires_only_matching_nodes_and_prunes():
    m = _model()
    assert count_qdq(m, "^/a/") == 2
    assert restore(m, "^/a/") == 2
    onnx.checker.check_model(m)
    conv = next(n for n in m.graph.node if n.name == "/a/Conv")
    assert list(conv.input) == ["x", "w"]
    names = {n.name for n in m.graph.node}
    assert not {"x_Q", "x_DQ", "w_Q", "w_DQ"} & names  # orphaned pairs removed
    assert {"h_Q", "h_DQ"} <= names  # downstream consumers keep their quantized input
    assert count_qdq(m, "^/b/") == 1


def test_shared_qdq_kept_while_another_consumer_needs_it():
    m = _model()
    restore(m, "^/b/")
    names = {n.name for n in m.graph.node}
    assert {"h_Q", "h_DQ"} <= names  # /c/Sigmoid still reads h_dq
    relu = next(n for n in m.graph.node if n.name == "/b/Relu")
    assert list(relu.input) == ["h"]


def test_restore_everything_leaves_no_qdq():
    m = _model()
    restore(m, ".")
    assert not any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in m.graph.node)
    assert not any(i.name.endswith(("_s", "_z")) for i in m.graph.initializer)
