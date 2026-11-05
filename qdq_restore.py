"""Restore selected nodes of an explicit-Q/DQ ONNX graph to high precision.

A node is "restored" when none of its inputs passes through a
QuantizeLinear -> DequantizeLinear pair any more: each such input is rewired
to the tensor that entered the QuantizeLinear (an FP16/FP32 activation or a
float weight initializer). Every other node keeps exactly the Q/DQ pairs and
scales it had, so restoring a block never re-calibrates the rest of the graph.
Q/DQ pairs left without consumers are removed.

Usage:
    python qdq_restore.py --onnx in_int8.onnx --out out.onnx --regex "^/model\\.23/"
"""
import argparse
import re

import onnx


def restore(model, pattern):
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    rx = re.compile(pattern)
    n_rewired = 0
    for node in g.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") or not rx.search(node.name):
            continue
        for k, name in enumerate(node.input):
            dq = prod.get(name)
            if dq is None or dq.op_type != "DequantizeLinear":
                continue
            q = prod.get(dq.input[0])
            if q is None or q.op_type != "QuantizeLinear":
                continue
            node.input[k] = q.input[0]
            n_rewired += 1
    _prune(model)
    return n_rewired


def _prune(model):
    """Drop nodes whose outputs are consumed by nobody (iterates to a fixpoint)."""
    g = model.graph
    outputs = {o.name for o in g.output}
    while True:
        used = {i for n in g.node for i in n.input} | outputs
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        if not dead:
            break
        for n in dead:
            g.node.remove(n)
    used = {i for n in g.node for i in n.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)


def count_qdq(model, pattern):
    """Number of quantized inputs (DQ-fed) of nodes matching pattern."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    rx = re.compile(pattern)
    return sum(1 for n in g.node if n.op_type not in ("QuantizeLinear", "DequantizeLinear")
               and rx.search(n.name) for i in n.input
               if i in prod and prod[i].op_type == "DequantizeLinear")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regex", required=True)
    a = ap.parse_args()
    m = onnx.load(a.onnx)
    before = count_qdq(m, a.regex)
    n = restore(m, a.regex)
    onnx.checker.check_model(m)
    onnx.save(m, a.out)
    print(f"rewired {n} inputs ({before} DQ-fed inputs matched); remaining={count_qdq(m, a.regex)} -> {a.out}")
