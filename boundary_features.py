"""Phase B step 0 (README 11.7): conv-level precision-boundary features of the Orin Q/DQ graphs."""
import glob, json, os, sys
import numpy as np
import onnx
from onnx import shape_inference


def features(path):
    m = onnx.load(path)
    m = shape_inference.infer_shapes(m)
    g = m.graph
    inits = {i.name for i in g.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(g.value_info) + list(g.input) + list(g.output)}
    prod = {o: n for n in g.node for o in n.output}
    numel = lambda t: int(np.prod(shapes[t])) if t in shapes and shapes[t] and all(shapes[t]) else 0
    convs = [n for n in g.node if n.op_type == "Conv"]
    prec, macs = {}, {}
    for c in convs:
        x = c.input[0]
        prec[c.name] = "int8" if x in prod and prod[x].op_type == "DequantizeLinear" else "fp16"
        y = c.output[0]
        w = None
        wt = c.input[1]
        # weight shape: through DQ to initializer
        if wt in prod and prod[wt].op_type == "DequantizeLinear":
            wt = prod[wt].input[0]
        for i in g.initializer:
            if i.name == wt:
                w = list(i.dims); break
        if w is None:
            w = shapes.get(c.input[1])
        macs[c.name] = numel(y) * int(np.prod(w[1:])) if w else 0
    edges = []
    for c in convs:
        x = c.input[0]
        seen, stack = set(), [x]
        while stack:
            t = stack.pop()
            if t in seen or t in inits or t not in prod:
                continue
            seen.add(t)
            p = prod[t]
            if p.op_type == "Conv":
                edges.append((p.name, c.name, numel(x)))
                continue
            ins = p.input[:1] if p.op_type in ("QuantizeLinear", "DequantizeLinear") else p.input
            stack.extend(i for i in ins if i and i not in inits)
    tot_mac = sum(macs.values())
    mac = sum(v for k, v in macs.items() if prec[k] == "fp16") / tot_mac
    tr = [e for e in edges if prec[e[0]] != prec[e[1]]]
    tot_b = sum(e[2] for e in edges)
    return {"mac": mac, "T": len(tr) / len(edges), "Bytes": sum(e[2] for e in tr) / tot_b,
            "n_edges": len(edges), "n_trans": len(tr), "n_conv": len(convs),
            "n_fp16_conv": sum(1 for v in prec.values() if v == "fp16")}


if __name__ == "__main__":
    out = {}
    for model in sys.argv[1:]:
        for p in sorted(glob.glob(f"exports/orin/{model}/*.onnx")):
            tag = os.path.basename(p)[:-5]
            out[f"{model}/{tag}"] = features(p)
            print(model, tag, out[f"{model}/{tag}"], flush=True)
    dst = "metrics/phaseB_boundary_features.json"
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    json.dump(out, open(dst, "w"), indent=1)
    print("FEAT_DONE")
