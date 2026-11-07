"""rules_11_12d.md section 6 + add1: paired val2017 bootstrap for RT-DETR-l, then the C1/C2/C3 reading.
Prediction files are found by restored regex, as fx_boot.py does, in the evaluation directory of each arm's base.
Identical prediction files are packed once (one set name per file). Pair X:Y means Y - X.
Usage: python rd_boot.py            (writes metrics/bootstrap/rd_rtdetr-l.json and metrics/rd_rtdetr-l_verdict.json)"""
import glob
import json
import os
import subprocess
import sys

import rd_common

os.chdir(rd_common.ROOT)
M = "rtdetr-l"
BUD = ("0.03", "0.05", "0.1", "0.15", "0.2")
GT = "data/coco_src/annotations/instances_val2017.json"
OUT = os.path.join("metrics", "bootstrap", "rd_rtdetr-l.json")
VER = os.path.join("metrics", "rd_rtdetr-l_verdict.json")
ARMS = {  # label: (base, config, criterion)
    "mean": ("int8ent", f"metrics/alloc_{M}_int8ent_ho.json", "mean"),
    "sqnr": ("int8ent", f"configs/fx_{M}_int8ent_ho_noise_n2000.json", "sqnrq"),
    "pandey": ("int8ent", f"configs/fx_{M}_int8ent_ho_pandey_n256.json", "pandey"),
    "hyb": ("int8ent", f"configs/fx_{M}_int8ent_ho_hyb.json", "hyb"),
    "sqnrbs": ("int8ent", f"configs/fx_{M}_int8ent_ho_noise_n2000bs.json", "sqnrq"),
    "greedy": ("int8ent", f"configs/greedy_{M}_int8ent_ho.json", "greedy"),
    "F": ("int8ent", f"configs/manual_{M}_int8ent.json", None),
    "meanF": ("int8entF", f"metrics/alloc_{M}_int8entF_ho.json", "mean"),
}
P_PRIMARY = ("sqnr", "pandey", "hyb", "sqnrbs")  # proxy family P (rules_11_12d.md 6 + add1 A3)


def evaluated(base):
    d = os.path.join("metrics", "alloc_eval", f"{M}_{base}")
    known = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json")), key=lambda f: (os.path.basename(f).startswith(("fx", "hyb")), f)):
        if f.endswith(("_val.json", "_tune.json")):
            continue
        r = json.load(open(f))
        assert all(v.get("arch") == rd_common.ARCH for v in r["splits"].values()), f
        pred = f[:-5] + "_val.pred.json.gz"
        if "val" in r["splits"] and os.path.exists(pred):
            known.setdefault(r["regex"], (pred, r))
    return known


def regex(plan, key, blocks):
    sel = plan["allocations"][key]
    rx = sel.get("regex") or "|".join(f"(?:{blocks[b]})" for b in sel["blocks"])
    if plan.get("force_regex"):
        rx = f"(?:{plan['force_regex']})" + (f"|{rx}" if rx else "")
    return rx


def pexact(p_le0, n):
    return max(2 * min(p_le0, 1 - p_le0), 2 / (n + 1))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    known = {b: evaluated(b) for b in ("int8ent", "int8entF")}
    fp32 = os.path.join("metrics", "alloc_eval", f"{M}_int8ent", "fp32_val.pred.json.gz")
    assert os.path.exists(fp32), fp32
    name_of, sets, arm = {fp32: "fp32"}, {"fp32": fp32}, {}
    meta = {}
    for label, (base, cfg, crit) in ARMS.items():
        if not os.path.exists(cfg):
            print(f"{label}: {cfg} missing, arm skipped", flush=True)
            continue
        plan = json.load(open(cfg))
        assert plan.get("base", base) == base, (cfg, plan.get("base"))
        blocks = json.load(open(os.path.join("metrics", f"sens_{M}_{base}.json")))["blocks"]
        for B in BUD:
            key = "F@0" if crit is None else f"{crit}@{B}"
            if key not in plan["allocations"]:
                print(f"{label}@{B}: not in {cfg} (infeasible or absent)", flush=True)
                continue
            rx = regex(plan, key, blocks)
            if rx not in known[base]:
                raise SystemExit(f"{label}@{B}: allocation not evaluated yet")
            path, r = known[base][rx]
            name_of.setdefault(path, f"s{len(name_of)}")
            sets[name_of[path]] = path
            arm[(label, B)] = name_of[path]
            meta[f"{label}@{B}"] = {"file": os.path.basename(path), "base": base,
                                   "cost_macs_frac": r.get("cost_macs_frac"),
                                   "tune_AP": 100 * r["splits"]["tune"]["stats"]["AP"] if "tune" in r["splits"] else None,
                                   "val_AP_eval": 100 * r["splits"]["val"]["stats"]["AP"]}
    pairs = set()
    for B in BUD:
        for tgt in ("mean", "meanF"):
            if (tgt, B) not in arm:
                continue
            for x in (*P_PRIMARY, "greedy", "F") + (("mean",) if tgt == "meanF" else ()):
                if (x, B) in arm and arm[(x, B)] != arm[(tgt, B)]:
                    pairs.add(f"{arm[(x, B)]}:{arm[(tgt, B)]}")
    if not os.path.exists(OUT):
        cmd = [sys.executable, "bootstrap_fast.py", "--gt", GT, "--sets", ",".join(f"{k}={v}" for k, v in sets.items()),
               "--ref", "fp32", "--pairs", ",".join(sorted(pairs)), "--n", "1000", "--out", OUT]
        if subprocess.call(["nice", "-n", "10"] + cmd):
            raise SystemExit("bootstrap failed")
    bs = json.load(open(OUT))
    n = bs["n"]
    pt = {k: v["AP"] for k, v in bs["point"].items()}

    def diff(x, tgt, B):
        a, b = arm.get((x, B)), arm.get((tgt, B))
        if a is None or b is None:
            return None
        if a == b:
            return {"mean": 0.0, "ci95": [0.0, 0.0], "p": 1.0, "same_allocation": True}
        d = bs["diff_b_minus_a"][f"{a}:{b}"]["AP"]
        return {"mean": d["mean"], "ci95": d["ci95"], "p": pexact(d["p_le0"], n)}

    verdict = {"n": n, "fp32_val_AP": pt["fp32"], "rules": "rules_11_12d.md section 7 + rules_11_12d_add1.md",
               "point_val_AP": {f"{l}@{B}": pt[s] for (l, B), s in arm.items()}, "arms": meta}
    for tgt in ("mean", "meanF"):
        rows, ps = [], []
        for B in BUD:
            if (tgt, B) not in arm:
                continue
            mem = [x for x in P_PRIMARY if (x, B) in arm]
            strongest = max(mem, key=lambda x: pt[arm[(x, B)]])
            d = diff(strongest, tgt, B)
            row = {"B": B, "target_AP": pt[arm[(tgt, B)]], "strongest": strongest,
                   "strongest_AP": pt[arm[(strongest, B)]], "vs_strongest": d,
                   "vs": {x: diff(x, tgt, B) for x in (*P_PRIMARY, "greedy", "F", "mean") if x != tgt}}
            row["behind_any_P_uncorrected"] = [x for x in mem if (v := row["vs"][x]) and v["p"] < 0.05 and v["mean"] < 0]
            rows.append(row)
            ps.append(d["p"])
        for row, pa in zip(rows, holm(ps)):
            row["vs_strongest"]["p_holm"] = pa
            row["ahead_after_holm"] = pa < 0.05 and row["vs_strongest"]["mean"] > 0
        k = sum(r["ahead_after_holm"] for r in rows)
        behind = any(r["behind_any_P_uncorrected"] for r in rows)
        c = "C3" if (k == 0 or behind) else ("C1" if k >= 3 else "C2")
        if tgt == "meanF" and c == "C1":
            c = "C2 (secondary arm capped, add1 A2)"
        verdict[tgt] = {"rows": rows, "n_budgets_ahead_holm": k, "behind_any": behind, "class": c}
    verdict["D_outcome"] = verdict["mean"]["class"]  # primary arm decides; meanF is secondary (cannot lift above C2)
    json.dump(verdict, open(VER, "w"), indent=1)
    print(json.dumps({t: (verdict[t]["class"], verdict[t]["n_budgets_ahead_holm"]) for t in ("mean", "meanF") if t in verdict}))


if __name__ == "__main__":
    main()
