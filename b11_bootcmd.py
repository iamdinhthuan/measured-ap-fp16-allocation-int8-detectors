"""rules_11_11b.md B: build the bootstrap_groups command (ExDark test, lighting groups); identical restored regexes share one set."""
import json, os, shlex
os.chdir("/home/thuan/isci_method_20261025")
D = "metrics/alloc_eval/yolo11s_int8ent_exdarkcal"; D0 = "metrics/alloc_eval/yolo11s_int8ent"
sets, by_rx, alias = {"fp32": "metrics/exdark/yolo11s_fp32_test.pred.json.gz"}, {}, {}
def add(label, d, tag):
    rx = json.load(open(f"{d}/{tag}.json"))["regex"]
    key = (d, rx)
    if key in by_rx: alias[label] = by_rx[key]; return
    by_rx[key] = label; alias[label] = label; sets[label] = f"{d}/{tag}_exdark_test.pred.json.gz"
    assert os.path.exists(sets[label]), sets[label]
add("stem", D, "exdarkcal_stem_headout")
BS = ("0.03", "0.05", "0.1", "0.15", "0.2")
for B in BS:
    for k in ("mean", "minimax", "mean_matched", "sqnrpool", "sqnrpandey"):
        add(f"{k}_B{B}", D, f"exdarkcal_{k}_B{B}")
    add(f"coco_mean_B{B}", D0, f"exdark_mean_B{B}")
pairs = []
for B in BS:
    raw = [(f"mean_matched_B{B}", f"minimax_B{B}"), (f"sqnrpool_B{B}", f"mean_B{B}"), (f"sqnrpandey_B{B}", f"mean_B{B}"),
           ("stem", f"mean_B{B}"), (f"coco_mean_B{B}", f"mean_B{B}")]
    for a, b in raw:
        a2, b2 = alias[a], alias[b]
        print(f"# {a}:{b} -> {a2}:{b2}" + ("  IDENTICAL" if a2 == b2 else ""))
        if a2 != b2 and f"{a2}:{b2}" not in pairs: pairs.append(f"{a2}:{b2}")
print(f"# {len(sets)} sets, {len(pairs)} pairs")
cmd = ["nice", "-n", "10", "/home/thuan/miniconda3/envs/qtsd/bin/python", "bootstrap_groups.py", "--gt", "data/exdark/exdark_test.json",
       "--sets", ",".join(f"{k}={v}" for k, v in sets.items()), "--ref", "fp32", "--pairs", ",".join(pairs),
       "--groups", "attr:light", "--group-names", "Ambient,Strong,Single,Weak,Twilight,Object,Low,Window", "--n", "1000",
       "--out", "metrics/bootstrap/g_yolo11s_exdarkcal_b11.json"]
json.dump({"alias": alias}, open("logs/b11_boot_alias.json", "w"), indent=1)
print(" ".join(shlex.quote(c) for c in cmd))
