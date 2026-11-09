#!/usr/bin/env python3
"""Evaluate 3-seed reward (mean/max) vs single-seed on the TCRvdb validation set."""
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path("/disk1/zhuhb/useful_tools/esmfold2")
meta = {r["id"]: r for r in csv.DictReader(open(REPO / "tcrvdb_validation_meta.csv"))}

# collect mean pep-TCR per (id, seed)
per_seed = defaultdict(dict)
for seed, d in [("s0", "out_tcrvdb"), ("s1", "out_tcrvdb_s1"), ("s2", "out_tcrvdb_s2")]:
    for r in csv.DictReader(open(REPO / d / "results.csv")):
        a, b = float(r["iptm_peptide_tcra"]), float(r["iptm_peptide_tcrb"])
        per_seed[r["id"]][seed] = (a + b) / 2

rows = []
for rid, seeds in per_seed.items():
    m = meta[rid]
    vals = list(seeds.values())
    rows.append({
        "id": rid, "ep": m["epitope"], "lab": m["label"], "score": float(m["vdjdb_score"]),
        "s0": seeds.get("s0"), "s1": seeds.get("s1"), "s2": seeds.get("s2"),
        "mean": st.mean(vals), "max": max(vals),
        "sd": st.pstdev(vals) if len(vals) > 1 else 0.0,
    })
print(f"n = {len(rows)} (all with 3 seeds: {sum(1 for r in rows if r['s1'] is not None and r['s2'] is not None)})")


def auc(pos, neg):
    w = sum(1 for x in pos for y in neg if x > y) + 0.5 * sum(1 for x in pos for y in neg if x == y)
    return w / (len(pos) * len(neg))


def mw_p(pos, neg):
    u = sum(1 for x in pos for y in neg if x > y) + 0.5 * sum(1 for x in pos for y in neg if x == y)
    n1, n2 = len(pos), len(neg)
    z = (u - n1 * n2 / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return 0.5 * (1 + math.erf(-abs(z) / math.sqrt(2)))


by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r["ep"]][r["lab"]].append(r)

variants = ["s0", "mean", "max"]
print(f"\n{'表位':<12}" + "".join(f"{'AUC ' + v:>12}" for v in variants) + f"{'p(mean)':>10}")
for ep, g in sorted(by.items()):
    line = f"{ep:<12}"
    for v in variants:
        line += f"{auc([r[v] for r in g['pos']], [r[v] for r in g['neg']]):>12.3f}"
    line += f"{mw_p([r['mean'] for r in g['pos']], [r['mean'] for r in g['neg']]):>10.4f}"
    print(line)
for v in variants:
    line = f"{'pooled':<12}"
    for vv in variants:
        line += f"{auc([r[vv] for r in rows if r['lab']=='pos'], [r[vv] for r in rows if r['lab']=='neg']):>12.3f}"
    print(line + f"{'':>10}")

# acc/prec/recall sweep on 3-seed mean (global threshold)
print(f"\n全局阈值扫描（3-seed mean）:")
print(f"{'阈值':>6}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}")
y = [1 if r["lab"] == "pos" else 0 for r in rows]
best = (0, None)
for thr in [round(0.25 + 0.05 * i, 2) for i in range(14)]:
    p = [r["mean"] for r in rows]
    tp = sum(1 for yi, s in zip(y, p) if s >= thr and yi == 1)
    fp = sum(1 for yi, s in zip(y, p) if s >= thr and yi == 0)
    fn = sum(1 for yi, s in zip(y, p) if s < thr and yi == 1)
    tn = sum(1 for yi, s in zip(y, p) if s < thr and yi == 0)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    acc = (tp + tn) / len(y)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    if f1 == f1 and f1 > best[0]:
        best = (f1, thr, acc, prec, rec)
    print(f"{thr:>6.2f}{acc:>8.1%}{prec:>8.1%}{rec:>8.1%}{f1:>8.3f}")
f1, thr, acc, prec, rec = best
print(f">>> 最优F1阈值 {thr}: Acc={acc:.1%} Prec={prec:.1%} Recall={rec:.1%} F1={f1:.3f}")

# per-epitope z on 3-seed mean
bym = defaultdict(list)
for r in rows:
    bym[r["ep"]].append(r["mean"])
mus = {e: st.mean(v) for e, v in bym.items()}
sds = {e: st.pstdev(v) for e, v in bym.items()}
zrows = []
for r in rows:
    zrows.append({**r, "z": (r["mean"] - mus[r["ep"]]) / sds[r["ep"]]})
print(f"\n逐表位 z-score 阈值扫描（3-seed mean）:")
bestz = (0, None)
for thr in [round(-1.4 + 0.2 * i, 1) for i in range(13)]:
    y = [1 if r["lab"] == "pos" else 0 for r in zrows]
    p = [r["z"] for r in zrows]
    tp = sum(1 for yi, s in zip(y, p) if s >= thr and yi == 1)
    fp = sum(1 for yi, s in zip(y, p) if s >= thr and yi == 0)
    fn = sum(1 for yi, s in zip(y, p) if s < thr and yi == 1)
    tn = sum(1 for yi, s in zip(y, p) if s < thr and yi == 0)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    acc = (tp + tn) / len(y)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    if f1 == f1 and f1 > bestz[0]:
        bestz = (f1, thr, acc, prec, rec)
    print(f"{thr:>6.1f}{acc:>8.1%}{prec:>8.1%}{rec:>8.1%}{f1:>8.3f}")
f1, thr, acc, prec, rec = bestz
print(f">>> 最优F1 z阈值 {thr}: Acc={acc:.1%} Prec={prec:.1%} Recall={rec:.1%} F1={f1:.3f}")

# per-sample noise stats
print(f"\n每样本噪声（3 seed 的标准差分布）:")
for lab in ("pos", "neg"):
    sds = [r["sd"] for r in rows if r["lab"] == lab]
    print(f"  {lab}: 中位={st.median(sds):.3f}  P90={sorted(sds)[int(len(sds)*0.9)]:.3f}  最大={max(sds):.3f}")
