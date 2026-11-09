#!/usr/bin/env python3
"""Evaluate ESMFold2 ipTM on the TCRvdb validation sample (experimental labels)."""
import csv
import statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path("/disk1/zhuhb/useful_tools/esmfold2")
meta = {r["id"]: r for r in csv.DictReader(open(REPO / "tcrvdb_validation_meta.csv"))}

rows = []
for r in csv.DictReader(open(REPO / "out_tcrvdb" / "results.csv")):
    m = meta[r["id"]]
    a, b = float(r["iptm_peptide_tcra"]), float(r["iptm_peptide_tcrb"])
    rows.append({
        "id": r["id"], "ep": m["epitope"], "lab": m["label"], "score": float(m["vdjdb_score"]),
        "iptm": float(r["iptm"]), "ptm": float(r["ptm"]), "plddt": float(r["mean_plddt"]),
        "mpt": (a + b) / 2, "pep_mhc": float(r["iptm_peptide_mhc"]),
    })


def auc(pairs):
    wins = ties = 0
    for x, y in pairs:
        wins += x > y
        ties += x == y
    return (wins + 0.5 * ties) / len(pairs) if pairs else float("nan")


print(f"n = {len(rows)} ({sum(1 for r in rows if r['lab']=='pos')} pos / "
      f"{sum(1 for r in rows if r['lab']=='neg')} neg)\n")

print(f"{'epitope':<12}{'lab':<5}{'med mPTCR':>10}{'min':>7}{'max':>7}{'med ipTM':>10}")
by_ep = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_ep[r["ep"]][r["lab"]].append(r)
for ep, g in sorted(by_ep.items()):
    for lab in ("pos", "neg"):
        v = [x["mpt"] for x in g[lab]]
        vi = [x["iptm"] for x in g[lab]]
        print(f"{ep:<12}{lab:<5}{st.median(v):>10.3f}{min(v):>7.3f}{max(v):>7.3f}{st.median(vi):>10.3f}")

print(f"\n{'epitope':<12}{'AUC mPTCR':>10}{'AUC ipTM':>10}")
all_p, all_n, all_pi, all_ni = [], [], [], []
for ep, g in sorted(by_ep.items()):
    p = [x["mpt"] for x in g["pos"]]
    n = [x["mpt"] for x in g["neg"]]
    pi = [x["iptm"] for x in g["pos"]]
    ni = [x["iptm"] for x in g["neg"]]
    all_p += p; all_n += n; all_pi += pi; all_ni += ni
    print(f"{ep:<12}{auc([(x,y) for x in p for y in n]):>10.3f}{auc([(x,y) for x in pi for y in ni]):>10.3f}")
print(f"{'pooled':<12}{auc([(x,y) for x in all_p for y in all_n]):>10.3f}"
      f"{auc([(x,y) for x in all_pi for y in all_ni]):>10.3f}")

# threshold on mean pep-TCR within epitope (per-epitope threshold at max of negs)
print("\n按各 epitope 阈值(=阴性最大值)的阳性召回：")
for ep, g in sorted(by_ep.items()):
    neg_max = max(x["mpt"] for x in g["neg"])
    rec = sum(1 for x in g["pos"] if x["mpt"] > neg_max) / len(g["pos"])
    print(f"  {ep:<12} 阈值={neg_max:.3f}  召回 {rec:.0%}")

# vdjdb_score vs score among positives
print("\n阳性内部 vdjdb_score 与 mean pep-TCR:")
groups = defaultdict(list)
for r in rows:
    if r["lab"] == "pos":
        groups[r["score"]].append(r["mpt"])
for s in sorted(groups):
    v = groups[s]
    print(f"  score {s:.0f}: n={len(v):>2}  med={st.median(v):.3f}")
