#!/usr/bin/env python3
"""Build a stratified validation sample from the TCRvdb dataset.

5 strata (4 epitopes on HLA-A*02:01 + 1 on A*01:01) x {pos, neg} x 12 rows.
Positive rows prefer high vdjdb_score (better evidence). Ambiguous rows and
label clashes are excluded. Chains: official IMGT/HLA A1/A2 ectodomains
(first 276 aa of mature), TCR V-domain sequences from the dataset.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SRC = Path("/disk1/zhuhb/project/immrep26/data/ttd_gil_ylq_nlv_glc_01-10-2026_TCRvdb.csv")
CHAINS = json.load(open("/tmp/hla_chains.json"))
MHC = {"A1": CHAINS["a1_ecto"], "A2": CHAINS["a2_ecto"]}
N_PER = 12
SEED = 0

rows = list(csv.DictReader(open(SRC)))
by_stratum = defaultdict(list)
for r in rows:
    if r["pos"] == "1":
        lab = "pos"
    elif r["neg"] == "1":
        lab = "neg"
    else:
        continue  # ambiguous
    by_stratum[(r["epitope_aa"], r["hla_short"], lab)].append(r)

rng = random.Random(SEED)
picked = []
for (ep, hla, lab), rs in sorted(by_stratum.items()):
    rs = rs.copy()
    if lab == "pos":
        hi = [r for r in rs if float(r["vdjdb_score"]) >= 1]
        lo = [r for r in rs if float(r["vdjdb_score"]) < 1]
        rng.shuffle(hi)
        rng.shuffle(lo)
        take = (hi + lo)[:N_PER]
    else:
        rng.shuffle(rs)
        take = rs[:N_PER]
    for i, r in enumerate(take):
        picked.append({
            "id": f"{lab}_{ep[:6]}_{i:02d}",
            "peptide": r["epitope_aa"],
            "mhc": MHC[r["hla_short"]],
            "tcralpha": r["full_seq_alpha_aa"],
            "tcrbeta": r["full_seq_beta_aa"],
            "label": lab, "epitope": ep, "hla": r["hla_long"],
            "vdjdb_score": r["vdjdb_score"],
            "cdr3a": r["cdr3_alpha_aa"], "cdr3b": r["cdr3_beta_aa"],
        })

out = Path("/disk1/zhuhb/useful_tools/esmfold2/tcrvdb_validation.csv")
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "peptide", "mhc", "tcralpha", "tcrbeta"])
    for r in picked:
        w.writerow([r["id"], r["peptide"], r["mhc"], r["tcralpha"], r["tcrbeta"]])

meta = Path("/disk1/zhuhb/useful_tools/esmfold2/tcrvdb_validation_meta.csv")
with meta.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "label", "epitope", "hla", "vdjdb_score", "cdr3a", "cdr3b"], extrasaction="ignore")
    w.writeheader()
    w.writerows(picked)

from collections import Counter
print("picked:", Counter((r["epitope"], r["label"]) for r in picked))
print("vdjdb_score of picked pos:", Counter(r["vdjdb_score"] for r in picked if r["label"] == "pos"))
print(f"wrote {len(picked)} rows -> {out}")
