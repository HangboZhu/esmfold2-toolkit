#!/usr/bin/env python3
"""Merge all pMHC-TCR benchmark results and report positive/negative separation.

Usage: uv run python summarize_benchmark.py
"""
import csv
from pathlib import Path

# legacy positives without the POS_ prefix
LEGACY_POS = {"1G4_NYESO1_HLA-A2"}

METRICS = [
    ("iptm", "global ipTM"),
    ("ptm", "global pTM"),
    ("iptm_peptide_mhc", "pep-MHC"),
    ("iptm_peptide_tcra", "pep-TCRa"),
    ("iptm_peptide_tcrb", "pep-TCRb"),
    ("mean_pTCR", "mean pep-TCR"),
    ("composite", "0.5*mean_pTCR + 0.5*iptm"),
]


def is_positive(pid: str) -> bool:
    return pid.startswith("POS_") or pid in LEGACY_POS


def load_rows() -> list[dict]:
    rows = []
    for path in ["out_test/results.csv", "out_negatives/results.csv",
                 "out_expanded/results.csv", "out_cohort3/results.csv"]:
        if not Path(path).is_file():
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                r["kind"] = "POS" if is_positive(r["id"]) else "NEG"
                a, b = float(r["iptm_peptide_tcra"]), float(r["iptm_peptide_tcrb"])
                r["mean_pTCR"] = (a + b) / 2
                r["composite"] = 0.5 * r["mean_pTCR"] + 0.5 * float(r["iptm"])
                rows.append(r)
    return rows


def auc(pos: list[float], neg: list[float]) -> float:
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> None:
    rows = load_rows()
    pos = [r for r in rows if r["kind"] == "POS"]
    neg = [r for r in rows if r["kind"] == "NEG"]
    print(f"n = {len(rows)} complexes: {len(pos)} positives / {len(neg)} negatives\n")

    header = f"{'id':<26}{'kind':<5}" + "".join(f"{lbl:>13}" for _, lbl in METRICS)
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: -float(r["composite"])):
        cells = "".join(f"{float(r[k]):>13.3f}" for k, _ in METRICS)
        print(f"{r['id']:<26}{r['kind']:<5}{cells}")

    print(f"\n{'metric':<22}{'min POS':>10}{'max NEG':>10}{'gap':>9}{'AUC':>8}")
    print("-" * 62)
    for key, lbl in METRICS:
        p = [float(r[key]) for r in pos]
        n = [float(r[key]) for r in neg]
        gap = min(p) - max(n)
        print(f"{lbl:<22}{min(p):>10.3f}{max(n):>10.3f}{gap:>+9.3f}{auc(p, n):>8.3f}")


if __name__ == "__main__":
    main()
