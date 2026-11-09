#!/usr/bin/env python3
"""Build cohort-3 benchmark CSV from RCSB-verified cognate pairs (candidates.tsv).

Positives: 16 cognate complexes, one per selected PDB entry (own MHC chain used).
Negatives: for TCRs whose MHC is an HLA-A2 construct variant, a same-allele
peptide-swap (foreign A2-restricted peptide from a different antigen family);
for the rest, a cross-unit pair (foreign pMHC from a different entry).
Foreign peptides never come from the TCR's own pathogen family.
"""
import csv
from pathlib import Path

HLA_A2 = ("GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEP")

PEP = {
    "NYESO": "SLLMWITQC",
    "TAX": "LLFGYPVYV",
    "M1": "GILGFVFTL",
    "HIV": "ILKEPVHGV",
}

# entry -> (family tag, [swap1, swap2]) ; swaps must avoid the same family.
SWAPS = {
    "3GSN": ("CMV", ["NYESO", "M1"]),
    "5D2L": ("CMV", ["TAX", "HIV"]),
    "5NME": ("HIV", ["TAX", "M1"]),
    "5MEN": ("EBV", ["HIV", "TAX"]),
    "7N1F": ("SARS2", ["NYESO", "M1"]),
    "7RM4": ("SARS2", ["NYESO", "TAX"]),
    "4JFF": ("MART1", ["TAX", "HIV"]),
    "6DKP": ("MART1", ["NYESO", "HIV"]),
    "2YPL": ("HIV", []),
    "7RK7": ("TUMOR", ["M1", "HIV"]),
    "8DNT": ("OTHER", ["NYESO", "TAX"]),
    "5EUO": ("FLU", ["NYESO", "HIV"]),
    "3DXA": ("OTHER", ["TAX", "M1"]),
    "6TRO": ("SARS2", ["NYESO", "TAX"]),
    "9WC2": ("OTHER", []),
    "8RJ5": ("SARS2", []),
}

# entries that always get a cross-unit negative instead of same-allele swaps
CROSS_UNIT = {"2YPL": "NYESO", "8DNT": "TAX", "9WC2": "NYESO", "8RJ5": "TAX"}

# entries getting a second same-allele swap (famous families, more negatives)
EXTRA_SECOND_SWAPS = {"3GSN", "4JFF", "5NME", "5EUO"}


def is_a2_variant(seq: str, ref: str = HLA_A2) -> bool:
    """True if seq is a (possibly edge-trimmed) contiguous variant of ref."""
    if seq == ref:
        return True
    if seq in ref or ref in seq:
        return True
    # tolerate 1-2 edge mismatches of equal-length constructs
    if len(seq) == len(ref):
        diffs = sum(a != b for a, b in zip(seq, ref))
        return diffs <= 2
    # slide ref over seq allowing trim at either end
    for i in range(0, len(ref) - len(seq) + 1):
        window = ref[i:i + len(seq)]
        if sum(a != b for a, b in zip(window, seq)) <= 1:
            return True
    return False


def main() -> None:
    with open(Path(__file__).resolve().parent / "candidates.tsv") as fh:
        cands = {r[0]: r for r in list(csv.reader(fh, delimiter="\t"))[1:]}

    rows = []
    for entry, (family, swaps) in SWAPS.items():
        r = cands[entry]
        _, _, pep, _, _, _, _, tcra, tcrb, mhc = r
        rows.append([f"POS_{entry}", pep, mhc, tcra, tcrb])

        if entry in CROSS_UNIT:
            tag = CROSS_UNIT[entry]
            rows.append([f"NEG_{entry}_{tag}", PEP[tag], HLA_A2, tcra, tcrb])
            continue

        if not is_a2_variant(mhc):
            tag = swaps[0]
            rows.append([f"NEG_{entry}_{tag}", PEP[tag], HLA_A2, tcra, tcrb])
            continue

        chosen = swaps[:1] + ([swaps[1]] if entry in EXTRA_SECOND_SWAPS else [])
        for tag in chosen:
            rows.append([f"NEG_{entry}_{tag}", PEP[tag], HLA_A2, tcra, tcrb])

    out = Path(__file__).resolve().parent / "cohort3.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "peptide", "mhc", "tcralpha", "tcrbeta"])
        w.writerows(rows)
    n_pos = sum(1 for r in rows if r[0].startswith("POS_"))
    print(f"wrote {len(rows)} rows ({n_pos} POS / {len(rows) - n_pos} NEG) to {out}")


if __name__ == "__main__":
    main()
