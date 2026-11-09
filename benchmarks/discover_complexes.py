#!/usr/bin/env python3
"""Systematically discover human pMHC-I:TCR complexes in the PDB.

Searches RCSB with the human TCR-alpha constant domain sequence, fetches every
hit's FASTA, and keeps entries that contain a complete tri-molecular complex:
human TCR alpha + TCR beta + MHC class I heavy chain + beta-2-microglobulin +
one short canonical peptide (8-16 aa). Deduplicates on
(tcra, tcrb, peptide, mhc) sequences and writes candidates.tsv.
"""
import csv
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
FASTA_URL = "https://www.rcsb.org/fasta/entry/{}"

TCAC_QUERY = ("NPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKCVLDMRSMDFKSNSAVAWSNKSDFACANAFNNSIIPEDTFFPS")

EXCLUDE_WORDS = ("ANTIBODY", "SCFV", "FAB ", "TCR MIMIC", "IMAGAB", "NANOBODY")


def http_get(url: str, tries: int = 3) -> str:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pMHC-TCR-benchmark/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.5)
    raise RuntimeError("unreachable")


def search_all_entries() -> list[str]:
    ids: list[str] = []
    start = 0
    while True:
        body = json.dumps({
            "query": {"type": "terminal", "service": "sequence", "parameters": {
                "evalue_cutoff": 1, "identity_cutoff": 0.9,
                "sequence_type": "protein", "value": TCAC_QUERY}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": start, "rows": 100}},
        }).encode()
        req = urllib.request.Request(SEARCH_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        batch = [r["identifier"] for r in data.get("result_set", [])]
        ids.extend(batch)
        start += 100
        if start >= data.get("total_count", 0) or not batch:
            break
    return ids


def parse_fasta(text: str):
    """Yield (chain_ids, description, sequence) per FASTA record."""
    records = []
    cur_desc, cur_seq = None, []
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            if cur_desc is not None:
                records.append((cur_desc, "".join(cur_seq)))
            cur_desc, cur_seq = line[1:], []
        elif cur_desc is not None:
            cur_seq.append(line.strip())
    if cur_desc is not None:
        records.append((cur_desc, "".join(cur_seq)))
    out = []
    for desc, seq in records:
        # ">1BD2_4|Chain D|T CELL RECEPTOR ALPHA|HOMO SAPIENS (9606)"
        parts = desc.split("|")
        chain_field = parts[1] if len(parts) > 1 else ""
        chains = re.findall(r"[A-Z](?:\[auth [A-Z]+\])?", chain_field)
        chains = [re.sub(r"\[auth [A-Z]+\]", "", c) for c in chains]
        label = parts[2] if len(parts) > 2 else ""
        organism = parts[3] if len(parts) > 3 else ""
        out.append((chains, label, organism, seq))
    return out


def norm(s: str) -> str:
    return s.upper().replace("-", " ")


def classify_entry(entry_id: str) -> dict | None:
    """Return candidate info if entry is a human pMHC-I:TCR complex, else None."""
    try:
        text = http_get(FASTA_URL.format(entry_id))
    except Exception as exc:
        return {"entry": entry_id, "error": str(exc)}
    if "No fasta" in text:
        return None
    recs = parse_fasta(text)
    all_chains = [c for rec in recs for c in rec[0]]
    all_desc = " | ".join(rec[1] for rec in recs)
    if any(w in norm(all_desc) for w in EXCLUDE_WORDS):
        return None
    if "MUS MUSCULUS" in all_desc.upper() or "MUS MUSCULUS" in all_desc:
        return None

    tcr_a = tcr_b = mhc = b2m = None
    peptides = []
    for chains, label, organism, seq in recs:
        nlabel = norm(label)
        if any(w in nlabel for w in EXCLUDE_WORDS):
            continue
        clean = seq.replace(" ", "").upper()
        if "T CELL RECEPTOR ALPHA" in nlabel or "TCR ALPHA" in nlabel:
            if tcr_a is None and len(clean) > 100:
                tcr_a = clean
        elif "T CELL RECEPTOR BETA" in nlabel or "TCR BETA" in nlabel:
            if tcr_b is None and len(clean) > 100:
                tcr_b = clean
        elif ("HLA" in nlabel or "MHC CLASS I" in nlabel) and "CLASS II" not in nlabel:
            if mhc is None and 240 <= len(clean) <= 340:
                mhc = clean
        elif "BETA 2 MICROGLOBULIN" in nlabel or nlabel.startswith("BETA 2 MICRO"):
            b2m = clean
        elif 8 <= len(clean) <= 16 and set(clean) <= STANDARD_AA:
            peptides.append((label, clean))

    if not (tcr_a and tcr_b and mhc and b2m and peptides):
        return None
    # one peptide chain per complex; entries with several distinct peptide
    # sequences are ambiguous -> require a single distinct sequence
    distinct = {p for _, p in peptides}
    if len(distinct) != 1:
        return None
    if not all("HOMO SAPIENS" in organism.upper() for _, _, organism, seq in recs
               if seq in (tcr_a, tcr_b, mhc) or seq == b2m):
        return None
    return {"entry": entry_id, "peptide": peptides[0][1],
            "tcra": tcr_a, "tcrb": tcr_b, "mhc": mhc}


def main() -> None:
    print("[1/3] searching RCSB ...", flush=True)
    entry_ids = search_all_entries()
    print(f"      {len(entry_ids)} entries contain a human TCR alpha chain", flush=True)

    print("[2/3] fetching FASTAs (8 threads) ...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(classify_entry, entry_ids):
            if res:
                results.append(res)
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    print(f"      {len(ok)} complete human pMHC-I:TCR complexes "
          f"({len(err)} fetch errors)", flush=True)

    print("[3/3] deduplicating ...", flush=True)
    seen = set()
    unique = []
    for r in ok:
        key = (r["tcra"], r["tcrb"], r["peptide"], r["mhc"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    print(f"      {len(unique)} unique (TCR, pMHC) cognate pairs", flush=True)

    out = Path(__file__).resolve().parent / "candidates.tsv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["entry", "peptide_len", "peptide", "tcra_len", "tcrb_len", "mhc_len",
                    "peptide", "tcra", "tcrb", "mhc"])
        for r in sorted(unique, key=lambda r: (len(r["peptide"]), r["entry"])):
            w.writerow([r["entry"], len(r["peptide"]), r["peptide"],
                        len(r["tcra"]), len(r["tcrb"]), len(r["mhc"]),
                        r["peptide"], r["tcra"], r["tcrb"], r["mhc"]])
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
