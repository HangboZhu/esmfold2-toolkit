"""Input/output helpers: FASTA parsing, forgiving CSV reading, resume bookkeeping."""
from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return (name, sequence) pairs. Name = first whitespace token of the header."""
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                header = line[1:].strip()
                name = header.split()[0] if header else f"sequence{len(records) + 1}"
                chunks = []
            elif name is not None:
                chunks.append(line)
            else:
                raise ValueError(f"FASTA file {path} has sequence data before any '>' header.")
    if name is not None:
        records.append((name, "".join(chunks)))
    if not records:
        raise ValueError(f"No sequences found in FASTA file {path}.")
    return records


def normalize_header(header: str) -> str:
    """'TCR alpha'/'TCR_alpha'/'TcrAlpha' -> 'tcralpha' for forgiving CSV matching."""
    return re.sub(r"[^a-z0-9]", "", header.strip().lower())


def read_complex_csv(path: Path, chain_names: Sequence[str]) -> list[dict[str, str]]:
    """Read a complexes CSV into dicts keyed by chain name (plus 'id'/'_line_no').

    Headers are matched forgivingly (see normalize_header). Required columns
    raise SystemExit listing what is missing. An optional 'id' column names the
    row (else row_1, row_2, ...). Blank lines are skipped.
    """
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"Input CSV {path} is empty.")
        names = [normalize_header(h) for h in header]
        missing = [name for name in chain_names if name not in names]
        if missing:
            raise SystemExit(
                f"Input CSV {path} is missing required column(s): {missing}. "
                f"Found headers: {header}"
            )
        id_idx = names.index("id") if "id" in names else None
        col_idx = {name: names.index(name) for name in chain_names}

        rows: list[dict[str, str]] = []
        for line_no, raw in enumerate(reader, start=2):
            if not any(cell.strip() for cell in raw):
                continue  # blank line
            row = {
                name: (raw[idx].strip() if idx < len(raw) else "")
                for name, idx in col_idx.items()
            }
            if id_idx is not None and id_idx < len(raw) and raw[id_idx].strip():
                row_id = raw[id_idx].strip()
            else:
                row_id = f"row_{len(rows) + 1}"
            row["id"] = row_id
            row["_line_no"] = str(line_no)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No data rows found in input CSV {path}.")
    return rows


def clean_sequence(seq: str) -> str:
    """Uppercase, drop whitespace and trailing stop codons."""
    return seq.replace(" ", "").replace("\t", "").replace("*", "").upper()


def sanitize_name(name: str, fallback: str = "sequence") -> str:
    """Filesystem-safe identifier (used for PDB filenames and result ids)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return cleaned or fallback


def read_done_ids(results_csv: Path) -> set[str]:
    """Ids already present in a results.csv (for resume support)."""
    if not results_csv.is_file():
        return set()
    with results_csv.open(newline="") as fh:
        return {row["id"] for row in csv.DictReader(fh) if row.get("id")}


def open_results_csv(results_csv: Path, fieldnames: Sequence[str]):
    """Open results.csv for appending, writing the header for a new file."""
    new_file = not results_csv.is_file()
    fh = results_csv.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
    if new_file:
        writer.writeheader()
        fh.flush()
    return fh, writer
