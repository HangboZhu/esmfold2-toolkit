"""Batch pipelines with resume support: fold / complex / reward.

Each pipeline takes plain keyword arguments (no argparse namespace), sets up
the runtime environment, validates local model files, loads the model once,
and loops over the input rows -- skipping rows that are already done so runs
are resumable. Failed rows are logged and skipped without stopping the batch.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .data import (
    clean_sequence,
    open_results_csv,
    parse_fasta,
    read_complex_csv,
    read_done_ids,
    sanitize_name,
)
from .env import setup_runtime
from .folding import (
    PMHC_TCR_CHAINS,
    RESULT_COLUMNS,
    STANDARD_AA,
    check_chain_order,
    complex_to_pdb_string,
    extract_complex_metrics,
    fold_complex,
    fold_single,
    score_binding,
)
from .model import DEFAULT_ESMC_DIR, DEFAULT_ESMFOLD2_DIR, load_model, validate_model_files, resolve_options


def _check_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. ESMFold2 local inference requires a GPU.")


def _load_and_report(esmfold2_dir: Path, esmc_dir: Path, *, mode="full", kernel_backend="auto", chunk_size=64):
    import torch  # noqa: F401 - ensure CUDA is initialised after GPU pinning

    print(f"[info] GPU: cuda (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    print(f"[info] Loading ESMFold2 from {esmfold2_dir}")
    print(f"[info] Loading ESMC backbone from {esmc_dir}")
    t0 = time.time()
    model, builder = load_model(esmfold2_dir, esmc_dir, mode=mode, kernel_backend=kernel_backend, chunk_size=chunk_size)
    print(f"[info] Models loaded in {time.time() - t0:.1f}s")
    return model, builder


def run_fold(input_fasta: Path, output_dir: Path, *, gpu: int = 0,
             esmfold2_dir: Path | None = None, esmc_dir: Path = DEFAULT_ESMC_DIR,
             mode: str = "full", kernel_backend: str = "auto", chunk_size: int = 64,
             num_loops: int = 3, num_sampling_steps: int = 50,
             num_diffusion_samples: int = 1, seed: int = 0, force: bool = False,
             errors: Path | None = None) -> None:
    """Fold every sequence in a FASTA file; one PDB per sequence in output_dir.

    Sequences whose PDB already exists are skipped unless force=True. Failed
    sequences are logged and (if errors is given) written to that FASTA.
    """
    setup_runtime(gpu)
    esmfold2_dir, kernel_backend = resolve_options(esmfold2_dir, mode=mode, kernel_backend=kernel_backend, chunk_size=chunk_size)
    if not input_fasta.is_file():
        raise SystemExit(f"Input FASTA not found: {input_fasta}")
    validate_model_files(esmfold2_dir, esmc_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _guard_mode(output_dir / "inference_mode.json", output_dir.glob("*.pdb"), mode, kernel_backend, esmfold2_dir, chunk_size)
    records = parse_fasta(input_fasta)

    # Resume support: which sequences already have a PDB on disk.
    existing = {p.name for p in output_dir.glob("*.pdb")}
    todo = [(n, s) for (n, s) in records if force or f"{sanitize_name(n)}.pdb" not in existing]
    n_skipped = len(records) - len(todo)
    if n_skipped:
        print(f"[info] {n_skipped}/{len(records)} already folded -> skipping (use --force to redo)")
    print(f"[info] {len(todo)}/{len(records)} to fold from {input_fasta}")

    if not todo:
        print("[done] nothing to do -- all sequences already processed")
        return

    _check_cuda()
    model, builder = _load_and_report(esmfold2_dir, esmc_dir, mode=mode, kernel_backend=kernel_backend, chunk_size=chunk_size)

    from tqdm import tqdm

    done = failed = 0
    failed_records: list[tuple[str, str]] = []
    pbar = tqdm(todo, desc="Folding", unit="seq", dynamic_ncols=True)
    for name, raw_seq in pbar:
        sequence = clean_sequence(raw_seq)
        out_path = output_dir / f"{sanitize_name(name)}.pdb"
        pbar.set_postfix_str(f"{out_path.stem} len={len(sequence)}")
        if not sequence:
            pbar.write(f"[warn] '{name}': empty sequence, skipping.")
            continue
        nonstandard = sorted(set(sequence) - STANDARD_AA)
        if nonstandard:
            pbar.write(f"[warn] '{name}': non-standard residue(s) {nonstandard}; attempting fold anyway.")
        t1 = time.time()
        try:
            pdb_str, plddt, ptm = fold_single(
                model, builder, sequence,
                num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=num_diffusion_samples, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - keep folding remaining sequences
            failed += 1
            failed_records.append((name, raw_seq))
            pbar.write(f"[error] '{name}' failed: {exc}")
            continue
        out_path.write_text(pdb_str)
        done += 1
        plddt_pct = plddt * 100 if plddt <= 1.0 else plddt  # pLDDT is conventionally 0-100
        pbar.write(
            f"[ok] {out_path.name} | mean pLDDT={plddt_pct:.1f} | pTM={ptm:.3f} | {time.time() - t1:.1f}s"
        )
    pbar.close()

    if errors is not None:
        if failed_records:
            with errors.open("w") as fh:
                for fname, fseq in failed_records:
                    fh.write(f">{fname}\n{fseq}\n")
            print(f"[info] wrote {len(failed_records)} failed sequence(s) to {errors}")
        else:
            print(f"[info] no failures; nothing written to {errors}")

    print(f"[done] folded={done} skipped={n_skipped} failed={failed}")


def run_complex(input_csv: Path, output_dir: Path, *, gpu: int = 0,
                esmfold2_dir: Path | None = None, esmc_dir: Path = DEFAULT_ESMC_DIR,
             mode: str = "full", kernel_backend: str = "auto", chunk_size: int = 64,
                num_loops: int = 3, num_sampling_steps: int = 50,
                num_diffusion_samples: int = 1, seed: int = 0, force: bool = False,
                save_pae: bool = False) -> None:
    """Fold every row of a complexes CSV (peptide,mhc,tcralpha,tcrbeta[,id]).

    Writes <id>.pdb per row plus results.csv with iptm/ptm/mean_plddt and the
    six chain-pair ipTMs. Rows already in results.csv are skipped unless
    force=True. With save_pae, also writes <id>_pae.npz per row.
    """
    import numpy as np

    setup_runtime(gpu)
    esmfold2_dir, kernel_backend = resolve_options(esmfold2_dir, mode=mode, kernel_backend=kernel_backend, chunk_size=chunk_size)
    if not input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    validate_model_files(esmfold2_dir, esmc_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"

    _guard_mode(output_dir / "inference_mode.json", [results_csv], mode, kernel_backend, esmfold2_dir, chunk_size)
    rows = read_complex_csv(input_csv, [name for name, _ in PMHC_TCR_CHAINS])

    # Resume support: which row ids already have a result line in results.csv.
    done = set() if force else read_done_ids(results_csv)
    todo = [row for row in rows if row["id"] not in done]
    n_skipped = len(rows) - len(todo)
    if n_skipped:
        print(f"[info] {n_skipped}/{len(rows)} already have results -> skipping (use --force to redo)")
    print(f"[info] {len(todo)}/{len(rows)} complexes to fold from {input_csv}")

    if not todo:
        print("[done] nothing to do -- all rows already processed")
        return

    _check_cuda()
    model, builder = _load_and_report(esmfold2_dir, esmc_dir, mode=mode, kernel_backend=kernel_backend, chunk_size=chunk_size)

    results_fh, results_writer = open_results_csv(results_csv, RESULT_COLUMNS)
    done_count = failed = 0
    try:
        for row in todo:
            row_id = sanitize_name(row["id"], fallback="complex")
            sequences = [clean_sequence(row[name]) for name, _ in PMHC_TCR_CHAINS]
            total_len = sum(len(s) for s in sequences)
            print(f"[fold] {row_id} | " + " ".join(
                f"{name}={len(seq)}aa" for (name, _), seq in zip(PMHC_TCR_CHAINS, sequences)
            ) + f" | total={total_len}aa")
            if any(not seq for seq in sequences):
                print(f"[error] {row_id}: empty sequence in one of the chain columns, skipping.")
                failed += 1
                continue
            for (name, _), seq in zip(PMHC_TCR_CHAINS, sequences):
                nonstandard = sorted(set(seq) - STANDARD_AA)
                if nonstandard:
                    print(f"[warn] {row_id}/{name}: non-standard residue(s) {nonstandard}; attempting fold anyway.")

            t1 = time.time()
            try:
                result = fold_complex(
                    model, builder, sequences,
                    num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                    num_diffusion_samples=num_diffusion_samples, seed=seed,
                )
                order_ok = check_chain_order(result.complex, sequences)
                if not order_ok:
                    print(f"[warn] {row_id}: chain token counts do not match input lengths "
                          "in input order; per-pair ipTM columns may be misassigned.")
                metrics = extract_complex_metrics(result)
                pdb_str = complex_to_pdb_string(result.complex)
            except Exception as exc:  # noqa: BLE001 - keep folding remaining rows
                failed += 1
                print(f"[error] {row_id} failed: {exc}")
                continue

            (output_dir / f"{row_id}.pdb").write_text(pdb_str)
            if save_pae:
                pae = getattr(result, "pae", None)
                if pae is not None:
                    np.savez_compressed(output_dir / f"{row_id}_pae.npz", pae=np.asarray(pae))

            record = {"id": row_id, **metrics,
                      **{f"len_{name}": len(seq) for (name, _), seq in zip(PMHC_TCR_CHAINS, sequences)},
                      "num_residues": total_len,
                      "fold_seconds": f"{time.time() - t1:.1f}"}
            results_writer.writerow(record)
            results_fh.flush()
            done_count += 1
            print(
                f"[ok] {row_id} | ipTM={metrics['iptm']:.4f} | pTM={metrics['ptm']:.4f} "
                f"| mean pLDDT={metrics['mean_plddt']:.1f} "
                f"| pep-MHC={metrics['iptm_peptide_mhc']:.3f} "
                f"| pep-TCRa={metrics['iptm_peptide_tcra']:.3f} "
                f"| pep-TCRb={metrics['iptm_peptide_tcrb']:.3f} "
                f"| {time.time() - t1:.1f}s"
            )
    finally:
        results_fh.close()

    print(f"[done] folded={done_count} skipped={n_skipped} failed={failed}")
    print(f"[done] results written to {results_csv}")


def run_reward(input_csv: Path, output_csv: Path, *, gpu: int = 0,
               decoys: Path | None = None, num_samples: int = 3, lam: float = 1.0,
               esmfold2_dir: Path | None = None, esmc_dir: Path = DEFAULT_ESMC_DIR,
             mode: str = "full", kernel_backend: str = "auto", chunk_size: int = 64,
               num_loops: int = 3, num_sampling_steps: int = 50, seed: int = 0,
               force: bool = False) -> None:
    """Score candidate TCRs against one target pMHC (+ optional decoy pMHCs).

    reward = r_bind - lam * max(0, max_decoy_r_bind), where r_bind is the
    seed-averaged mean(peptide-TCR ipTM) from score_binding. Incremental:
    existing ids in output_csv are skipped unless force=True.
    """
    setup_runtime(gpu)
    esmfold2_dir, kernel_backend = resolve_options(esmfold2_dir, mode=mode, kernel_backend=kernel_backend, chunk_size=chunk_size)
    validate_model_files(esmfold2_dir, esmc_dir)
    _guard_mode(output_csv.with_suffix(output_csv.suffix + ".mode.json"), [output_csv], mode, kernel_backend, esmfold2_dir, chunk_size)
    chains = [name for name, _ in PMHC_TCR_CHAINS]
    cands = read_complex_csv(input_csv, chains)
    decoy_rows = read_complex_csv(decoys, ["peptide", "mhc"]) if decoys else []

    if output_csv.is_file() and force:
        output_csv.unlink()
    done = read_done_ids(output_csv)
    todo = [r for r in cands if sanitize_name(r["id"], fallback="cand") not in done]
    print(f"[info] {len(todo)}/{len(cands)} candidates to score ({len(decoy_rows)} decoys each, "
          f"{num_samples} samples/fold)")

    _check_cuda()
    model, builder = _load_and_report(esmfold2_dir, esmc_dir, mode=mode, kernel_backend=kernel_backend, chunk_size=chunk_size)

    fields = ["id", "reward", "r_bind", "r_bind_std", "r_decoy_max", "iptm", "plddt"]
    results_fh, writer = open_results_csv(output_csv, fields)

    for i, cand in enumerate(todo):
        cid = sanitize_name(cand["id"], fallback="cand")
        seqs = {k: clean_sequence(cand[k]) for k in chains}
        t0 = time.time()
        try:
            tgt = score_binding(model, builder, seqs["peptide"], seqs["mhc"],
                                seqs["tcralpha"], seqs["tcrbeta"], num_samples=num_samples,
                                num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                                seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {cid} target fold failed: {exc}")
            continue
        decoy_max = None
        if decoy_rows:
            dvals = []
            for d in decoy_rows:
                try:
                    dvals.append(score_binding(model, builder, clean_sequence(d["peptide"]),
                                               clean_sequence(d["mhc"]), seqs["tcralpha"],
                                               seqs["tcrbeta"], num_samples=num_samples,
                                               num_loops=num_loops,
                                               num_sampling_steps=num_sampling_steps,
                                               seed=seed)["r_bind"])
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] {cid} decoy fold failed: {exc}")
            if dvals:
                decoy_max = max(dvals)
        reward = tgt["r_bind"] - lam * max(0.0, decoy_max or 0.0)
        rec = {"id": cid, "reward": round(reward, 4), "r_bind": round(tgt["r_bind"], 4),
               "r_bind_std": round(tgt["r_bind_std"], 4),
               "r_decoy_max": round(decoy_max, 4) if decoy_max is not None else "",
               "iptm": round(tgt["iptm"], 4), "plddt": round(tgt["plddt"], 1)}
        writer.writerow(rec)
        results_fh.flush()
        extra = f" decoy_max={decoy_max:.3f}" if decoy_max is not None else ""
        print(f"[ok] {cid} | reward={reward:.4f} | r_bind={tgt['r_bind']:.4f} "
              f"±{tgt['r_bind_std']:.3f}{extra} | {time.time()-t0:.0f}s ({i+1}/{len(todo)})")
    results_fh.close()
    print(f"[done] rewards -> {output_csv}")


def _guard_mode(path, outputs, mode, backend, directory, chunk_size):
    """An output containing full predictions must not be resumed as Fast."""
    import json
    contract = dict(mode=mode, kernel_backend=backend, model_dir=str(Path(directory).resolve()), chunk_size=chunk_size)
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Inference mode changed; choose a new output path")
        return
    if any(p.exists() for p in outputs):
        if mode != "full" or backend != "reference":
            raise ValueError("Existing outputs have no mode manifest; choose a new output path for Fast/fused")
        return  # Preserve the behavior of legacy full/reference output folders.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2) + "\n")
