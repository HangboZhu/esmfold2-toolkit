#!/usr/bin/env python3
"""ESMFold2 toolkit: local structure prediction & pMHC-TCR binding scoring.

Three subcommands, all running ESMFold2 (with a local ESMC backbone) on a
single GPU, fully offline -- models are loaded from local directories and
nothing is downloaded from HuggingFace at runtime:

  fold     Fold every sequence in a FASTA file -> one PDB per sequence.
  complex  Fold peptide-MHC-TCR complexes from a CSV -> PDBs + ipTM metrics.
  reward   Score candidate TCRs against a target pMHC (+ decoys) for RL.

Usage:
    uv run main.py fold    -i examples/example.fasta        -o out_fold/
    uv run main.py complex -i examples/example_pmhc_tcr.csv -o out_complex/
    uv run main.py reward  -i candidates.csv -o rewards.csv --decoys decoys.csv

Each subcommand is resumable: already-processed rows are skipped on re-run
(pass -f/--force to redo). The underlying components are importable from the
`esmfold2_toolkit` package for use in your own scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=("full", "fast"), default="full",
                   help="Model mode; Fast shares the existing environment and ESMC backbone.")
    p.add_argument("--kernel-backend", choices=("auto", "reference", "fused"), default="auto",
                   help="auto: reference for full, fused for fast.")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--esmfold2-dir", type=Path, default=None,
                   help="Local ESMFold2 model dir (model.safetensors, config.json, ccd.pkl).")
    p.add_argument("--esmc-dir", type=Path, default=REPO / "models" / "ESMC-6B",
                   help="Local ESMC model dir (sharded weights, index, config, tokenizer).")


def _add_folding_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-loops", type=int, default=3, help="Folding refinement loops.")
    p.add_argument("--num-sampling-steps", type=int, default=50, help="Diffusion sampling steps.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ESMFold2 local toolkit: fold / complex / reward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="Persistent peptide/MHC/b2m/TCRbeta scoring on multiple GPUs.")
    _add_model_args(p)
    _add_folding_args(p)
    p.add_argument("--gpus", default="4,5", help="Comma-separated physical GPU indices or UUIDs.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7878)
    p.add_argument("--project-root", type=Path, default=Path("/disk1/zhuhb/project/tcr-change-base"))
    p.add_argument("--archive", type=Path, default=Path("/disk1/zhuhb/project/tcr-change-base/data/rl/esmfold"))
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_serve)

    # --- fold -----------------------------------------------------------
    p = sub.add_parser("fold", help="Single-chain folding: FASTA -> one PDB per sequence.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="Input FASTA file (one or more protein sequences).")
    p.add_argument("-o", "--output", required=True, type=Path,
                   help="Output directory (created if it does not exist).")
    p.add_argument("-g", "--gpu", type=int, default=0,
                   help="GPU device id to use for single-card inference.")
    _add_model_args(p)
    _add_folding_args(p)
    p.add_argument("--num-diffusion-samples", type=int, default=1, help="Number of diffusion samples.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    p.add_argument("-f", "--force", action="store_true",
                   help="Re-fold sequences whose PDB already exists (default: skip to resume).")
    p.add_argument("-e", "--errors", type=Path, default=None,
                   help="Write sequences that failed to fold to this FASTA file (for retry).")
    p.set_defaults(func=lambda a: _run("run_fold", a))

    # --- complex --------------------------------------------------------
    p = sub.add_parser("complex", help="pMHC-TCR complex folding: CSV -> PDBs + ipTM metrics.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="Input CSV; required columns peptide,mhc,tcralpha,tcrbeta (case-insensitive), optional id.")
    p.add_argument("-o", "--output", required=True, type=Path,
                   help="Output directory (PDBs, PAE npz, results.csv); created if missing.")
    p.add_argument("-g", "--gpu", type=int, default=0,
                   help="GPU device id for single-card inference.")
    _add_model_args(p)
    _add_folding_args(p)
    p.add_argument("--num-diffusion-samples", type=int, default=1, help="Number of diffusion samples.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    p.add_argument("-f", "--force", action="store_true",
                   help="Re-fold rows already present in results.csv (default: skip to resume).")
    p.add_argument("--save-pae", action="store_true",
                   help="Save each row's full PAE matrix to <id>_pae.npz.")
    p.set_defaults(func=lambda a: _run("run_complex", a))

    # --- reward ---------------------------------------------------------
    p = sub.add_parser("reward", help="pMHC-TCR reward model: candidates CSV -> rewards CSV.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="candidates csv: peptide,mhc,tcralpha,tcrbeta[,id]")
    p.add_argument("-o", "--output", required=True, type=Path,
                   help="output csv (incremental; existing ids are skipped unless --force)")
    p.add_argument("-g", "--gpu", type=int, default=0)
    p.add_argument("--decoys", type=Path, default=None,
                   help="decoy pMHCs csv: peptide,mhc (specificity penalty)")
    p.add_argument("--num-samples", type=int, default=3,
                   help="diffusion samples per fold; reward = mean (noise drops as 1/sqrt(n))")
    p.add_argument("--lam", type=float, default=1.0,
                   help="weight of the decoy specificity penalty")
    _add_model_args(p)
    _add_folding_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-f", "--force", action="store_true")
    p.set_defaults(func=lambda a: _run("run_reward", a))

    return parser


def _run(pipeline_name: str, args: argparse.Namespace) -> None:
    """Forward parsed CLI args to the named esmfold2_toolkit pipeline."""
    import esmfold2_toolkit.pipelines as pipelines

    kwargs = {k: v for k, v in vars(args).items() if k not in ("command", "func")}
    # CLI arg names -> pipeline parameter names (library API uses descriptive ones).
    command = args.command
    if "input" in kwargs:
        kwargs["input_fasta" if command == "fold" else "input_csv"] = kwargs.pop("input")
    if "output" in kwargs:
        kwargs["output_csv" if command == "reward" else "output_dir"] = kwargs.pop("output")
    getattr(pipelines, pipeline_name)(**kwargs)


def _serve(args):
    import sys
    sys.path.insert(0, str(args.project_root))
    from structure_scoring.service import serve
    serve(args)


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
