# Benchmarks (archived)

Internal benchmark scripts from the development of this toolkit, kept for reference.
**These are archival, not supported** — they were written for one-off experiments and
some depend on data or paths that are not part of this repo:

- `build_tcrvdb_sample.py` reads a private local copy of the TCRvdb/VDJdb dataset and a
  `/tmp/hla_chains.json` cache; the derived `tcrvdb_validation*.csv` files were removed.
- `build_metrics_workbook.py` / `add_tcrvdb_sheet.py` rely on a private local xlsx helper
  (not shipped) to write Excel workbooks.
- `discover_complexes.py` scrapes the RCSB web API to find complete human pMHC-I:TCR
  complexes (network-dependent) — this one is standalone and should still run.
- `build_cohort3.py` / `build_expanded_csv.py` build benchmark CSVs from hard-coded
  cognate/negative complexes (`build_cohort3.py` also needs a `candidates.tsv` from the
  discovery step above).
- `evaluate_tcrvdb.py` / `evaluate_3seed.py` / `summarize_benchmark.py` read the
  `results.csv` files produced by `uv run main.py complex` (they expect them under
  `../output/out_*/results.csv`) and print AUC / separation summaries to the console.

The historical benchmark outputs (cohort CSVs, `benchmark_metrics.xlsx`, per-complex
PDBs) are not included in this repo.
