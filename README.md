# ESMFold2 Toolkit

Local inference toolkit for [ESMFold2](https://huggingface.co/biohub/ESMFold2) — protein
structure prediction and pMHC–TCR binding scoring that runs **entirely on one local GPU**.
Model weights are loaded from local directories; **nothing is downloaded from HuggingFace
at runtime** (works on air-gapped / firewalled machines).

Three subcommands behind one entry point (`main.py`):

| Command | What it does | Input → Output |
| --- | --- | --- |
| `fold` | Single-chain structure prediction | FASTA → one PDB per sequence |
| `complex` | pMHC–TCR 4-chain complex folding + interface scoring | CSV (`peptide,mhc,tcralpha,tcrbeta`) → PDBs + `results.csv` with ipTM metrics |
| `reward` | Binding reward model for RL / sequence optimization | candidates CSV → rewards CSV |

All batch runs are **resumable**: already-processed rows are skipped on re-run, so you can
Ctrl-C and restart freely.

## Requirements

- **Linux + NVIDIA GPU** with ≥ 24 GB VRAM recommended (weights use ~13 GB: ESMFold2
  ≈ 0.9 GB fp32 + ESMC-6B ≈ 12 GB bf16; verified on A100 80 GB)
- **[uv](https://docs.astral.sh/uv/)** (installs Python 3.12 for you — the `esm` package
  requires `>=3.12,<3.13`)
- **~30 GB disk**: ~26 GB model weights + ~3 GB environment / vendored sources
- Internet access to **PyPI** and **huggingface.co** — only for the one-time setup below

## Setup (5 minutes)

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Get this repo

```bash
git clone https://github.com/HangboZhu/esmfold2-toolkit.git
cd esmfold2-toolkit
```

### 3. Download the model weights (~26 GB)

| Directory | HuggingFace repo | Contents |
| --- | --- | --- |
| `models/ESMFold2/` | [`biohub/ESMFold2`](https://huggingface.co/biohub/ESMFold2) | `model.safetensors`, `config.json`, `ccd.pkl` (~1.3 GB) |
| `models/ESMC-6B/` | [`biohub/ESMC-6B`](https://huggingface.co/biohub/ESMC-6B) | 6-sharded weights, index, config, tokenizer (~24 GB) |

```bash
uvx --from "huggingface_hub[cli]" hf download biohub/ESMFold2 --local-dir models/ESMFold2
uvx --from "huggingface_hub[cli]" hf download biohub/ESMC-6B  --local-dir models/ESMC-6B
```

<details>
<summary>Alternatives & troubleshooting</summary>

Via Python instead of the CLI:

```bash
uv run --with huggingface_hub python -c "\
from huggingface_hub import snapshot_download as d; \
d('biohub/ESMFold2', local_dir='models/ESMFold2'); \
d('biohub/ESMC-6B', local_dir='models/ESMC-6B')"
```

- If `huggingface.co` is blocked on your network (common in mainland China), point the
  tool at the mirror: export `HF_ENDPOINT=https://hf-mirror.com` before the commands above.
- A faster, non-MSA-conditioned variant `biohub/ESMFold2-Fast` also exists on the Hub;
  this toolkit targets the full `ESMFold2`.
- Other ESMC sizes (300M/600M) work too — pass `--esmc-dir` pointing at them (much lower
  VRAM, lower accuracy).

</details>

### 4. Create the environment

```bash
uv sync          # creates .venv with Python 3.12 and installs everything
```

The `esm` package and the Biohub `transformers` fork (which provides the
`ESMFold2Model` implementation) are **vendored in this repo under `external/`** and wired
in via `[tool.uv.sources]`, so `uv sync` never needs github.com / raw.githubusercontent.com.

### 5. Smoke test

```bash
uv run main.py fold -i examples/example.fasta -o out_smoke/ -g 0
```

You should see ubiquitin fold to mean pLDDT ≈ 80 / pTM ≈ 0.74 and GB1 (134 aa, two
domains) to pLDDT ≈ 81 / pTM ≈ 0.52. Two PDBs land in `out_smoke/` — you can open
them in PyMOL/ChimeraX; per-residue pLDDT is in the B-factor column.

## Usage

> Pick an idle GPU with `nvidia-smi` first and pass its id via `-g` (the toolkit pins
> itself to that single card).

### Single-chain folding

```bash
uv run main.py fold -i sequences.fasta -o out_fold/ -g 0
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `-i / --input` | input FASTA (one or more `>name` sequences) | required |
| `-o / --output` | output directory (created if missing) | required |
| `-g / --gpu` | physical GPU id for single-card inference | `0` |
| `--esmfold2-dir` | ESMFold2 model dir | `models/ESMFold2` |
| `--esmc-dir` | ESMC model dir | `models/ESMC-6B` |
| `--num-loops` | folding refinement loops (more = better, slower) | `3` |
| `--num-sampling-steps` | diffusion sampling steps | `50` |
| `--num-diffusion-samples` | diffusion samples per fold call | `1` |
| `--seed` | RNG seed (reproducible) | `0` |
| `-f / --force` | re-fold sequences whose PDB already exists | off |
| `-e / --errors` | write sequences that failed to fold to this FASTA (retryable) | off |

Each FASTA record becomes `<name>.pdb` (first token of the `>` header, sanitized), with
per-residue **pLDDT in the B-factor column** (0–100). Mean pLDDT and pTM print per
sequence. Non-standard residues are flagged but folding is still attempted.

Retry just the failures:

```bash
uv run main.py fold -i big.fasta -o out/ -g 0 -e failed.fasta    # first pass
uv run main.py fold -i failed.fasta -o out/ -g 0 -e failed.fasta # retry failures
```

### pMHC–TCR complex ipTM prediction

```bash
uv run main.py complex -i complexes.csv -o out_complex/ -g 0 --save-pae
```

Input CSV needs columns `peptide, mhc, tcralpha, tcrbeta` (case-insensitive;
`TCR_alpha`-style spellings match) plus an optional `id` column (else `row_1, row_2, ...`).
Chains are fed in fixed order **A=peptide, B=mhc, C=tcralpha, D=tcrbeta** and folded as
one complex — no MSAs, nothing downloaded. See `examples/example_pmhc_tcr.csv`: the 1G4
TCR / NY-ESO-1 (SLLMWITQC) / HLA-A*02:01 complex (PDB 2BNR, minus β2-microglobulin).

Per row it writes `<id>.pdb` and, with `--save-pae`, `<id>_pae.npz`. `results.csv` gets
one line per row: `iptm` (the main binding-strength score), `ptm`, `mean_plddt`, the six
chain-pair ipTMs (`iptm_peptide_mhc`, `iptm_peptide_tcra`, `iptm_peptide_tcrb`,
`iptm_mhc_tcra`, `iptm_mhc_tcrb`, `iptm_tcra_tcrb`), chain lengths and fold time.

| Extra flags | Meaning |
| --- | --- |
| `--save-pae` | save each row's full PAE matrix to `<id>_pae.npz` |
| `-f / --force` | re-fold rows already present in `results.csv` |

### Reward model (for RL / sequence optimization)

```bash
uv run main.py reward -i candidates.csv -o rewards.csv -g 0 \
    --decoys decoys.csv --num-samples 3 --lam 1.0
```

Per candidate TCR against one target pMHC:

- `r_bind` — mean over diffusion samples of mean(`iptm_peptide_tcra`, `iptm_peptide_tcrb`);
  the primary binding-strength signal (validated against experimental TCR–pMHC labels)
- `r_decoy_max` — same metric against decoy pMHCs (from `--decoys` CSV, columns
  `peptide,mhc`), max over decoys → specificity: a good binder scores high on target, low on decoys
- `reward` — `r_bind - lam * max(0, r_decoy_max)` (λ configurable via `--lam`)
- `r_bind_std` — std over samples (sampling-noise estimate)

Throughput: ~16 s per fold for V-domain TCRs (~110 aa), ~3× for full-length. Per candidate
cost = `num_samples × (1 target fold + len(decoys) folds)`. For large sweeps screen with
`--num-samples 1`, then re-score survivors with 3–5 independent seeds.

## Python API

Every component is importable for your own pipelines:

```python
import esmfold2_toolkit as ef2

ef2.setup_runtime(gpu=0)                      # pin GPU + offline HF; call BEFORE importing torch
model, builder = ef2.load_model()             # local ESMFold2 + ESMC-6B, ~13 GB VRAM

# Single chain
pdb_str, plddt, ptm = ef2.fold_single(model, builder, "MKTAYIAKQRQISFVKSHFSRQ")
# -> (PDB string, mean pLDDT on 0-1 scale, pTM)

# pMHC-TCR complex: result has .complex, .iptm, .ptm, .plddt, .pair_chains_iptm, .pae
result = ef2.fold_complex(model, builder, [peptide, mhc, tcra, tcrb])
metrics = ef2.extract_complex_metrics(result)   # iptm/ptm/mean_plddt + 6 chain-pair ipTMs
pdb_str = ef2.complex_to_pdb_string(result.complex)

# Binding score for RL (seed-averaged mean pep-TCR ipTM)
score = ef2.score_binding(model, builder, peptide, mhc, tcra, tcrb, num_samples=3)
# -> {"r_bind": ..., "r_bind_std": ..., "iptm": ..., "plddt": ..., "r_samples": [...]}

# Batch with resume (same functions the CLI uses)
ef2.run_fold(input_fasta="in.fasta", output_dir="out/", gpu=0)
ef2.run_complex(input_csv="complexes.csv", output_dir="out/", gpu=0, save_pae=True)
ef2.run_reward(input_csv="cands.csv", output_csv="rewards.csv", gpu=0, decoys="decoys.csv")

# Input parsing utilities
records = ef2.parse_fasta(path)               # [(name, sequence), ...]
rows = ef2.read_complex_csv(path, ["peptide", "mhc", "tcralpha", "tcrbeta"])
```

## How local loading works

```python
from esmfold2_toolkit import load_model

model, builder = load_model("models/ESMFold2", "models/ESMC-6B")
# internally:
#   ESMFold2Model.from_pretrained(dir, local_files_only=True, load_esmc=False).cuda().eval()
#   model.load_esmc(esmc_dir, precision="bf16")       # attach the local ESMC-6B backbone
#   ESMFold2InputBuilder(ccd_cache=esmfold2_dir)      # dir containing ccd.pkl
```

`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are set automatically by `setup_runtime()`.

## Performance notes

- **VRAM:** ~13 GB weights; activation memory scales with complex size. Anything from a
  24 GB card upward is fine for pMHC–TCR work.
- **Speed:** the run logs warnings that `transformer-engine`, `xformers`, and `flash-attn`
  are absent — folding still works, just slower. To speed up:
  `uv pip install xformers flash-attn` (needs a CUDA toolchain matching your torch build).
- Multiple runs in parallel: launch each on a different free GPU (`-g`).

## Repository layout

```
esmfold2-toolkit/
├── main.py                  # unified CLI: fold | complex | reward
├── esmfold2_toolkit/        # reusable library (import this)
│   ├── env.py               # GPU pinning + offline HF env setup
│   ├── model.py             # offline model loading & file validation
│   ├── folding.py           # fold_single / fold_complex / score_binding / metrics
│   ├── data.py              # FASTA + CSV parsing, resume bookkeeping
│   └── pipelines.py         # resumable batch loops behind the CLI
├── examples/                # tiny demo inputs (ubiquitin/GB1 FASTA, 1G4/NY-ESO-1 pMHC-TCR)
├── benchmarks/              # archived internal benchmark scripts (see benchmarks/README.md)
├── external/                # vendored esm + Biohub transformers fork (installed by uv sync)
├── models/                  # downloaded weights (gitignored) -- see Setup step 3
├── pyproject.toml           # uv project; Python 3.12; path-pinned vendored deps
└── .python-version          # pins 3.12
```

## Credits

- [ESMFold2](https://huggingface.co/biohub/ESMFold2) & [ESMC-6B](https://huggingface.co/biohub/ESMC-6B)
  models by Biohub (weights follow the models' own license — see each model card).
- `esm` and the Biohub `transformers` fork are vendored unmodified under `external/`
  (their licenses are in the respective directories).

## License

Code in this repo: [MIT](LICENSE). Model weights: see the
[ESMFold2](https://huggingface.co/biohub/ESMFold2) /
[ESMC-6B](https://huggingface.co/biohub/ESMC-6B) model cards for their terms.

## Full / Fast 模式与多卡常驻服务

使用同一个 `.venv`、同一个 ESMC 骨干；`--mode full` 为默认原版，`--mode fast` 选择 `models/ESMFold2-Fast` 并默认启用融合内核。`--kernel-backend reference|fused|auto` 可显式选择，chunk 默认 64。原来的三个批处理子命令也接受这些参数；更换模式必须使用不同输出路径，避免把原版结果误当成 Fast 缓存。

```bash
.venv/bin/python main.py fold -i examples/example.fasta -o out_fast --mode fast --gpu 4
.venv/bin/python main.py serve --mode fast --gpus 4,5 --host 127.0.0.1 --port 7878
```

`serve` 为 peptide/MHC/b2m/TCRβ 四链打分接口，每张卡一个常驻模型，空闲卡领取下一项；提高候选批次吞吐，不把一个结构拆到多卡。支持 `/health`、`/metadata`、`POST /predict`（单个四链 JSON）和 `POST /score`（`{"rows":[...]}`，结果顺序与输入一致）。每行必需字符串字段为 `peptide,mhc,b2m,tcrbeta`；可传 `candidate` 和 `provenance.split` 等来源信息。未知来源保持 unknown，不能默认加入 train。

该服务使用 `/disk1/zhuhb/project/tcr-change-base/structure_scoring/service.py`，可用 `--project-root` 指定源码目录。模型加载前固定每张物理 GPU；数据默认保存在项目 `data/rl/esmfold/`，包含结构、原始 PAE logits、PAE、pLDDT 与坐标。日志/PID 在 `structure_scoring/output_esmfold_service/`。相同教师、序列和 seed 的重复请求复用已校验的结果；模式不同不能混用缓存。在线服务的分数使用经独立验证的 PAE pair-ipTM 公式。

原独立 Fast 试验环境仅保留在 `.deployment_history/fast_trial/`，不作为运行入口；旧实验路径迁移见项目数据集的 `audits/unified_deployment_migration.json`。
