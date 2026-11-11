"""Local ESMFold2 model loading (ESMC backbone attached from local files, no HF)."""
from __future__ import annotations

from pathlib import Path

# Repo layout: esmfold2_toolkit/model.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ESMFOLD2_DIR = REPO_ROOT / "models" / "ESMFold2"
DEFAULT_ESMC_DIR = REPO_ROOT / "models" / "ESMC-6B"


def resolve_options(esmfold2_dir=None, *, mode="full", kernel_backend="auto", chunk_size=64):
    """Select weights and kernels without creating a CUDA context."""
    if mode not in ("full", "fast"):
        raise ValueError("mode must be full or fast")
    if kernel_backend not in ("auto", "reference", "fused"):
        raise ValueError("kernel_backend must be auto, reference or fused")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    directory = Path(esmfold2_dir) if esmfold2_dir is not None else REPO_ROOT / "models" / ("ESMFold2-Fast" if mode == "fast" else "ESMFold2")
    backend = ("fused" if mode == "fast" else "reference") if kernel_backend == "auto" else kernel_backend
    return directory, backend

REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "ccd.pkl",
)
REQUIRED_ESMC_FILES = (
    "config.json",
    "model.safetensors.index.json",
)


def validate_model_files(esmfold2_dir: Path = DEFAULT_ESMFOLD2_DIR,
                         esmc_dir: Path = DEFAULT_ESMC_DIR) -> None:
    """Check the local model files exist before pulling in heavy libraries."""
    required = [esmfold2_dir / name for name in REQUIRED_FILES]
    required += [esmc_dir / name for name in REQUIRED_ESMC_FILES]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing local model files:\n" + "\n".join(f"  - {p}" for p in missing))


def load_model(esmfold2_dir: Path | str | None = None,
               esmc_dir: Path | str = DEFAULT_ESMC_DIR, *, mode="full",
               kernel_backend="auto", chunk_size=64):
    """Load ESMFold2 + local ESMC backbone, fully offline.

    Requires CUDA. Returns (model, builder) where builder is an
    ESMFold2InputBuilder ready to `builder.fold(model, ...)`.

    Note: set esmfold2_toolkit.env.setup_runtime(gpu) first, and make sure
    torch is imported afterwards (it is, lazily, inside here).
    """
    esmfold2_dir, backend = resolve_options(esmfold2_dir, mode=mode,
        kernel_backend=kernel_backend, chunk_size=chunk_size)
    validate_model_files(esmfold2_dir, Path(esmc_dir))
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from esm.models.esmfold2 import ESMFold2InputBuilder

    # Load ESMFold2 WITHOUT auto-downloading the ESMC backbone (HF is offline).
    model = ESMFold2Model.from_pretrained(
        str(esmfold2_dir),
        local_files_only=True,
        load_esmc=False,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    # Attach the ESMC language-model backbone from its own local directory.
    model.load_esmc(str(esmc_dir), precision="bf16")
    model.eval()
    model.set_kernel_backend(None if backend == "reference" else backend)
    model.set_chunk_size(chunk_size)

    # ccd_cache is the DIRECTORY containing ccd.pkl, not the file path itself.
    builder = ESMFold2InputBuilder(ccd_cache=str(esmfold2_dir))
    return model, builder
