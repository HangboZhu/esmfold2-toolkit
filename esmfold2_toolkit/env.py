"""Runtime environment setup: GPU pinning + offline HuggingFace."""
from __future__ import annotations

import os
import sys


def setup_runtime(gpu: int) -> None:
    """Prepare the process for single-GPU, fully-offline ESMFold2 inference.

    - Pins the process to one GPU via CUDA_VISIBLE_DEVICES.
    - Forces HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE so nothing reaches HuggingFace.
    - Line-buffers stdout/stderr so progress shows up live when redirected.

    Call this BEFORE importing torch (CUDA context) -- e.g. at the top of your
    script or entry point. The bundled pipelines call it as their first step.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:  # noqa: BLE001
            pass

    # Pin to a single GPU BEFORE importing torch / CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Force offline so nothing tries to reach HuggingFace (may be unreachable).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
