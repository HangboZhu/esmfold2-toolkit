"""esmfold2-toolkit: local ESMFold2 inference utilities.

Reusable components for running ESMFold2 (with a local ESMC backbone) on a
single GPU, fully offline:

- model loading / validation            (esmfold2_toolkit.model)
- single-chain & complex folding        (esmfold2_toolkit.folding)
- FASTA / CSV input parsing             (esmfold2_toolkit.data)
- batch pipelines with resume           (esmfold2_toolkit.pipelines)
"""

from .data import (
    clean_sequence,
    normalize_header,
    open_results_csv,
    parse_fasta,
    read_complex_csv,
    read_done_ids,
    sanitize_name,
)
from .env import setup_runtime
from .folding import (
    CHAIN_PAIRS,
    PMHC_TCR_CHAINS,
    RESULT_COLUMNS,
    complex_to_pdb_string,
    extract_complex_metrics,
    fold_complex,
    fold_single,
    score_binding,
)
from .model import DEFAULT_ESMC_DIR, DEFAULT_ESMFOLD2_DIR, load_model, validate_model_files
from .pipelines import run_complex, run_fold, run_reward

__version__ = "0.1.0"

__all__ = [
    "CHAIN_PAIRS",
    "DEFAULT_ESMC_DIR",
    "DEFAULT_ESMFOLD2_DIR",
    "PMHC_TCR_CHAINS",
    "RESULT_COLUMNS",
    "__version__",
    "clean_sequence",
    "complex_to_pdb_string",
    "extract_complex_metrics",
    "fold_complex",
    "fold_single",
    "load_model",
    "normalize_header",
    "open_results_csv",
    "parse_fasta",
    "read_complex_csv",
    "read_done_ids",
    "run_complex",
    "run_fold",
    "run_reward",
    "sanitize_name",
    "score_binding",
    "setup_runtime",
    "validate_model_files",
]
