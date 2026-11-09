"""Folding primitives: single chains, fixed-chain-order complexes, scoring."""
from __future__ import annotations

from .data import clean_sequence  # re-exported for convenience

# Standard 20 amino acids. Anything else is flagged (folding is still attempted).
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Fixed chain order for pMHC-TCR complexes: index i in
# StructurePredictionInput.sequences becomes chain id i in the model output,
# so pair_chains_iptm[i, j] is the ipTM between these chain pairs.
PMHC_TCR_CHAINS = [
    ("peptide", "A"),
    ("mhc", "B"),
    ("tcralpha", "C"),
    ("tcrbeta", "D"),
]
CHAIN_PAIRS = [
    (0, 1, "peptide_mhc"),
    (0, 2, "peptide_tcra"),
    (0, 3, "peptide_tcrb"),
    (1, 2, "mhc_tcra"),
    (1, 3, "mhc_tcrb"),
    (2, 3, "tcra_tcrb"),
]

RESULT_COLUMNS = (
    ["id", "iptm", "ptm", "mean_plddt"]
    + [f"iptm_{label}" for _, _, label in CHAIN_PAIRS]
    + [f"len_{name}" for name, _ in PMHC_TCR_CHAINS]
    + ["num_residues", "fold_seconds"]
)


def complex_to_pdb_string(complex_obj) -> str:
    """Render a fold result's structure as a PDB string.

    result.complex is a MolecularComplex: it has to_mmcif() and
    to_protein_complex(), but no direct to_pdb(). ProteinComplex / ProteinChain
    expose to_pdb_string(). Handle every shape defensively.
    """
    import io

    # MolecularComplex -> ProteinComplex (atom37) -> PDB string.
    if hasattr(complex_obj, "to_protein_complex"):
        return complex_obj.to_protein_complex().to_pdb_string()
    if hasattr(complex_obj, "to_pdb_string"):
        return complex_obj.to_pdb_string()
    if hasattr(complex_obj, "to_pdb"):
        buf = io.StringIO()
        complex_obj.to_pdb(buf)
        return buf.getvalue()
    raise RuntimeError(
        "Cannot serialize structure to PDB: the .complex object has none of "
        "to_protein_complex()/to_pdb_string()/to_pdb()."
    )


def fold_single(model, builder, sequence: str, *, num_loops: int = 3,
                num_sampling_steps: int = 50, num_diffusion_samples: int = 1,
                seed: int = 0) -> tuple[str, float, float]:
    """Fold one protein sequence. Returns (pdb_string, mean_pLDDT_0_1, pTM)."""
    import torch
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=clean_sequence(sequence))]
    )
    with torch.inference_mode():
        result = builder.fold(
            model,
            spi,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
        )

    pdb_str = complex_to_pdb_string(result.complex)

    plddt = float(result.plddt.mean()) if getattr(result, "plddt", None) is not None else float("nan")
    ptm = float(result.ptm) if getattr(result, "ptm", None) is not None else float("nan")
    return pdb_str, plddt, ptm


def check_chain_order(complex_obj, sequences: list[str]) -> bool:
    """Verify the fold result's per-chain token counts match the input order.

    pair_chains_iptm is indexed by internal chain id, which is assigned in
    input order; this spot-check guards that assumption. Returns True if OK.
    """
    import numpy as np

    chain_id = getattr(complex_obj, "chain_id", None)
    if chain_id is None:
        return True
    counts = [int((chain_id == c).sum()) for c in np.unique(chain_id)]
    return counts == [len(s) for s in sequences]


def fold_complex(model, builder, sequences: list[str], *, num_loops: int = 3,
                 num_sampling_steps: int = 50, num_diffusion_samples: int = 1,
                 seed: int = 0):
    """Fold one multi-chain complex (one sequence per chain, in input order).

    Returns the MolecularComplexResult (.complex, .iptm, .ptm, .plddt,
    .pair_chains_iptm, .pae).
    """
    import torch
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id=chain_id, sequence=clean_sequence(seq))
            for (_, chain_id), seq in zip(PMHC_TCR_CHAINS, sequences)
        ]
    )
    with torch.inference_mode():
        result = builder.fold(
            model,
            spi,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
        )
    return result


def extract_complex_metrics(result) -> dict[str, object]:
    """Pull iptm/ptm/mean_plddt and the 6 chain-pair iptms out of a fold result.

    mean_plddt is returned on the conventional 0-100 scale.
    """
    import numpy as np

    def _f(value) -> float:
        return float(value) if value is not None else float("nan")

    plddt = getattr(result, "plddt", None)
    mean_plddt = float(np.asarray(plddt).mean()) if plddt is not None else float("nan")
    if not np.isnan(mean_plddt) and mean_plddt <= 1.0:
        mean_plddt *= 100.0  # pLDDT is conventionally 0-100

    pair_matrix = getattr(result, "pair_chains_iptm", None)
    pair_matrix = np.asarray(pair_matrix) if pair_matrix is not None else None
    pair_values: dict[str, float] = {}
    for i, j, label in CHAIN_PAIRS:
        if pair_matrix is not None and i < pair_matrix.shape[0] and j < pair_matrix.shape[1]:
            pair_values[label] = float(pair_matrix[i, j])
        else:
            pair_values[label] = float("nan")

    return {
        "iptm": _f(getattr(result, "iptm", None)),
        "ptm": _f(getattr(result, "ptm", None)),
        "mean_plddt": mean_plddt,
        **{f"iptm_{label}": pair_values[label] for _, _, label in CHAIN_PAIRS},
    }


def score_binding(model, builder, peptide: str, mhc: str, tcra: str, tcrb: str,
                  *, num_samples: int = 3, num_loops: int = 3,
                  num_sampling_steps: int = 50, seed: int = 0) -> dict:
    """Score a pMHC-TCR quaternary for binding: mean pep-TCR ipTM over seeds.

    Folds the 4-chain complex num_samples times with independent seeds and
    returns r_bind = mean over samples of mean(iptm_peptide_tcra,
    iptm_peptide_tcrb), its std, global iptm and pLDDT.

    NOTE: num_diffusion_samples batching inside one fold call yields nearly
    identical samples (measured: std ~0.003 across in-call samples), so real
    noise reduction needs separate fold calls with distinct seeds.
    """
    import numpy as np
    import statistics as st
    import torch
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    spi = StructurePredictionInput(sequences=[
        ProteinInput(id="A", sequence=clean_sequence(peptide)),
        ProteinInput(id="B", sequence=clean_sequence(mhc)),
        ProteinInput(id="C", sequence=clean_sequence(tcra)),
        ProteinInput(id="D", sequence=clean_sequence(tcrb)),
    ])
    scores, iptms, plddt = [], [], None
    with torch.inference_mode():
        for k in range(max(1, num_samples)):
            result = builder.fold(model, spi, num_loops=num_loops,
                                  num_sampling_steps=num_sampling_steps,
                                  num_diffusion_samples=1, seed=seed + k)
            if isinstance(result, list):
                result = result[0]
            mat = np.asarray(result.pair_chains_iptm)
            scores.append((float(mat[0, 2]) + float(mat[0, 3])) / 2)
            iptms.append(float(result.iptm))
            if plddt is None:
                plddt = float(np.asarray(result.plddt).mean())
    return {"r_bind": st.mean(scores), "r_bind_std": st.pstdev(scores) if len(scores) > 1 else 0.0,
            "iptm": st.mean(iptms), "plddt": plddt * 100 if plddt <= 1 else plddt,
            "r_samples": scores}
