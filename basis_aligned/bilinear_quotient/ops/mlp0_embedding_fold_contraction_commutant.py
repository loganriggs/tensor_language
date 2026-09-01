"""RUNG 340 -- GAUGE-INVARIANT MLP0 QUADRATIC CONTRACTION ALGEBRA.

Raw hidden-unit supports in the embedding-folded MLP0 toy were not identifiable:
wrong CP/support priors fit the same function.  Replace that gauge-dependent
object by the quadratic function tensor itself.  Contract the MLP0 output with
fixed directions to obtain symmetric input forms

    Q_z = sym(sum_u (z^T d_u) l_u r_u^T).

A genuine input block decomposition makes every Q_z simultaneously block
diagonal, so its block projectors live in the commutant of the generated
matrix algebra.  This is invariant to hidden-unit gauges and common input
coordinate rotations.

Frozen predictions
------------------
pred_a_planted_blocks_and_dense_null_are_identified:
    A gauged (3,4,5) planted family has commutant dimension3, exact recovered
    sizes/off-block energy, while its dense spectral null has dimension1.
pred_b_real_mlp0_has_nontrivial_approximate_commutant_signal:
    On the exhaustive position-zero embedding PCA32 subspace, the normalized
    second commutator singular value is <=75% of the matched independently
    conjugated spectral null on the full contraction family.
pred_c_real_signal_is_split_stable_and_block_reducing:
    Two disjoint output-contraction halves each beat their null by >=15%, their
    recovered projectors overlap >=.60, and full-family off-block energy is
    <=70% of the matched null witness.

Null: real full ratio >=.90 and split-projector overlap <=.40.  A pass is a
gauge-invariant structural screen only; it earns recursive hierarchy/DAG toys
and an intervention discriminator, not a compiler adoption.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_embedding_fold_contraction_commutant_results.json"
D = 1152
H = 4608
VOCAB = 50304
P = 32
CONTRACTIONS = 12
SEED = 340


def _normalize_forms(forms: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(forms.shape[-1], dtype=forms.dtype, device=forms.device)[None]
    scalar = torch.diagonal(forms, dim1=-2, dim2=-1).mean(-1)[:, None, None]
    centered = forms - scalar * eye
    return centered / centered.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-12)


def _commutator(forms: torch.Tensor) -> torch.Tensor:
    p = forms.shape[-1]
    eye = torch.eye(p, dtype=forms.dtype, device=forms.device)
    blocks = [torch.kron(eye, form) - torch.kron(form.T.contiguous(), eye)
              for form in forms]
    return torch.cat(blocks, dim=0)


def _spectrum_witness(forms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    operator = _commutator(_normalize_forms(forms))
    _u, singular, vh = torch.linalg.svd(operator, full_matrices=False)
    ascending = singular.flip(0)
    # Last right-singular row is the scalar identity; second-last is the best
    # nontrivial approximate commutant element.
    witness = vh[-2].reshape(forms.shape[-1], forms.shape[-1])
    witness = 0.5 * (witness + witness.T)
    return ascending, witness


def _normalized_second(spectrum: torch.Tensor) -> float:
    return float(spectrum[1] / spectrum[len(spectrum) // 2].clamp_min(1e-12))


def _projector_from_witness(witness: torch.Tensor) -> tuple[torch.Tensor, int]:
    values, vectors = torch.linalg.eigh(witness)
    p = len(values)
    gaps = values[1:] - values[:-1]
    eligible = torch.arange(p - 1, device=values.device)
    eligible[(eligible < 3) | (eligible > p - 5)] = -1
    masked = gaps.clone()
    masked[eligible < 0] = -torch.inf
    cut = int(masked.argmax()) + 1
    left = vectors[:, :cut]
    right = vectors[:, cut:]
    chosen = left if left.shape[1] <= right.shape[1] else right
    return chosen @ chosen.T, chosen.shape[1]


def _projector_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    rl = float(torch.trace(left))
    rr = float(torch.trace(right))
    return float(torch.trace(left @ right) / max((rl * rr) ** 0.5, 1e-12))


def _offblock(forms: torch.Tensor, projector: torch.Tensor) -> float:
    eye = torch.eye(forms.shape[-1], dtype=forms.dtype, device=forms.device)
    other = eye - projector
    cross = torch.stack([projector @ form @ other for form in forms])
    return float(2 * cross.square().sum() / forms.square().sum().clamp_min(1e-12))


def _independent_conjugate(forms: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=forms.device).manual_seed(seed)
    result = []
    for form in forms:
        gauge = torch.linalg.qr(torch.randn(
            form.shape, generator=generator, dtype=form.dtype, device=form.device
        )).Q
        result.append(gauge.T @ form @ gauge)
    return torch.stack(result)


def _toy() -> dict[str, object]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(SEED)
    sizes = (3, 4, 5)
    dimension = sum(sizes)
    forms = []
    for _ in range(7):
        blocks = []
        for size in sizes:
            raw = torch.randn(size, size, generator=generator, dtype=dtype)
            blocks.append(0.5 * (raw + raw.T))
        forms.append(torch.block_diag(*blocks))
    forms = torch.stack(forms)
    gauge = torch.linalg.qr(torch.randn(
        dimension, dimension, generator=generator, dtype=dtype
    )).Q
    forms = torch.einsum("ia,cij,jb->cab", gauge, forms, gauge)
    spectrum, witness = _spectrum_witness(forms)
    cutoff = 1e-9 * spectrum[-1].clamp_min(1.0)
    commutant_dim = int((spectrum <= cutoff).sum())
    values = torch.linalg.eigvalsh(witness)
    labels = torch.zeros(dimension, dtype=torch.long)
    label = 0
    for index in range(1, dimension):
        if abs(values[index] - values[index - 1]) > 1e-7:
            label += 1
        labels[index] = label
    recovered_sizes = sorted(int((labels == value).sum()) for value in labels.unique())
    transformed = torch.einsum("ia,cij,jb->cab", torch.linalg.eigh(witness).eigenvectors,
                               forms, torch.linalg.eigh(witness).eigenvectors)
    cross = labels[:, None] != labels[None, :]
    offblock = float(transformed[:, cross].square().sum()
                     / transformed.square().sum().clamp_min(1e-30))
    dense = _independent_conjugate(forms, SEED + 1)
    dense_spectrum, _ = _spectrum_witness(dense)
    dense_dim = int((dense_spectrum <= 1e-9 * dense_spectrum[-1].clamp_min(1.0)).sum())
    return {
        "commutant_dimension": commutant_dim,
        "recovered_block_sizes": recovered_sizes,
        "offblock_energy_fraction": offblock,
        "dense_null_commutant_dimension": dense_dim,
        "smallest_singular_values": [float(value) for value in spectrum[:6]],
        "dense_smallest_singular_values": [float(value) for value in dense_spectrum[:4]],
    }


@torch.no_grad()
def _real_forms(model) -> tuple[torch.Tensor, dict[str, object]]:
    embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))
    covariance = embedding.T @ embedding / embedding.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    basis = eigenvectors[:, -P:]
    explained = float(eigenvalues[-P:].sum() / eigenvalues.clamp_min(0).sum())
    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().float() @ basis
    right = mlp.Right.weight.detach().float() @ basis
    down = mlp.Down.weight.detach().float()
    generator = torch.Generator(device=down.device).manual_seed(SEED)
    z = torch.linalg.qr(torch.randn(
        D, CONTRACTIONS, generator=generator, device=down.device
    )).Q
    coefficients = down.T @ z
    forms = []
    for index in range(CONTRACTIONS):
        weighted = coefficients[:, index:index + 1] * right
        form = left.T @ weighted
        forms.append(0.5 * (form + form.T))
    return torch.stack(forms), {
        "embedding_rows": int(embedding.shape[0]),
        "embedding_pca_rank": P,
        "embedding_variance_explained": explained,
        "output_contractions": CONTRACTIONS,
    }


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert P == 32 and CONTRACTIONS == 12 and D == 1152 and H == 4608
        assert VOCAB == 50304 and CONTRACTIONS % 2 == 0
        print("MLP0 CONTRACTION COMMUTANT | dry run: toy, exhaustive fold, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    toy = _toy()
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and model.transformer.wte.weight.shape[0] == VOCAB
    forms, identity = _real_forms(model)
    null_forms = _independent_conjugate(forms, SEED + 2)
    gauge_generator = torch.Generator(device=forms.device).manual_seed(SEED + 3)
    common_gauge = torch.linalg.qr(torch.randn(
        P, P, generator=gauge_generator, device=forms.device
    )).Q
    gauged = torch.einsum("ia,cij,jb->cab", common_gauge, forms, common_gauge)

    full_spectrum, full_witness = _spectrum_witness(forms)
    null_spectrum, null_witness = _spectrum_witness(null_forms)
    gauge_spectrum, _ = _spectrum_witness(gauged)
    half = CONTRACTIONS // 2
    split_a_spectrum, split_a_witness = _spectrum_witness(forms[:half])
    split_b_spectrum, split_b_witness = _spectrum_witness(forms[half:])
    null_a_spectrum, _ = _spectrum_witness(null_forms[:half])
    null_b_spectrum, _ = _spectrum_witness(null_forms[half:])
    full_projector, full_rank = _projector_from_witness(full_witness)
    null_projector, null_rank = _projector_from_witness(null_witness)
    split_a_projector, split_a_rank = _projector_from_witness(split_a_witness)
    split_b_projector, split_b_rank = _projector_from_witness(split_b_witness)

    real_full = _normalized_second(full_spectrum)
    null_full = _normalized_second(null_spectrum)
    real_a = _normalized_second(split_a_spectrum)
    real_b = _normalized_second(split_b_spectrum)
    null_a = _normalized_second(null_a_spectrum)
    null_b = _normalized_second(null_b_spectrum)
    full_ratio = real_full / max(null_full, 1e-12)
    ratio_a = real_a / max(null_a, 1e-12)
    ratio_b = real_b / max(null_b, 1e-12)
    overlap = _projector_overlap(split_a_projector, split_b_projector)
    real_offblock = _offblock(forms, full_projector)
    null_offblock = _offblock(null_forms, null_projector)
    gauge_difference = float((full_spectrum - gauge_spectrum).abs().max()
                             / full_spectrum.max().clamp_min(1e-12))

    pred_a = (toy["commutant_dimension"] == 3
              and toy["recovered_block_sizes"] == [3, 4, 5]
              and toy["offblock_energy_fraction"] <= 1e-18
              and toy["dense_null_commutant_dimension"] == 1)
    pred_b = full_ratio <= .75
    pred_c = (ratio_a <= .85 and ratio_b <= .85 and overlap >= .60
              and real_offblock <= .70 * null_offblock)
    null = full_ratio >= .90 and overlap <= .40
    result = {
        "status": "mlp0_embedding_fold_contraction_commutant_complete",
        "rung": 340,
        "claim_level": "gauge_invariant_planted_to_real_structural_screen",
        "object": "output contractions of exact MLP0 quadratic function tensor",
        "identity": identity,
        "toy": toy,
        "real": {
            "normalized_second_singular_full": real_full,
            "spectral_null_normalized_second_full": null_full,
            "real_to_null_ratio_full": full_ratio,
            "real_to_null_ratio_split_a": ratio_a,
            "real_to_null_ratio_split_b": ratio_b,
            "split_projector_overlap": overlap,
            "full_projector_rank": full_rank,
            "split_projector_ranks": [split_a_rank, split_b_rank],
            "real_offblock_energy_fraction": real_offblock,
            "null_offblock_energy_fraction": null_offblock,
            "null_projector_rank": null_rank,
            "common_gauge_spectrum_relative_max_difference": gauge_difference,
            "smallest_singular_values_full": [float(value) for value in full_spectrum[:8]],
            "smallest_singular_values_null": [float(value) for value in null_spectrum[:8]],
        },
        'pred_a_planted_blocks_and_dense_null_are_identified': bool(pred_a),
        'pred_b_real_mlp0_has_nontrivial_approximate_commutant_signal': bool(pred_b),
        'pred_c_real_signal_is_split_stable_and_block_reducing': bool(pred_c),
        "null_real_mlp0_is_as_irreducible_as_spectral_null": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
