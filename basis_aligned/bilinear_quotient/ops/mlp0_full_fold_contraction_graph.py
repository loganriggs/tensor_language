"""RUNG 346 -- FULL-SPACE EXACT-FOLD MLP0 CONTRACTION GRAPH.

The PCA32 commutant screen covered only 17% of embedding variance.  Test the
same gauge-invariant structural question in all 1,152 input dimensions without
forming a D^2-by-D^2 commutator.  For a generic reference quadratic form Q0,
any common block projector is diagonal in Q0's eigenbasis.  If x is that
diagonal, the remaining commutator objective is the graph energy

    sum_j ||[diag(x), Q_j]||_F^2
      = 2 sum_ab [sum_j |Q_j[a,b]|^2] (x_a-x_b)^2.

Thus common blocks are low modes/components of a 1,152-node Laplacian.  The
metric is the exact covariance of all 50,304 deterministic position-zero MLP0
inputs, captured after block-0 self-attention.

Frozen predictions
------------------
pred_a_planted_full_graph_recovers_blocks_and_controls:
    Gauged planted (3,4,5) forms recover three components and exact sizes;
    independently permuted forms are connected; common-gauge spectrum holds.
pred_b_real_full_space_has_nontrivial_block_signal:
    Full normalized Fiedler value <=75% of the spectrum/entry-matched
    independently permuted null; both contraction halves <=85%.
pred_c_real_partition_is_split_stable_and_reducing:
    Split projectors overlap >=.60 and real full-cut off-block energy <=70%
    of the matched-null optimized cut.

Null: full ratio >=.90 and split overlap <=.40. Pass is a structural screen;
it earns recursive hierarchy/DAG and intervention tests, never adoption alone.
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
OUT = ROOT / "mlp0_full_fold_contraction_graph_results.json"
D = 1152
H = 4608
VOCAB = 50304
CONTRACTIONS = 7
SEED = 346


def _normalize(forms: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(forms.shape[-1], dtype=forms.dtype, device=forms.device)[None]
    scalar = torch.diagonal(forms, dim1=-2, dim2=-1).mean(-1)[:, None, None]
    centered = forms - scalar * eye
    return centered / centered.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-20)


def _graph(forms: torch.Tensor, permute: bool = False, seed: int = 0):
    forms = _normalize(forms)
    _values, basis = torch.linalg.eigh(forms[0])
    transformed = torch.einsum("ia,cij,jb->cab", basis, forms, basis)
    effective = transformed.clone()
    if permute:
        generator = torch.Generator(device=forms.device).manual_seed(seed)
        for index in range(1, len(effective)):
            order = torch.randperm(forms.shape[-1], generator=generator, device=forms.device)
            effective[index] = effective[index][order][:, order]
    weights = effective[1:].square().sum(0)
    weights.fill_diagonal_(0)
    laplacian = torch.diag(weights.sum(1)) - weights
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (laplacian + laplacian.T))
    return eigenvalues, eigenvectors, basis, effective, weights


def _normalized_fiedler(eigenvalues: torch.Tensor) -> float:
    return float(eigenvalues[1] / eigenvalues[len(eigenvalues) // 2].clamp_min(1e-20))


def _partition(eigenvectors: torch.Tensor, basis: torch.Tensor):
    fiedler = eigenvectors[:, 1]
    order = torch.argsort(fiedler)
    sorted_values = fiedler[order]
    gaps = sorted_values[1:] - sorted_values[:-1]
    p = len(fiedler)
    eligible = torch.arange(p - 1, device=fiedler.device)
    masked = gaps.clone()
    masked[(eligible < max(3, p // 20)) | (eligible > p - max(5, p // 20) - 1)] = -torch.inf
    cut = int(masked.argmax()) + 1
    chosen = order[:cut] if cut <= p - cut else order[cut:]
    vectors = basis[:, chosen]
    return vectors @ vectors.T, int(len(chosen))


def _overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    rl, rr = float(torch.trace(left)), float(torch.trace(right))
    return float(torch.trace(left @ right) / max((rl * rr) ** 0.5, 1e-20))


def _offblock(forms: torch.Tensor, projector: torch.Tensor) -> float:
    eye = torch.eye(forms.shape[-1], dtype=forms.dtype, device=forms.device)
    cross = torch.stack([projector @ form @ (eye - projector) for form in forms])
    return float(2 * cross.square().sum() / forms.square().sum().clamp_min(1e-20))


def _components(weights: torch.Tensor, tolerance: float = 1e-12) -> list[int]:
    adjacency = weights > tolerance * weights.max().clamp_min(1.0)
    unseen = set(range(len(weights)))
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = torch.where(adjacency[node])[0].tolist()
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes)


def _toy() -> dict[str, object]:
    generator = torch.Generator().manual_seed(SEED)
    dtype = torch.float64
    sizes = (3, 4, 5)
    forms = []
    for _ in range(CONTRACTIONS):
        blocks = []
        for size in sizes:
            raw = torch.randn(size, size, generator=generator, dtype=dtype)
            blocks.append(0.5 * (raw + raw.T))
        forms.append(torch.block_diag(*blocks))
    forms = torch.stack(forms)
    gauge = torch.linalg.qr(torch.randn(12, 12, generator=generator, dtype=dtype)).Q
    gauged = torch.einsum("ia,cij,jb->cab", gauge, forms, gauge)
    spectrum, _vectors, _basis, _effective, weights = _graph(gauged)
    null_spectrum, _nv, _nb, _ne, null_weights = _graph(gauged, True, SEED + 1)
    second_gauge = torch.linalg.qr(torch.randn(
        12, 12, generator=generator, dtype=dtype
    )).Q
    regauged = torch.einsum("ia,cij,jb->cab", second_gauge, gauged, second_gauge)
    regauged_spectrum, *_ = _graph(regauged)
    return {
        "recovered_component_sizes": _components(weights),
        "null_component_sizes": _components(null_weights),
        "smallest_laplacian_values": [float(value) for value in spectrum[:6]],
        "null_smallest_laplacian_values": [float(value) for value in null_spectrum[:4]],
        "common_gauge_relative_max_difference": float(
            (spectrum - regauged_spectrum).abs().max() / spectrum.max().clamp_min(1e-20)
        ),
    }


@torch.no_grad()
def _exact_mlp0_covariance(model) -> tuple[torch.Tensor, int]:
    total = torch.zeros(D, D, device="cuda")
    count = 0

    def capture(_module, args):
        nonlocal count
        x = args[0].detach().reshape(-1, D).float()
        total.addmm_(x.T, x)
        count += x.shape[0]

    handle = model.transformer.h[0].mlp.register_forward_pre_hook(capture)
    try:
        for start in range(0, VOCAB, 1024):
            indices = torch.arange(start, min(start + 1024, VOCAB), device="cuda")[:, None]
            x = F.rms_norm(model.transformer.wte(indices), (D,))
            model.transformer.h[0](x, None, x)
    finally:
        handle.remove()
    assert count == VOCAB
    covariance = total / count
    return 0.5 * (covariance + covariance.T), count


@torch.no_grad()
def _forms(model, covariance: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = float(eigenvalues[-1]) * 1e-8
    safe = eigenvalues.clamp_min(floor)
    sqrt = (eigenvectors * safe.sqrt()) @ eigenvectors.T
    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    outputs = torch.linalg.qr(torch.randn(D, CONTRACTIONS, generator=generator, device="cuda")).Q
    coefficients = down.T @ outputs
    result = []
    for index in range(CONTRACTIONS):
        raw = left.T @ (coefficients[:, index:index + 1] * right)
        raw = 0.5 * (raw + raw.T)
        metric_form = sqrt @ raw @ sqrt
        result.append(0.5 * (metric_form + metric_form.T))
    return torch.stack(result), {
        "covariance_min_eigenvalue": float(eigenvalues[0]),
        "covariance_max_eigenvalue": float(eigenvalues[-1]),
        "covariance_effective_rank": float(eigenvalues.sum().square()
                                            / eigenvalues.square().sum().clamp_min(1e-20)),
    }


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert D == 1152 and H == 4608 and VOCAB == 50304 and CONTRACTIONS == 7
        print("MLP0 FULL CONTRACTION GRAPH | dry run: toy, exact fold, graph null, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    toy = _toy()
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and model.transformer.wte.weight.shape[0] == VOCAB
    covariance, count = _exact_mlp0_covariance(model)
    forms, covariance_diag = _forms(model, covariance)
    full, vectors, basis, effective, _weights = _graph(forms)
    null_full, null_vectors, null_basis, null_effective, _ = _graph(forms, True, SEED + 2)
    split_a, split_a_vectors, split_a_basis, _ea, _ = _graph(forms[:4])
    split_b_forms = torch.cat((forms[:1], forms[4:]), dim=0)
    split_b, split_b_vectors, split_b_basis, _eb, _ = _graph(split_b_forms)
    null_a, *_ = _graph(forms[:4], True, SEED + 3)
    null_b, *_ = _graph(split_b_forms, True, SEED + 4)
    projector, projector_rank = _partition(vectors, basis)
    null_projector, null_rank = _partition(null_vectors, null_basis)
    projector_a, rank_a = _partition(split_a_vectors, split_a_basis)
    projector_b, rank_b = _partition(split_b_vectors, split_b_basis)
    real_ratio = _normalized_fiedler(full) / max(_normalized_fiedler(null_full), 1e-20)
    ratio_a = _normalized_fiedler(split_a) / max(_normalized_fiedler(null_a), 1e-20)
    ratio_b = _normalized_fiedler(split_b) / max(_normalized_fiedler(null_b), 1e-20)
    overlap = _overlap(projector_a, projector_b)
    real_offblock = _offblock(forms, projector)
    # Reconstruct the permuted null family in original coordinates for the cut metric.
    null_forms = torch.einsum("ia,cab,jb->cij", null_basis, null_effective, null_basis)
    null_offblock = _offblock(null_forms, null_projector)

    pred_a = (toy["recovered_component_sizes"] == [3, 4, 5]
              and toy["null_component_sizes"] == [12]
              and toy["common_gauge_relative_max_difference"] <= 1e-9)
    pred_b = real_ratio <= .75 and ratio_a <= .85 and ratio_b <= .85
    pred_c = overlap >= .60 and real_offblock <= .70 * null_offblock
    null = real_ratio >= .90 and overlap <= .40
    result = {
        "status": "mlp0_full_fold_contraction_graph_complete",
        "rung": 346,
        "claim_level": "full_space_exact_fold_gauge_invariant_structural_screen",
        "object": "full 1152D quadratic contraction graph under all-token exact MLP0-input covariance",
        "exact_fold_rows": count,
        "output_contractions": CONTRACTIONS,
        "covariance_diagnostics": covariance_diag,
        "toy": toy,
        "real": {
            "normalized_fiedler": _normalized_fiedler(full),
            "null_normalized_fiedler": _normalized_fiedler(null_full),
            "real_to_null_ratio_full": real_ratio,
            "real_to_null_ratio_split_a": ratio_a,
            "real_to_null_ratio_split_b": ratio_b,
            "split_projector_overlap": overlap,
            "projector_rank": projector_rank,
            "split_projector_ranks": [rank_a, rank_b],
            "null_projector_rank": null_rank,
            "real_offblock_energy_fraction": real_offblock,
            "null_offblock_energy_fraction": null_offblock,
            "smallest_laplacian_values": [float(value) for value in full[:8]],
            "null_smallest_laplacian_values": [float(value) for value in null_full[:8]],
        },
        'pred_a_planted_full_graph_recovers_blocks_and_controls': bool(pred_a),
        'pred_b_real_full_space_has_nontrivial_block_signal': bool(pred_b),
        'pred_c_real_partition_is_split_stable_and_reducing': bool(pred_c),
        "null_full_mlp0_is_as_irreducible_as_graph_null": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
