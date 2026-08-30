"""Known-answer toy for sparse routed interaction-tensor decompositions.

This is deliberately small enough to inspect and run on CPU.  It verifies the algebra
needed before applying the proposed tensor-similarity objective to a bilin18 MLP:

* folding a linear combination of native bilinear gates into a symmetric quadratic;
* the Gaussian functional metric, including amplitude rather than cosine alone;
* the CP permutation/rescaling/L-R-swap gauges;
* a four-arm (Mobius) interaction difference which cancels both main effects;
* the fact that an identical atom bank with a wrong router is a different function; and
* gradient recovery of a planted low-rank interaction using tensor distance plus a
  teacher-distribution cross-entropy term.

It is a correctness toy, not evidence that the real model has the planted structure.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "toy_sparse_routed_interaction_tensor_receipt.json"


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetrize the two repeated input legs of a quadratic tensor."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def fold_gate_encoder(L: torch.Tensor, R: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Fold scores E[(Lx)*(Rx)] into Q with score_a(x)=x^T Q_a x.

    L,R: [native_gate, input]; E: [atom, native_gate]; result: [atom,input,input].
    """
    raw = torch.einsum("ah,hi,hj->aij", E, L, R)
    return symmetrize(raw)


def block_tensor(decoder: torch.Tensor, quadratics: torch.Tensor) -> torch.Tensor:
    """Return T[o,i,j] = sum_a decoder[o,a] * Q[a,i,j]."""
    return torch.einsum("oa,aij->oij", decoder, symmetrize(quadratics))


def tensor_forward(tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate y_o=x^T T_o x for a batch x[n,i]."""
    return torch.einsum("ni,oij,nj->no", x, symmetrize(tensor), x)


def gaussian_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """E[y_left(x).y_right(x)] for x~N(0,I), by Isserlis' theorem."""
    left, right = symmetrize(left), symmetrize(right)
    traces = torch.einsum("oii->o", left) @ torch.einsum("oii->o", right)
    frobenius = 2.0 * torch.sum(left * right)
    return traces + frobenius


def tensor_cosine(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Scale-free Gaussian functional cosine; useful for direction, not replacement."""
    cross = gaussian_inner(target, candidate)
    norms = gaussian_inner(target, target) * gaussian_inner(candidate, candidate)
    return cross / norms.clamp_min(1e-30).sqrt()


def relative_tensor_error(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """E||y-yhat||^2/E||y||^2 under N(0,I), including norm/amplitude error."""
    delta = target - candidate
    return gaussian_inner(delta, delta) / gaussian_inner(target, target).clamp_min(1e-30)


def optimal_scalar(target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Least-squares scalar for a candidate direction in the Gaussian metric."""
    return gaussian_inner(target, candidate) / gaussian_inner(candidate, candidate).clamp_min(1e-30)


def mobius_interaction(
    both: torch.Tensor,
    left_only: torch.Tensor,
    right_only: torch.Tensor,
    neither: torch.Tensor,
) -> torch.Tensor:
    """Second finite difference: both-left_only-right_only+neither."""
    return both - left_only - right_only + neither


def routed_forward(
    decoder_atoms: torch.Tensor,
    quadratics: torch.Tensor,
    x: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Evaluate selected quadratic blocks.

    decoder_atoms: [atom, output], quadratics: [atom,input,input],
    support: Boolean [sample,atom].  No claim is made that this discrete node is one
    global polynomial; each fixed-support piece is quadratic.
    """
    scores = torch.einsum("ni,aij,nj->na", x, symmetrize(quadratics), x)
    return torch.einsum("na,ao->no", scores * support.to(scores.dtype), decoder_atoms)


def _planted_interaction(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    """Return a small rank-2 CP tensor and its factors."""
    D = torch.tensor([[0.55, -0.35], [-0.30, 0.45], [0.20, 0.25]], dtype=dtype)
    L = torch.tensor([[0.50, -0.20, 0.15, 0.30], [-0.25, 0.40, 0.20, -0.10]], dtype=dtype)
    R = torch.tensor([[-0.10, 0.35, 0.25, 0.15], [0.30, 0.10, -0.35, 0.20]], dtype=dtype)
    Q = symmetrize(torch.einsum("hi,hj->hij", L, R))
    return block_tensor(D, Q), D, L, R


@dataclass(frozen=True)
class OptimizationReceipt:
    seed: int
    steps: int
    initial_tensor_error: float
    final_tensor_error: float
    initial_teacher_ce: float
    final_teacher_ce: float
    final_tensor_cosine: float
    passed: bool


def run_planted_optimization(seed: int = 0, steps: int = 1200) -> OptimizationReceipt:
    """Recover a rank-2 planted interaction with Adam under a compatible hybrid loss."""
    torch.set_num_threads(1)
    dtype = torch.float64
    target, _, _, _ = _planted_interaction(dtype)
    generator = torch.Generator().manual_seed(seed + 101)
    x = torch.randn(768, target.shape[-1], generator=generator, dtype=dtype)
    teacher_logits = tensor_forward(target, x).detach()
    teacher_prob = teacher_logits.softmax(dim=-1)

    rank = 2
    D = (0.15 * torch.randn(target.shape[0], rank, generator=generator, dtype=dtype)).requires_grad_()
    L = (0.15 * torch.randn(rank, target.shape[-1], generator=generator, dtype=dtype)).requires_grad_()
    R = (0.15 * torch.randn(rank, target.shape[-1], generator=generator, dtype=dtype)).requires_grad_()
    optimizer = torch.optim.Adam([D, L, R], lr=2e-2)

    def losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = block_tensor(D, torch.einsum("hi,hj->hij", L, R))
        tensor_loss = relative_tensor_error(target, candidate)
        student_logits = tensor_forward(candidate, x)
        # Cross-entropy against the original function's full output distribution.  The
        # omitted constant teacher entropy does not affect gradients.
        teacher_ce = -(teacher_prob * F.log_softmax(student_logits, dim=-1)).sum(-1).mean()
        return tensor_loss, teacher_ce, candidate

    with torch.no_grad():
        initial_tensor, initial_ce, _ = losses()
    for _ in range(steps):
        tensor_loss, teacher_ce, _ = losses()
        loss = tensor_loss + 0.10 * teacher_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_tensor, final_ce, candidate = losses()
        cosine = tensor_cosine(target, candidate)

    # Both terms must improve; the tensor term has the stricter known-answer threshold.
    passed = (
        final_tensor < 1e-5
        and final_tensor < initial_tensor * 1e-4
        and final_ce < initial_ce
        and cosine > 0.99999
    )
    return OptimizationReceipt(
        seed=seed,
        steps=steps,
        initial_tensor_error=float(initial_tensor),
        final_tensor_error=float(final_tensor),
        initial_teacher_ce=float(initial_ce),
        final_teacher_ce=float(final_ce),
        final_tensor_cosine=float(cosine),
        passed=bool(passed),
    )


def algebraic_checks() -> dict[str, bool | float]:
    """Run deterministic known-answer checks used by both pytest and the receipt."""
    dtype = torch.float64
    generator = torch.Generator().manual_seed(7)

    # Exact fold from native gates to atom quadratic forms.
    L = torch.randn(5, 4, generator=generator, dtype=dtype)
    R = torch.randn(5, 4, generator=generator, dtype=dtype)
    E = torch.randn(3, 5, generator=generator, dtype=dtype)
    x = torch.randn(19, 4, generator=generator, dtype=dtype)
    Q = fold_gate_encoder(L, R, E)
    native_scores = ((x @ L.T) * (x @ R.T)) @ E.T
    folded_scores = torch.einsum("ni,aij,nj->na", x, Q, x)

    target, D, Lp, Rp = _planted_interaction(dtype)
    Qp = symmetrize(torch.einsum("hi,hj->hij", Lp, Rp))

    # True CP gauges: atom permutation, reciprocal scaling, and input-leg swap.
    permutation = torch.tensor([1, 0])
    a = torch.tensor([1.7, 0.6], dtype=dtype)
    b = torch.tensor([0.8, 1.9], dtype=dtype)
    gauged = block_tensor(
        (D / (a * b)[None, :])[:, permutation],
        torch.einsum(
            "hi,hj->hij",
            (a[:, None] * Lp)[permutation],
            (b[:, None] * Rp)[permutation],
        ),
    )
    swapped = block_tensor(D, torch.einsum("hi,hj->hij", Rp, Lp))

    # Cosine is blind to amplitude; relative distance is not.  A free scalar fixes it.
    scaled = 2.5 * target
    scale_fit = optimal_scalar(target, scaled)

    # Main effects can be arbitrarily complex; the Mobius difference must isolate I.
    base = torch.randn(target.shape, generator=generator, dtype=dtype)
    main_left = torch.randn(target.shape, generator=generator, dtype=dtype)
    main_right = torch.randn(target.shape, generator=generator, dtype=dtype)
    neither = base
    left_only = base + main_left
    right_only = base + main_right
    both = base + main_left + main_right + target
    recovered = mobius_interaction(both, left_only, right_only, neither)
    null = mobius_interaction(base + main_left + main_right, left_only, right_only, neither)

    # Same atoms, wrong route: bank comparison cannot detect this functional failure.
    decoder_atoms = torch.tensor([[1.0, 0.2], [-0.3, 0.8]], dtype=dtype)
    route_q = Qp
    xr = torch.randn(256, 4, generator=generator, dtype=dtype)
    true_support = torch.stack([xr[:, 0] >= 0, xr[:, 0] < 0], dim=1)
    wrong_support = true_support.flip(1)
    routed_true = routed_forward(decoder_atoms, route_q, xr, true_support)
    routed_wrong = routed_forward(decoder_atoms, route_q, xr, wrong_support)
    wrong_router_relative_mse = float(
        ((routed_true - routed_wrong) ** 2).sum()
        / (routed_true**2).sum().clamp_min(1e-30)
    )

    return {
        "fold_exact": bool(torch.allclose(native_scores, folded_scores, atol=1e-11, rtol=1e-11)),
        "permutation_rescaling_gauge_exact": bool(torch.allclose(target, gauged, atol=1e-11, rtol=1e-11)),
        "input_leg_swap_exact": bool(torch.allclose(target, swapped, atol=1e-11, rtol=1e-11)),
        "scaled_cosine": float(tensor_cosine(target, scaled)),
        "scaled_relative_error": float(relative_tensor_error(target, scaled)),
        "optimal_scale": float(scale_fit),
        "scale_corrected_error": float(relative_tensor_error(target, scale_fit * scaled)),
        "mobius_recovery_error": float((recovered - target).abs().max()),
        "null_interaction_norm": float(null.norm()),
        "identical_bank_similarity": float(tensor_cosine(target, target)),
        "wrong_router_relative_mse": wrong_router_relative_mse,
    }


def main() -> None:
    started = time.monotonic()
    algebra = algebraic_checks()
    optimization = [run_planted_optimization(seed) for seed in range(3)]
    algebra_passed = (
        algebra["fold_exact"]
        and algebra["permutation_rescaling_gauge_exact"]
        and algebra["input_leg_swap_exact"]
        and math.isclose(algebra["scaled_cosine"], 1.0, abs_tol=1e-12)
        and algebra["scaled_relative_error"] > 1.0
        and algebra["scale_corrected_error"] < 1e-20
        and algebra["mobius_recovery_error"] < 1e-12
        and algebra["null_interaction_norm"] < 1e-12
        and algebra["wrong_router_relative_mse"] > 0.1
    )
    payload = {
        "schema": "toy_sparse_routed_interaction_tensor_v1",
        "purpose": "known-answer code validation only; not real-model evidence",
        "algebraic_checks": algebra,
        "optimization_checks": [asdict(item) for item in optimization],
        "all_passed": bool(algebra_passed and all(item.passed for item in optimization)),
        "runtime_s": time.monotonic() - started,
    }
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
