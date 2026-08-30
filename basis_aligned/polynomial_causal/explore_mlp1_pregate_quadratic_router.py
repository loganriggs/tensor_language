#!/usr/bin/env python3
"""Discovery screen for a low-rank pre-gate router for sparse MLP1.

The exact P512 sparse program scores E[(Lx)*(Rx)], which still computes every native
bilinear gate.  This runner folds each encoder row into a symmetric, generally
indefinite quadratic Q_a and approximates Q_a by its largest-magnitude signed
eigenmodes.  It reuses the already-opened SELECT role, never opens FINAL, and measures
the resulting executable replacement in the full model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for root in (ROOT, HERE, BQ):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bilin18_observed_model_facade as facade
import mlp1_sparse_down_program_v1 as sparse
import run_mlp1_sparse_c512_continue_factorial_v1_fit as base


BUNDLE = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_bundle.pt"
P512_RESULT = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_result.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
OUTPUT = HERE / "mlp1_pregate_quadratic_router_v2_discovery.json"

RANKS = (1, 2, 4, 8)
MAX_RANK = max(RANKS)
OVERSAMPLE = 8
POWER_ITERS = 3
ATOM_BATCH = 16
STATE_CHUNK = 256
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
P = 512
K = 32
D = 1152
H = 4608


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def router_price(rank: int) -> dict[str, int | float]:
    router_vectors = P * rank * D
    signed_eigenvalues = P * rank
    decoder = D * P
    intercept = D
    bias_folded_stored = router_vectors + signed_eigenvalues + decoder + intercept
    executed_artifact_stored = bias_folded_stored + D
    native_full = 3 * D * H + D
    multiplies = router_vectors + signed_eigenvalues + K * D
    return {
        "rank_per_quadratic": rank,
        "router_vector_reals": router_vectors,
        "signed_eigenvalue_reals": signed_eigenvalues,
        "decoder_reals": decoder,
        "intercept_reals": intercept,
        "bias_folded_stored_reals": bias_folded_stored,
        "executed_artifact_stored_reals": executed_artifact_stored,
        "native_down_bias_reals_in_executed_artifact": D,
        "bias_folding_is_exact": True,
        "native_full_mlp_reals": native_full,
        "full_mlp_storage_saved_reals": native_full - bias_folded_stored,
        "full_mlp_storage_saved_fraction": (
            native_full - bias_folded_stored
        ) / native_full,
        "router_score_and_decode_multiplies_per_token": multiplies,
        "full_mlp_dense_map_multiply_saved_fraction": (
            3 * D * H - multiplies
        ) / (3 * D * H),
        "topk_comparisons_indices_squares_and_additions_charged_separately": True,
    }


def _randomized_signed_factors(
    matvec: Callable[[torch.Tensor], torch.Tensor], batch: int, dimension: int,
    max_rank: int, device: torch.device, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest-magnitude signed eigenmodes of a batch of symmetric operators."""

    width = max_rank + OVERSAMPLE
    generator = torch.Generator(device=device).manual_seed(seed)
    probes = torch.randn(batch, dimension, width, device=device, generator=generator)
    basis = torch.linalg.qr(matvec(probes), mode="reduced").Q
    for _ in range(POWER_ITERS):
        basis = torch.linalg.qr(matvec(matvec(basis)), mode="reduced").Q
    projected = basis.transpose(1, 2) @ matvec(basis)
    projected = 0.5 * (projected + projected.transpose(1, 2))
    values, vectors = torch.linalg.eigh(projected)
    order = values.abs().argsort(dim=1, descending=True)[:, :max_rank]
    values = values.gather(1, order)
    vectors = vectors.gather(2, order[:, None, :].expand(-1, width, -1))
    factors = basis @ vectors
    return factors, values


def explicit_randomized_signed_factors(
    matrices: torch.Tensor, max_rank: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public explicit-matrix wrapper used by the known-answer tests."""

    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("matrices must be [batch,dimension,dimension]")
    return _randomized_signed_factors(
        lambda vectors: matrices @ vectors,
        matrices.shape[0], matrices.shape[1], max_rank, matrices.device, seed,
    )


def implicit_quadratic_factors(
    left: torch.Tensor, right: torch.Tensor, encoder: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Factor Q_a=.5(L' diag(e_a) R + R' diag(e_a) L) without materializing Q."""

    factors, values = [], []
    maximum_relative_residual = 0.0
    maximum_orthogonality_error = 0.0
    for start in range(0, len(encoder), ATOM_BATCH):
        weights = encoder[start:start + ATOM_BATCH]

        def matvec(vectors: torch.Tensor) -> torch.Tensor:
            rv = torch.matmul(right, vectors)
            lv = torch.matmul(left, vectors)
            first = torch.matmul(left.T, weights[:, :, None] * rv)
            second = torch.matmul(right.T, weights[:, :, None] * lv)
            return 0.5 * (first + second)

        u, lam = _randomized_signed_factors(
            matvec, len(weights), D, MAX_RANK, left.device, 31_000 + start,
        )
        qu = matvec(u)
        residual = (qu - u * lam[:, None, :]).norm(dim=(1, 2)) \
            / qu.norm(dim=(1, 2)).clamp_min(1e-30)
        gram = u.transpose(1, 2) @ u
        identity = torch.eye(MAX_RANK, device=u.device)[None]
        orthogonality = (gram - identity).abs().amax(dim=(1, 2))
        maximum_relative_residual = max(maximum_relative_residual, float(residual.max()))
        maximum_orthogonality_error = max(
            maximum_orthogonality_error, float(orthogonality.max()),
        )
        factors.append(u.cpu()); values.append(lam.cpu())
        print(f"factored atoms {start}:{start + len(weights)}", flush=True)
    return torch.cat(factors), torch.cat(values), {
        "maximum_relative_ritz_residual": maximum_relative_residual,
        "maximum_orthogonality_error": maximum_orthogonality_error,
    }


def quadratic_scores(
    states: torch.Tensor, factors: torch.Tensor, values: torch.Tensor, rank: int,
) -> torch.Tensor:
    """Evaluate all signed low-rank x'Q_a x scores."""

    if factors.ndim != 3 or factors.shape[1] != states.shape[1] \
            or values.shape != (factors.shape[0], factors.shape[2]):
        raise ValueError("quadratic factor/state/value shapes disagree")
    projections = torch.einsum("nd,adr->nar", states.float(), factors[:, :, :rank])
    return (projections.square() * values[:, :rank]).sum(-1)


@torch.no_grad()
def capture_states(
    model, rows: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    captured = []
    calls = {"forwards": 0, "attention": 0, "site1": 0, "other_mlp": 0}
    for start in range(0, len(rows), DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)

        def attention(event: facade.AttentionEvent):
            calls["attention"] += 1
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            if event.site == 1:
                captured.append(event.state[:, SCORING].detach().float().cpu().reshape(-1, D))
                calls["site1"] += 1
            else:
                calls["other_mlp"] += 1
            return event.block.mlp(event.state)

        facade.forward_with_dispatch(model, tokens, attention, mlp)
        calls["forwards"] += 1
    expected = len(rows) // DOCUMENT_BATCH
    if calls != {
        "forwards": expected, "attention": 18 * expected,
        "site1": expected, "other_mlp": 17 * expected,
    }:
        raise RuntimeError(f"state capture census changed: {calls}")
    result = torch.cat(captured)
    if tuple(result.shape) != (len(rows) * 192, D):
        raise RuntimeError("captured state shape changed")
    return result, calls


@torch.no_grad()
def route_metrics(
    states: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
    encoder: torch.Tensor, factors: torch.Tensor, values: torch.Tensor,
    decoder: torch.Tensor, intercept: torch.Tensor, device: torch.device,
) -> dict[str, dict[str, float]]:
    accum = {rank: {
        "sse": 0.0, "den": 0.0, "intersection": 0, "top1": 0, "rows": 0,
        "positive_exact": 0, "positive_approximate": 0, "positive_intersection": 0,
        "code_sse": 0.0, "code_den": 0.0, "write_sse": 0.0, "write_den": 0.0,
    }
             for rank in RANKS}
    for start in range(0, len(states), STATE_CHUNK):
        x = states[start:start + STATE_CHUNK].to(device)
        exact = ((x @ left.T) * (x @ right.T)) @ encoder.T
        exact_values, exact_top = exact.topk(K, dim=1)
        exact_top1 = exact.argmax(1)
        exact_code = sparse.topk_relu(exact)
        exact_write = exact_code @ decoder.T + intercept
        for rank in RANKS:
            approximate = quadratic_scores(x, factors, values, rank)
            approximate_values, top = approximate.topk(K, dim=1)
            intersection = (top[:, :, None] == exact_top[:, None, :]).any(2).sum()
            exact_positive = exact_values > 0
            approximate_positive = approximate_values > 0
            positive_matches = (
                (top[:, :, None] == exact_top[:, None, :])
                & approximate_positive[:, :, None]
                & exact_positive[:, None, :]
            ).any(2).sum()
            approximate_code = sparse.topk_relu(approximate)
            approximate_write = approximate_code @ decoder.T + intercept
            item = accum[rank]
            item["sse"] += float((approximate.double() - exact.double()).square().sum())
            item["den"] += float(exact.double().square().sum())
            item["intersection"] += int(intersection)
            item["top1"] += int((approximate.argmax(1) == exact_top1).sum())
            item["positive_exact"] += int(exact_positive.sum())
            item["positive_approximate"] += int(approximate_positive.sum())
            item["positive_intersection"] += int(positive_matches)
            item["code_sse"] += float((approximate_code.double() - exact_code.double()).square().sum())
            item["code_den"] += float(exact_code.double().square().sum())
            item["write_sse"] += float((approximate_write.double() - exact_write.double()).square().sum())
            item["write_den"] += float(exact_write.double().square().sum())
            item["rows"] += len(x)
    return {
        str(rank): {
            "relative_score_mse": item["sse"] / max(item["den"], 1e-30),
            "topk_recall": item["intersection"] / (item["rows"] * K),
            "top1_agreement": item["top1"] / item["rows"],
            "positive_topk_recall": item["positive_intersection"] / max(item["positive_exact"], 1),
            "positive_topk_precision": item["positive_intersection"] / max(item["positive_approximate"], 1),
            "positive_topk_jaccard": item["positive_intersection"] / max(
                item["positive_exact"] + item["positive_approximate"]
                - item["positive_intersection"], 1,
            ),
            "relative_routed_code_mse": item["code_sse"] / max(item["code_den"], 1e-30),
            "relative_frozen_decoder_write_mse": item["write_sse"] / max(item["write_den"], 1e-30),
        }
        for rank, item in accum.items()
    }


@torch.no_grad()
def physical_ce(
    model, rows: torch.Tensor, exact_program: sparse.SparseDownProgram,
    factors: torch.Tensor, values: torch.Tensor, decoder: torch.Tensor,
    intercept: torch.Tensor, device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    arms = ("NATIVE", "ZERO", "P512_EXACT", *(f"Q_RANK_{rank}" for rank in RANKS))
    sums = {arm: 0.0 for arm in arms}; counts = {arm: 0 for arm in arms}
    calls = {arm: {"forwards": 0, "attention": 0, "site1_native": 0,
                   "site1_replacement": 0, "other_mlp": 0,
                   "native_left": 0, "native_right": 0, "native_down": 0}
             for arm in arms}
    factors = factors.to(device); values = values.to(device)
    current_arm = {"name": None}

    def count_submodule(name: str):
        def hook(_module, _inputs, _output):
            arm = current_arm["name"]
            if arm is None:
                raise RuntimeError("native MLP1 submodule called outside a scored arm")
            calls[arm][name] += 1
        return hook

    hooks = [
        model.transformer.h[1].mlp.Left.register_forward_hook(count_submodule("native_left")),
        model.transformer.h[1].mlp.Right.register_forward_hook(count_submodule("native_right")),
        model.transformer.h[1].mlp.Down.register_forward_hook(count_submodule("native_down")),
    ]
    try:
        for start in range(0, len(rows), DOCUMENT_BATCH):
            batch = rows[start:start + DOCUMENT_BATCH]
            tokens = batch[:, :-1].to(device); targets = batch[:, 1:].to(device)
            for arm in arms:
                current_arm["name"] = arm
                def attention(event: facade.AttentionEvent, arm=arm):
                    calls[arm]["attention"] += 1
                    return event.block.attn(event.state, event.first_value)

                def mlp(event: facade.EarlyMLPEvent, arm=arm):
                    if event.site != 1:
                        calls[arm]["other_mlp"] += 1
                        return event.block.mlp(event.state)
                    if arm == "NATIVE":
                        calls[arm]["site1_native"] += 1
                        return event.block.mlp(event.state)
                    calls[arm]["site1_replacement"] += 1
                    if arm == "ZERO":
                        action = torch.zeros_like(event.state)
                    elif arm == "P512_EXACT":
                        gate = event.block.mlp.Left(event.state) * event.block.mlp.Right(event.state)
                        action = exact_program(gate)
                    else:
                        rank = int(arm.rsplit("_", 1)[1])
                        flat = event.state.float().reshape(-1, D)
                        scores = quadratic_scores(flat, factors, values, rank)
                        action = (
                            sparse.topk_relu(scores) @ decoder.T + intercept
                        ).reshape_as(event.state).to(event.state.dtype)
                    return action + event.block.mlp.Down_bias

                logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
                loss = F.cross_entropy(
                    logits[:, SCORING].reshape(-1, logits.shape[-1]),
                    targets[:, SCORING].reshape(-1), reduction="sum",
                )
                sums[arm] += float(loss); counts[arm] += targets[:, SCORING].numel()
                calls[arm]["forwards"] += 1
                current_arm["name"] = None
    finally:
        current_arm["name"] = None
        for hook in hooks:
            hook.remove()
    expected = len(rows) // DOCUMENT_BATCH
    for arm in arms:
        wanted = {
            "forwards": expected, "attention": 18 * expected,
            "site1_native": expected if arm == "NATIVE" else 0,
            "site1_replacement": 0 if arm == "NATIVE" else expected,
            "other_mlp": 17 * expected,
            "native_left": expected if arm in ("NATIVE", "P512_EXACT") else 0,
            "native_right": expected if arm in ("NATIVE", "P512_EXACT") else 0,
            "native_down": expected if arm == "NATIVE" else 0,
        }
        if calls[arm] != wanted:
            raise RuntimeError(f"physical CE census changed for {arm}: {calls[arm]}")
    ce = {arm: sums[arm] / counts[arm] for arm in arms}
    benefit = ce["ZERO"] - ce["NATIVE"]
    if not benefit > 0:
        raise RuntimeError("MLP1 deletion denominator is not positive")
    for arm in arms:
        if arm not in ("NATIVE", "ZERO"):
            ce[f"{arm}_RECOVERY"] = (ce["ZERO"] - ce[arm]) / benefit
    return ce, calls


def main() -> None:
    started = time.time()
    receipt = json.loads(ROWS_RECEIPT.read_text())
    select_entry = receipt["entries"]["SELECT"]
    rows = base.load_role(select_entry)
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    state = sparse.validate_state(bundle["program"])
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    left = model.transformer.h[1].mlp.Left.weight.detach().float()
    right = model.transformer.h[1].mlp.Right.weight.detach().float()
    encoder = state["encoder"].to(device)
    decoder = state["decoder"].to(device)
    intercept = state["intercept"].to(device)
    factors, values, factorization_diagnostics = implicit_quadratic_factors(
        left, right, encoder,
    )
    exact_program = sparse.SparseDownProgram(state, device).eval()
    states, capture_calls = capture_states(model, rows, device)
    metrics = route_metrics(
        states, left, right, encoder, factors.to(device), values.to(device),
        decoder, intercept, device,
    )
    ce, score_calls = physical_ce(
        model, rows, exact_program, factors, values, decoder, intercept, device,
    )
    expected = json.loads(P512_RESULT.read_text())["select_ce"]
    if abs(ce["NATIVE"] - expected["NATIVE"]) > 2e-6 \
            or abs(ce["ZERO"] - expected["ZERO"]) > 2e-6 \
            or abs(ce["P512_EXACT"] - expected["SPARSE"]) > 2e-6:
        raise RuntimeError("P512 physical CE anchors did not replay")
    output = {
        "schema": "mlp1_pregate_quadratic_router_discovery_v2",
        "status": "discovery_complete",
        "claim_boundary": (
            "Weight-space randomized low-rank screen plus reused-SELECT physical CE; "
            "FINAL never opened, no confirmation/OOD/composition/semantic claim."
        ),
        "method": {
            "ranks": list(RANKS), "oversample": OVERSAMPLE,
            "power_iterations_on_q_squared": POWER_ITERS,
            "qr_after_every_subspace_iteration": True,
            "signed_largest_magnitude_eigenmodes": True,
            "grammar": "signed symmetric matrix-eigenrank; one stored vector and eigenvalue per mode",
            "metric": "coefficient Frobenius control, not empirical fourth-moment optimum",
            "quadratics_materialized": False,
        },
        "documents": {"SELECT": len(rows), "FINAL_opened": 0},
        "route_metrics": metrics,
        "factorization_diagnostics": factorization_diagnostics,
        "physical_ce": ce,
        "prices": {str(rank): router_price(rank) for rank in RANKS},
        "calls": {"capture": capture_calls, "score": score_calls},
        "checkpoint": checkpoint.__dict__,
        "runtime_seconds": time.time() - started,
        "parents": {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "bundle_sha256": file_sha256(BUNDLE),
            "p512_result_sha256": file_sha256(P512_RESULT),
            "rows_receipt_sha256": file_sha256(ROWS_RECEIPT),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
