#!/usr/bin/env python3
"""Run the preregistered post-validation MLP2 response-context diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys
for root in (ROOT, HERE):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bilin18_observed_model_facade as facade

PREREG = HERE / "MLP2_FINITE_RESPONSE_CONTEXT_PREREGISTRATION.md"
ROWS = HERE / "mlp2_cmr_v1_validation_rows.pt"
LEDGER = HERE / "mlp2_cmr_v1_validation_ledger.pt"
RESULT = HERE / "mlp2_finite_response_context_result.json"

ARMS = ("SUFFIX", "LOCAL", "RMS", "MASS", "DERANGED", "HASH_RANDOM")
RANKS = (4, 8, 16, 32)
LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float) -> tuple[torch.Tensor, float]:
    xm, ym = x.mean(0), y.mean()
    xc, yc = x - xm, y - ym
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    w = torch.linalg.solve(xc.T @ xc + lam * eye, xc.T @ yc)
    return w, float(ym - xm @ w)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((a @ b) / den) if float(den) > 0 else 0.0


def metrics(y: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "pearson": corr(y, pred),
        "mse": sse / y.numel(),
        "nrmse": (sse / max(sst, 1e-30)) ** 0.5,
        "r2": 1.0 - sse / max(sst, 1e-30),
    }


def standardize_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-8)
    return (x - mean) / scale, mean, scale


def pca_design(
    train: torch.Tensor, test: torch.Tensor, rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z, mean, scale = standardize_fit(train)
    _, _, vh = torch.linalg.svd(z, full_matrices=False)
    basis = vh[: min(rank, vh.shape[0])]
    a, b = z @ basis.T, ((test - mean) / scale) @ basis.T
    pc_scale = a.std(0, unbiased=False).clamp_min(1e-8)
    return a / pc_scale, b / pc_scale


def choose_state_hyperparameters(
    x: torch.Tensor, y: torch.Tensor, doc_ids: torch.Tensor,
) -> tuple[int, float]:
    fold_id = torch.arange(doc_ids.numel()) % 4
    designs = []
    for fold in range(4):
        va = fold_id == fold
        tr = ~va
        a, b = pca_design(x[tr], x[va], max(RANKS))
        designs.append((tr, va, a, b))
    scored = []
    for rank in RANKS:
        for lam in LAMBDAS:
            losses = []
            for tr, va, a, b in designs:
                w, bias = ridge_fit(a[:, :rank], y[tr], lam)
                losses.append(float(
                    ((y[va] - (b[:, :rank] @ w + bias)) ** 2).sum()
                ))
            scored.append((sum(losses), rank, lam))
    _, rank, lam = min(scored)
    return rank, lam


def choose_baseline_lambda(
    x: torch.Tensor, y: torch.Tensor, doc_ids: torch.Tensor,
) -> float:
    fold_id = torch.arange(doc_ids.numel()) % 4
    scored = []
    for lam in LAMBDAS:
        losses = []
        for fold in range(4):
            va = fold_id == fold
            tr = ~va
            a, mean, scale = standardize_fit(x[tr])
            b = (x[va] - mean) / scale
            w, bias = ridge_fit(a, y[tr], lam)
            losses.append(float(((y[va] - (b @ w + bias)) ** 2).sum()))
        scored.append((sum(losses), lam))
    return min(scored)[1]


def main() -> None:
    if RESULT.exists():
        raise RuntimeError("result namespace already exists")
    started = time.time()
    device = torch.device("cuda")
    rows_bundle = torch.load(ROWS, map_location="cpu", weights_only=True)
    ledger = torch.load(LEDGER, map_location="cpu", weights_only=True)["ledgers"]
    rows = rows_bundle["rows"]
    eligible = rows_bundle["eligible_mask"]
    arm_names = tuple(ledger["arm_names"])
    cell_names = tuple(ledger["cell_names"])
    float_fields = tuple(ledger["float_fields"])
    all_cell = cell_names.index("all_scored")
    candidate_field = float_fields.index("candidate_nll_sum")
    native_field = float_fields.index("native_nll_sum")
    counts = ledger["counts"][:, :, all_cell].double()
    sums = ledger["float_sums"][:, :, all_cell]
    dce = (sums[:, :, candidate_field] - sums[:, :, native_field]) / counts.clamp_min(1)
    supported = counts[arm_names.index("NATIVE")] > 0
    zero = dce[arm_names.index("ZERO")]
    response = torch.stack([dce[arm_names.index(a)] - zero for a in ARMS], 1)

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    state_features = torch.empty(rows.shape[0], 2304, dtype=torch.float64)
    native_calls = [0] * 18
    capture_calls = 0
    with torch.inference_mode():
        for start in range(0, rows.shape[0], 4):
            stop = start + 4
            tokens = rows[start:stop, :-1].to(device).contiguous()
            mask = eligible[start:stop].to(device)
            captured: list[torch.Tensor] = []

            def attention(event: facade.AttentionEvent):
                return event.block.attn(event.state, event.first_value)

            def mlp(event: facade.EarlyMLPEvent):
                nonlocal capture_calls
                native_calls[event.site] += 1
                if event.site == 2:
                    if captured:
                        raise RuntimeError("MLP2 state captured twice")
                    m = mask.unsqueeze(-1)
                    denom = m.sum(1).clamp_min(1)
                    z = event.state.float()
                    mean = (z * m).sum(1) / denom
                    second = (z.square() * m).sum(1) / denom
                    captured.append(torch.cat((mean, second), 1).double().cpu())
                    capture_calls += 1
                return event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            if len(captured) != 1:
                raise RuntimeError("MLP2 state capture missing")
            state_features[start:stop] = captured[0]
            del logits, tokens, mask, captured
    torch.cuda.synchronize(device)
    if capture_calls != 48 or native_calls != [48] * 18:
        raise RuntimeError("forward/capture call census changed")
    del model
    torch.cuda.empty_cache()

    token_features = torch.empty(rows.shape[0], 7, dtype=torch.float64)
    cell_count = ledger["counts"][arm_names.index("NATIVE")].double()
    for d in range(rows.shape[0]):
        mask = eligible[d]
        toks = rows[d, :-1][mask]
        n = max(int(toks.numel()), 1)
        seen = set()
        repeated = 0
        for value in toks.tolist():
            repeated += value in seen
            seen.add(value)
        adjacent = int((toks[1:] == toks[:-1]).sum()) if n > 1 else 0
        denom = max(float(cell_count[d, all_cell]), 1.0)
        token_features[d] = torch.tensor([
            n / 256.0,
            len(seen) / n,
            adjacent / max(n - 1, 1),
            repeated / n,
            float(cell_count[d, cell_names.index("copy_positive")]) / denom,
            float(cell_count[d, cell_names.index("repeat_negative")]) / denom,
            float(cell_count[d, cell_names.index("nonrepeat")]) / denom,
        ], dtype=torch.float64)

    ids = torch.arange(rows.shape[0])[supported]
    x_state, x_base, y6 = state_features[supported], token_features[supported], response[supported]
    state_oof = torch.empty(ids.numel(), dtype=torch.float64)
    base_oof = torch.empty_like(state_oof)
    target_oof = torch.empty_like(state_oof)
    perm_oof = torch.empty_like(state_oof)
    fold_reports = []
    generator = torch.Generator().manual_seed(2026082919)
    permutation = torch.randperm(ids.numel(), generator=generator)

    for parity in (0, 1):
        test = (ids % 2) == parity
        train = ~test
        arm_mean = y6[train].mean(0)
        arm_scale = y6[train].std(0, unbiased=False).clamp_min(1e-8)
        target = ((y6 - arm_mean) / arm_scale).mean(1)
        rank, lam = choose_state_hyperparameters(x_state[train], target[train], ids[train])
        a, b = pca_design(x_state[train], x_state[test], rank)
        w, bias = ridge_fit(a, target[train], lam)
        state_oof[test] = b @ w + bias
        base_lam = choose_baseline_lambda(x_base[train], target[train], ids[train])
        ba, bm, bs = standardize_fit(x_base[train])
        bw, bb = ridge_fit(ba, target[train], base_lam)
        base_oof[test] = ((x_base[test] - bm) / bs) @ bw + bb
        target_oof[test] = target[test]

        perm_target = target[permutation]
        prank, plam = choose_state_hyperparameters(
            x_state[train], perm_target[train], ids[train]
        )
        pa, pb = pca_design(x_state[train], x_state[test], prank)
        pw, pbias = ridge_fit(pa, perm_target[train], plam)
        perm_oof[test] = pb @ pw + pbias
        fold_reports.append({
            "test_parity": parity,
            "train_documents": int(train.sum()),
            "test_documents": int(test.sum()),
            "state_rank": rank,
            "state_lambda": lam,
            "baseline_lambda": base_lam,
            "state_metrics": metrics(target[test], state_oof[test]),
            "baseline_metrics": metrics(target[test], base_oof[test]),
        })

    state_metrics = metrics(target_oof, state_oof)
    baseline_metrics = metrics(target_oof, base_oof)
    perm_corr = corr(target_oof, perm_oof)
    mse_gain = 1.0 - state_metrics["mse"] / baseline_metrics["mse"]
    gates = {
        "pooled_pearson_at_least_0p50": state_metrics["pearson"] >= 0.50,
        "both_fold_correlations_positive": all(
            f["state_metrics"]["pearson"] > 0 for f in fold_reports
        ),
        "mse_improvement_over_baseline_at_least_0p20": mse_gain >= 0.20,
        "permutation_abs_correlation_below_0p20": abs(perm_corr) < 0.20,
    }
    result = {
        "schema": "mlp2_finite_response_context_diagnostic_v1",
        "status": "promising" if all(gates.values()) else "not_sufficient_at_document_moments",
        "scope": "post_validation_exploratory_no_strict_ledger_move",
        "documents": int(ids.numel()),
        "state_metrics": state_metrics,
        "baseline_metrics": baseline_metrics,
        "mse_improvement_over_baseline": mse_gain,
        "permutation_control_pooled_correlation": perm_corr,
        "folds": fold_reports,
        "gates": gates,
        "capture_call_census": {"full_native_forwards": 48, "mlp2_captures": capture_calls},
        "checkpoint": checkpoint.__dict__,
        "runtime_seconds": time.time() - started,
        "parents": {"prereg": sha256(PREREG), "rows": sha256(ROWS), "ledger": sha256(LEDGER)},
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "raw_states_published": False,
        "raw_targets_published": False,
        "per_document_predictions_published": False,
    }
    RESULT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
