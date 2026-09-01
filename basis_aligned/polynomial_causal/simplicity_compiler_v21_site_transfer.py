#!/usr/bin/env python3
"""Rung 443 two-phase historical transfer screen for compiler-v2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
SITE0 = BQ / "early_mlp_state_complete_compiler_v21_site0_ledger.pt"
SITE1 = BQ / "early_mlp_state_complete_compiler_v21_site1_ledger.pt"
SITE0_RECEIPT = BQ / "early_mlp_state_complete_compiler_v21_site0_ledger_receipt.json"
SITE1_RECEIPT = BQ / "early_mlp_state_complete_compiler_v21_site1_ledger_receipt.json"
HERE = Path(__file__).resolve().parent
FIT_PATH = HERE / "simplicity_compiler_v21_site0_fit.json"
RESULT_PATH = HERE / "simplicity_compiler_v21_site_transfer_results.json"
FEATURE_NAMES = [
    "log_total_reals",
    "log_inference_multiplies",
    "log_capacity",
    "log1p_regularization",
    "affine",
    "state_complete",
    "causal_metric",
]
ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_accuracy(score: np.ndarray, outcome: np.ndarray) -> float:
    earned = 0.0
    total = 0
    for i in range(len(score)):
        for j in range(i + 1, len(score)):
            dy = outcome[i] - outcome[j]
            if dy == 0:
                continue
            ds = score[i] - score[j]
            total += 1
            if ds == 0:
                earned += 0.5
            elif ds * dy > 0:
                earned += 1.0
    return earned / max(1, total)


def load_bank(site: int, kind: str) -> dict[str, dict[str, Any]]:
    path = SITE0 if site == 0 else SITE1
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["candidate_ledgers"][f"{kind}_site{site}"]


def declared_row(candidate_id: str, record: dict[str, Any]) -> tuple[list[float], str, float]:
    state = record["state"]
    price = record["metrics"]["price"]
    family = str(state["family"])
    capacity = state.get("rank", state.get("k"))
    regularization = state.get("lambda", state.get("source_lambda_ratio", 0.0))
    values = [
        math.log(float(price["total_reals"])),
        math.log(float(price["inference_multiplies_per_token"])),
        math.log(float(capacity)),
        math.log1p(float(regularization)),
        float(state["grammar"] == "affine"),
        float(state["interface"] == "state_complete_p"),
        float("causal" in family.lower()),
    ]
    recovery = float(record["metrics"]["recovery"])
    if not all(math.isfinite(x) for x in values + [recovery]):
        raise RuntimeError(f"nonfinite candidate {candidate_id}")
    if price["total_reals"] <= 0 or price["inference_multiplies_per_token"] <= 0 or capacity <= 0:
        raise RuntimeError(f"nonpositive structural field {candidate_id}")
    return values, family, recovery


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    if alpha == 0:
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
    else:
        coef = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)
    return mean, scale, coef


def predict(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ coef


def choose_alpha(x: np.ndarray, y: np.ndarray, families: list[str]) -> tuple[float, dict[str, float]]:
    unique = sorted(set(families))
    scores: dict[str, float] = {}
    fam_array = np.asarray(families)
    for alpha in ALPHAS:
        fold_scores = []
        for family in unique:
            train = fam_array != family
            test = ~train
            mean, scale, coef = fit_ridge(x[train], y[train], alpha)
            fold_scores.append(spearman(predict(x[test], mean, scale, coef), y[test]))
        scores[str(alpha)] = float(np.mean(fold_scores))
    best = max(ALPHAS, key=lambda a: (scores[str(a)], -a))
    return best, scores


def command_fit() -> None:
    receipt = json.loads(SITE0_RECEIPT.read_text())
    if sha256(SITE0) != receipt["artifact_sha256"]:
        raise RuntimeError("site0 ledger hash mismatch")
    true_bank = load_bank(0, "true")
    shuffle_bank = load_bank(0, "shuffle")
    ids = sorted(true_bank)
    if len(ids) != 108 or ids != sorted(shuffle_bank):
        raise RuntimeError("site0 true/shuffle candidate IDs do not match the frozen 108 bank")
    rows = [declared_row(cid, true_bank[cid]) for cid in ids]
    x = np.asarray([row[0] for row in rows], dtype=np.float64)
    families = [row[1] for row in rows]
    y_true = np.asarray([row[2] for row in rows], dtype=np.float64)
    y_shuffle = np.asarray([declared_row(cid, shuffle_bank[cid])[2] for cid in ids], dtype=np.float64)
    alpha_true, cv_true = choose_alpha(x, y_true, families)
    alpha_shuffle, cv_shuffle = choose_alpha(x, y_shuffle, families)
    mean, scale, coef_true = fit_ridge(x, y_true, alpha_true)
    _, _, coef_shuffle = fit_ridge(x, y_shuffle, alpha_shuffle)
    output = {
        "schema": "simplicity_compiler_v21_site0_fit_v1",
        "site0_ledger_sha256": sha256(SITE0),
        "site0_receipt_sha256": sha256(SITE0_RECEIPT),
        "candidate_ids": ids,
        "feature_names": FEATURE_NAMES,
        "features": x.tolist(),
        "families": families,
        "site0_true_recovery": y_true.tolist(),
        "site0_shuffle_recovery": y_shuffle.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "true_alpha": alpha_true,
        "true_leave_family_out_spearman": cv_true,
        "true_coefficients": coef_true.tolist(),
        "shuffle_alpha": alpha_shuffle,
        "shuffle_leave_family_out_spearman": cv_shuffle,
        "shuffle_coefficients": coef_shuffle.tolist(),
    }
    write_json(FIT_PATH, output)
    print(json.dumps({"fit_path": str(FIT_PATH), "fit_sha256": sha256(FIT_PATH), "true_alpha": alpha_true, "shuffle_alpha": alpha_shuffle}))


def command_score(fit_sha: str) -> None:
    if sha256(FIT_PATH) != fit_sha:
        raise RuntimeError("frozen site0 fit hash mismatch")
    fit = json.loads(FIT_PATH.read_text())
    receipt0 = json.loads(SITE0_RECEIPT.read_text())
    receipt1 = json.loads(SITE1_RECEIPT.read_text())
    if fit["site0_ledger_sha256"] != receipt0["artifact_sha256"] or sha256(SITE1) != receipt1["artifact_sha256"]:
        raise RuntimeError("ledger receipt hash mismatch")
    true_bank = load_bank(1, "true")
    shuffle_bank = load_bank(1, "shuffle")
    ids = fit["candidate_ids"]
    exact_ids = len(ids) == 108 and ids == sorted(true_bank) == sorted(shuffle_bank)
    if not exact_ids:
        raise RuntimeError("site0/site1 true/shuffle IDs do not match")
    site1_rows = [declared_row(cid, true_bank[cid]) for cid in ids]
    x1 = np.asarray([row[0] for row in site1_rows], dtype=np.float64)
    y1 = np.asarray([row[2] for row in site1_rows], dtype=np.float64)
    ys1 = np.asarray([declared_row(cid, shuffle_bank[cid])[2] for cid in ids], dtype=np.float64)
    y0 = np.asarray(fit["site0_true_recovery"], dtype=np.float64)
    mean = np.asarray(fit["feature_mean"], dtype=np.float64)
    scale = np.asarray(fit["feature_scale"], dtype=np.float64)
    learned = predict(x1, mean, scale, np.asarray(fit["true_coefficients"], dtype=np.float64))
    shuffle_learned = predict(x1, mean, scale, np.asarray(fit["shuffle_coefficients"], dtype=np.float64))
    price_score = x1[:, FEATURE_NAMES.index("log_total_reals")]
    rank_score = x1[:, FEATURE_NAMES.index("log_capacity")]
    metrics = {
        "learned_spearman": spearman(learned, y1),
        "learned_pairwise_accuracy": pairwise_accuracy(learned, y1),
        "price_spearman": spearman(price_score, y1),
        "rank_spearman": spearman(rank_score, y1),
        "site0_direct_spearman": spearman(y0, y1),
        "shuffle_learned_spearman": spearman(shuffle_learned, y1),
    }
    top_n = math.ceil(0.10 * len(ids))
    top_indices = np.argsort(learned, kind="mergesort")[-top_n:]
    top_gap = float(np.median(y1[top_indices]) - np.median(ys1[top_indices]))
    rng = np.random.default_rng(443)
    nulls = np.asarray([spearman(learned, rng.permutation(y1)) for _ in range(1000)])
    null_p = float((1 + np.sum(nulls >= metrics["learned_spearman"])) / (1 + len(nulls)))
    pred_a = bool(exact_ids and np.isfinite(x1).all() and (np.exp(x1[:, :3]) > 0).all())
    pred_b = bool(metrics["learned_spearman"] >= 0.50 and metrics["learned_pairwise_accuracy"] >= 0.70)
    pred_c = bool(
        metrics["learned_spearman"] >= max(metrics["price_spearman"], metrics["rank_spearman"]) + 0.10
        and metrics["learned_spearman"] >= metrics["shuffle_learned_spearman"] + 0.15
    )
    pred_d = bool(top_gap >= 0.15)
    strong_null = bool(
        metrics["learned_spearman"] <= 0.20
        or metrics["learned_spearman"] <= metrics["price_spearman"]
        or metrics["learned_spearman"] <= metrics["rank_spearman"]
        or top_gap <= 0.0
    )
    result = {
        "status": "complete",
        "rung": 443,
        "claim_level": "historical_statistical_transfer_screen",
        "fit_sha256": fit_sha,
        "site0_ledger_sha256": sha256(SITE0),
        "site1_ledger_sha256": sha256(SITE1),
        "candidate_count": len(ids),
        "family_count": len(set(fit["families"])),
        "feature_names": FEATURE_NAMES,
        "true_alpha": fit["true_alpha"],
        "shuffle_alpha": fit["shuffle_alpha"],
        "metrics": metrics,
        "predicted_top_decile_ids": [ids[i] for i in top_indices],
        "predicted_top_decile_true_minus_shuffle_median_recovery": top_gap,
        "permutation_null_spearman_mean": float(nulls.mean()),
        "permutation_null_spearman_p95": float(np.quantile(nulls, 0.95)),
        "permutation_p_one_sided": null_p,
        "pred_a_instrument_and_role_separation": pred_a,
        "pred_b_learned_score_transfers": pred_b,
        "pred_c_beats_simple_and_shuffle_controls": pred_c,
        "pred_d_top_decile_is_true_data_specific": pred_d,
        "strong_null_no_structural_transfer": strong_null,
        "routing": "historical_screen_only_design_new_prospective_family" if pred_a and pred_b and pred_c and pred_d and not strong_null else "no_prospective_simplicity_credit_from_this_bank",
        "literal_deployed_model_values": 0,
        "native_model_calls": 0,
    }
    write_json(RESULT_PATH, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fit")
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--fit-sha", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        command_fit()
    else:
        command_score(args.fit_sha)


if __name__ == "__main__":
    main()
