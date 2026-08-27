"""Matched-cost causal test of the mlp11 question quadratic slice.

The selected rank-2 scalar slice has one positive and one negative eigenmode.
Its exact real product program uses one multiplication by pairing those modes.
The matched baseline is the best one-square program fitted on separate natural
activations. Both use one multiplication; only the allowed product geometry differs.

The hook replaces only this rank-2 contribution to the question output direction;
the rest of mlp11 and the model remain live. This is a causal replacement test of a
slice, not a claim that the full MLP has been compressed.

Registered before execution:
  A. paired-fp32 relative scalar RMSE <= 1e-5 on both evaluation splits;
  B. paired-bf16 relative scalar RMSE <= .02 and question-class KL <= 1e-4 on
     both splits;
  C. the best discovery-fitted square has relative scalar RMSE >= .25 and
     question-class KL >= 1e-4 on both held-out splits;
  D. exploratory causal share: report square KL / zero-rank2 KL without a gate.

The square is optimized over all signed multiples s*(cos(theta) z+ +
sin(theta) z-)^2 on the fit rows, not merely chosen from the two eigen-squares.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
QK = HERE.parent / "qk_mdl"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(QK))

from data import fineweb_rows  # noqa: E402
from tier2_model import load_elriggs, reference_forward  # noqa: E402


D = 1152
T = 256
SITE = 11
BATCH = 8
FIT_ROWS = 96
EVAL_ROWS = 240
FIT_SKIP = 80
DISCOVERY_SKIP = 7000
HELDOUT_SKIP = 11000
GRID = 4096
OUT = HERE / "question_one_product_results.json"
DEV = "cuda"

MODEL = None
DIRECTION = None
U2 = None
EIG = None
PAIR_A = None
PAIR_B = None
SQUARE_W = None
SQUARE_SCALE = None
MODE = "base"
SCALAR_ACC = None
ENC = tiktoken.get_encoding("gpt2")


def token_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(50257, dtype=torch.bool)
    for token in range(50257):
        if re.match(pattern, ENC.decode([token])):
            mask[token] = True
    return mask


def build_slice(mask_v: torch.Tensor) -> dict:
    global DIRECTION, U2, EIG, PAIR_A, PAIR_B
    unembed = MODEL.lm_head.weight.float()[:50257]
    DIRECTION = unembed[mask_v.to(DEV)].mean(0)
    DIRECTION = DIRECTION / DIRECTION.norm()
    mlp = MODEL.transformer.h[SITE].mlp
    output_weight = DIRECTION @ mlp.Down.weight.float()
    raw = mlp.Left.weight.float().T @ (output_weight[:, None] * mlp.Right.weight.float())
    symmetric = 0.5 * (raw + raw.T)
    values, vectors = torch.linalg.eigh(symmetric)
    positive = int(values.argmax())
    negative = int(values.argmin())
    lp, ln = values[positive], values[negative]
    if not (lp > 0 and ln < 0):
        raise RuntimeError("selected question form is not indefinite")
    U2 = torch.stack([vectors[:, positive], vectors[:, negative]], dim=1).contiguous()
    EIG = torch.stack([lp, ln])
    PAIR_A = math.sqrt(float(lp)) * U2[:, 0] + math.sqrt(float(-ln)) * U2[:, 1]
    PAIR_B = math.sqrt(float(lp)) * U2[:, 0] - math.sqrt(float(-ln)) * U2[:, 1]
    return {
        "positive": float(lp),
        "negative": float(ln),
        "positive_index": positive,
        "negative_index": negative,
        "full_numerical_rank_1e-6": int((values.abs() > values.abs().max() * 1e-6).sum()),
    }


def direct_q(coords: torch.Tensor) -> torch.Tensor:
    return EIG[0] * coords[..., 0].square() + EIG[1] * coords[..., 1].square()


def replacement_q(x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if MODE == "pair_fp32":
        return (x.float() @ PAIR_A) * (x.float() @ PAIR_B)
    if MODE == "pair_bf16":
        left = x.bfloat16() @ PAIR_A.bfloat16()
        right = x.bfloat16() @ PAIR_B.bfloat16()
        return (left * right).float()
    if MODE == "square":
        return SQUARE_SCALE * (x.float() @ SQUARE_W).square()
    if MODE == "zero_rank2":
        return torch.zeros_like(coords[..., 0])
    return direct_q(coords)


def replacement_hook(module, args, output):
    x = args[0]
    coords = x.float() @ U2
    target = direct_q(coords)
    replacement = replacement_q(x, coords)
    if SCALAR_ACC is not None:
        error = replacement - target
        SCALAR_ACC["error_sq"] += float(error.double().square().sum())
        SCALAR_ACC["target_sq"] += float(target.double().square().sum())
        SCALAR_ACC["n"] += target.numel()
    if MODE == "base":
        return None
    return (output.float() + (replacement - target).unsqueeze(-1) * DIRECTION).to(output.dtype)


@torch.no_grad()
def capture_coords(rows: torch.Tensor) -> torch.Tensor:
    captured = []

    def hook(module, args, output):
        captured.append((args[0].float() @ U2).reshape(-1, 2).cpu())

    handle = MODEL.transformer.h[SITE].mlp.register_forward_hook(hook)
    for start in range(0, len(rows), BATCH):
        reference_forward(MODEL, rows[start:start + BATCH, :-1].to(DEV))
    handle.remove()
    return torch.cat(captured).to(DEV)


@torch.no_grad()
def fit_best_square(coords: torch.Tensor) -> dict:
    global SQUARE_W, SQUARE_SCALE
    q = direct_q(coords)
    best = None
    chunk = 128
    for start in range(0, GRID, chunk):
        theta = torch.arange(start, min(start + chunk, GRID), device=DEV, dtype=torch.float64)
        theta = theta * math.pi / GRID
        w = torch.stack([theta.cos(), theta.sin()], dim=0).float()
        h = (coords @ w).square()
        scale = (h * q[:, None]).sum(0) / h.square().sum(0).clamp_min(1e-30)
        mse = ((h * scale - q[:, None]).square()).mean(0)
        value, index = mse.min(0)
        record = (float(value), float(theta[index]), float(scale[index]))
        if best is None or record[0] < best[0]:
            best = record
    mse, theta, scale = best
    vector2 = torch.tensor([math.cos(theta), math.sin(theta)], device=DEV)
    SQUARE_W = U2 @ vector2
    SQUARE_SCALE = scale
    rel_rmse = math.sqrt(mse / float(q.double().square().mean()))
    return {"theta": theta, "scale": scale, "fit_mse": mse, "fit_relative_rmse": rel_rmse}


def new_totals():
    return defaultdict(float)


@torch.no_grad()
def evaluate(rows: torch.Tensor, mask_v: torch.Tensor) -> dict:
    global MODE, SCALAR_ACC
    modes = ("base", "pair_fp32", "pair_bf16", "square", "zero_rank2")
    totals = {mode: new_totals() for mode in modes}
    handle = MODEL.transformer.h[SITE].mlp.register_forward_hook(replacement_hook)
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH].to(DEV)
        idx, targets = batch[:, :-1], batch[:, 1:]
        MODE = "base"
        SCALAR_ACC = None
        clean = reference_forward(MODEL, idx).float()
        clean_logp = clean.log_softmax(-1)
        clean_prob = clean_logp.exp()
        valid = torch.ones_like(targets, dtype=torch.bool)
        valid[:, :64] = False
        question = mask_v.to(DEV)[targets] & valid
        for mode in modes:
            MODE = mode
            SCALAR_ACC = totals[mode] if mode != "base" else None
            logits = clean if mode == "base" else reference_forward(MODEL, idx).float()
            logp = logits.log_softmax(-1)
            ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            kl = (clean_prob * (clean_logp - logp)).sum(-1)
            true_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for prefix, select in (("global", valid), ("question", question)):
                totals[mode][f"{prefix}_ce_sum"] += float(ce[select].sum())
                totals[mode][f"{prefix}_kl_sum"] += float(kl[select].sum())
                totals[mode][f"{prefix}_true_logit_sum"] += float(true_logit[select].sum())
                totals[mode][f"{prefix}_n"] += int(select.sum())
    handle.remove()
    MODE = "base"
    SCALAR_ACC = None
    result = {}
    for mode, total in totals.items():
        row = {}
        for prefix in ("global", "question"):
            n = max(total[f"{prefix}_n"], 1)
            row[f"{prefix}_ce"] = total[f"{prefix}_ce_sum"] / n
            row[f"{prefix}_kl"] = total[f"{prefix}_kl_sum"] / n
            row[f"{prefix}_true_logit"] = total[f"{prefix}_true_logit_sum"] / n
            row[f"{prefix}_n"] = int(total[f"{prefix}_n"])
        if mode != "base":
            row["scalar_relative_rmse"] = math.sqrt(total["error_sq"] / max(total["target_sq"], 1e-30))
        result[mode] = row
    base = result["base"]
    for mode in modes[1:]:
        for prefix in ("global", "question"):
            result[mode][f"{prefix}_ce_excess"] = result[mode][f"{prefix}_ce"] - base[f"{prefix}_ce"]
            result[mode][f"{prefix}_true_logit_change"] = result[mode][f"{prefix}_true_logit"] - base[f"{prefix}_true_logit"]
    return result


@torch.no_grad()
def main() -> None:
    global MODEL
    start = time.time()
    MODEL, _ = load_elriggs("bilin18")
    mask_v = token_mask(r"^\?$| \?$")
    spectrum = build_slice(mask_v)
    fit_rows = fineweb_rows(FIT_ROWS, skip=FIT_SKIP)[:, :T + 1].contiguous()
    fit = fit_best_square(capture_coords(fit_rows))
    print(f"slice eigs {spectrum['positive']:.4f}, {spectrum['negative']:.4f}; square fit relRMSE={fit['fit_relative_rmse']:.4f}", flush=True)

    splits = {
        "discovery": fineweb_rows(EVAL_ROWS, skip=DISCOVERY_SKIP)[:, :T + 1].contiguous(),
        "heldout": fineweb_rows(EVAL_ROWS, skip=HELDOUT_SKIP)[:, :T + 1].contiguous(),
    }
    evaluations = {}
    for name, rows in splits.items():
        evaluations[name] = evaluate(rows, mask_v)
        r = evaluations[name]
        print(f"{name}: pair={r['pair_fp32']['scalar_relative_rmse']:.2e} bf16={r['pair_bf16']['scalar_relative_rmse']:.4f} square={r['square']['scalar_relative_rmse']:.4f}; qKL square={r['square']['question_kl']:.6f}", flush=True)

    pred_a = all(evaluations[s]["pair_fp32"]["scalar_relative_rmse"] <= 1e-5 for s in splits)
    pred_b = all(evaluations[s]["pair_bf16"]["scalar_relative_rmse"] <= 0.02 and evaluations[s]["pair_bf16"]["question_kl"] <= 1e-4 for s in splits)
    pred_c = all(evaluations[s]["square"]["scalar_relative_rmse"] >= 0.25 and evaluations[s]["square"]["question_kl"] >= 1e-4 for s in splits)
    causal_share = {
        s: evaluations[s]["square"]["question_kl"] / max(evaluations[s]["zero_rank2"]["question_kl"], 1e-30)
        for s in splits
    }
    result = {
        "config": {
            "model": "bilin18",
            "site": "mlp11 question-unembedding rank-2 spectral slice",
            "fit_rows": FIT_ROWS,
            "eval_rows_per_split": EVAL_ROWS,
            "fit_skip": FIT_SKIP,
            "discovery_skip": DISCOVERY_SKIP,
            "heldout_skip": HELDOUT_SKIP,
            "square_grid": GRID,
            "product_count_pair": 1,
            "product_count_square": 1,
        },
        "spectrum": spectrum,
        "best_square": fit,
        "evaluations": evaluations,
        "square_question_kl_fraction_of_zero_rank2": causal_share,
        "predictions": {
            "A_pair_fp32_exact": pred_a,
            "B_pair_bf16_stable": pred_b,
            "C_pair_geometry_beats_best_square": pred_c,
        },
        "runtime_s": round(time.time() - start, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["predictions"], indent=2), flush=True)
    print(f"wrote {OUT} ({result['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
