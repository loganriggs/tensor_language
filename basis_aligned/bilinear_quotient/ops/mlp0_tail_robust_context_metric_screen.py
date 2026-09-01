"""RUNG 350 -- TAIL-ROBUST CONTEXT METRIC FOR MLP0 SHARED INPUT RANK.

Test whether the rare-row failures at p384 arise from fitting a mean covariance.
For each of two fit halves, whiten exact contextual MLP0 inputs, select the top
10% by leverage, and form C_rob=.75*C+.25*C_tail.  Compare ordinary and robust
RRR at p384/p448 on rowwise FineWeb and a non-overlapping WikiText-103-train
segment.  This is a screen only; no physical price or adoption is claimed.

Frozen predictions
------------------
pred_a_robust_p384_reduces_both_tail_distributions:
    Robust p384 p95/max <=85%/80% of ordinary on BOTH corpora.
pred_b_robust_p384_preserves_mean_prediction:
    Robust mean <=120% of ordinary nonnegative mean, and absolute mean
    <=.020 FineWeb / .025 WikiText.
pred_c_metric_is_split_stable_and_p448_safe:
    Selected mass is 8--12%, robust split overlap >=.60 at both ranks, and
    robust p448 maxima do not exceed ordinary on either corpus.

Null: robust maxima fail to improve at both ranks on both corpora, OR both
robust split overlaps <=.30.
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
OUT = ROOT / "mlp0_tail_robust_context_metric_screen_results.json"
DEV = "cuda"
D = 1152
RANKS = (384, 448)
FIT_A = (0, 24)
FIT_B = (24, 48)
FINEWEB_EVAL = (120, 160)
WIKI_SKIP = 120 * 257
EVAL_ROWS = 40
TAIL_FRACTION = .10
TAIL_MIX = .25


@torch.no_grad()
def _inputs(model, rows, manual_logits) -> torch.Tensor:
    pieces = []

    def hook(_module, args, _output):
        pieces.append(args[0].detach().reshape(-1, D).float())

    handle = model.transformer.h[0].mlp.register_forward_hook(hook)
    try:
        for start in range(0, len(rows), 2):
            manual_logits(model, rows[start:start + 2, :-1].to(DEV))
    finally:
        handle.remove()
    value = torch.cat(pieces)
    assert value.shape == (len(rows) * 256, D)
    return value


@torch.no_grad()
def _metrics(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    covariance = inputs.T @ inputs / len(inputs)
    covariance = .5 * (covariance + covariance.T)
    values, vectors = torch.linalg.eigh(covariance)
    floor = float(values[-1]) * 1e-6
    safe = values.clamp_min(floor)
    whitened = (inputs @ vectors) * safe.rsqrt()
    leverage = whitened.square().sum(dim=1)
    threshold = torch.quantile(leverage, 1.0 - TAIL_FRACTION)
    selected = leverage >= threshold
    tail = inputs[selected]
    tail_covariance = tail.T @ tail / len(tail)
    tail_covariance = .5 * (tail_covariance + tail_covariance.T)
    robust = (1.0 - TAIL_MIX) * covariance + TAIL_MIX * tail_covariance
    diagnostics = {
        "selected_fraction": float(selected.float().mean()),
        "leverage_threshold": float(threshold),
        "selected_mean_leverage_ratio": float(leverage[selected].mean() / leverage.mean()),
        "covariance_condition_after_floor": float(values[-1] / safe[0]),
    }
    return covariance, robust, diagnostics


@torch.no_grad()
def _score_rows(model, rows, manual_logits, program=None) -> torch.Tensor:
    handle = None
    if program is not None:
        encoder = program["encoder"]
        left = program["left"]
        right = program["right"]
        down = program["down"]
        bias = program["bias"]

        def hook(_module, args, output):
            x = args[0].float()
            z = x @ encoder.T
            hidden = (z @ left.T) * (z @ right.T)
            return (hidden @ down.T + bias).to(output.dtype)

        handle = model.transformer.h[0].mlp.register_forward_hook(hook)
    values = []
    try:
        for row in rows:
            index = row[:-1].unsqueeze(0).to(DEV)
            target = row[1:].to(DEV)
            logits = manual_logits(model, index)[0].float()
            values.append(float(F.cross_entropy(logits, target)))
    finally:
        if handle is not None:
            handle.remove()
    return torch.tensor(values, dtype=torch.float64)


def _summary(damage: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(damage.mean()),
        "p95": float(torch.quantile(damage, .95)),
        "max": float(damage.max()),
    }


def _nonnegative_ratio(robust: float, ordinary: float) -> float:
    return max(0.0, robust) / max(1e-6, max(0.0, ordinary))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FIT_A[1] == FIT_B[0] and FIT_B[1] <= FINEWEB_EVAL[0]
        assert WIKI_SKIP == 30_840 and EVAL_ROWS * 257 == 10_280
        print("MLP0 TAIL-ROBUST METRIC | dry run: splits, ranks, corpus segment, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    fit_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    eval_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    eval_cached = eval_cached["rows"] if isinstance(eval_cached, dict) else eval_cached
    fit_rows = {
        "a": fit_cached[FIT_A[0]:FIT_A[1], :257].long().contiguous(),
        "b": fit_cached[FIT_B[0]:FIT_B[1], :257].long().contiguous(),
    }
    fineweb = eval_cached[FINEWEB_EVAL[0]:FINEWEB_EVAL[1], :257].long().contiguous()
    wikitext, fingerprint, token_count = _wikitext103_train_rows(
        n=EVAL_ROWS, width=257, skip=WIKI_SKIP)
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and fineweb.shape == wikitext.shape == (EVAL_ROWS, 257)

    programs = {}
    bases = {}
    metric_diagnostics = {}
    for split, rows in fit_rows.items():
        x = _inputs(model, rows, _manual_logits)
        ordinary, robust, metric_diagnostics[split] = _metrics(x)
        del x
        for metric_name, covariance in (("ordinary", ordinary), ("robust", robust)):
            for rank in RANKS:
                program, basis, diagnostic = _rrr_program(
                    model.transformer.h[0].mlp, covariance, rank=rank)
                key = f"{split}_{metric_name}_{rank}"
                programs[key] = program
                bases[key] = basis
                metric_diagnostics[split][f"{metric_name}_{rank}"] = diagnostic
        del ordinary, robust
        torch.cuda.empty_cache()

    native = {
        "fineweb": _score_rows(model, fineweb, _manual_logits),
        "wikitext": _score_rows(model, wikitext, _manual_logits),
    }
    summaries = {}
    for split in ("a", "b"):
        for metric_name in ("ordinary", "robust"):
            for rank in RANKS:
                key = f"{split}_{metric_name}_{rank}"
                summaries[key] = {}
                for corpus, rows in (("fineweb", fineweb), ("wikitext", wikitext)):
                    ce = _score_rows(model, rows, _manual_logits, programs[key])
                    summaries[key][corpus] = _summary(ce - native[corpus])
                print(f"{key}: FW {summaries[key]['fineweb']} WT {summaries[key]['wikitext']}",
                      flush=True)

    split_overlap = {}
    for rank in RANKS:
        a = bases[f"a_robust_{rank}"]
        b = bases[f"b_robust_{rank}"]
        split_overlap[str(rank)] = float((a.T @ b).square().sum() / rank)

    primary_ordinary = summaries["a_ordinary_384"]
    primary_robust = summaries["a_robust_384"]
    pred_a = all(
        primary_robust[corpus]["p95"] <= .85 * primary_ordinary[corpus]["p95"]
        and primary_robust[corpus]["max"] <= .80 * primary_ordinary[corpus]["max"]
        for corpus in ("fineweb", "wikitext")
    )
    pred_b = (
        _nonnegative_ratio(primary_robust["fineweb"]["mean"],
                           primary_ordinary["fineweb"]["mean"]) <= 1.20
        and _nonnegative_ratio(primary_robust["wikitext"]["mean"],
                               primary_ordinary["wikitext"]["mean"]) <= 1.20
        and primary_robust["fineweb"]["mean"] <= .020
        and primary_robust["wikitext"]["mean"] <= .025
    )
    pred_c = (
        all(.08 <= metric_diagnostics[split]["selected_fraction"] <= .12
            for split in ("a", "b"))
        and all(split_overlap[str(rank)] >= .60 for rank in RANKS)
        and all(summaries["a_robust_448"][corpus]["max"]
                <= summaries["a_ordinary_448"][corpus]["max"]
                for corpus in ("fineweb", "wikitext"))
    )
    all_max_fail = all(
        summaries[f"a_robust_{rank}"][corpus]["max"]
        >= summaries[f"a_ordinary_{rank}"][corpus]["max"]
        for rank in RANKS for corpus in ("fineweb", "wikitext")
    )
    null = all_max_fail or all(split_overlap[str(rank)] <= .30 for rank in RANKS)
    result = {
        "status": "mlp0_tail_robust_context_metric_screen_complete",
        "rung": 350,
        "claim_level": "split_fit_two_corpus_row_tail_metric_screen_only",
        "metric": {"tail_fraction": TAIL_FRACTION, "tail_mix": TAIL_MIX,
                   "formula": "C_rob=(1-tail_mix)*C+tail_mix*C_top_leverage"},
        "fit_rows": {"cache": "fineweb_n192_skip11000.pt",
                     "fit_a": list(FIT_A), "fit_b": list(FIT_B)},
        "evaluation": {
            "fineweb_cache": "fineweb_n192_skip7000.pt",
            "fineweb_rows_half_open": list(FINEWEB_EVAL),
            "wikitext103_train_token_skip": WIKI_SKIP,
            "wikitext_rows": EVAL_ROWS,
            "dataset_fingerprint": fingerprint,
            "source_token_count": token_count,
        },
        "metric_diagnostics": metric_diagnostics,
        "robust_split_subspace_overlap": split_overlap,
        "row_damage_summaries": summaries,
        'pred_a_robust_p384_reduces_both_tail_distributions': bool(pred_a),
        'pred_b_robust_p384_preserves_mean_prediction': bool(pred_b),
        'pred_c_metric_is_split_stable_and_p448_safe': bool(pred_c),
        "null_tail_robust_metric_has_no_replicable_advantage": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "split_overlap": split_overlap, "runtime_s": result["runtime_s"]},
                     indent=2), flush=True)
    print("MLP0 TAIL-ROBUST CONTEXT METRIC SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
