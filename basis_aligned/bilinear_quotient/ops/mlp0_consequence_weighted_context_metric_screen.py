"""RUNG 353 -- CONSEQUENCE-WEIGHTED CONTEXT METRIC FOR MLP0.

Weight each contextual MLP0 input by the norm of the complete suffix-loss
gradient at the corresponding MLP0 output.  Compare p384/p448 paired RRR under
this metric against ordinary covariance on two independent fit halves and two
row-tail populations.  This is distinct from the failed leverage weighting:
the variable is downstream consequence, not input rarity.

Frozen predictions
------------------
pred_a_gradient_p384_reduces_both_tail_distributions:
    Gradient p384 p95/max <=90% of ordinary on BOTH corpora.
pred_b_gradient_p384_preserves_mean_prediction:
    Gradient mean <=120% of ordinary nonnegative mean and absolute mean
    <=.020 FineWeb / .025 WikiText.
pred_c_metric_is_split_stable_clipped_and_p448_safe:
    Gradient split overlap >=.55 at both ranks, <=25% weights hit upper clip,
    and p448 gradient maxima do not exceed ordinary on either corpus.

Null: no gradient maximum improves at either rank/corpus, OR both gradient
split overlaps <=.25.  No weight-clip or rank tuning follows the result.
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
OUT = ROOT / "mlp0_consequence_weighted_context_metric_screen_results.json"
DEV = "cuda"
D = 1152
RANKS = (384, 448)
FIT_A = (0, 24)
FIT_B = (24, 48)
FINEWEB_EVAL = (80, 120)
WIKI_SKIP = 400 * 257
EVAL_ROWS = 40
WEIGHT_CLIP = (.25, 4.0)


def _gradient_metric_inputs(model, rows, manual_logits):
    inputs = []
    weights = []
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for start in range(0, len(rows), 2):
        state = {}

        def hook(_module, args, output):
            state["input"] = args[0].detach().reshape(-1, D).float()
            leaf = output.detach().float().requires_grad_(True)
            state["leaf"] = leaf
            return leaf.to(output.dtype)

        handle = model.transformer.h[0].mlp.register_forward_hook(hook)
        try:
            batch = rows[start:start + 2]
            index = batch[:, :-1].to(DEV)
            target = batch[:, 1:].to(DEV)
            logits = manual_logits(model, index).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   target.reshape(-1), reduction="sum")
            loss.backward()
        finally:
            handle.remove()
        leaf = state["leaf"]
        if leaf.grad is None:
            raise RuntimeError("MLP0 output leaf received no suffix-loss gradient")
        inputs.append(state["input"])
        weights.append(leaf.grad.detach().reshape(-1, D).float().norm(dim=1))
        del logits, loss, leaf, state
    x = torch.cat(inputs)
    gradient_norm = torch.cat(weights)
    assert x.shape == (len(rows) * 256, D) and gradient_norm.shape == (len(rows) * 256,)
    median = gradient_norm.median().clamp_min(1e-12)
    weight = (gradient_norm / median).clamp(*WEIGHT_CLIP)
    diagnostics = {
        "raw_gradient_norm_median": float(median),
        "raw_gradient_norm_p95": float(torch.quantile(gradient_norm, .95)),
        "weight_mean_before_normalization": float(weight.mean()),
        "lower_clip_fraction": float((weight == WEIGHT_CLIP[0]).float().mean()),
        "upper_clip_fraction": float((weight == WEIGHT_CLIP[1]).float().mean()),
    }
    weight = weight / weight.mean()
    return x, weight, diagnostics


def _covariances(x, weight):
    ordinary = x.T @ x / len(x)
    consequence = (x * weight[:, None]).T @ x / weight.sum()
    return .5 * (ordinary + ordinary.T), .5 * (consequence + consequence.T)


def _nonnegative_ratio(changed: float, control: float) -> float:
    return max(0.0, changed) / max(1e-6, max(0.0, control))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FIT_A[1] == FIT_B[0] and FIT_B[1] <= FINEWEB_EVAL[0]
        assert WIKI_SKIP == 102_800 and WIKI_SKIP + EVAL_ROWS * 257 == 113_080
        assert WEIGHT_CLIP == (.25, 4.0)
        print("MLP0 CONSEQUENCE METRIC | dry run: splits, ranks, corpus, clips, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp0_tail_robust_context_metric_screen import _score_rows, _summary
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
    diagnostics = {}
    for split, rows in fit_rows.items():
        with torch.enable_grad():
            x, weight, diagnostics[split] = _gradient_metric_inputs(model, rows, _manual_logits)
        ordinary, consequence = _covariances(x, weight)
        del x, weight
        for metric_name, covariance in (("ordinary", ordinary),
                                        ("consequence", consequence)):
            for rank in RANKS:
                program, basis, diag = _rrr_program(
                    model.transformer.h[0].mlp, covariance, rank=rank)
                key = f"{split}_{metric_name}_{rank}"
                programs[key] = program
                bases[key] = basis
                diagnostics[split][f"{metric_name}_{rank}"] = diag
        del ordinary, consequence
        torch.cuda.empty_cache()

    native = {
        "fineweb": _score_rows(model, fineweb, _manual_logits),
        "wikitext": _score_rows(model, wikitext, _manual_logits),
    }
    summaries = {}
    for split in ("a", "b"):
        for metric_name in ("ordinary", "consequence"):
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
        a = bases[f"a_consequence_{rank}"]
        b = bases[f"b_consequence_{rank}"]
        split_overlap[str(rank)] = float((a.T @ b).square().sum() / rank)

    control = summaries["a_ordinary_384"]
    changed = summaries["a_consequence_384"]
    pred_a = all(
        changed[corpus]["p95"] <= .90 * control[corpus]["p95"]
        and changed[corpus]["max"] <= .90 * control[corpus]["max"]
        for corpus in ("fineweb", "wikitext")
    )
    pred_b = (
        _nonnegative_ratio(changed["fineweb"]["mean"], control["fineweb"]["mean"]) <= 1.20
        and _nonnegative_ratio(changed["wikitext"]["mean"], control["wikitext"]["mean"]) <= 1.20
        and changed["fineweb"]["mean"] <= .020
        and changed["wikitext"]["mean"] <= .025
    )
    pred_c = (
        all(split_overlap[str(rank)] >= .55 for rank in RANKS)
        and all(diagnostics[split]["upper_clip_fraction"] <= .25 for split in ("a", "b"))
        and all(summaries["a_consequence_448"][corpus]["max"]
                <= summaries["a_ordinary_448"][corpus]["max"]
                for corpus in ("fineweb", "wikitext"))
    )
    no_max_improvement = all(
        summaries[f"a_consequence_{rank}"][corpus]["max"]
        >= summaries[f"a_ordinary_{rank}"][corpus]["max"]
        for rank in RANKS for corpus in ("fineweb", "wikitext")
    )
    null = no_max_improvement or all(split_overlap[str(rank)] <= .25 for rank in RANKS)
    result = {
        "status": "mlp0_consequence_weighted_context_metric_screen_complete",
        "rung": 353,
        "claim_level": "split_fit_two_corpus_suffix_gradient_metric_screen_only",
        "metric": {"formula": "C_grad=sum clip(||dL/dy0||/median,.25,4) xxT / sum w",
                   "weight_clip": list(WEIGHT_CLIP)},
        "fit_rows": {"cache": "fineweb_n192_skip11000.pt",
                     "fit_a": list(FIT_A), "fit_b": list(FIT_B)},
        "evaluation": {
            "fineweb_cache": "fineweb_n192_skip7000.pt",
            "fineweb_rows_half_open": list(FINEWEB_EVAL),
            "wikitext103_train_token_span_half_open": [WIKI_SKIP,
                                                         WIKI_SKIP + EVAL_ROWS * 257],
            "dataset_fingerprint": fingerprint,
            "source_token_count": token_count,
        },
        "weight_and_fit_diagnostics": diagnostics,
        "consequence_split_subspace_overlap": split_overlap,
        "row_damage_summaries": summaries,
        'pred_a_gradient_p384_reduces_both_tail_distributions': bool(pred_a),
        'pred_b_gradient_p384_preserves_mean_prediction': bool(pred_b),
        'pred_c_metric_is_split_stable_clipped_and_p448_safe': bool(pred_c),
        "null_consequence_metric_has_no_replicable_advantage": bool(null),
        "stop_rule": "no_weight_clip_or_rank_tuning_after_result",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "split_overlap": split_overlap, "runtime_s": result["runtime_s"]},
                     indent=2), flush=True)
    print("MLP0 CONSEQUENCE-WEIGHTED CONTEXT METRIC SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
