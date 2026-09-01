"""RUNG 324 -- MLP0 CONTEXT-METRIC SHARED-INPUT p512/p640 FRONTIER.

Context-covariance RRR repaired the late-layer Frobenius failure by 10--300x,
but multi-site composition paid the common damage tax.  Apply that metric to
MLP0 alone, where p768 weight SVD is already fully adopted and no new
multi-site interaction is introduced.

Fit p512/p640 paired Left/Right maps on two independent contextual MLP0-input
second moments from FineWeb skip11000 rows0:24 and 24:48.  Compare matched
weight SVD at identical rank and literal price.  Evaluate on untouched FineWeb
skip7000 rows176:188 and WikiText after token90000.

Frozen predictions
------------------
pred_a_context_p512_is_a_low_damage_price_point:
    Primary p512 RRR adds <=.012 on both corpora and its mean nonnegative
    damage is <=70% of matched p512 weight SVD.
pred_b_context_p640_is_near_adoption_quality:
    Primary p640 RRR adds <=.006 on both corpora and its mean nonnegative
    damage is <=70% of matched p640 weight SVD.
pred_c_both_ranks_are_split_stable:
    Whitened-subspace overlap is >=.70 at both ranks, while fit-B max damage is
    <=.015 at p512 and <=.009 at p640.

Null: both primary RRR ranks add >=.030 on at least one corpus, OR neither rank
improves matched SVD mean damage by >=20%.  This is a two-fit/two-corpus
single-site screen; a pass still needs physical mixed104 composition, census,
certificates, price, OOD, and signed intervention gates.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_context_metric_shared_input_frontier_results.json"
DEV = "cuda"
D = 1152
H = 4608
RANKS = (512, 640)
FIT_A = (0, 24)
FIT_B = (24, 48)
EVAL_SLICE = (176, 188)
WIKI_SKIP = 90000
NATIVE_MLP = 3 * H * D + D


def _price(rank: int) -> int:
    return D * rank + 2 * H * rank + H * D + D


def _mean_nonnegative(arm) -> float:
    return (max(0.0, arm["fineweb_damage"]) + max(0.0, arm["wikitext_damage"])) / 2


@torch.no_grad()
def _covariance(model, rows, manual_logits):
    total = torch.zeros(D, D, device=DEV)
    count = 0

    def hook(_module, args, _output):
        nonlocal count
        x = args[0].detach().reshape(-1, D).float()
        total.addmm_(x.T, x)
        count += x.shape[0]

    handle = model.transformer.h[0].mlp.register_forward_hook(hook)
    try:
        for start in range(0, len(rows), 2):
            manual_logits(model, rows[start:start + 2, :-1].to(DEV))
    finally:
        handle.remove()
    assert count == len(rows) * 256
    value = total / count
    return 0.5 * (value + value.T)


@torch.no_grad()
def _weight_program(mlp, rank):
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    stacked = torch.cat((left, right), dim=0)
    gram = stacked.T @ stacked
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    basis = vectors[:, torch.argsort(values, descending=True)[:rank]]
    coefficient = stacked @ basis
    return {
        "encoder": basis.T.contiguous(),
        "left": coefficient[:H].contiguous(),
        "right": coefficient[H:].contiguous(),
        "down": down,
        "bias": bias,
    }


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FIT_A[1] == FIT_B[0] and FIT_B[1] <= EVAL_SLICE[0]
        assert NATIVE_MLP - _price(512) == 5_308_416
        assert NATIVE_MLP - _price(640) == 3_981_312
        print("MLP0 CONTEXT-METRIC FRONTIER | dry run: splits, ranks, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mlp0_signed_response_rank_screen import _wikitext_rows
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits, _score
    from tier2_model import load_elriggs

    fit_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    eval_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    eval_cached = eval_cached["rows"] if isinstance(eval_cached, dict) else eval_cached
    fit_a = fit_cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    fit_b = fit_cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    fineweb = eval_cached[EVAL_SLICE[0]:EVAL_SLICE[1], :257].long().contiguous()
    wikitext, fingerprint = _wikitext_rows(12, skip=WIKI_SKIP)
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    covariance_a = _covariance(model, fit_a, _manual_logits)
    covariance_b = _covariance(model, fit_b, _manual_logits)
    native = {"fineweb": _score(model, fineweb), "wikitext": _score(model, wikitext)}

    arms = {}
    diagnostics = {}
    mlp = model.transformer.h[0].mlp
    for rank in RANKS:
        program_a, basis_a, diag_a = _rrr_program(mlp, covariance_a, rank=rank)
        program_b, basis_b, diag_b = _rrr_program(mlp, covariance_b, rank=rank)
        weight = _weight_program(mlp, rank)
        overlap = float((basis_a.T @ basis_b).square().sum() / rank)
        diagnostics[str(rank)] = {"fit_a": diag_a, "fit_b": diag_b,
                                  "whitened_subspace_overlap": overlap}
        arms[str(rank)] = {}
        for name, program in (("context_rrr_fit_a", program_a),
                              ("context_rrr_fit_b", program_b),
                              ("weight_svd", weight)):
            ce_fw = _score(model, fineweb, 0, program)
            ce_wt = _score(model, wikitext, 0, program)
            arm = {
                "fineweb_damage": ce_fw - native["fineweb"],
                "wikitext_damage": ce_wt - native["wikitext"],
                "literal_mlp0_scalars": _price(rank),
                "saving_scalars": NATIVE_MLP - _price(rank),
            }
            arms[str(rank)][name] = arm
            print(f"p{rank} {name}: FW/WT {arm['fineweb_damage']:+.6f}/"
                  f"{arm['wikitext_damage']:+.6f}", flush=True)
        del program_a, program_b, weight, basis_a, basis_b
        torch.cuda.empty_cache()

    p512 = arms["512"]
    p640 = arms["640"]
    pred_a = (max(p512["context_rrr_fit_a"]["fineweb_damage"],
                  p512["context_rrr_fit_a"]["wikitext_damage"]) <= .012
              and _mean_nonnegative(p512["context_rrr_fit_a"])
              <= .70 * _mean_nonnegative(p512["weight_svd"]))
    pred_b = (max(p640["context_rrr_fit_a"]["fineweb_damage"],
                  p640["context_rrr_fit_a"]["wikitext_damage"]) <= .006
              and _mean_nonnegative(p640["context_rrr_fit_a"])
              <= .70 * _mean_nonnegative(p640["weight_svd"]))
    pred_c = (all(diagnostics[str(rank)]["whitened_subspace_overlap"] >= .70
                  for rank in RANKS)
              and max(p512["context_rrr_fit_b"]["fineweb_damage"],
                      p512["context_rrr_fit_b"]["wikitext_damage"]) <= .015
              and max(p640["context_rrr_fit_b"]["fineweb_damage"],
                      p640["context_rrr_fit_b"]["wikitext_damage"]) <= .009)
    improving20 = [rank for rank in RANKS
                   if _mean_nonnegative(arms[str(rank)]["context_rrr_fit_a"])
                   <= .80 * _mean_nonnegative(arms[str(rank)]["weight_svd"])]
    null = (all(max(arms[str(rank)]["context_rrr_fit_a"]["fineweb_damage"],
                        arms[str(rank)]["context_rrr_fit_a"]["wikitext_damage"]) >= .030
                for rank in RANKS) or not improving20)
    result = {
        "status": "mlp0_context_metric_shared_input_frontier_complete",
        "rung": 324,
        "claim_level": "single_site_two_fit_two_corpus_context_metric_frontier_screen",
        "convention": "CE added above native; lower is better",
        "fit_rows": {"cache": "fineweb_n192_skip11000.pt", "fit_a": list(FIT_A),
                     "fit_b": list(FIT_B)},
        "evaluation_rows": {"fineweb_cache": "fineweb_n192_skip7000.pt",
                            "fineweb_slice": list(EVAL_SLICE), "wikitext_skip": WIKI_SKIP,
                            "wikitext_fingerprint": str(fingerprint)},
        "native_ce": native,
        "arms": arms,
        "diagnostics": diagnostics,
        "ranks_improving_weight_svd_by_20pct": improving20,
        'pred_a_context_p512_is_a_low_damage_price_point': bool(pred_a),
        'pred_b_context_p640_is_near_adoption_quality': bool(pred_b),
        'pred_c_both_ranks_are_split_stable': bool(pred_c),
        "null_context_metric_does_not_improve_mlp0": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"improving20": improving20,
                      "predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP0 CONTEXT-METRIC FRONTIER DONE", flush=True)


if __name__ == "__main__":
    main()
