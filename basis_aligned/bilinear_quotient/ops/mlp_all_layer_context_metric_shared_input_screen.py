"""RUNG 366 -- ALL-LAYER CONTEXT-METRIC SHARED BILINEAR INPUT SCREEN.

Extend the successful paired-Left/Right contextual reduced-rank regression
from MLP0 and layers15--17 to every MLP independently.  Fit two disjoint
context covariances, test ranks512/768 on two untouched corpora, and rescore
the fixed matched Frobenius weight-SVD construction on the same rows.

Frozen predictions
------------------
pred_a_context_metric_shared_input_rank_is_broad:
    p768 adds <=.02 on both corpora at >=14/18 layers and p512 adds <=.04 at
    >=12/18 layers.
pred_b_context_metric_beats_matched_weight_svd_broadly:
    p768 mean-nonnegative damage <=50% of weight SVD at layers15--17 and
    <=80% at >=15/18 layers.
pred_c_context_metric_is_split_stable_and_priced:
    p768 overlap>=.70 and fit mean gap<=.02 at >=14/18; p768 no worse than
    p512 at every p512 qualifier; exact ranks, shapes, splits, and prices.

Null: <=8 p768 qualifiers or any p768 layer/corpus damage >=.20.  Screen
only; a pass licenses a CPU allocator and one frozen-rule composition.
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
OUT = ROOT / "mlp_all_layer_context_metric_shared_input_screen_results.json"
DEV = "cuda"
D = 1152
H = 4608
LAYERS = tuple(range(18))
RANKS = (512, 768)
FIT_A = (0, 24)
FIT_B = (24, 48)
FINEWEB_SLICE = (160, 176)
WIKI_SKIP = 205_600
EVAL_ROWS = 16
WIKI_STOP = WIKI_SKIP + EVAL_ROWS * 257
NATIVE_MLP = 3 * H * D + D


def _price(rank: int) -> int:
    return D * rank + 2 * H * rank + H * D + D


def _mean_nonnegative(arm) -> float:
    return (max(0.0, arm["fineweb_damage"]) + max(0.0, arm["wikitext_damage"])) / 2


@torch.no_grad()
def _covariances(model, rows, manual_logits):
    sums = {layer: torch.zeros(D, D, device=DEV) for layer in LAYERS}
    counts = {layer: 0 for layer in LAYERS}
    handles = []
    for layer in LAYERS:
        def hook(_module, args, _output, layer=layer):
            x = args[0].detach().reshape(-1, D).float()
            sums[layer].addmm_(x.T, x)
            counts[layer] += x.shape[0]
        handles.append(model.transformer.h[layer].mlp.register_forward_hook(hook))
    try:
        for start in range(0, len(rows), 2):
            manual_logits(model, rows[start:start + 2, :-1].to(DEV))
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer in LAYERS:
        assert counts[layer] == len(rows) * 256
        covariance = sums[layer] / counts[layer]
        result[layer] = 0.5 * (covariance + covariance.T)
    return result


@torch.no_grad()
def _rrr_programs(mlp, covariance):
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    stacked = torch.cat((left, right), dim=0)
    values_c, vectors_c = torch.linalg.eigh(covariance)
    order_c = torch.argsort(values_c, descending=True)
    values_c, vectors_c = values_c[order_c], vectors_c[:, order_c]
    floor = float(values_c[0]) * 1e-6
    safe = values_c.clamp_min(floor)
    covariance_sqrt = (vectors_c * safe.sqrt()) @ vectors_c.T
    covariance_inv_sqrt = (vectors_c * safe.rsqrt()) @ vectors_c.T
    gram = stacked.T @ stacked
    metric = covariance_sqrt @ gram @ covariance_sqrt
    values, vectors = torch.linalg.eigh(0.5 * (metric + metric.T))
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    total = values.clamp_min(0).sum()
    programs, bases, diagnostics = {}, {}, {}
    for rank in RANKS:
        basis = vectors[:, :rank]
        encoder = basis.T @ covariance_inv_sqrt
        coefficient = stacked @ covariance_sqrt @ basis
        programs[rank] = {
            "encoder": encoder.contiguous(),
            "left": coefficient[:H].contiguous(),
            "right": coefficient[H:].contiguous(),
            "down": down,
            "bias": bias,
        }
        bases[rank] = basis
        diagnostics[str(rank)] = {
            "metric_retained_energy": float(values[:rank].clamp_min(0).sum() / total),
            "covariance_floor": floor,
            "covariance_condition_after_floor": float(values_c[0] / safe[-1]),
        }
    return programs, bases, diagnostics


@torch.no_grad()
def _weight_programs(mlp):
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    stacked = torch.cat((left, right), dim=0)
    gram = stacked.T @ stacked
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    vectors = vectors[:, torch.argsort(values, descending=True)]
    programs = {}
    for rank in RANKS:
        basis = vectors[:, :rank]
        coefficient = stacked @ basis
        programs[rank] = {
            "encoder": basis.T.contiguous(),
            "left": coefficient[:H].contiguous(),
            "right": coefficient[H:].contiguous(),
            "down": down,
            "bias": bias,
        }
    return programs


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip11000.pt").exists()
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FIT_A[1] == FIT_B[0] and FIT_B[1] <= FINEWEB_SLICE[0]
        assert WIKI_STOP == 209_712
        assert NATIVE_MLP - _price(512) == 5_308_416
        assert NATIVE_MLP - _price(768) == 2_654_208
        print("ALL-LAYER CONTEXT-METRIC MLP | dry run: splits, ranks, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp_shared_input_svd_all_layers_screen import _manual_logits, _score
    from tier2_model import load_elriggs

    fit_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip11000.pt", map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    eval_cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    eval_cached = eval_cached["rows"] if isinstance(eval_cached, dict) else eval_cached
    fit_a = fit_cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    fit_b = fit_cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    fineweb = eval_cached[FINEWEB_SLICE[0]:FINEWEB_SLICE[1], :257].long().contiguous()
    wikitext, fingerprint, token_count = _wikitext103_train_rows(
        n=EVAL_ROWS, width=257, skip=WIKI_SKIP)
    assert fit_a.shape == fit_b.shape == (24, 257)
    assert fineweb.shape == wikitext.shape == (EVAL_ROWS, 257)

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == len(LAYERS)
    covariance_a = _covariances(model, fit_a, _manual_logits)
    covariance_b = _covariances(model, fit_b, _manual_logits)
    native = {"fineweb": _score(model, fineweb), "wikitext": _score(model, wikitext)}

    arms, diagnostics = {}, {}
    for layer in LAYERS:
        mlp = model.transformer.h[layer].mlp
        programs_a, bases_a, diag_a = _rrr_programs(mlp, covariance_a[layer])
        programs_b, bases_b, diag_b = _rrr_programs(mlp, covariance_b[layer])
        weight_programs = _weight_programs(mlp)
        arms[str(layer)] = {}
        diagnostics[str(layer)] = {}
        for rank in RANKS:
            for name, program in (("context_rrr_fit_a", programs_a[rank]),
                                  ("context_rrr_fit_b", programs_b[rank]),
                                  ("weight_svd", weight_programs[rank])):
                expected_shapes = {
                    "encoder": (rank, D), "left": (H, rank), "right": (H, rank),
                    "down": (D, H), "bias": (D,),
                }
                assert all(tuple(program[key].shape) == shape
                           for key, shape in expected_shapes.items())
                fw = _score(model, fineweb, layer, program) - native["fineweb"]
                wt = _score(model, wikitext, layer, program) - native["wikitext"]
                arms[str(layer)][f"{name}_p{rank}"] = {
                    "fineweb_damage": fw, "wikitext_damage": wt,
                    "saving_scalars_if_installed": NATIVE_MLP - _price(rank),
                }
                print(f"L{layer:02d} {name} p{rank}: FW/WT {fw:+.6f}/{wt:+.6f}",
                      flush=True)
            diagnostics[str(layer)][str(rank)] = {
                "fit_a": diag_a[str(rank)], "fit_b": diag_b[str(rank)],
                "whitened_subspace_overlap": float(
                    (bases_a[rank].T @ bases_b[rank]).square().sum() / rank),
            }
        del programs_a, programs_b, weight_programs, bases_a, bases_b
        torch.cuda.empty_cache()

    qualifiers = {}
    for rank, bar in ((512, .04), (768, .02)):
        qualifiers[str(rank)] = [layer for layer in LAYERS
                                 if max(arms[str(layer)][f"context_rrr_fit_a_p{rank}"]["fineweb_damage"],
                                        arms[str(layer)][f"context_rrr_fit_a_p{rank}"]["wikitext_damage"])
                                 <= bar]
    dominance80, late50, stable768 = [], [], []
    for layer in LAYERS:
        context768 = _mean_nonnegative(arms[str(layer)]["context_rrr_fit_a_p768"])
        weight768 = _mean_nonnegative(arms[str(layer)]["weight_svd_p768"])
        if context768 <= .80 * weight768:
            dominance80.append(layer)
        if layer in (15, 16, 17) and context768 <= .50 * weight768:
            late50.append(layer)
        fit_a = arms[str(layer)]["context_rrr_fit_a_p768"]
        fit_b = arms[str(layer)]["context_rrr_fit_b_p768"]
        if (diagnostics[str(layer)]["768"]["whitened_subspace_overlap"] >= .70
                and abs(_mean_nonnegative(fit_a) - _mean_nonnegative(fit_b)) <= .02):
            stable768.append(layer)
    monotone = all(
        _mean_nonnegative(arms[str(layer)]["context_rrr_fit_a_p768"])
        <= _mean_nonnegative(arms[str(layer)]["context_rrr_fit_a_p512"])
        for layer in qualifiers["512"]
    )
    p768_max = max(max(arms[str(layer)]["context_rrr_fit_a_p768"]["fineweb_damage"],
                       arms[str(layer)]["context_rrr_fit_a_p768"]["wikitext_damage"])
                   for layer in LAYERS)
    pred_a = len(qualifiers["768"]) >= 14 and len(qualifiers["512"]) >= 12
    pred_b = len(late50) == 3 and len(dominance80) >= 15
    pred_c = (len(stable768) >= 14 and monotone
              and NATIVE_MLP - _price(512) == 5_308_416
              and NATIVE_MLP - _price(768) == 2_654_208)
    null = len(qualifiers["768"]) <= 8 or p768_max >= .20
    result = {
        "status": "mlp_all_layer_context_metric_shared_input_screen_complete",
        "rung": 366,
        "claim_level": "all_layer_split_fit_context_metric_two_corpus_screen_only",
        "convention": "CE added above native; lower is better",
        "fit_rows": {"cache": "fineweb_n192_skip11000.pt", "fit_a": list(FIT_A),
                     "fit_b": list(FIT_B)},
        "evaluation": {"fineweb_cache": "fineweb_n192_skip7000.pt",
                       "fineweb_rows_half_open": list(FINEWEB_SLICE),
                       "wikitext103_train_span_half_open": [WIKI_SKIP, WIKI_STOP],
                       "wikitext_fingerprint": fingerprint, "source_tokens": token_count},
        "native_ce": native,
        "ranks": list(RANKS),
        "saving_scalars_per_installed_layer": {
            "512": NATIVE_MLP - _price(512), "768": NATIVE_MLP - _price(768)},
        "arms": arms,
        "diagnostics": diagnostics,
        "qualifying_layers": qualifiers,
        "p768_layers_beating_weight_svd_by_20pct": dominance80,
        "late_layers_beating_weight_svd_by_50pct": late50,
        "p768_split_stable_layers": stable768,
        "p768_no_worse_at_every_p512_qualifier": monotone,
        "p768_max_damage": p768_max,
        'pred_a_context_metric_shared_input_rank_is_broad': bool(pred_a),
        'pred_b_context_metric_beats_matched_weight_svd_broadly': bool(pred_b),
        'pred_c_context_metric_is_split_stable_and_priced': bool(pred_c),
        "null_context_metric_shared_input_rank_is_not_broad": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualifiers": qualifiers, "dominance80": dominance80, "late50": late50,
        "stable768": stable768, "p768_max": p768_max,
        "predicates": [pred_a, pred_b, pred_c], "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ALL-LAYER CONTEXT-METRIC MLP SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
