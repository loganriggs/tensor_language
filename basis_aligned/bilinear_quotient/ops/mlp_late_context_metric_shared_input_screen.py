"""RUNG 321 -- CONTEXT-METRIC SHARED-INPUT RANK AT LATE DEPTH.

Rungs318/318B show that ordinary Frobenius SVD of [Left;Right] is blind to
late-layer functional load: p1024 fails badly at MLP15--17 while p1152 is
exact.  Test a genuinely different input-side object rather than another rank
or subset.  Fit reduced-rank regression under the empirical contextual MLP
input second moment C:

    A=[Left;Right],  G=C^(1/2) A^T A C^(1/2),
    encoder = V_p^T C^(-1/2), coefficient = A C^(1/2) V_p.

This minimizes paired linear-map error under contextual inputs, not Frobenius
weight error.  Fit independent programs on FineWeb skip11000 rows 0:24 and
24:48.  Evaluate both, plus matched weight SVD, on untouched FineWeb skip7000
rows 176:188 and WikiText after token 90000.  Rank and literal price remain
p768 and identical across methods.

Frozen predictions
------------------
pred_a_context_metric_opens_a_late_layer_regime:
    Primary fit-A RRR adds <=.08 on both corpora at >=2/3 late layers.
pred_b_context_metric_beats_frobenius_at_every_late_layer:
    At every layer, fit-A RRR mean nonnegative damage is <=70% of matched
    weight-SVD damage.
pred_c_context_metric_is_split_stable:
    At >=2/3 layers, whitened rank-768 subspace overlap between fit halves is
    >=.70 and the fit-A/fit-B mean-damage difference is <=.03.

Null: all primary RRR arms add >=.10 on at least one corpus, OR no layer
improves matched SVD mean damage by >=20%.  This is a split-fit two-corpus
screen only; any pass still needs a fixed physical composition, price,
certificates, OOD, and signed intervention transfer.
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
OUT = ROOT / "mlp_late_context_metric_shared_input_screen_results.json"
DEV = "cuda"
D = 1152
H = 4608
LAYERS = (15, 16, 17)
RANK = 768
FIT_A = (0, 24)
FIT_B = (24, 48)
EVAL_SLICE = (176, 188)
WIKI_SKIP = 90000


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
def _rrr_program(mlp, covariance):
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
    metric = 0.5 * (metric + metric.T)
    values, vectors = torch.linalg.eigh(metric)
    whitened_basis = vectors[:, torch.argsort(values, descending=True)[:RANK]]
    encoder = whitened_basis.T @ covariance_inv_sqrt
    coefficient = stacked @ covariance_sqrt @ whitened_basis
    program = {
        "encoder": encoder.contiguous(),
        "left": coefficient[:H].contiguous(),
        "right": coefficient[H:].contiguous(),
        "down": down,
        "bias": bias,
    }
    diagnostics = {
        "context_cov_top768_energy": float(values_c[:RANK].clamp_min(0).sum()
                                              / values_c.clamp_min(0).sum()),
        "context_cov_condition_after_floor": float(values_c[0] / safe[-1]),
        "covariance_floor": floor,
    }
    return program, whitened_basis, diagnostics


@torch.no_grad()
def _weight_program(mlp):
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    stacked = torch.cat((left, right), dim=0)
    gram = stacked.T @ stacked
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    basis = vectors[:, torch.argsort(values, descending=True)[:RANK]]
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
        saving = 2 * H * D - RANK * (D + 2 * H)
        assert saving == 2_654_208 and LAYERS == (15, 16, 17)
        print("LATE CONTEXT-METRIC INPUT RANK | dry run: splits, price, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mlp0_signed_response_rank_screen import _wikitext_rows
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
    assert fit_a.shape == fit_b.shape == (24, 257) and fineweb.shape == (12, 257)

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    covariance_a = _covariances(model, fit_a, _manual_logits)
    covariance_b = _covariances(model, fit_b, _manual_logits)
    native = {"fineweb": _score(model, fineweb), "wikitext": _score(model, wikitext)}

    arms = {}
    diagnostics = {}
    for layer in LAYERS:
        mlp = model.transformer.h[layer].mlp
        program_a, basis_a, diag_a = _rrr_program(mlp, covariance_a[layer])
        program_b, basis_b, diag_b = _rrr_program(mlp, covariance_b[layer])
        weight_program = _weight_program(mlp)
        overlap = float((basis_a.T @ basis_b).square().sum() / RANK)
        diagnostics[str(layer)] = {"fit_a": diag_a, "fit_b": diag_b,
                                   "whitened_subspace_overlap": overlap}
        arms[str(layer)] = {}
        for name, program in (("context_rrr_fit_a", program_a),
                              ("context_rrr_fit_b", program_b),
                              ("weight_svd", weight_program)):
            ce_fw = _score(model, fineweb, layer, program)
            ce_wt = _score(model, wikitext, layer, program)
            arm = {"fineweb_damage": ce_fw - native["fineweb"],
                   "wikitext_damage": ce_wt - native["wikitext"]}
            arms[str(layer)][name] = arm
            print(f"L{layer} {name}: FW/WT {arm['fineweb_damage']:+.6f}/"
                  f"{arm['wikitext_damage']:+.6f}", flush=True)
        del program_a, program_b, weight_program, basis_a, basis_b
        torch.cuda.empty_cache()

    primary_qualifying = [layer for layer in LAYERS
                          if max(arms[str(layer)]["context_rrr_fit_a"].values()) <= .08]
    all_dominate = all(
        _mean_nonnegative(arms[str(layer)]["context_rrr_fit_a"])
        <= .70 * _mean_nonnegative(arms[str(layer)]["weight_svd"])
        for layer in LAYERS
    )
    stable = []
    improving20 = []
    for layer in LAYERS:
        arm_a = arms[str(layer)]["context_rrr_fit_a"]
        arm_b = arms[str(layer)]["context_rrr_fit_b"]
        weight = arms[str(layer)]["weight_svd"]
        if (diagnostics[str(layer)]["whitened_subspace_overlap"] >= .70
                and abs(_mean_nonnegative(arm_a) - _mean_nonnegative(arm_b)) <= .03):
            stable.append(layer)
        if _mean_nonnegative(arm_a) <= .80 * _mean_nonnegative(weight):
            improving20.append(layer)
    pred_a = len(primary_qualifying) >= 2
    pred_b = all_dominate
    pred_c = len(stable) >= 2
    null = (all(max(arms[str(layer)]["context_rrr_fit_a"].values()) >= .10 for layer in LAYERS)
            or not improving20)
    result = {
        "status": "mlp_late_context_metric_shared_input_screen_complete",
        "rung": 321,
        "claim_level": "split_fit_late_layer_context_metric_two_corpus_screen_only",
        "convention": "CE added above native; lower is better",
        "fit_rows": {"cache": "fineweb_n192_skip11000.pt", "fit_a": list(FIT_A),
                     "fit_b": list(FIT_B)},
        "evaluation_rows": {"fineweb_cache": "fineweb_n192_skip7000.pt",
                            "fineweb_slice": list(EVAL_SLICE), "wikitext_skip": WIKI_SKIP,
                            "wikitext_fingerprint": str(fingerprint)},
        "native_ce": native,
        "rank": RANK,
        "saving_scalars_per_installed_layer": 2_654_208,
        "arms": arms,
        "diagnostics": diagnostics,
        "primary_qualifying_layers": primary_qualifying,
        "split_stable_layers": stable,
        "layers_improving_weight_svd_by_20pct": improving20,
        'pred_a_context_metric_opens_a_late_layer_regime': bool(pred_a),
        'pred_b_context_metric_beats_frobenius_at_every_late_layer': bool(pred_b),
        'pred_c_context_metric_is_split_stable': bool(pred_c),
        "null_context_metric_does_not_repair_late_depth": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualifying": primary_qualifying, "stable": stable, "improving20": improving20,
        "predicates": [pred_a, pred_b, pred_c], "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("LATE CONTEXT-METRIC INPUT RANK SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
