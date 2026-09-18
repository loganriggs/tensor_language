#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linearization_remainder_small pred_c_direct_terms_carry_most pred_d_downstream_net_is_small pred_e_largest_responder_is_an_mlp
"""Pronoun gender he/she DoD (v74): RESPONSE CENSUS of the set removal (better_circuits §3.3) on the v71 fresh rows.

Lane: Claude circuit lane. Parents: v71 (set {10.1, 9.6, 12.4, 15.1} removes 0.96-1.11 of the margin on fresh rows; keep-only
raises the margin above native), v73 (77% on natural rows). The auxiliary families showed a 40-60% MLP relay share (v6); v71's
whole-margin removal suggests the pronoun set is read almost directly by the unembedding. This run splits the resid18 change
exactly by the residual recurrence x_{l+1} = lambda0_l x_l + lambda1_l x_0 + attn_l + mlp_l (blocks 9-17), attributes the margin
linearly (gradient of he-she at the native resid18), and reports the nonlinear remainder. DIRECT terms = the attention modules of
the set's own blocks (attn:09, attn:10, attn:12, attn:15); the caveat that each of these block deltas also contains the other
heads' response to the earlier edits is stated, not hidden (v6 had the same caveat within one block). Evidence tag: response.

PREDICTIONS (scored as written; failures preserved)
    pred_a_recurrence_closure              lambda-weighted module deltas equal the resid18 delta within relative L2 1e-3 per row
    pred_b_linearization_remainder_small   |mean exact change - mean linear total| <= 0.10 x |mean exact change|
    pred_c_direct_terms_carry_most         direct terms >= 0.80 of the pooled linear attribution (registered from v71's keep-only > 1)
    pred_d_downstream_net_is_small         |downstream net| <= 0.25 x direct
    pred_e_largest_responder_is_an_mlp     the largest |downstream| responder is an MLP module. Prior: unsure.

PRICE (registered maximum): 2 batches x (native trace + edited trace) = 4 forwards; gradients on captured 1152-d vectors only;
0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_gender_dod_battery_v71 as v71
import dod_battery

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/pronoun_gender_dod_response_census_v74_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_response_census_v74"
BATCH, FROM = 32, 9
CLOSURE_TOL, REMAINDER_RATIO, DIRECT_MIN, DOWNSTREAM_MAX = 1e-3, 0.10, 0.80, 0.25
FORWARDS_MAX = 8
MODULES = [f"{kind}:{layer:02d}" for layer in range(FROM, 18) for kind in ("attn", "mlp")]
DIRECT = [f"attn:{l:02d}" for l in sorted({l for l, _ in v71.HEADS})]
PREDICTIONS = {"pred_a_recurrence_closure": "<= 1e-3", "pred_b_linearization_remainder_small": "<= 0.10", "pred_c_direct_terms_carry_most": ">= 0.80",
               "pred_d_downstream_net_is_small": "<= 0.25 x direct", "pred_e_largest_responder_is_an_mlp": "mlp:*"}


def main() -> None:
    rows, he, she = v71.build()
    SET = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, v71.HEADS).set_components()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "set": list(v71.HEADS), "modules": MODULES, "direct": DIRECT,
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "remainder_ratio": REMAINDER_RATIO, "direct_min": DIRECT_MIN, "downstream_max": DOWNSTREAM_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    F, model = backend.F, backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, SET, he, she)
    forwards = 0
    native, edited = [], []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        native.extend(L.forward_trace(fw, chunk, from_layer=FROM)); forwards += 1
        edited.extend(L.forward_trace(fw, chunk, components=SET, mode="project", from_layer=FROM)); forwards += 1
    per_row = []
    for row, nat, ed in zip(rows, native, edited):
        weights = {}
        for layer in range(FROM, 18):
            w = 1.0
            for later in range(layer + 1, 18):
                w *= nat[f"lambda0:{later:02d}"]
            weights[f"attn:{layer:02d}"] = w; weights[f"mlp:{layer:02d}"] = w
        deltas = {m: ed[m] - nat[m] for m in MODULES}
        recon = sum(weights[m] * deltas[m] for m in MODULES)
        true_delta = ed["resid18"] - nat["resid18"]
        closure = float((recon - true_delta).norm() / true_delta.norm())
        x = nat["resid18"].clone().requires_grad_(True)
        l = 30.0 * (model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30.0).tanh()
        (l[row.answer_id] - l[row.foil_id]).backward()
        g = x.grad.detach()
        exact = L.final_margin(model, F, ed["resid18"], row.answer_id, row.foil_id) - L.final_margin(model, F, nat["resid18"], row.answer_id, row.foil_id)
        attribution = {m: float(g @ (weights[m] * deltas[m])) for m in MODULES}
        per_row.append({"row_id": row.row_id, "construction": row.construction, "closure": closure, "exact_change": exact, "linear_total": sum(attribution.values()), "attribution": attribution})
    n = len(per_row)
    mean = {m: sum(i["attribution"][m] for i in per_row) / n for m in MODULES}
    exact_mean = sum(i["exact_change"] for i in per_row) / n; linear_mean = sum(mean.values())
    direct = sum(mean[m] for m in DIRECT); downstream = linear_mean - direct
    down = {m: v for m, v in mean.items() if m not in DIRECT}; largest = max(down, key=lambda m: abs(down[m]))
    max_closure = max(i["closure"] for i in per_row)
    summary = {"n": n, "exact_change_mean": exact_mean, "linear_total_mean": linear_mean, "remainder": exact_mean - linear_mean, "attribution_mean": mean,
               "direct_mean": direct, "direct_share_of_linear": direct / linear_mean if linear_mean else None, "downstream_net_mean": downstream, "largest_downstream_responder": largest}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k != "attribution_mean"}, indent=1)); print({m: round(v, 4) for m, v in mean.items()})
    predictions = {"pred_a_recurrence_closure": max_closure <= CLOSURE_TOL, "pred_b_linearization_remainder_small": abs(summary["remainder"]) <= REMAINDER_RATIO * abs(exact_mean),
                   "pred_c_direct_terms_carry_most": summary["direct_share_of_linear"] is not None and summary["direct_share_of_linear"] >= DIRECT_MIN,
                   "pred_d_downstream_net_is_small": abs(downstream) <= DOWNSTREAM_MAX * abs(direct), "pred_e_largest_responder_is_an_mlp": largest.startswith("mlp")}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_response_census_result_v74", "candidate_id": CANDIDATE_ID, "plan": plan, "max_closure_relative_error": max_closure, "summary": summary,
                               "per_row": per_row, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
