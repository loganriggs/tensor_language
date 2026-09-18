#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linearization_remainder_small pred_c_reader_blocks_carry_most pred_d_later_mlps_oppose pred_e_pre_reader_blocks_small
"""Pronoun number they/he DoD (v224): RESPONSE CENSUS of a UNIT edit -- why do edits run 2-4x below their carrier shares? v195: zeroing the five MLP-5
number carriers {1036, 2403, 2538, 3274, 3034} at the noun removed 6% of u_2483 (carriage said 22%), 3.9% of u_829 and 1.3% of the they - he margin.
Here the exact lambda-recurrence split (v74 / v84's body) of the resid18 change at the FINAL token under that edit, over every block 0-17 (the edit
is at the noun; it reaches the final token only through attention, so blocks < 5 are unchanged and blocks 5-8 carry it before the readers do):
per-block linear attribution g . (w_l delta_l) with g the margin gradient at the native resid18, closure of the recurrence, and the remainder
against the exact margin change. DIRECT = the reader blocks attn:09, attn:10, attn:12, attn:15. The registered hypothesis for the 2-4x gap is
that later MLPs (9-17) respond against the change (compensation).
PREDICTIONS (scored as written; failures preserved; priors unsure except a / b from v84)
    pred_a_recurrence_closure         recurrence reconstructs the resid18 change within relative 1e-3, every row
    pred_b_linearization_remainder_small  |exact - linear| <= 0.10 x |exact| (mean)
    pred_c_reader_blocks_carry_most   the four reader attention blocks carry >= 0.60 of the linear total
    pred_d_later_mlps_oppose          the net of mlp:09..mlp:17 has the sign opposite to the reader term and |net| >= 0.10 x |reader term|
    pred_e_pre_reader_blocks_small    blocks 5-8 (attn + mlp at the final) carry <= 0.20 of the linear total in absolute value
PRICE (registered maximum): 3 batches x (native + edited trace) = 6 forwards; gradients on captured 1152-d vectors only; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_unit_edit_response_census_v224_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_unit_edit_response_census_v224"
UNITS, LAYER, BATCH = (1036, 2403, 2538, 3274, 3034), 5, 32
CLOSURE_TOL, REMAINDER_RATIO, DIRECT_MIN, OPPOSE_MIN, PRE_MAX = 1e-3, 0.10, 0.60, 0.10, 0.20
FORWARDS_MAX = 8
MODULES = [f"{kind}:{layer:02d}" for layer in range(18) for kind in ("attn", "mlp")]
DIRECT = ["attn:09", "attn:10", "attn:12", "attn:15"]; LATER_MLPS = [f"mlp:{l:02d}" for l in range(9, 18)]; PRE = [f"{k}:{l:02d}" for l in range(5, 9) for k in ("attn", "mlp")]
PREDICTIONS = {"pred_a_recurrence_closure": "<= 1e-3", "pred_b_linearization_remainder_small": "<= 0.10", "pred_c_reader_blocks_carry_most": ">= 0.60", "pred_d_later_mlps_oppose": "opposite sign, >= 0.10", "pred_e_pre_reader_blocks_small": "<= 0.20"}


def trace(backend, fw, chunk, edit_positions):
    """Per-row: attn/mlp outputs at row.final for every block, lambda0 per block, resid18; the block-LAYER hidden units zeroed at edit_positions(row) if given."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens = fw._tokens(chunk); fin = [r.final for r in chunk]; idx = torch.arange(len(chunk)); out = [dict() for _ in chunk]
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == LAYER and edit_positions is not None:
                h = dod_units.hidden(model, block.mlp, xin).clone(); ui = torch.tensor(list(UNITS), device=h.device)
                for i, row in enumerate(chunk): h[i, edit_positions(row), ui] = 0
                m = block.mlp.Down(h) + block.mlp.Down_bias
            else:
                m = block.mlp(xin)
            for i in range(len(chunk)):
                out[i][f"attn:{l:02d}"] = attention[i, fin[i]].detach().float().clone(); out[i][f"mlp:{l:02d}"] = m[i, fin[i]].detach().float().clone(); out[i][f"lambda0:{l:02d}"] = float(block.lambdas[0])
            x = x + m
        for i in range(len(chunk)): out[i]["resid18"] = x[i, fin[i]].detach().float().clone()
    return out


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "layer": LAYER, "modules": MODULES, "direct": DIRECT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "remainder_ratio": REMAINDER_RATIO, "direct_min": DIRECT_MIN, "oppose_min": OPPOSE_MIN, "pre_max": PRE_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); F, model = backend.F, backend.model
    fw = L.ManualForward(backend); forwards = 0; native, edited = [], []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        native.extend(trace(backend, fw, chunk, None)); forwards += 1
        edited.extend(trace(backend, fw, chunk, noun_of)); forwards += 1
    per_row = []
    for row, nat, ed in zip(rows, native, edited):
        weights = {}
        for layer in range(18):
            w = 1.0
            for later in range(layer + 1, 18): w *= nat[f"lambda0:{later:02d}"]
            weights[f"attn:{layer:02d}"] = w; weights[f"mlp:{layer:02d}"] = w
        deltas = {m: ed[m] - nat[m] for m in MODULES}
        recon = sum(weights[m] * deltas[m] for m in MODULES); true_delta = ed["resid18"] - nat["resid18"]
        closure = float((recon - true_delta).norm() / max(float(true_delta.norm()), 1e-9))
        x = nat["resid18"].clone().requires_grad_(True)
        lg = 30.0 * (model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30.0).tanh(); (lg[he] - lg[she]).backward(); gvec = x.grad.detach()
        sign = 1.0 if row.present else -1.0          # orient: change of the (they - he) margin toward the row's number
        exact = sign * (L.final_margin(model, F, ed["resid18"], he, she) - L.final_margin(model, F, nat["resid18"], he, she))
        attribution = {m: sign * float(gvec @ (weights[m] * deltas[m])) for m in MODULES}
        per_row.append({"row_id": row.row_id, "construction": row.construction, "closure": closure, "exact_change": exact, "linear_total": sum(attribution.values()), "attribution": attribution})
    n = len(per_row); mean = {m: sum(i["attribution"][m] for i in per_row) / n for m in MODULES}
    exact_mean = sum(i["exact_change"] for i in per_row) / n; linear_mean = sum(mean.values())
    direct = sum(mean[m] for m in DIRECT); later = sum(mean[m] for m in LATER_MLPS); pre = sum(mean[m] for m in PRE); other = linear_mean - direct - later - pre
    largest = max((m for m in MODULES if m not in DIRECT), key=lambda m: abs(mean[m])); max_closure = max(i["closure"] for i in per_row)
    summary = {"n": n, "exact_change_mean": exact_mean, "linear_total_mean": linear_mean, "remainder": exact_mean - linear_mean, "attribution_mean": mean, "reader_blocks": direct, "reader_share_of_linear": direct / linear_mean if linear_mean else None,
               "later_mlps_net": later, "pre_reader_blocks": pre, "other_net": other, "largest_non_reader_responder": largest, "max_closure": max_closure}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k != "attribution_mean"}); print({m: round(v, 4) for m, v in mean.items() if abs(v) > 1e-3})
    predictions = {"pred_a_recurrence_closure": max_closure <= CLOSURE_TOL, "pred_b_linearization_remainder_small": abs(summary["remainder"]) <= REMAINDER_RATIO * abs(exact_mean),
                   "pred_c_reader_blocks_carry_most": bool(summary["reader_share_of_linear"] is not None and summary["reader_share_of_linear"] >= DIRECT_MIN),
                   "pred_d_later_mlps_oppose": (later * direct < 0) and abs(later) >= OPPOSE_MIN * abs(direct), "pred_e_pre_reader_blocks_small": abs(pre) <= PRE_MAX * abs(linear_mean)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_unit_edit_response_census_result_v224", "candidate_id": CANDIDATE_ID, "plan": plan, "summary": summary, "per_row": per_row, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
