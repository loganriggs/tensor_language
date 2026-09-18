#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_gradient_closure pred_b_1779_sign_flips pred_c_1779_is_residual_aligned pred_d_mlp7_total_differs pred_e_top_units_change
"""Pronoun number they/he DoD (v237): MLP 7's unit carriers into the plural detector 829 with the TRUE gradient. v229 (fixed-rms carrier identity) made unit 1779
49% of MLP 7's relay into 829; v234-v236 showed 1779's write restores 829 when 1779's own contrast shrinks, and that the detector is scale-invariant (radial
writes invisible). The fixed-rms identity a_w = (L . C_w)/rms treats every writer as if the norm were fixed; a writer aligned with the residual then gets a contrast the
normalised detector never sees. Here per unit j: term_j = g . (lambda0_8 (h7_j^P - h7_j^S) Down7[:, j]) pooled over pairs with g = the exact gradient of u_829 through
rms_norm at the pair-mean block-8 input (first-order, scale-invariant), against v229's fixed-rms terms for the same units; also cos(Down7[:, 1779], x_8_hat) per row.
PREDICTIONS (scored as written; failures preserved)
    pred_a_gradient_closure          the true-gradient linearisation of MLP 7's whole plural - singular write reproduces the exact u_829 difference (with the other writers at their pair mean) within relative 0.25 (first-order; product is quadratic)
    pred_b_1779_sign_flips           unit 1779's true-gradient carrier has the sign OPPOSITE to its fixed-rms carrier (+424, v229)
    pred_c_1779_is_residual_aligned  mean |cos(Down7[:, 1779], x_8_hat)| over rows >= 0.30
    pred_d_mlp7_total_differs        |true-gradient MLP-7 total - fixed-rms MLP-7 total| >= 0.30 x |fixed-rms total| (858, v229)
    pred_e_top_units_change          fewer than 5 of v229's top-10 units are in the true-gradient top-10
PRICE (registered maximum): 3 batches x 1 forward = 3 forwards; gradients on captured 1152-d vectors only; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp7_units_true_gradient_v237_result.json"
V229 = ROOT / "circuits/followups/pronoun_number_dod_mlp7_unit_carriers_into_829_v229_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp7_units_true_gradient_v237"
UNIT, U1779, SRC, DST, CLOSURE_TOL, COS_MIN, DIFF_MIN, BATCH = 829, 1779, 7, 8, 0.25, 0.30, 0.30, 32
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_gradient_closure": "<= 0.25", "pred_b_1779_sign_flips": "opposite to +424", "pred_c_1779_is_residual_aligned": ">= 0.30", "pred_d_mlp7_total_differs": ">= 0.30 x 858", "pred_e_top_units_change": "< 5 of 10 shared"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    v229 = json.loads(V229.read_text()); fixed_total = v229["carrier_total"]; fixed_top = [j for j, _ in v229["top_units"][:10]]; fixed_1779 = dict((j, v) for j, v in v229["top_units"])[1779]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "cos_min": COS_MIN, "diff_min": DIFF_MIN}, "v229": {"total": fixed_total, "top10": fixed_top, "u1779": fixed_1779}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h
    Lrow, Rrow = blocks[DST].mlp.Left.weight.detach().float()[UNIT], blocks[DST].mlp.Right.weight.detach().float()[UNIT]
    D7 = blocks[SRC].mlp.Down.weight.detach().float(); lam = float(blocks[DST].lambdas[0])
    per_row, forwards = [], 0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk))
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_, h7, m7 = x, None, None, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == SRC: h7 = dod_units.hidden(model, block.mlp, xin)[idx, pos].float(); m7 = block.mlp(xin)[idx, pos].float()
                if l == DST: x8 = x[idx, pos].float(); break
                x = x + block.mlp(xin)
            forwards += 1
            for i in range(len(chunk)): per_row.append({"h7": h7[i].cpu(), "m7": m7[i].cpu(), "x8": x8[i].cpu()})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    dev = Lrow.device
    def u_of(x):
        xin = F.rms_norm(x, (x.shape[-1],)); return (Lrow @ xin) * (Rrow @ xin)
    terms = torch.zeros(D7.shape[1]); closure_num = closure_den = 0.0; cos_list = []
    for i, j in pairs:
        P, S = per_row[i], per_row[j]; xm = ((P["x8"] + S["x8"]) / 2).to(dev).clone().requires_grad_(True)
        u = u_of(xm); u.backward(); gvec = xm.grad.detach()
        dh = (P["h7"] - S["h7"]).to(dev); gD = (gvec @ D7) * lam; terms += (gD * dh).cpu()
        # closure: the exact difference of u_829 when only MLP 7's write differs (other writers at the pair mean) vs its linearisation
        dm = lam * (P["m7"] - S["m7"]).to(dev)
        with torch.no_grad(): exact = float(u_of(xm.detach() + dm / 2) - u_of(xm.detach() - dm / 2))
        closure_num += abs(float(gvec @ dm) - exact); closure_den += abs(exact)
        xh = P["x8"] / P["x8"].norm(); dcol = D7[:, U1779].cpu(); cos_list.append(abs(float(dcol @ xh)) / float(dcol.norm()))
    total = float(terms.sum()); order = torch.argsort(terms.abs(), descending=True); top = [(int(k), float(terms[k])) for k in order[:40]]
    shared = len(set(fixed_top) & {k for k, _ in top[:10]}); mean_cos = sum(cos_list) / len(cos_list)
    report = {"true_gradient_total": total, "fixed_rms_total": fixed_total, "u1779_true": float(terms[U1779]), "u1779_fixed": fixed_1779, "mean_abs_cos_down1779_x8": mean_cos, "closure_rel": closure_num / max(closure_den, 1e-9), "top_units": top, "shared_top10_with_v229": shared}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k != "top_units"}); print([(k, round(v, 1)) for k, v in top[:10]])
    predictions = {"pred_a_gradient_closure": report["closure_rel"] <= CLOSURE_TOL, "pred_b_1779_sign_flips": float(terms[U1779]) * fixed_1779 < 0, "pred_c_1779_is_residual_aligned": mean_cos >= COS_MIN,
                   "pred_d_mlp7_total_differs": abs(total - fixed_total) >= DIFF_MIN * abs(fixed_total), "pred_e_top_units_change": shared < 5}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp7_units_true_gradient_result_v237", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
