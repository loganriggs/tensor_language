#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_joint_leave_out_matches_edit pred_b_first_order_overshoots pred_c_cross_terms_explain_gap pred_d_rms_effect_small
"""Pronoun number they/he DoD (v226): second calibration of v225's rule -- the exact in-place joint leave-out D = u(x) - u(x - c), from the native trace,
predicts a unit edit. Here the MLP-6 trio {2483, 2826, 4131} into the plural detector 829 at the noun: v182's carrier / leave-one-out census, v196's edit
(-6.3% of u_829's contrast; first-order carriage of MLP 6 into 829 was 20% x 57% = 11%). Same quantities as v225 with c = the trio's summed write as it
reaches block 8 (scaled by lambda0 of blocks 7 and 8).
PREDICTIONS (scored as written; failures preserved)
    pred_a_joint_leave_out_matches_edit  D with rms recomputed, pooled, is within 0.02 of v196's -0.063
    pred_b_first_order_overshoots        the carrier term (first-order) is >= 1.5 x |D| in magnitude
    pred_c_cross_terms_explain_gap       carrier - D (rms held) accounts for >= 0.70 of the carrier - edit gap
    pred_d_rms_effect_small              |D(rms recomputed) - D(rms held)| <= 0.10 x |D(rms held)|
PRICE (registered maximum): 3 batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_joint_leave_out_mlp6_v226_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_joint_leave_out_mlp6_v226"
UNITS, SRC, DST, UNIT, V195_EDIT, MATCH_TOL, OVERSHOOT, EXPLAIN_MIN, RMS_MAX, BATCH = (2483, 2826, 4131), 6, 8, 829, -0.0633, 0.02, 1.5, 0.70, 0.10, 32
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_joint_leave_out_matches_edit": "within 0.02 of -0.063", "pred_b_first_order_overshoots": ">= 1.5 x", "pred_c_cross_terms_explain_gap": ">= 0.70 of the gap", "pred_d_rms_effect_small": "<= 0.10"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"match_tol": MATCH_TOL, "overshoot": OVERSHOOT, "explain_min": EXPLAIN_MIN, "rms_max": RMS_MAX, "v196_edit": V195_EDIT}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h
    Lrow, Rrow = blocks[DST].mlp.Left.weight.detach().float()[UNIT], blocks[DST].mlp.Right.weight.detach().float()[UNIT]
    D5 = blocks[SRC].mlp.Down.weight.detach().float(); scale = 1.0
    for l in range(SRC + 1, DST + 1): scale *= float(blocks[l].lambdas[0])
    ui = torch.tensor(list(UNITS), device=D5.device)
    per_row, forwards = [], 0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk))
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_, h5 = x, None, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == SRC: h5 = dod_units.hidden(model, block.mlp, xin)[idx, pos].float()
                if l == DST: x6 = x[idx, pos].float(); break
                x = x + block.mlp(xin)
            forwards += 1
            for i in range(len(chunk)):
                xi = x6[i].to(Lrow.device); c = scale * (D5[:, ui] @ h5[i].to(Lrow.device)[ui])
                rms = float(xi.pow(2).mean().sqrt()); rms_c = float((xi - c).pow(2).mean().sqrt())
                Lx, Rx, Lc, Rc = float(Lrow @ xi), float(Rrow @ xi), float(Lrow @ c), float(Rrow @ c)
                u = Lx * Rx / rms ** 2; u_edit_held = (Lx - Lc) * (Rx - Rc) / rms ** 2; u_edit_rms = (Lx - Lc) * (Rx - Rc) / rms_c ** 2
                per_row.append({"u": u, "D_held": u - u_edit_held, "D_rms": u - u_edit_rms, "a": Lc / rms, "b": Rc / rms, "A": Lx / rms, "B": Rx / rms})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    contrast = sum(per_row[i]["u"] - per_row[j]["u"] for i, j in pairs)
    D_held = sum(per_row[i]["D_held"] - per_row[j]["D_held"] for i, j in pairs) / contrast; D_rms = sum(per_row[i]["D_rms"] - per_row[j]["D_rms"] for i, j in pairs) / contrast
    carrier = sum((per_row[i]["a"] - per_row[j]["a"]) * (per_row[i]["B"] + per_row[j]["B"]) / 2 + (per_row[i]["A"] + per_row[j]["A"]) / 2 * (per_row[i]["b"] - per_row[j]["b"]) for i, j in pairs) / contrast
    gap = carrier - (-V195_EDIT); explained = (carrier - D_held) / gap if gap else float("nan")
    report = {"u2483_contrast": contrast, "carrier_first_order": carrier, "D_joint_rms_held": D_held, "D_joint_rms_recomputed": D_rms, "v196_edit": -V195_EDIT, "gap_carrier_minus_edit": gap, "cross_terms_share_of_gap": explained}
    print({k: round(v, 4) for k, v in report.items()})
    predictions = {"pred_a_joint_leave_out_matches_edit": abs(D_rms - (-V195_EDIT)) <= MATCH_TOL, "pred_b_first_order_overshoots": abs(carrier) >= OVERSHOOT * abs(D_held), "pred_c_cross_terms_explain_gap": explained >= EXPLAIN_MIN, "pred_d_rms_effect_small": abs(D_rms - D_held) <= RMS_MAX * abs(D_held)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_joint_leave_out_mlp6_result_v226", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
