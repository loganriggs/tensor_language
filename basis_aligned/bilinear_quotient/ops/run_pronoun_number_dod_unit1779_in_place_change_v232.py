#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_in_place_change_has_response_sign pred_b_first_order_has_feed_sign pred_c_mean_terms_dominate pred_d_rms_effect_small
"""Pronoun number they/he DoD (v232): HOW unit 1779 pushes back. v228: removing the MLP-6 trio at the noun changes MLP-7 unit 1779 so that its contribution to the
plural detector 829 RISES (+0.038 of 829's contrast; 40% of MLP 7's response). v230 / v231: the trio feeds 1779 with the number (first-order, same sign) --
so the first-order prediction of the removal is a FALL of 1779's number contrast. The two agree only if the finite, mean-driven terms of the product change
dominate: with c = the trio's write as it reaches block 7 and x = the block-7 input at the noun,
    D = u_1779(x) - u_1779(x - c) = (L.c)(R.x) + (L.x)(R.c) - (L.c)(R.c)   over rms^2   (exact in-place, pooled plural - singular)
    first-order (carrier) term = da mB + mA db  with a = (L.c)/rms, b = (R.c)/rms  (v188's identity)
    mean-driven remainder = D - carrier  (the trio's pair-MEAN write times 1779's pair-DIFFERENCE partner, and the quadratic term).
Pooled over the 48 pairs, as fractions of 1779's native contrast (-715). Sign convention: a positive fraction = the removal reduces |contrast| in the native
direction; the v228 response corresponds to the removal moving 1779's contrast AGAINST its native sign (a negative fraction here).
PREDICTIONS (scored as written; failures preserved)
    pred_a_in_place_change_has_response_sign  D (rms recomputed), pooled, is < 0 (the removal pushes 1779's contrast against its native sign, as v228's response requires)
    pred_b_first_order_has_feed_sign          the carrier term is > 0 (first-order, the trio feeds 1779: v231)
    pred_c_mean_terms_dominate                |D - carrier| >= 1.5 x |carrier|
    pred_d_rms_effect_small                   |D(rms recomputed) - D(rms held)| <= 0.25 x |D(rms held)|
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_unit1779_in_place_change_v232_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_unit1779_in_place_change_v232"
UNITS, SRC, DST, UNIT, DOMINATE, RMS_MAX, BATCH = (2483, 2826, 4131), 6, 7, 1779, 1.5, 0.25, 32
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_in_place_change_has_response_sign": "< 0", "pred_b_first_order_has_feed_sign": "> 0", "pred_c_mean_terms_dominate": ">= 1.5 x", "pred_d_rms_effect_small": "<= 0.25"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"dominate": DOMINATE, "rms_max": RMS_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h
    Lrow, Rrow = blocks[DST].mlp.Left.weight.detach().float()[UNIT], blocks[DST].mlp.Right.weight.detach().float()[UNIT]
    D5 = blocks[SRC].mlp.Down.weight.detach().float(); scale = float(blocks[DST].lambdas[0]); ui = torch.tensor(list(UNITS), device=D5.device)
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
    remainder = D_held - carrier
    report = {"u1779_contrast": contrast, "carrier_first_order": carrier, "D_rms_held": D_held, "D_rms_recomputed": D_rms, "mean_driven_remainder": remainder}
    print({k: round(v, 4) for k, v in report.items()})
    predictions = {"pred_a_in_place_change_has_response_sign": D_rms < 0, "pred_b_first_order_has_feed_sign": carrier > 0, "pred_c_mean_terms_dominate": abs(remainder) >= DOMINATE * abs(carrier), "pred_d_rms_effect_small": abs(D_rms - D_held) <= RMS_MAX * abs(D_held)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_unit1779_in_place_change_result_v232", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
