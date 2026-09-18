#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_mlp6_carries_half pred_c_embedding_carries_020 pred_d_attn6_and_8_1_carry_little
"""Pronoun number they/he DoD (v188): CARRIER SPLIT of the plural detector 829's pair fold (v181). v187 showed a writer can hold a large pair share
(head 6.3, 32% of 2483) with a write that is constant across the contrast -- pair shares measure product mass, not who carries the difference.
For every aligned pair (P, S), with a_w = (L . C_w)/rms and b_w = (R . C_w)/rms per writer, the contrast of the pair term is, exactly,
    T_ab(P) - T_ab(S) = da_a mb_b + ma_a db_b + (da_a db_b - (da_a db_b)) ... written as da_a mb_b + ma_a db_b with m = pair mean, d = P - S
(the identity  xP yP - xS yS = dx my + mx dy  holds exactly for pair means). Writer w's CARRIER share = [sum_b da_w mb_b + sum_a ma_a db_w] / contrast:
the part of the contrast that exists because w's own write changed. Sums to 1 over writers (closure). Compared with v181's mass shares.
PREDICTIONS (scored as written; failures preserved; priors from v181-v187)
    pred_a_carrier_closure           carrier shares sum to 1 within 1e-3 and per-pair identity holds within relative 1e-3
    pred_b_mlp6_carries_half         mlp:06 carrier share >= 0.50 (mass share was 0.45; v182-v187 say the number is MLP-borne)
    pred_c_embedding_carries_020     embed carrier share >= 0.20 (the noun token itself changes between plural and singular)
    pred_d_attn6_and_8_1_carry_little attn:06 and attnhead:08:1 carrier shares each <= 0.05 (constant multipliers, as 6.3 into 2483)
PRICE (registered maximum): 2 batches x 1 positional trace = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_unit829_carrier_split_v188_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_unit829_carrier_split_v188"
UNIT, CLOSURE_TOL, MLP6_MIN, EMBED_MIN, LITTLE_MAX = 829, 1e-3, 0.50, 0.20, 0.05
FORWARDS_MAX = 4
WRITERS = v16.WRITERS
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-3", "pred_b_mlp6_carries_half": ">= 0.50", "pred_c_embedding_carries_020": ">= 0.20", "pred_d_attn6_and_8_1_carry_little": "<= 0.05 x 2"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "mlp6_min": MLP6_MIN, "embed_min": EMBED_MIN, "little_max": LITTLE_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend)
    mlp = model.transformer.h[8].mlp; Lrow, Rrow = mlp.Left.weight.detach().float()[UNIT], mlp.Right.weight.detach().float()[UNIT]
    forwards, traces = 0, []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: [noun_of(rw)], upto_layer=9, head_write_layers=(8,))); forwards += 1
    n = len(WRITERS); closure, factors = 0.0, []
    for row, tr in zip(rows, traces):
        pos = noun_of(row); C = v16.writers_at_block8_input(tr, pos)
        x = sum(C.values()); rms = float(x.pow(2).mean().sqrt())
        M = torch.stack([C[w] for w in WRITERS]).to(Lrow.device)
        factors.append(((M @ Lrow) / rms, (M @ Rrow) / rms))
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    carrier, mass = torch.zeros(n), torch.zeros(n); contrast = 0.0
    for i, row in enumerate(rows):
        if not row.present: continue
        j = partner[(row.construction, row.group, False)]
        aP, bP = factors[i]; aS, bS = factors[j]
        ref = float(aP.sum() * bP.sum() - aS.sum() * bS.sum())
        da, db, ma, mb = aP - aS, bP - bS, (aP + aS) / 2, (bP + bS) / 2
        left, right = da * float(mb.sum()), db * float(ma.sum())          # per-writer: own-change x mean partner total, for each factor
        closure = max(closure, abs(float(left.sum() + right.sum()) - ref) / max(abs(ref), 1e-6))
        carrier += (left + right).cpu(); contrast += ref
        Tm = torch.outer(aP, bP) - torch.outer(aS, bS); mass += ((Tm + Tm.T) / 2).sum(1).cpu()
    carrier_share = {w: float(carrier[k]) / contrast for k, w in enumerate(WRITERS)}
    mass_share = {w: float(mass[k]) / contrast for k, w in enumerate(WRITERS)}            # = v181's factor_share (pairs containing w, diagonal once)
    ranked = sorted(carrier_share, key=lambda w: -abs(carrier_share[w]))
    report = {"contrast": contrast, "carrier_share": carrier_share, "mass_share": mass_share, "carrier_sum": sum(carrier_share.values()), "ranked": ranked}
    print("contrast", round(contrast, 2), "carrier sum", round(report["carrier_sum"], 4)); print("carrier", [(w, round(carrier_share[w], 3), "mass", round(mass_share[w], 3)) for w in ranked[:10]])
    predictions = {"pred_a_carrier_closure": closure <= CLOSURE_TOL and abs(report["carrier_sum"] - 1) <= CLOSURE_TOL, "pred_b_mlp6_carries_half": carrier_share["mlp:06"] >= MLP6_MIN, "pred_c_embedding_carries_020": carrier_share["embed"] >= EMBED_MIN,
                   "pred_d_attn6_and_8_1_carry_little": abs(carrier_share["attn:06"]) <= LITTLE_MAX and abs(carrier_share["attnhead:08:1"]) <= LITTLE_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_unit829_carrier_split_result_v188", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
