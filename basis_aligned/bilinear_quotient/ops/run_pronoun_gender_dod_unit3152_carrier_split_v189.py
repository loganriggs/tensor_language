#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_embedding_carries_040 pred_c_mlp7_carries_020 pred_d_mlp6_carrier_negative pred_e_8_1_carries_little
"""Pronoun gender he/she DoD (v189): CARRIER split of the male-noun detector 3152's product contrast at the noun (v178's pair fold re-read the way v188
re-read 829). v178 (mass shares) said: embedding x MLP-7 interaction (63% of pair mass involves MLP 7, 57% the embedding), damped by MLP 6 (-21%),
head 8.1's copy 5% -- the story in for_logan/in_depth_circuit.md's appendix. v187 / v188 showed mass shares can misattribute (a constant write holds
mass; the change lives in its partner). Carrier share of writer w = [sum_b da_w mb_b + sum_a ma_a db_w] / contrast over aligned male - female pairs,
exact and summing to 1. The registered question: is the male detector's contrast carried by the noun token (embedding, 8.1's copy) or by the MLP stack?
PREDICTIONS (scored as written; failures preserved; priors from v178 and v188)
    pred_a_carrier_closure       carrier shares sum to 1 within 1e-3; per-pair identity within relative 1e-3
    pred_b_embedding_carries_040 embed carrier share >= 0.40 (the gender detectors were read as token products)
    pred_c_mlp7_carries_020      mlp:07 carrier share >= 0.20
    pred_d_mlp6_carrier_negative mlp:06 carrier share < 0 (v177's damping, if it is a carrier and not a constant multiplier)
    pred_e_8_1_carries_little    attnhead:08:1 carrier share <= 0.10 (8.1's copy was 22% of each factor but 5% of pair mass)
PRICE (registered maximum): 2 batches x 1 positional trace = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_unit3152_carrier_split_v189_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_unit3152_carrier_split_v189"
UNIT, CLOSURE_TOL, EMBED_MIN, MLP7_MIN, H81_MAX = 3152, 1e-3, 0.40, 0.20, 0.10
FORWARDS_MAX = 4
WRITERS = v16.WRITERS
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-3", "pred_b_embedding_carries_040": ">= 0.40", "pred_c_mlp7_carries_020": ">= 0.20", "pred_d_mlp6_carrier_negative": "< 0", "pred_e_8_1_carries_little": "<= 0.10"}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "embed_min": EMBED_MIN, "mlp7_min": MLP7_MIN, "h81_max": H81_MAX}}
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
    predictions = {"pred_a_carrier_closure": closure <= CLOSURE_TOL and abs(report["carrier_sum"] - 1) <= CLOSURE_TOL, "pred_b_embedding_carries_040": carrier_share["embed"] >= EMBED_MIN, "pred_c_mlp7_carries_020": carrier_share["mlp:07"] >= MLP7_MIN,
                   "pred_d_mlp6_carrier_negative": carrier_share["mlp:06"] < 0, "pred_e_8_1_carries_little": abs(carrier_share["attnhead:08:1"]) <= H81_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_unit3152_carrier_split_result_v189", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
