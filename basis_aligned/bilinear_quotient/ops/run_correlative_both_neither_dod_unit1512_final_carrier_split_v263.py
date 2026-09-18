#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_embedding_carries_040 pred_c_mlp_stack_at_most_040 pred_d_attention_at_most_020 pred_e_top_carrier_is_embedding
"""Correlative both/neither DoD (v263): what carries the neither-context detector, MLP-8 unit 1512, at the FINAL token? v260 / v262: 1512 carries 59% of MLP 8's write on
16.8's and - nor direction at the final, fires on neither-rows (-108 vs +2), and its removal costs 1.8% of the margin. At the final the cue (both / neither, four tokens
back) can only arrive by attention: head 8.1 copies cue tokens to the final query in every family (v146). Carrier split (exact; `dod_units.carrier_split` identity) of
u_1512's both - neither contrast at the final over the 48 aligned pairs of the v120 rows, writers = embedding (the final token, identical across a pair), attn / mlp
totals of blocks 0-7, the nine heads of block 8.
PREDICTIONS (scored as written; failures preserved)
    pred_a_carrier_closure         carrier shares sum to 1 within 1e-3; per-pair identity within relative 1e-3
    pred_b_embedding_carries_040   the embedding carries >= 0.40 -- registered to FAIL: the final token is the same in both members, so its carriage must be ~0
    pred_c_mlp_stack_at_most_040   mlp:00..07 together carry <= 0.40
    pred_d_attention_at_most_020   attention writers together carry <= 0.20 -- registered to FAIL if the cue arrives by attention (expected)
    pred_e_top_carrier_is_embedding  the embedding is the single largest carrier -- registered to FAIL; the reading is that head 8.1 or another attention writer leads
PRICE (registered maximum): 3 batches x 1 positional trace = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
import run_correlative_both_neither_dod_battery_v120 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/correlative_both_neither_dod_unit1512_final_carrier_split_v263_result.json"
CANDIDATE_ID = "correlative_both_neither.and_vs_nor.dod_unit1512_final_carrier_split_v263"
UNIT, CLOSURE_TOL, EMBED_MIN, MLP_MAX, ATTN_MAX = 1512, 1e-3, 0.40, 0.40, 0.20
FORWARDS_MAX = 4
WRITERS = v16.WRITERS
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-3", "pred_b_embedding_carries_040": ">= 0.40", "pred_c_mlp_stack_at_most_040": "<= 0.40", "pred_d_attention_at_most_020": "<= 0.20", "pred_e_top_carrier_is_embedding": "embed largest"}


def main() -> None:
    rows, he, she, *_ = g.build()          # he = and-token (positive), she = nor-token (negative): names kept
    noun_of = lambda row: row.final         # the FINAL position (the query)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "embed_min": EMBED_MIN, "mlp_max": MLP_MAX, "attn_max": ATTN_MAX}}
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
    mlps = sum(v for w, v in carrier_share.items() if w.startswith("mlp")); attn = sum(v for w, v in carrier_share.items() if w.startswith("attn")); report.update({"mlp_total": mlps, "attention_total": attn})
    print("mlps", round(mlps, 3), "attention", round(attn, 3), "embed", round(carrier_share["embed"], 3))
    predictions = {"pred_a_carrier_closure": closure <= CLOSURE_TOL and abs(report["carrier_sum"] - 1) <= CLOSURE_TOL, "pred_b_embedding_carries_040": carrier_share["embed"] >= EMBED_MIN, "pred_c_mlp_stack_at_most_040": mlps <= MLP_MAX,
                   "pred_d_attention_at_most_020": attn <= ATTN_MAX, "pred_e_top_carrier_is_embedding": ranked[0] == "embed"}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_unit3152_carrier_split_result_v263", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
