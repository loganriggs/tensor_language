#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_linear_closure pred_b_top_10_units_carry_half pred_c_top_50_units_carry_080 pred_d_largest_unit_supports
"""Pronoun number they/he DoD (v184): one more layer down the number chain. v182 named MLP-6 unit 2483 as the largest MLP-6 input of the plural
detector 829; v183 showed 2483's product at the noun is built on MLP 4 (52% of pair mass) and MLP 3 / MLP 5 (33% / 9%), not the embedding (20%).
Which MLP-5 and MLP-4 units? `dod_units.product_unit_census` (the v182 body, sign-carrying: exact leave-one-unit-out change of u_2483 at the
noun with rms held) for source blocks 5 and 4 into unit 2483 of block 6, pooled plural - singular over aligned v76 pairs.
PREDICTIONS (scored as written; failures preserved; priors unsure): pred_a closure <= 1e-3; pred_b top-10 units carry >= 0.50 of the pooled sum of
D_j for both sources; pred_c top-50 >= 0.80 for both; pred_d the largest |pooled D_j| is positive (supports 2483's plural response) for both.
PRICE (registered maximum): 3 batches x 1 forward (both sources captured in the same pass) = 3 forwards; bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp45_units_into_2483_v184_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp45_units_into_2483_v184"
UNIT, SRCS, DST, CLOSURE_TOL, TOP10_MIN, TOP50_MIN = 2483, (5, 4), 6, 1e-3, 0.50, 0.80
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_linear_closure": "<= 1e-3", "pred_b_top_10_units_carry_half": ">= 0.50 x 2", "pred_c_top_50_units_carry_080": ">= 0.80 x 2", "pred_d_largest_unit_supports": "> 0 x 2"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    positions_of = lambda row: {"noun": next(i for i, t in enumerate(row.ids) if t in nouns)}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "source_layers": SRCS, "dst_layer": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "top50_min": TOP50_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fw = L.ManualForward(backend)
    per_row, closure, forwards = dod_units.product_unit_census(backend, fw, rows, SRCS, DST, UNIT, positions_of)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report = {}
    for s in SRCS:
        total = dod_units.pooled_contrast(rows, per_row[s], partner, "noun"); pooled = float(total.sum())
        order = torch.argsort(total.abs(), descending=True); share = lambda k: float(total[order[:k]].sum()) / pooled
        top = [(int(j), float(total[j])) for j in order[:40]]
        report[str(s)] = {"pooled_sum": pooled, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)}, "top_units": top}
        print("mlp", s, "pooled sum of D_j", round(pooled, 3), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "top", [(j, round(v, 2)) for j, v in top[:10]])
    predictions = {"pred_a_linear_closure": closure <= CLOSURE_TOL, "pred_b_top_10_units_carry_half": all(r["shares"]["10"] >= TOP10_MIN for r in report.values()),
                   "pred_c_top_50_units_carry_080": all(r["shares"]["50"] >= TOP50_MIN for r in report.values()), "pred_d_largest_unit_supports": all(r["top_units"][0][1] > 0 for r in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp45_units_into_2483_result_v184", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "sources": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
