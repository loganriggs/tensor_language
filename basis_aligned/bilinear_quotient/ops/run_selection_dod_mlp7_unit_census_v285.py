#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_cue_top10_carry_half pred_c_final_top10_carry_half pred_d_top_units_stable_across_constructions
"""Selection DoD (v285): the "look one block lower" check for the selection port. v261b: MLP 8's write on 13.8's reader direction is spread at the adjective cue (top-10
29%) and at the final (8%) -- a port; the temporal family's MLP-8 port had concentrated units at MLP 7 (v278). Same exact census for MLP 7 on the same reader direction
at the CUE (the adjective where the pair members differ) and the FINAL token of the v85 rows. Kill: top-10 < 0.30 at both positions -> port at MLP 7 as well.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_unit_closure                        sum_j T_j + bias = r . mlp(x) within 1e-3, every row and position
    pred_b_cue_top10_carry_half                at the cue, the top-10 |pooled| units carry >= 0.50 of MLP 7's pooled write on r
    pred_c_final_top10_carry_half              at the final token, likewise >= 0.50
    pred_d_top_units_stable_across_constructions  at the cue, min pairwise Jaccard of the three verbs' top-50 sets >= 1/3
PRICE (registered maximum): 3 batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_selection_dod_battery_v85 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/selection_dod_mlp7_unit_census_v285_result.json"
CANDIDATE_ID = "selection.particle.dod_mlp8_unit_census_v261b"
LAYER, HEAD, CLOSURE_TOL, TOP10_MIN, JACCARD_MIN = 7, (13, 8), 1e-3, 0.50, 1 / 3
CUES = {L._single(" remained"), L._single(" looked"), L._single(" grew")}
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_cue_top10_carry_half": ">= 0.50", "pred_c_final_top10_carry_half": ">= 0.50", "pred_d_top_units_stable_across_constructions": "min Jaccard >= 1/3"}


def main() -> None:
    rows, has, had, *_ = g.build()          # has = myself-token (positive), had = yourself-token (negative): names kept from v251
    partner0 = {(row.construction, row.group, row.present): row for row in rows}
    cue_pos = {}
    for row in rows:
        other = partner0[(row.construction, row.group, not row.present)]
        cue_pos[row.row_id] = max(i for i, (a_, b_) in enumerate(zip(row.ids, other.ids)) if a_ != b_)
    positions_of = lambda row: {"cue": cue_pos[row.row_id], "final": row.final}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "layer": LAYER, "reader_head": f"{HEAD[0]}.{HEAD[1]}", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "jaccard_min": JACCARD_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    fw = L.ManualForward(backend)
    comp = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, has, had, (HEAD,)).set_components())
    fw.directions = L.readout_directions(model, (comp,), has, had)
    r = L.reader_directions(model, comp, fw.directions)[HEAD[1]].float()
    per_row, closure, forwards = dod_units.unit_census(backend, fw, rows, LAYER, r, positions_of)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report = {}
    for label in ("cue", "final"):
        total = dod_units.pooled_contrast(rows, per_row, partner, label); contrast = float(total.sum())
        order = torch.argsort(total.abs(), descending=True); share = lambda k: float(total[order[:k]].sum()) / contrast
        by_con = {}
        for con in sorted({rw.construction for rw in rows}):
            acc = None
            for i, rw in enumerate(rows):
                if rw.construction != con or not rw.present: continue
                j = partner[(rw.construction, rw.group, False)]; d = per_row[i][label] - per_row[j][label]; acc = d.clone() if acc is None else acc + d
            by_con[con] = set(torch.argsort(acc.abs(), descending=True)[:50].tolist())
        cons = list(by_con); jac = min(len(by_con[a_] & by_con[b_]) / len(by_con[a_] | by_con[b_]) for a_ in cons for b_ in cons if a_ < b_)
        report[label] = {"contrast": contrast, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)}, "top_units": [(int(j), float(total[j])) for j in order[:40]], "jaccard_fronted_report_top50": jac}
        print(label, "contrast", round(contrast, 2), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "jaccard", round(jac, 3), "top", [(int(j), round(float(total[j]), 1)) for j in order[:8]])
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_cue_top10_carry_half": report["cue"]["shares"]["10"] >= TOP10_MIN, "pred_c_final_top10_carry_half": report["final"]["shares"]["10"] >= TOP10_MIN,
                   "pred_d_top_units_stable_across_constructions": report["cue"]["jaccard_fronted_report_top50"] >= JACCARD_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit_census_result_v261b", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "positions": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
