#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_bank_top10_carry_half pred_c_one_unit_leads_bank pred_d_top_units_stable_across_constructions
"""Aspectual has/had DoD (v278): the temporal family ONE BLOCK DOWN -- MLP 7 at unit grain at the bank. v253: MLP 8's write on 9.1's has - had reader
direction at the bank (last / period / the) is spread (top-10 19%) -- a port; but the bank state is written 10-17% by MLP 7 and 10-12% by MLP 6 (scorecard rows 26-27).
Same exact census (`dod_units.unit_census`) for MLP 7's write on the same reader direction at the three bank positions and their sum, pooled since - by.
Kill criterion (review 29): top-10 < 0.30 at the bank for both MLP 7 and MLP 6 -> the temporal family is a port at every MLP grain.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_unit_closure                        sum_j T_j + bias = r . mlp(x) within 1e-3, every row and position
    pred_b_bank_top10_carry_half               summed over the three bank positions, the top-10 |pooled| units carry >= 0.50 of MLP 7's pooled write on r
    pred_c_one_unit_leads_bank                 the largest unit carries >= 0.15 of the bank total
    pred_d_top_units_stable_across_constructions  Jaccard(top-50 of the fronted rows, top-50 of the report rows) >= 1/3 at the bank
PRICE (registered maximum): 2 batches x 1 forward = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/aspectual_dod_mlp7_unit_census_bank_v278_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_mlp7_unit_census_bank_v278"
LAYER, HEAD, CLOSURE_TOL, TOP10_MIN, LEAD_MIN, JACCARD_MIN = 7, (9, 1), 1e-3, 0.50, 0.15, 1 / 3
CUES = {L._single(" since"), L._single(" by"), L._single("Since"), L._single("By")}
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_bank_top10_carry_half": ">= 0.50", "pred_c_one_unit_leads_bank": ">= 0.15", "pred_d_top_units_stable_across_constructions": "Jaccard >= 1/3"}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    has, had = L._single(" has"), L._single(" had")
    positions_of = lambda row: {"last": row.source_positions[0], "period": row.source_positions[1], "the": row.source_positions[2]}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "layer": LAYER, "reader_head": f"{HEAD[0]}.{HEAD[1]}", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "lead_min": LEAD_MIN, "jaccard_min": JACCARD_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    fw = L.ManualForward(backend)
    comp = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, has, had, (HEAD,)).set_components())
    fw.directions = L.readout_directions(model, (comp,), has, had)
    r = L.reader_directions(model, comp, fw.directions)[HEAD[1]].float()
    per_row, closure, forwards = dod_units.unit_census(backend, fw, rows, LAYER, r, positions_of)
    for entry in per_row: entry["bank"] = entry["last"] + entry["period"] + entry["the"]
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report = {}
    for label in ("last", "period", "the", "bank"):
        total = dod_units.pooled_contrast(rows, per_row, partner, label); contrast = float(total.sum())
        order = torch.argsort(total.abs(), descending=True); share = lambda k: float(total[order[:k]].sum()) / contrast
        by_con = {}
        for con in sorted({rw.construction for rw in rows}):
            acc = None
            for i, rw in enumerate(rows):
                if rw.construction != con or not rw.present: continue
                j = partner[(rw.construction, rw.group, False)]; d = per_row[i][label] - per_row[j][label]; acc = d.clone() if acc is None else acc + d
            by_con[con] = set(torch.argsort(acc.abs(), descending=True)[:50].tolist())
        cons = list(by_con); jac = len(by_con[cons[0]] & by_con[cons[1]]) / len(by_con[cons[0]] | by_con[cons[1]])
        report[label] = {"contrast": contrast, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)}, "top_units": [(int(j), float(total[j])) for j in order[:40]], "jaccard_fronted_report_top50": jac}
        print(label, "contrast", round(contrast, 2), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "jaccard", round(jac, 3), "top", [(int(j), round(float(total[j]), 1)) for j in order[:8]])
    bank = report["bank"]; lead_share = bank["top_units"][0][1] / bank["contrast"]
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_bank_top10_carry_half": bank["shares"]["10"] >= TOP10_MIN, "pred_c_one_unit_leads_bank": lead_share >= LEAD_MIN,
                   "pred_d_top_units_stable_across_constructions": bank["jaccard_fronted_report_top50"] >= JACCARD_MIN}
    print("bank leader share", round(lead_share, 3))
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit_census_bank_result_v278", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "positions": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
