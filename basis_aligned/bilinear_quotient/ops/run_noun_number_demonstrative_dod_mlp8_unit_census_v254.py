#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_number_detectors_in_cue_top5 pred_c_cue_top10_carry_half pred_d_cue_top50_overlaps_pronoun_number_top50
"""Noun-number demonstrative these/this DoD (v254): are the MLP-8 number detectors SHARED across the two number families? On the pronoun-number line, MLP 8's write on
9.6's they - he direction at the noun is two units, 829 (plural) and 953 (singular) [v168, v170]. The noun-number family ("Yesterday the messenger wanted these shiny" ->
ones / one; readers {11.2, 7.8, 6.3, 15.1}, v157) has a number-marked DETERMINER as its cue. Exact census (`dod_units.unit_census`) of MLP 8's write on 11.2's weight-only
ones - one reader direction at the CUE (these / this) and at the FINAL token of the v157 rows, pooled these - this; overlap with v168's pronoun-number top-50.
PREDICTIONS (scored as written; failures preserved; priors unsure -- Composes at unit grain)
    pred_a_unit_closure                          sum_j T_j + bias = r . mlp(x) within 1e-3, every row and position
    pred_b_number_detectors_in_cue_top5          829 or 953 is among the top-5 |pooled| units at the cue
    pred_c_cue_top10_carry_half                  at the cue the top-10 units carry >= 0.50 of MLP 8's pooled write on r
    pred_d_cue_top50_overlaps_pronoun_number_top50  Jaccard(cue top-50, v168's noun top-50) >= 1/3
PRICE (registered maximum): 3 batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_noun_number_demonstrative_dod_battery_v157 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/noun_number_demonstrative_dod_mlp8_unit_census_v254_result.json"
CANDIDATE_ID = "noun_number_demonstrative.these_vs_this.dod_mlp8_unit_census_v254"
LAYER, HEAD, CLOSURE_TOL, TOP10_MIN, JACCARD_MIN = 8, (11, 2), 1e-3, 0.50, 1 / 3
V168 = ROOT / "circuits/followups/pronoun_number_dod_mlp8_unit_census_v168_result.json"
CUES = {L._single(" these"), L._single(" this")}
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_number_detectors_in_cue_top5": "829 or 953 in top-5", "pred_c_cue_top10_carry_half": ">= 0.50", "pred_d_cue_top50_overlaps_pronoun_number_top50": "Jaccard >= 1/3"}


def main() -> None:
    rows, has, had, *_ = g.build()          # has = ones-token (positive), had = one-token (negative): names kept from v251
    positions_of = lambda row: {"cue": next(i for i, t in enumerate(row.ids) if t in CUES), "final": row.final}
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
        v168_top50 = {j for j, _ in json.loads(V168.read_text())["top_units"][:50]}; mine = set(order[:50].tolist()); jac = len(mine & v168_top50) / len(mine | v168_top50)
        report[label] = {"contrast": contrast, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)}, "top_units": [(int(j), float(total[j])) for j in order[:40]], "jaccard_with_v168_top50": jac, "rank_829": int((total.abs() > abs(total[829])).sum()) + 1, "rank_953": int((total.abs() > abs(total[953])).sum()) + 1}
        print(label, "contrast", round(contrast, 2), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "jaccard v168", round(jac, 3), "top", [(int(j), round(float(total[j]), 1)) for j in order[:8]], "rank 829 / 953", report[label]["rank_829"] if label in report else "")
    cue_top5 = [j for j, _ in report["cue"]["top_units"][:5]]
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_number_detectors_in_cue_top5": (829 in cue_top5) or (953 in cue_top5), "pred_c_cue_top10_carry_half": report["cue"]["shares"]["10"] >= TOP10_MIN,
                   "pred_d_cue_top50_overlaps_pronoun_number_top50": report["cue"]["jaccard_with_v168_top50"] >= JACCARD_MIN}
    print("ranks at the cue: 829", report["cue"]["rank_829"], "953", report["cue"]["rank_953"], "| at the final:", report["final"]["rank_829"], report["final"]["rank_953"])
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit_census_result_v254", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "positions": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
