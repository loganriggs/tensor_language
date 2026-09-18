#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_top_10_units_carry_half pred_c_top_50_units_carry_080
"""Pronoun gender he/she DoD (v173): MLP 6 at UNIT grain as an input of MLP-8 unit 3152 (v171: MLP 6 writes 0.28 of the Left factor and 0.17 of the
Right factor at the noun). `dod_units.unit_census` on block 6 with reader = the Left row and, separately, the Right row of unit 3152 (both 1152-d),
at the noun position: T_j = (row . Down6[:, j]) h6_j, exact. If a few MLP-6 units carry the factor contrast, the port opens one more named layer;
if spread, MLP 6 is declared at unit grain.
PREDICTIONS (scored as written; failures preserved; priors unsure): pred_a closure <= 1e-3 (both factors); pred_b top-10 units >= 0.50 of the pooled
contrast for both factors; pred_c top-50 >= 0.80 for both.
PRICE (registered maximum): 2 batches x 1 forward x 2 factors = 4 forwards; bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_mlp6_unit_census_for_unit_v173_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_mlp6_unit_census_for_unit_v173"
UNIT, LAYER, CLOSURE_TOL, TOP10_MIN, TOP50_MIN = 3152, 6, 1e-3, 0.50, 0.80
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_top_10_units_carry_half": ">= 0.50 x 2", "pred_c_top_50_units_carry_080": ">= 0.80 x 2"}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    positions_of = lambda row: {"noun": next(i for i, t in enumerate(row.ids) if t in nouns)}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "top50_min": TOP50_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    fw = L.ManualForward(backend)
    mlp8 = model.transformer.h[8].mlp
    readers = {"L": mlp8.Left.weight.detach().float()[UNIT], "R": mlp8.Right.weight.detach().float()[UNIT]}
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    forwards, report, closure_max = 0, {}, 0.0
    for name, reader in readers.items():
        per_row, closure, n = dod_units.unit_census(backend, fw, rows, LAYER, reader, positions_of); forwards += n; closure_max = max(closure_max, closure)
        total = dod_units.pooled_contrast(rows, per_row, partner, "noun"); contrast = float(total.sum())
        order = fw.torch.argsort(total.abs(), descending=True); share = lambda k: float(total[order[:k]].sum()) / contrast
        report[name] = {"contrast": contrast, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)}, "top_units": [(int(j), float(total[j])) for j in order[:30]]}
        print(name, "contrast", round(contrast, 3), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "top", [(int(j), round(float(total[j]), 2)) for j in order[:8]])
    predictions = {"pred_a_unit_closure": closure_max <= CLOSURE_TOL, "pred_b_top_10_units_carry_half": all(float(r["shares"]["10"]) >= TOP10_MIN for r in report.values()), "pred_c_top_50_units_carry_080": all(float(r["shares"]["50"]) >= TOP50_MIN for r in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_mlp6_unit_census_for_unit_result_v173", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure_max, "factors": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
