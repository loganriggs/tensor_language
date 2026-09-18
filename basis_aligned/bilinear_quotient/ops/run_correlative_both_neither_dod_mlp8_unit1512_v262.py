#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_separates_both_neither pred_b_down_aligns_with_reader pred_c_unit_fires_at_final pred_d_edit_beats_random_units
"""Correlative both/neither DoD (v262): what MLP-8 unit 1512 is. v260: at the FINAL token unit 1512 carries 59% of MLP 8's write on 16.8's and - nor direction (top-10
93%, stable across verbs), while at the cue (both / neither) MLP 8 is spread. Readings on the v120 rows: (i) u_1512 at the final and at the cue per row -- sign agreement
of (both - neither) over the 48 aligned pairs; (ii) cos(Down8[:, 1512], r_16.8); (iii) the pooled and - nor margin change when 1512 is zeroed at the FINAL, against 16
seeded random single units zeroed at the final, and at all positions.
PREDICTIONS (scored as written; failures preserved; priors from v255: a detector's edit ~1%)
    pred_a_unit_separates_both_neither   u_1512(final) has the same sign of (both - neither) on >= 90% of the aligned pairs
    pred_b_down_aligns_with_reader       |cos(Down8[:, 1512], r_16.8)| >= 0.30
    pred_c_unit_fires_at_final           the pooled |both - neither| contrast of u_1512 at the final is >= 1.5x that at the cue
    pred_d_edit_beats_random_units       zeroing 1512 at the final removes > 0 of the pooled and - nor margin and more than every one of the 16 random units
PRICE (registered maximum): 3 batches x (native + final edit + all-positions edit + 16 random) = 57 forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_correlative_both_neither_dod_battery_v120 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/correlative_both_neither_dod_mlp8_unit1512_v262_result.json"
CANDIDATE_ID = "correlative_both_neither.and_vs_nor.dod_mlp8_unit1512_v262"
LAYER, UNIT, HEAD, AGREE_MIN, COS_MIN, RATIO_MIN, N_NULL, SEED, BATCH = 8, 1512, (16, 8), 0.90, 0.30, 1.5, 16, 262, 32
CUES = {L._single(" both"), L._single(" neither")}
FORWARDS_MAX = 60
PREDICTIONS = {"pred_a_unit_separates_both_neither": ">= 0.90", "pred_b_down_aligns_with_reader": ">= 0.30", "pred_c_unit_fires_at_final": ">= 1.5x the cue", "pred_d_edit_beats_random_units": "> 0 and > 16 random"}


def main() -> None:
    rows, has, had, *_ = g.build()          # has = ones-token (positive), had = one-token (negative): names kept from v252
    cue_of = lambda row: next(i for i, t in enumerate(row.ids) if t in CUES)
    positions_of = lambda row: {"cue": cue_of(row), "final": row.final}
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}; readers["target"] = (has, had)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"agree_min": AGREE_MIN, "cos_min": COS_MIN, "ratio_min": RATIO_MIN, "n_null": N_NULL, "seed": SEED}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    fw = L.ManualForward(backend)
    comp = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, has, had, (HEAD,)).set_components())
    fw.directions = L.readout_directions(model, (comp,), has, had); r = L.reader_directions(model, comp, fw.directions)[HEAD[1]].float()
    Down = model.transformer.h[LAYER].mlp.Down.weight.detach().float(); col = Down[:, UNIT]; cos = float(col @ r.to(col.device)) / float(col.norm() * r.norm())
    per_row, closure, forwards = dod_units.unit_census(backend, fw, rows, LAYER, r, positions_of)      # T_j = (r . Down_j) h_j: h_13 = T_13 / (r . Down_13)
    rD13 = float(r.to(col.device) @ col)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    h = lambda i, label: float(per_row[i][label][UNIT]) / rD13
    diffs_final = [h(i, "final") - h(j, "final") for i, j in pairs]; diffs_cue = [h(i, "cue") - h(j, "cue") for i, j in pairs]
    agree = max(sum(1 for d in diffs_final if d > 0), sum(1 for d in diffs_final if d < 0)); pooled_final, pooled_cue = sum(diffs_final), sum(diffs_cue)
    import random
    rng = random.Random(SEED); pool = [j for j in range(Down.shape[1]) if j != UNIT]; null_units = [rng.choice(pool) for _ in range(N_NULL)]
    fin_of = lambda rw: rw.final
    conds = [("native", None), ("cue", ((UNIT,), fin_of)), ("all", ((UNIT,), lambda rw: None))] + [(f"null{k}", ((u,), fin_of)) for k, u in enumerate(null_units)]   # "cue" key = the FINAL-position edit here
    outs = {name: [] for name, _ in conds}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, edits in conds:
            outs[name].extend(dod_units.forward_margins(backend, fw, chunk, LAYER, edits, readers)); forwards += 1
    pooled = lambda lst: sum(lst[i]["target"] - lst[j]["target"] for i, j in pairs)
    m_native = pooled(outs["native"]); change = {name: (m_native - pooled(outs[name])) / m_native for name in outs if name != "native"}
    null_max = max(change[f"null{k}"] for k in range(N_NULL))
    agree_final = max(sum(1 for d in diffs_final if d > 0), sum(1 for d in diffs_final if d < 0)) / len(pairs)
    report = {"cos_down_reader": cos, "r_dot_down": rD13, "u_both_minus_neither_pooled_final": pooled_final, "u_both_minus_neither_pooled_cue": pooled_cue, "sign_agreement_final": agree_final, "pairs": len(pairs),
              "u_final_means": {"both": sum(h(i, "final") for i, j in pairs) / len(pairs), "neither": sum(h(j, "final") for i, j in pairs) / len(pairs)}, "margin_native_pooled": m_native, "margin_change_final": change["cue"], "margin_change_all": change["all"], "null_max": null_max, "null_units": null_units}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k != "null_units"})
    predictions = {"pred_a_unit_separates_both_neither": agree_final >= AGREE_MIN, "pred_b_down_aligns_with_reader": abs(cos) >= COS_MIN, "pred_c_unit_fires_at_final": abs(pooled_final) >= RATIO_MIN * abs(pooled_cue),
                   "pred_d_edit_beats_random_units": change["cue"] > 0 and change["cue"] > null_max}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit1512_result_v262", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
