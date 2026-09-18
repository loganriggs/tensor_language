#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit1250_separates_since_by pred_b_down_aligns_with_reader pred_c_unit1250_fires_at_bank pred_d_edit_live_and_beats_random_units
"""Aspectual has/had DoD (v280): is MLP-7 unit 1250 the temporal family's detector? v278: at the bank (last / period / the) unit 1250 carries 23% of MLP 7's write on
9.1's has - had reader direction (top-10 73%, stable across constructions); MLP 8 there is a port (v253). Readings on the v1 rows: (i) u_1250 summed over the three bank
positions and at the cue, per row -- sign agreement of (since - by) over the 32 aligned pairs; (ii) cos(Down7[:, 1250], r_9.1); (iii) the pooled has - had margin change
when 1250 is zeroed at the three bank positions (plain forward, `dod_units.forward_margins` with a position list), against 16 seeded random single MLP-7 units zeroed at
the same positions, and at all positions.
PREDICTIONS (scored as written; failures preserved; priors from the number line's detectors)
    pred_a_unit1250_separates_since_by  u_1250(bank) has the same sign of (since - by) on >= 26 of the 32 aligned pairs
    pred_b_down_aligns_with_reader      |cos(Down7[:, 1250], r_9.1)| >= 0.20
    pred_c_unit1250_fires_at_bank       the pooled |since - by| contrast of u_1250 at the bank is >= 1.5x that at the cue
    pred_d_edit_live_and_beats_random_units  zeroing 1250 at the bank removes >= 0.02 of the pooled has - had margin and more than every one of the 16 random units
PRICE (registered maximum): 2 batches x (native + bank edit + all-positions edit + 16 random) = 38 forwards; 0 backwards; 0 fits. Bar <= 40.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/aspectual_dod_mlp7_unit1250_v280_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_mlp7_unit1250_v280"
LAYER, UNIT, HEAD, AGREE_MIN, COS_MIN, RATIO_MIN, EDIT_MIN, N_NULL, SEED, BATCH = 7, 1250, (9, 1), 26, 0.20, 1.5, 0.02, 16, 280, 32
CUES = {L._single(" since"), L._single(" by"), L._single("Since"), L._single("By")}
FORWARDS_MAX = 40
PREDICTIONS = {"pred_a_unit1250_separates_since_by": ">= 26 / 32", "pred_b_down_aligns_with_reader": ">= 0.20", "pred_c_unit1250_fires_at_bank": ">= 1.5x the cue", "pred_d_edit_live_and_beats_random_units": ">= 0.02 and > 16 random"}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    has, had = L._single(" has"), L._single(" had")
    cue_of = lambda row: next(i for i, t in enumerate(row.ids) if t in CUES)
    bank_of = lambda row: list(row.source_positions)
    positions_of = lambda row: {"cue": cue_of(row), "last": row.source_positions[0], "period": row.source_positions[1], "the": row.source_positions[2]}
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}; readers["target"] = (has, had)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"agree_min": AGREE_MIN, "cos_min": COS_MIN, "ratio_min": RATIO_MIN, "edit_min": EDIT_MIN, "n_null": N_NULL, "seed": SEED}}
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
    hb = lambda i: sum(h(i, lab) for lab in ("last", "period", "the"))
    diffs_bank = [hb(i) - hb(j) for i, j in pairs]; diffs_cue = [h(i, "cue") - h(j, "cue") for i, j in pairs]
    agree = max(sum(1 for d in diffs_bank if d > 0), sum(1 for d in diffs_bank if d < 0)); pooled_bank, pooled_cue = sum(diffs_bank), sum(diffs_cue)
    import random
    rng = random.Random(SEED); pool = [j for j in range(Down.shape[1]) if j != UNIT]; null_units = [rng.choice(pool) for _ in range(N_NULL)]
    conds = [("native", None), ("bank", ((UNIT,), bank_of)), ("all", ((UNIT,), lambda rw: None))] + [(f"null{k}", ((u,), bank_of)) for k, u in enumerate(null_units)]
    outs = {name: [] for name, _ in conds}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, edits in conds:
            outs[name].extend(dod_units.forward_margins(backend, fw, chunk, LAYER, edits, readers)); forwards += 1
    pooled = lambda lst: sum(lst[i]["target"] - lst[j]["target"] for i, j in pairs)
    m_native = pooled(outs["native"]); change = {name: (m_native - pooled(outs[name])) / m_native for name in outs if name != "native"}; null_max = max(change[f"null{k}"] for k in range(N_NULL))
    report = {"cos_down_reader": cos, "r_dot_down": rD13, "u_since_minus_by_pooled_bank": pooled_bank, "u_since_minus_by_pooled_cue": pooled_cue, "sign_agreement_bank": agree, "pairs": len(pairs),
              "u_bank_means": {"since": sum(hb(i) for i, j in pairs) / len(pairs), "by": sum(hb(j) for i, j in pairs) / len(pairs)}, "margin_native_pooled": m_native, "margin_change_bank": change["bank"], "margin_change_all": change["all"], "null_max": null_max, "null_units": null_units}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k != "null_units"})
    predictions = {"pred_a_unit1250_separates_since_by": agree >= AGREE_MIN, "pred_b_down_aligns_with_reader": abs(cos) >= COS_MIN, "pred_c_unit1250_fires_at_bank": abs(pooled_bank) >= RATIO_MIN * abs(pooled_cue), "pred_d_edit_live_and_beats_random_units": change["bank"] >= EDIT_MIN and change["bank"] > null_max}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit1250_result_v280", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
