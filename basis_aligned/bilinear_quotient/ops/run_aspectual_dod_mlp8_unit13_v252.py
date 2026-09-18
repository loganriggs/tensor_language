#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit13_separates_since_by pred_b_down_aligns_with_reader pred_c_unit13_fires_at_final_not_cue pred_d_edit_of_unit13_small
"""Aspectual has/had DoD (v252): is MLP-8 unit 13 a nameable has/had unit? v251: at the FINAL token, unit 13 leads MLP 8's write on 9.1's has - had direction (16%;
top-10 34%; stable across the two constructions), while at the cue MLP 8 is a port. Two forwards on the v1 rows: (i) u_13 at the final and at the cue, per row, and
whether it separates since-rows from by-rows (sign agreement over the 32 pairs); (ii) the cosine of Down8[:, 13] with 9.1's reader direction r; (iii) the pooled
has - had margin change when unit 13 is zeroed at the final (plain forward, `dod_units.forward_margins`), against v1's native margin. No null (a one-unit edit
sized by the census; the registered expectation is SMALL: 16% of MLP 8's part of one head's read).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_unit13_separates_since_by   u_13(final) has the same sign of (since - by) on >= 26 of the 32 aligned pairs
    pred_b_down_aligns_with_reader     |cos(Down8[:, 13], r_9.1)| >= 0.30
    pred_c_unit13_fires_at_final_not_cue  the pooled |since - by| contrast of u_13 at the final is >= 3x that at the cue
    pred_d_edit_of_unit13_small        zeroing unit 13 at the final changes the pooled has - had margin by <= 0.03 (fraction) -- and the change is a drop
PRICE (registered maximum): 2 batches x (native + edited) = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/aspectual_dod_mlp8_unit13_v252_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_mlp8_unit13_v252"
LAYER, UNIT, HEAD, AGREE_MIN, COS_MIN, RATIO_MIN, EDIT_MAX, BATCH = 8, 13, (9, 1), 26, 0.30, 3.0, 0.03, 32
CUES = {L._single(" since"), L._single(" by"), L._single("Since"), L._single("By")}
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_unit13_separates_since_by": ">= 26 / 32", "pred_b_down_aligns_with_reader": ">= 0.30", "pred_c_unit13_fires_at_final_not_cue": ">= 3x", "pred_d_edit_of_unit13_small": "<= 0.03, a drop"}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    has, had = L._single(" has"), L._single(" had")
    cue_of = lambda row: next(i for i, t in enumerate(row.ids) if t in CUES)
    positions_of = lambda row: {"cue": cue_of(row), "final": row.final}
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}; readers["target"] = (has, had)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"agree_min": AGREE_MIN, "cos_min": COS_MIN, "ratio_min": RATIO_MIN, "edit_max": EDIT_MAX}}
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
    native, edited = [], []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        native.extend(dod_units.forward_margins(backend, fw, chunk, LAYER, None, readers)); forwards += 1
        edited.extend(dod_units.forward_margins(backend, fw, chunk, LAYER, ((UNIT,), lambda rw: rw.final), readers)); forwards += 1
    pooled = lambda lst: sum(lst[i]["target"] - lst[j]["target"] for i, j in pairs)
    m_native, m_edit = pooled(native), pooled(edited); change = (m_native - m_edit) / m_native
    report = {"cos_down13_reader": cos, "r_dot_down13": rD13, "u13_since_minus_by_pooled_final": pooled_final, "u13_since_minus_by_pooled_cue": pooled_cue, "sign_agreement_final": agree, "pairs": len(pairs),
              "u13_final_means": {"since": sum(h(i, "final") for i, j in pairs) / len(pairs), "by": sum(h(j, "final") for i, j in pairs) / len(pairs)}, "margin_native_pooled": m_native, "margin_change_fraction_unit13_zeroed_at_final": change}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items()})
    predictions = {"pred_a_unit13_separates_since_by": agree >= AGREE_MIN, "pred_b_down_aligns_with_reader": abs(cos) >= COS_MIN, "pred_c_unit13_fires_at_final_not_cue": abs(pooled_final) >= RATIO_MIN * abs(pooled_cue), "pred_d_edit_of_unit13_small": 0 < change <= EDIT_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit13_result_v252", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
