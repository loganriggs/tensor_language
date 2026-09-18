#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_trio_separates_since_by pred_b_trio_edit_beats_random_sets pred_c_trio_edit_at_least_002 pred_d_trio_exceeds_1250_alone
"""Aspectual has/had DoD (v281): the MLP-7 TRIO {1250, 3364, 1884} (v278: 23% + 13% + 13% of MLP 7's write on 9.1's has - had direction at the bank) under EDIT at the three
bank positions on the v1 rows, against 16 seeded random 3-unit sets of MLP 7 at the same positions; v280: 1250 alone removes 0.56%. Sizing prior: the number line's
MLP-8 trio removes 5.3% at the noun (v169b); the temporal readout reads a bank state that MLP 7 writes only 10-17% of (row 26), so 1-3% is the honest expectation.
PREDICTIONS (scored as written; failures preserved)
    pred_a_trio_separates_since_by     the trio's summed bank value has the sign of (since - by) on >= 26 of the 32 aligned pairs
    pred_b_trio_edit_beats_random_sets zeroing the trio at the bank removes more of the pooled has - had margin than every one of the 16 random 3-unit sets
    pred_c_trio_edit_at_least_002      the trio's edit removes >= 0.02 of the margin
    pred_d_trio_exceeds_1250_alone     the trio's edit removes more than v280's 0.0056 (1250 alone)
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
OUT = ROOT / "circuits/followups/aspectual_dod_mlp7_trio_v281_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_mlp7_trio_v281"
LAYER, UNIT, HEAD, AGREE_MIN, COS_MIN, RATIO_MIN, EDIT_MIN, N_NULL, SEED, BATCH = 7, 1250, (9, 1), 26, 0.20, 1.5, 0.02, 16, 281, 32
TRIO = (1250, 3364, 1884); V280_ALONE = 0.0056
CUES = {L._single(" since"), L._single(" by"), L._single("Since"), L._single("By")}
FORWARDS_MAX = 40
PREDICTIONS = {"pred_a_trio_separates_since_by": ">= 26 / 32", "pred_b_trio_edit_beats_random_sets": "> 16 random 3-unit sets", "pred_c_trio_edit_at_least_002": ">= 0.02", "pred_d_trio_exceeds_1250_alone": "> 0.0056"}


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
    rDj = {j: float(r.to(col.device) @ Down[:, j]) for j in TRIO}
    h = lambda i, label: sum(float(per_row[i][label][j]) / rDj[j] for j in TRIO)          # the trio's summed hidden values
    hb = lambda i: sum(h(i, lab) for lab in ("last", "period", "the"))
    diffs_bank = [hb(i) - hb(j) for i, j in pairs]; diffs_cue = [h(i, "cue") - h(j, "cue") for i, j in pairs]
    agree = max(sum(1 for d in diffs_bank if d > 0), sum(1 for d in diffs_bank if d < 0)); pooled_bank, pooled_cue = sum(diffs_bank), sum(diffs_cue)
    import random
    rng = random.Random(SEED); pool = [j for j in range(Down.shape[1]) if j not in TRIO]; null_units = [tuple(sorted(rng.sample(pool, 3))) for _ in range(N_NULL)]
    conds = [("native", None), ("bank", (TRIO, bank_of)), ("all", (TRIO, lambda rw: None))] + [(f"null{k}", (u, bank_of)) for k, u in enumerate(null_units)]
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
    predictions = {"pred_a_trio_separates_since_by": agree >= AGREE_MIN, "pred_b_trio_edit_beats_random_sets": change["bank"] > null_max, "pred_c_trio_edit_at_least_002": change["bank"] >= EDIT_MIN, "pred_d_trio_exceeds_1250_alone": change["bank"] > V280_ALONE}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_mlp7_trio_result_v281", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
