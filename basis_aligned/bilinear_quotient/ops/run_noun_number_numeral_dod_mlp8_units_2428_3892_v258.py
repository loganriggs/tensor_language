#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_2428_separates_three_one pred_b_2428_down_aligns_with_reader pred_c_2428_cue_edit_beats_random pred_d_3892_final_edit_beats_random pred_e_3892_final_larger_than_2428_cue
"""Noun-number numeral three/one DoD (v258): the numeral line's own cue unit 2428 and the shared final-position unit 3892, under edit. v257: at the cue (three / one) MLP-8
unit 2428 leads MLP 8's write on 11.2's ones - one direction (13%, spread); at the final 3892 leads (+20%) as on the demonstrative line. Readings on the v158 rows:
(i) u_2428 at the cue per row -- sign agreement of (three - one) over the 48 aligned pairs; (ii) cos(Down8[:, 2428], r_11.2); (iii) the pooled ones - one margin change when
2428 is zeroed at the cue and when 3892 is zeroed at the final, each against 16 seeded random single units zeroed at the same position.
PREDICTIONS (scored as written; failures preserved; priors from v255 on the demonstrative line: a detector's edit ~0.7%)
    pred_a_2428_separates_three_one      u_2428(cue) has the same sign of (three - one) on >= 90% of the aligned pairs
    pred_b_2428_down_aligns_with_reader  |cos(Down8[:, 2428], r_11.2)| >= 0.20
    pred_c_2428_cue_edit_beats_random    zeroing 2428 at the cue removes > 0 of the margin and more than every one of its 16 random units
    pred_d_3892_final_edit_beats_random  zeroing 3892 at the final removes > 0 of the margin and more than every one of its 16 random units
    pred_e_3892_final_larger_than_2428_cue  the 3892-at-final edit removes more than the 2428-at-cue edit
PRICE (registered maximum): 3 batches x (native + 2 edits + 32 random) = 105 forwards; 0 backwards; 0 fits. Bar <= 108.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_noun_number_numeral_dod_battery_v158 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/noun_number_numeral_dod_mlp8_units_2428_3892_v258_result.json"
CANDIDATE_ID = "noun_number_numeral.three_vs_one.dod_mlp8_units_2428_3892_v258"
LAYER, UNIT, U3892, HEAD, AGREE_MIN, COS_MIN, N_NULL, SEED, BATCH = 8, 2428, 3892, (11, 2), 0.90, 0.20, 16, 258, 32
CUES = {L._single(" three"), L._single(" one")}
FORWARDS_MAX = 108
PREDICTIONS = {"pred_a_2428_separates_three_one": ">= 0.90", "pred_b_2428_down_aligns_with_reader": ">= 0.20", "pred_c_2428_cue_edit_beats_random": "> 0 and > 16 random", "pred_d_3892_final_edit_beats_random": "> 0 and > 16 random", "pred_e_3892_final_larger_than_2428_cue": "3892 > 2428"}


def main() -> None:
    rows, has, had, *_ = g.build()          # has = ones-token (positive), had = one-token (negative): names kept from v252
    cue_of = lambda row: next(i for i, t in enumerate(row.ids) if t in CUES)
    positions_of = lambda row: {"cue": cue_of(row), "final": row.final}
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}; readers["target"] = (has, had)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"agree_min": AGREE_MIN, "cos_min": COS_MIN, "n_null": N_NULL, "seed": SEED}}
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
    rng = random.Random(SEED); pool = [j for j in range(Down.shape[1]) if j not in (UNIT, U3892)]; null_cue = [rng.choice(pool) for _ in range(N_NULL)]; null_fin = [rng.choice(pool) for _ in range(N_NULL)]
    conds = [("native", None), ("2428_cue", ((UNIT,), cue_of)), ("3892_final", ((U3892,), lambda rw: rw.final))] + [(f"nullc{k}", ((u,), cue_of)) for k, u in enumerate(null_cue)] + [(f"nullf{k}", ((u,), lambda rw: rw.final)) for k, u in enumerate(null_fin)]
    outs = {name: [] for name, _ in conds}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, edits in conds:
            outs[name].extend(dod_units.forward_margins(backend, fw, chunk, LAYER, edits, readers)); forwards += 1
    pooled = lambda lst: sum(lst[i]["target"] - lst[j]["target"] for i, j in pairs)
    m_native = pooled(outs["native"]); change = {name: (m_native - pooled(outs[name])) / m_native for name in outs if name != "native"}
    null_cue_max = max(change[f"nullc{k}"] for k in range(N_NULL)); null_fin_max = max(change[f"nullf{k}"] for k in range(N_NULL))
    report = {"cos_down2428_reader": cos, "r_dot_down2428": rD13, "u2428_three_minus_one_pooled_cue": pooled_cue, "u2428_three_minus_one_pooled_final": pooled_final, "sign_agreement_cue": max(sum(1 for d in diffs_cue if d > 0), sum(1 for d in diffs_cue if d < 0)) / len(pairs), "pairs": len(pairs),
              "u2428_cue_means": {"three": sum(h(i, "cue") for i, j in pairs) / len(pairs), "one": sum(h(j, "cue") for i, j in pairs) / len(pairs)}, "margin_native_pooled": m_native, "margin_change_2428_cue": change["2428_cue"], "margin_change_3892_final": change["3892_final"],
              "null_cue_max": null_cue_max, "null_final_max": null_fin_max}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items()})
    predictions = {"pred_a_2428_separates_three_one": report["sign_agreement_cue"] >= AGREE_MIN, "pred_b_2428_down_aligns_with_reader": abs(cos) >= COS_MIN, "pred_c_2428_cue_edit_beats_random": change["2428_cue"] > 0 and change["2428_cue"] > null_cue_max,
                   "pred_d_3892_final_edit_beats_random": change["3892_final"] > 0 and change["3892_final"] > null_fin_max, "pred_e_3892_final_larger_than_2428_cue": change["3892_final"] > change["2428_cue"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_units_2428_3892_result_v258", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
