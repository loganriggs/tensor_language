#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_separates_plural_singular pred_b_down_opposes_reader pred_c_unit_fires_at_noun pred_d_edit_raises_margin_beyond_random
"""Pronoun number they/he DoD (v270): the plural unit 1738, read with the OPPOSITE sign by the pronoun set. v269: on the have/has line 1738 is a plural detector (53 vs
0) whose removal costs 1.3% of the margin; v168 / v250: on 9.6's and 12.4's they - he directions its pooled term is negative. Registered reading: zeroing 1738 at the noun
of the v76 rows RAISES the they - he margin (the unit fires on plural nouns but its output column projects against 'they'). Same readings as v267 / v269 on the v76 rows
with 9.6 as the reader head, 16 random single units as the null.
PREDICTIONS (scored as written; failures preserved)
    pred_a_unit_separates_plural_singular  u_1738(noun) has the same sign of (plural - singular) on >= 90% of the aligned pairs (it is the same detector)
    pred_b_down_opposes_reader             cos(Down8[:, 1738], r_9.6) <= -0.20 (opposite sign to its cosine with r_11.3, -0.53 -- note r_11.3's own sign convention; registered as: the product of the two cosines is < 0)
    pred_c_unit_fires_at_noun              the pooled |plural - singular| contrast of u_1738 at the noun is >= 1.5x that at the final
    pred_d_edit_raises_margin_beyond_random  zeroing 1738 at the noun changes the pooled they - he margin by <= -0.005 (a rise of >= 0.5%) and |change| > every one of the 16 random units
PRICE (registered maximum): 3 batches x (native + noun edit + all-positions edit + 16 random) = 57 forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp8_unit1738_v270_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp8_unit1738_v270"
LAYER, UNIT, HEAD, AGREE_MIN, COS_MAX, RATIO_MIN, RISE_MIN, N_NULL, SEED, BATCH = 8, 1738, (9, 6), 0.90, -0.20, 1.5, 0.005, 16, 270, 32
V269_COS = -0.5346
CUES = {L._single(" these"), L._single(" this")}
FORWARDS_MAX = 60
PREDICTIONS = {"pred_a_unit_separates_plural_singular": ">= 0.90", "pred_b_down_opposes_reader": "cos x cos(v269) < 0", "pred_c_unit_fires_at_noun": ">= 1.5x the final", "pred_d_edit_raises_margin_beyond_random": "<= -0.005 and beyond 16 random"}


def main() -> None:
    rows, has, had, *_ = g.build()          # has = they-token (positive), had = he-token (negative): names kept
    partner0 = {(row.construction, row.group, row.present): row for row in rows}
    cue_pos = {row.row_id: max(i for i, (a_, b_) in enumerate(zip(row.ids, partner0[(row.construction, row.group, not row.present)].ids)) if a_ != b_) for row in rows}
    cue_of = lambda row: cue_pos[row.row_id]
    positions_of = lambda row: {"cue": cue_of(row), "final": row.final}
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}; readers["target"] = (has, had)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"agree_min": AGREE_MIN, "cos_max": COS_MAX, "ratio_min": RATIO_MIN, "rise_min": RISE_MIN, "n_null": N_NULL, "seed": SEED, "v269_cos": V269_COS}}
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
    conds = [("native", None), ("cue", ((UNIT,), cue_of)), ("all", ((UNIT,), lambda rw: None))] + [(f"null{k}", ((u,), cue_of)) for k, u in enumerate(null_units)]
    outs = {name: [] for name, _ in conds}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, edits in conds:
            outs[name].extend(dod_units.forward_margins(backend, fw, chunk, LAYER, edits, readers)); forwards += 1
    pooled = lambda lst: sum(lst[i]["target"] - lst[j]["target"] for i, j in pairs)
    m_native = pooled(outs["native"]); change = {name: (m_native - pooled(outs[name])) / m_native for name in outs if name != "native"}
    null_max = max(change[f"null{k}"] for k in range(N_NULL))
    report = {"cos_down_reader": cos, "r_dot_down": rD13, "u_plural_minus_singular_pooled_final": pooled_final, "u_plural_minus_singular_pooled_cue": pooled_cue, "sign_agreement_cue": max(sum(1 for d in diffs_cue if d > 0), sum(1 for d in diffs_cue if d < 0)) / len(pairs), "pairs": len(pairs),
              "u_cue_means": {"plural": sum(h(i, "cue") for i, j in pairs) / len(pairs), "singular": sum(h(j, "cue") for i, j in pairs) / len(pairs)}, "margin_native_pooled": m_native, "margin_change_cue": change["cue"], "margin_change_all": change["all"], "null_max": null_max, "null_units": null_units}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k != "null_units"})
    predictions = {"pred_a_unit_separates_plural_singular": report["sign_agreement_cue"] >= AGREE_MIN, "pred_b_down_opposes_reader": cos * V269_COS < 0, "pred_c_unit_fires_at_noun": abs(pooled_cue) >= RATIO_MIN * abs(pooled_final),
                   "pred_d_edit_raises_margin_beyond_random": change["cue"] <= -RISE_MIN and abs(change["cue"]) > max(abs(change[f"null{k}"]) for k in range(N_NULL))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit1738_result_v270", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
