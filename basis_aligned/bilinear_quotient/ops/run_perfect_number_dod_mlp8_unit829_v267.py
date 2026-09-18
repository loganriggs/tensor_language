#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_separates_plural_singular pred_b_down_aligns_with_reader pred_c_unit_fires_at_noun pred_d_edit_live_and_beats_random_units
"""Perfect-number have/has DoD (v267): the shared plural detector 829 under EDIT on the verb-agreement line. v266: 829 leads MLP 8's write on 11.3's have - has direction
at the noun (35%). On the pronoun line zeroing {829, 953, 1030} at the noun cost 5.3% of the they - he margin (v169b) and 829 alone was plural-only on natural text
(v198). Readings on the v97 rows: (i) u_829 at the noun (the position where the pair members differ) and at the final -- sign agreement of (plural - singular); (ii)
cos(Down8[:, 829], r_11.3); (iii) the pooled have - has margin change when 829 is zeroed at the noun, against 16 seeded random single units zeroed at the noun, and at all
positions. If the edit is live here too, one unit is edit-decided in two behaviours.
PREDICTIONS (scored as written; failures preserved; priors from v169b / v170)
    pred_a_unit_separates_plural_singular  u_829(noun) has the same sign of (plural - singular) on >= 90% of the aligned pairs
    pred_b_down_aligns_with_reader         |cos(Down8[:, 829], r_11.3)| >= 0.30
    pred_c_unit_fires_at_noun              the pooled |plural - singular| contrast of u_829 at the noun is >= 1.5x that at the final
    pred_d_edit_live_and_beats_random_units  zeroing 829 at the noun removes >= 0.02 of the pooled have - has margin and more than every one of the 16 random units
PRICE (registered maximum): 3 batches x (native + noun edit + all-positions edit + 16 random) = 57 forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_perfect_number_dod_battery_v97 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/perfect_number_dod_mlp8_unit829_v267_result.json"
CANDIDATE_ID = "perfect_number.have_vs_has.dod_mlp8_unit829_v267"
LAYER, UNIT, HEAD, AGREE_MIN, COS_MIN, RATIO_MIN, EDIT_MIN, N_NULL, SEED, BATCH = 8, 829, (11, 3), 0.90, 0.30, 1.5, 0.02, 16, 267, 32
CUES = {L._single(" these"), L._single(" this")}
FORWARDS_MAX = 60
PREDICTIONS = {"pred_a_unit_separates_plural_singular": ">= 0.90", "pred_b_down_aligns_with_reader": ">= 0.30", "pred_c_unit_fires_at_noun": ">= 1.5x the final", "pred_d_edit_live_and_beats_random_units": ">= 0.02 and > 16 random"}


def main() -> None:
    rows, has, had, *_ = g.build()          # has = ones-token (positive), had = one-token (negative): names kept from v252
    partner0 = {(row.construction, row.group, row.present): row for row in rows}
    cue_pos = {row.row_id: max(i for i, (a_, b_) in enumerate(zip(row.ids, partner0[(row.construction, row.group, not row.present)].ids)) if a_ != b_) for row in rows}
    cue_of = lambda row: cue_pos[row.row_id]
    positions_of = lambda row: {"cue": cue_of(row), "final": row.final}
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
    predictions = {"pred_a_unit_separates_plural_singular": report["sign_agreement_cue"] >= AGREE_MIN, "pred_b_down_aligns_with_reader": abs(cos) >= COS_MIN, "pred_c_unit_fires_at_noun": abs(pooled_cue) >= RATIO_MIN * abs(pooled_final),
                   "pred_d_edit_live_and_beats_random_units": change["cue"] >= EDIT_MIN and change["cue"] > null_max}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "aspectual_dod_mlp8_unit829_result_v267", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
