#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_token_only_term_carries_the_three_heads pred_c_pattern_is_stable_within_cue pred_d_constant_pattern_transfers_pooled pred_e_constant_pattern_transfers_in_every_construction
"""Reflexive person DoD (v113): are the three token readers {8.1, 13.1, 15.1} a TOKEN-ONLY GENERATOR (better_circuits §3.7)?

Lane: Claude circuit lane. Parent: v112 (8.1 / 13.1 / 15.1 take 80-98% of their myself-yourself contrast from the I / you token, 72-95% through
the block-0 value branch). The token-only term of head h is p_h(final, pronoun) x lamb_h x v1_h(pronoun): v1 is a function of the pronoun
token alone, so the only context-dependent port is the pattern scalar p_h. Arms (v12's construction, three heads jointly, 10.5 left native):
ZERO the three slices (reference); REPLACE each slice by its token-only pronoun term with the NATIVE pattern (fold replay); REPLACE with a
CONSTANT pattern = the median of p_h over the OTHER two constructions' rows for the same cue token (leave-one-construction-out, a
per-(head, cue) number, not a fit), scored on the held-out construction. Retention = 1 - damage(arm) / damage(zero).

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native                       <= 1e-4
    pred_b_token_only_term_carries_the_three_heads         native-pattern token-only arm retention >= 0.80 (pooled)
    pred_c_pattern_is_stable_within_cue                    coefficient of variation of p_h over rows <= 0.35 for every (head, construction, cue)
    pred_d_constant_pattern_transfers_pooled               leave-one-construction-out constant-pattern retention >= 0.60 (pooled)
    pred_e_constant_pattern_transfers_in_every_construction  >= 0.50 in each held-out construction

PRICE (registered maximum): 3 batches x (native + producer + zero + 3 pattern folds + [3 captures + 1 arm] x 2) = 42 forwards; 0 backwards; 0 fits. Bar <= 48.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, statistics, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_person_reflexive_dod_battery_v104 as line
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/person_dod_token_only_generator_v113_result.json"
CANDIDATE_ID = "reflexive_person.i_vs_you.dod_token_only_generator_v113"
HEADS = ((8, 1), (13, 1), (15, 1))
NATIVE_MIN, CV_MAX, POOLED_MIN, PER_MIN, INSTRUMENT_TOL = 0.80, 0.35, 0.60, 0.50, 1e-4
FORWARDS_MAX = 48
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_token_only_term_carries_the_three_heads": ">= 0.80", "pred_c_pattern_is_stable_within_cue": "cv <= 0.35",
               "pred_d_constant_pattern_transfers_pooled": ">= 0.60", "pred_e_constant_pattern_transfers_in_every_construction": ">= 0.50 x 3"}


def main() -> None:
    rows, pos, neg, agents, objects = line.build()
    pronouns = {L._single(" I"), L._single("I"), L._single(" you"), L._single("you")}
    cue_pos = {row.row_id: next(i for i, t in enumerate(row.ids) if t in pronouns) for row in rows}
    is_cue = lambda r, s: s == cue_pos[r.row_id]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(HEADS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_min": NATIVE_MIN, "cv_max": CV_MAX, "pooled_min": POOLED_MIN, "per_min": PER_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    comps = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, pos, neg, HEADS).set_components()   # one component per layer, one head each
    fw.directions = L.readout_directions(backend.model, comps, pos, neg)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    zero, n = v1._run_arm(fw, rows, components=comps, mode="zero"); forwards += n
    pattern = {c.name: {} for c in comps}
    for comp in comps:
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
            for row, entry in zip(chunk, out):
                pattern[comp.name][row.row_id] = entry[comp.heads[0]]["pattern"][cue_pos[row.row_id]]
    constructions = sorted({r.construction for r in rows})
    stats, median = {}, {}
    for comp in comps:
        for c in constructions:
            for present in (True, False):
                v = [pattern[comp.name][r.row_id] for r in rows if r.construction == c and r.present == present]
                stats[f"{comp.name}/{c}/{'I' if present else 'you'}"] = {"median": statistics.median(v), "mean": statistics.mean(v), "cv": statistics.pstdev(v) / max(abs(statistics.mean(v)), 1e-9)}
        for c in constructions:   # leave-one-construction-out constant for rows of construction c
            for present in (True, False):
                others = [pattern[comp.name][r.row_id] for r in rows if r.construction != c and r.present == present]
                median[(comp.name, c, present)] = statistics.median(others)

    def arm_with(override_for):
        nonlocal forwards
        table = {}
        for comp in comps:
            ov = override_for(comp)
            for start in range(0, len(rows), v1.BATCH):
                chunk = rows[start:start + v1.BATCH]
                table.update(L.source_restricted_slices(fw, chunk, comp, is_cue, ("inherited",), ov)); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=comps, mode="replace"); forwards += n
        return arm
    native_arm = arm_with(lambda comp: None)
    const_arm = arm_with(lambda comp: (lambda r, s: median[(comp.name, r.construction, r.present)] if is_cue(r, s) else None))
    fw.use_subtract = False

    def retention(arm, idx):
        sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
        z = L.summarize(sub_rows, sub_native, [zero[i] for i in idx])["target_damage_mean"]; a = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])["target_damage_mean"]
        return {"zero_damage": z, "arm_damage": a, "retention": 1.0 - a / z}
    all_idx = list(range(len(rows)))
    scored = {"native_pattern_pooled": retention(native_arm, all_idx), "constant_pattern_pooled": retention(const_arm, all_idx)}
    for c in constructions:
        scored[f"constant_pattern_{c}"] = retention(const_arm, [i for i, r in enumerate(rows) if r.construction == c])
    for k, v in scored.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    print("pattern cv max", round(max(v["cv"] for v in stats.values()), 3))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_token_only_term_carries_the_three_heads": scored["native_pattern_pooled"]["retention"] >= NATIVE_MIN,
                   "pred_c_pattern_is_stable_within_cue": all(v["cv"] <= CV_MAX for v in stats.values()), "pred_d_constant_pattern_transfers_pooled": scored["constant_pattern_pooled"]["retention"] >= POOLED_MIN,
                   "pred_e_constant_pattern_transfers_in_every_construction": all(scored[f"constant_pattern_{c}"]["retention"] >= PER_MIN for c in constructions)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "person_dod_token_only_generator_result_v113", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "scored": scored, "pattern_stats": stats,
                               "constants": {f"{k[0]}/{k[1]}/{'I' if k[2] else 'you'}": v for k, v in median.items()}, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
