#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_015_of_frozen
"""Modal would/will DoD battery (v66): the census-led set {9.4, 11.3, 9.1, 15.5} on FRESH modal rows.

Lane: Claude circuit lane. Parent: v49b (family set, 59% on the line's authored rows). Frozen before access: F* = 0.59 ± 0.15.
Rows: 16 places new to the line (v28 places) x 16 adjectives new to the line (v42 focus list) in the line's two
constructions: "If the P closed early, the F route" / "When the P closes early, the F route" -> would/will; "The report
notes that if the P closed, the F route" / "... when the P closes, the F route". Readers: was−were, who−which, night−day.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_native_capability           all four (construction x mood) cells >= 0.85
    pred_c_set_live_and_beats_null     fraction >= 0.10, positive >= 0.75, > max of 16 norm-matched nulls
    pred_d_set_selective               three reader gates over the null
    pred_e_set_is_additive             |joint − Σ singles| <= 0.25 x min single
    pred_f_keep_only_retains_most      keep-only retention >= 0.70; random keeps <= 0.30
    pred_g_set_fraction_within_015_of_frozen   |fraction − 0.59| <= 0.15 in each construction

PRICE (registered maximum): 64 rows in 2 batches: native 2 + producer 2 + set 2 + nulls 32 + singles 8 + zero 2 + keep 2 + random
keeps 32 = 82 forwards; 0 backwards; 0 fits. Bar <= 90.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import circuit_fast_screen_candidates as lex
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28
import run_narrative_dod_sweep_and_set_v42 as v42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/modal_remoteness_dod_battery_v66_result.json"
CANDIDATE_ID = "modal_remoteness.would_vs_will.dod_battery_v66"
WOULD, WILL = L._single(" would"), L._single(" will")
FROZEN, BAND = 0.59, 0.15
NULL_SEEDS, KEEP_SEEDS = tuple(range(2101, 2117)), tuple(range(2151, 2167))
CAPABILITY_MIN, LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, ADD_RATIO, RETAIN_MIN, RANDOM_MAX, INSTRUMENT_TOL = 0.85, 0.10, 0.75, 0.25, 0.25, 0.70, 0.30, 1e-4
FORWARDS_MAX = 90
PLACES, ADJ = v28.PLACES, v42.FOCUS
SINGLES = (L.Component("attn9_h4_final", 9, "attn", (4,), "final"), L.Component("attn11_h3_final", 11, "attn", (3,), "final"), L.Component("attn9_h1_final", 9, "attn", (1,), "final"), L.Component("attn15_h5_final", 15, "attn", (5,), "final"))
SET = (SINGLES[1], L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"), SINGLES[3])


def build():
    if set(PLACES) & set(lex._OBJECTS) or set(ADJ) & set(lex._ADJECTIVES):
        raise SystemExit("lexicon overlaps the line's own lists")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction in ("fronted", "embedded"):
        for g in range(16):
            p, f = PLACES[g], ADJ[g]
            for remote in (True, False):
                if construction == "fronted":
                    text = (f"If the {p} closed early, the {f} route" if remote else f"When the {p} closes early, the {f} route")
                else:
                    text = (f"The report notes that if the {p} closed, the {f} route" if remote else f"The report notes that when the {p} closes, the {f} route")
                ids = L.ENCODING.encode(text); answer, foil = (" would", " will") if remote else (" will", " would")
                a_id, f_id = L._single(answer), L._single(foil)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise SystemExit(f"joint tokenization failed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["modal_v66", construction, g, remote, text]).encode()).hexdigest()[:24], construction, g, remote, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows


def main() -> None:
    rows = build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "frozen": {"fraction": FROZEN, "band": BAND}, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
                          "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SET + SINGLES, WOULD, WILL)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    capability = {}
    for c in ("fronted", "embedded"):
        for remote in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c and r.present == remote]
            capability[f"{c}/{'remote' if remote else 'factual'}"] = sum(cell) / len(cell)
    joint, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    per_c = {c: L.summarize([r for r in rows if r.construction == c], [native[i] for i, r in enumerate(rows) if r.construction == c], [joint[i] for i, r in enumerate(rows) if r.construction == c])["target_damage_fraction"] for c in ("fronted", "embedded")}
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls); null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * js["target_damage_mean"] for name in L.UNRELATED}
    singles = {}
    for comp in SINGLES:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        singles[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    gap = abs(js["target_damage_mean"] - sum(singles.values())); bar = ADD_RATIO * min(singles.values())
    zero, n = v1._run_arm(fw, rows, components=SET, mode="zero"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    keep, n = v1._run_arm(fw, rows, components=SET, mode="keep_only"); forwards += n
    retention = 1.0 - L.summarize(rows, native, keep)["target_damage_mean"] / zd
    random_ret = []
    for seed in KEEP_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="keep_only_random", seed=seed); forwards += n
        random_ret.append(1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd)
    print("capability", capability, "fractions", {k: round(v, 3) for k, v in per_c.items()}, "joint", round(js["target_damage_mean"], 3), "pos", js["target_damage_positive_fraction"], "null_max", round(null_max, 4), "gates", gates)
    print("singles", {k: round(v, 3) for k, v in singles.items()}, "gap/bar", round(gap, 4), round(bar, 4), "keep", round(retention, 3), "random keep max", round(max(random_ret), 3))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_native_capability": all(v >= CAPABILITY_MIN for v in capability.values()),
                   "pred_c_set_live_and_beats_null": js["target_damage_fraction"] >= LIVE_FRACTION and js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max,
                   "pred_d_set_selective": all(gates.values()), "pred_e_set_is_additive": gap <= bar, "pred_f_keep_only_retains_most": retention >= RETAIN_MIN and all(r <= RANDOM_MAX for r in random_ret),
                   "pred_g_set_fraction_within_015_of_frozen": all(abs(v - FROZEN) <= BAND for v in per_c.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "modal_remoteness_dod_battery_result_v66", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "instrument_max_abs_error": instrument, "capability": capability, "joint": js,
                               "fractions": per_c, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates, "singles": singles, "additivity": {"gap": gap, "bar": bar},
                               "keep_only": {"zero_damage": zd, "retention": retention, "random_retention": random_ret}, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
