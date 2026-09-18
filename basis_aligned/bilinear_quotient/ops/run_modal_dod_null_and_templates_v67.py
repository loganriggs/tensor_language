#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_set_beats_every_random_quadruple pred_c_random_quadruples_are_not_live pred_d_two_of_three_templates_capable pred_e_set_removal_transfers_to_every_capable_template pred_f_set_removal_selective_on_every_capable_template
"""Modal would/will DoD (v67): matched-count random 4-set null on the v66 rows + template-varying transfer.
Templates (v66 lexicon): "Had the P closed early, the F route" / "Once the P closes early, the F route"; "If the P had closed,
the F route" / "When the P has closed, the F route"; "Should the P close early, the F route" / "Whenever the P closes early, the
F route" (the last pair has no past/present antecedent; capability may fail — reported). Frozen band [0.5, 1.5] x 0.59.
PRICE (registered maximum): null panel 2 batches x (native + producer + set + 16 random) = 38; template panel 3 batches x (native +
producer + set + 16 nulls) = 57 -> 95 forwards; bar <= 104.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, os, random, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_random_head_set_null_v25 as v25
import run_modal_dod_battery_v66 as v66

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/modal_remoteness_dod_null_and_templates_v67_result.json"
CANDIDATE_ID = "modal_remoteness.would_vs_will.dod_null_and_templates_v67"
NULL_SEEDS = tuple(range(2201, 2217))
EXCLUDED = {(9, 4), (11, 3), (9, 1), (15, 5)}
POOL = [(l, h) for l in range(18) for h in range(9) if (l, h) not in EXCLUDED]
SETS = [tuple(sorted(random.Random(2026_09_18_67 + s).sample(POOL, 4))) for s in range(16)]
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, CAPABILITY_MIN, BAND, REF, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.85, (0.5, 1.5), 0.59, 1e-4
FORWARDS_MAX = 104
TEMPLATES = {"had_once": (lambda p, f: f"Had the {p} closed early, the {f} route", lambda p, f: f"Once the {p} closes early, the {f} route"),
             "perfect": (lambda p, f: f"If the {p} had closed, the {f} route", lambda p, f: f"When the {p} has closed, the {f} route"),
             "should_whenever": (lambda p, f: f"Should the {p} close early, the {f} route", lambda p, f: f"Whenever the {p} closes early, the {f} route")}


def build_templates():
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for name, (rm, fm) in TEMPLATES.items():
        for g in range(16):
            for remote in (True, False):
                text = (rm if remote else fm)(v66.PLACES[g], v66.ADJ[g]); ids = L.ENCODING.encode(text)
                answer, foil = (" would", " will") if remote else (" will", " would"); a_id, f_id = L._single(answer), L._single(foil)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise SystemExit(f"joint tokenization failed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["modal_t67", name, g, remote, text]).encode()).hexdigest()[:24], name, g, remote, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows


def main() -> None:
    rows, templates = v66.build(), build_templates()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "templates": len(templates), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, v66.SET + tuple(c for s in SETS for c in v25.components_for(s)), v66.WOULD, v66.WILL)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    arm, n = v1._run_arm(fw, rows, components=v66.SET, mode="project"); forwards += n
    comp = L.summarize(rows, native, arm)
    randoms = []
    for s in SETS:
        arm, n = v1._run_arm(fw, rows, components=v25.components_for(s), mode="project"); forwards += n
        sm = L.summarize(rows, native, arm); randoms.append({"set": [f"{l}.{h}" for l, h in s], "damage": sm["target_damage_mean"], "live": sm["target_damage_fraction"] >= LIVE_FRACTION and sm["target_damage_positive_fraction"] >= LIVE_POSITIVE})
    dmg = [r["damage"] for r in randoms]
    nat_t, n = v1._run_arm(fw, templates); forwards += n
    ref_t, n = v1._producer_native(backend, templates); forwards += n
    instrument = max(instrument, max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(nat_t, ref_t)))
    set_t, n = v1._run_arm(fw, templates, components=v66.SET, mode="project"); forwards += n
    nulls_t = []
    for seed in NULL_SEEDS:
        a2, n = v1._run_arm(fw, templates, components=v66.SET, mode="project_random", seed=seed); forwards += n; nulls_t.append(a2)
    report, capability = {}, {}
    for c in TEMPLATES:
        idx = [i for i, r in enumerate(templates) if r.construction == c]; sub, subn = [templates[i] for i in idx], [nat_t[i] for i in idx]
        for remote in (True, False):
            cell = [1.0 if nat_t[i]["answer"] > nat_t[i]["foil"] else 0.0 for i in idx if templates[i].present == remote]
            capability[f"{c}/{'remote' if remote else 'factual'}"] = sum(cell) / len(cell)
        s = L.summarize(sub, subn, [set_t[i] for i in idx]); ns = [L.summarize(sub, subn, [nl[i] for i in idx]) for nl in nulls_t]
        null_max = max(x["target_damage_mean"] for x in ns); null_moves = {name: sum(x[f"{name}_abs_move_mean"] for x in ns) / len(ns) for name in L.UNRELATED}
        gates = {name: s[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * s["target_damage_mean"] for name in L.UNRELATED}
        capable = all(capability[f"{c}/{k}"] >= CAPABILITY_MIN for k in ("remote", "factual"))
        transfers = s["target_damage_fraction"] >= LIVE_FRACTION and s["target_damage_positive_fraction"] >= LIVE_POSITIVE and s["target_damage_mean"] > null_max and BAND[0] * REF <= s["target_damage_fraction"] <= BAND[1] * REF
        report[c] = {"capable": capable, "set": s, "null_damage_max": null_max, "gates": gates, "selective": all(gates.values()), "transfers": transfers}
        print(c, "capable", capable, "fraction", round(s["target_damage_fraction"], 3), "pos", round(s["target_damage_positive_fraction"], 3), "null_max", round(null_max, 4), "gates", gates)
    capable = [c for c in TEMPLATES if report[c]["capable"]]
    print("random null: set", round(comp["target_damage_mean"], 3), "random max", round(max(dmg), 3), "live randoms", sum(r["live"] for r in randoms))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_set_beats_every_random_quadruple": comp["target_damage_mean"] > max(dmg), "pred_c_random_quadruples_are_not_live": not any(r["live"] for r in randoms),
                   "pred_d_two_of_three_templates_capable": len(capable) >= 2, "pred_e_set_removal_transfers_to_every_capable_template": bool(capable) and all(report[c]["transfers"] for c in capable),
                   "pred_f_set_removal_selective_on_every_capable_template": bool(capable) and all(report[c]["selective"] for c in capable)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "modal_remoteness_dod_null_and_templates_result_v67", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "set": comp, "random": randoms, "capability": capability, "templates": report,
                               "capable": capable, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
