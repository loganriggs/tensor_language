#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_pairwise_mobius_terms_below_gate pred_c_four_piece_gap_below_gate pred_d_module_split_no_less_additive_than_random_splits
"""Narrative tense DoD battery, step 6 (v47): COMPOSES with pairwise Möbius terms and a random-split null (as v32).

Set {15.5, 11.3, 9.4, 9.1} on `O_h^T(u_was - u_is)`, v43 rows. Six pairs; sixteen random coordinate splits of the same
total removal into four pieces.

PREDICTIONS: pred_a instrument <= 1e-4; pred_b |I(a,b)| <= 0.25 x min(dmg a, dmg b) for all six pairs (prior: the 11.3 x 9.1
serial term failed this on the temporal line only at the four-way level); pred_c |joint - sum singles| <= 0.25 x min single
(replay of v43's failure expected); pred_d normalized head-split gap <= median random-split gap.
PRICE (registered maximum): 2 batches x (native 1 + producer 1 + capture 1 + singles 4 + pairs 6 + joint 1 + 16 x 4) = 156; bar <= 170.
"""
from __future__ import annotations
from datetime import datetime, timezone
import itertools, json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_narrative_dod_sweep_and_set_v42 as v42
import run_narrative_dod_confirm_v43 as v43

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/narrative_tense_dod_composition_v47_result.json"
CANDIDATE_ID = "narrative_tense.past_vs_present.dod_composition_v47"
SPLIT_SEEDS = tuple(range(1401, 1417))
PAIR_RATIO, GAP_RATIO, INSTRUMENT_TOL = 0.25, 0.25, 1e-4
FORWARDS_MAX = 170
SINGLES = v43.SINGLES


def main() -> None:
    rows = v43.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SINGLES, v42.WAS, v42.IS)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    store = {}
    for start in range(0, len(rows), v1.BATCH):
        store.update(fw.capture(rows[start:start + v1.BATCH], SINGLES)); forwards += 1
    removal = {}
    for (rid, name, pos, head), w in store.items():
        v = fw.directions[(name, head)].float().to(w.device); v = v / v.norm()
        removal[(rid, name, pos, head)] = 2.0 * float(w.float() @ v) * v
    def damage(components, mode="project"):
        nonlocal forwards
        arm, n = v1._run_arm(fw, rows, components=components, mode=mode); forwards += n
        return L.summarize(rows, native, arm)["target_damage_mean"]
    singles = {c.name: damage((c,)) for c in SINGLES}
    pairs = {}
    for a, b in itertools.combinations(SINGLES, 2):
        j = damage((a, b)); inter = j - singles[a.name] - singles[b.name]; bar = PAIR_RATIO * min(singles[a.name], singles[b.name])
        pairs[f"{a.name}+{b.name}"] = {"joint": j, "interaction": inter, "bar": bar, "passes": abs(inter) <= bar}
    joint = damage(SINGLES); gap = abs(joint - sum(singles.values())); gap_bar = GAP_RATIO * min(singles.values()); module_g = gap / joint
    splits = []
    for seed in SPLIT_SEEDS:
        pieces = L.random_coordinate_split(removal, seed, pieces=4); pd = []
        for table in pieces:
            fw.subtract = table; pd.append(damage(SINGLES, mode="subtract"))
        splits.append({"seed": seed, "piece_damages": pd, "normalized_gap": abs(joint - sum(pd)) / joint})
    fw.use_subtract = False
    rg = sorted(s["normalized_gap"] for s in splits); rmed = rg[len(rg) // 2]
    print("singles", {k: round(v, 3) for k, v in singles.items()}, "joint", round(joint, 3), "gap/bar", round(gap, 4), round(gap_bar, 4), "pairs", {k: round(v["interaction"], 4) for k, v in pairs.items()}, "module_g", round(module_g, 4), "random med", round(rmed, 4))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_pairwise_mobius_terms_below_gate": all(p["passes"] for p in pairs.values()),
                   "pred_c_four_piece_gap_below_gate": gap <= gap_bar, "pred_d_module_split_no_less_additive_than_random_splits": module_g <= rmed}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "narrative_tense_dod_composition_result_v47", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "singles": singles, "pairs": pairs, "joint": joint, "gap": gap,
              "gap_bar": gap_bar, "module_normalized_gap": module_g, "random_splits": splits, "random_normalized_gap_median": rmed, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
