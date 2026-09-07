#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, clamp groups and bars fixed before the run; no fitting.
"""v105: do the seven greedy HEAD sets express their effect through the mlp7-10 band?

v104: hub head 07:08's indirect effect is read by the mlp7-10 band (MLP clamp shares 0.25-1.33 vs attention 0.05-0.23)
in four behaviours. The greedy sets are head-only, so whatever their heads write must be read by blocks OUTSIDE the set.
Instrument: exact-set interchange (all set heads from the donor) while one group of NON-set units is clamped to base:
    mlp_band = mlp:07..10   mlp_late = mlp:11..17   mlp_early = mlp:00..06   attn_other = every head not in the set
    all = every non-set unit (only the set's direct writes to the final norm survive).
survival(group) = clamped effect / exact-set effect (summed axis change over ODD A1 rows). v103 measured direct shares
(dative 0.77, modal 0.57, preposition 0.47, quantifier 0.40, polarity 0.29, complementizer 0.23, voice 0.10).

REGISTERED BEFORE THE RUN
    pred_a_band_carries_indirect   survival(mlp_band) <= 0.50 for voice, complementizer, polarity. Worked: 0.3 True; 0.7 False.
    pred_b_direct_sets_spared      survival(mlp_band) >= 0.70 for dative and modal. Worked: 0.8 True; 0.5 False.
    pred_c_instrument              |survival(all) - v103 direct share| <= 0.15 on >= 5 of 7 (v104 saw dative 0.98 vs 0.77).
    pred_d_band_over_late          1 - survival(mlp_band) > 1 - survival(mlp_late) on >= 5 of 7.
    pred_e_band_over_attention     1 - survival(mlp_band) > 1 - survival(attn_other) on >= 5 of 7.
    Prior: a 60%; b 55%; c 65%; d 70%; e 65%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23
import circuit_fast_screen_candidate_modal_remoteness as m_modal

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_seven_band_readers_v105_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
V103 = ROOT / "circuits/followups/unit_seven_head_direct_v103_result.json"
INDIRECT, DIRECT = ("voice_frame", "verb_complementizer", "polarity_licensing"), ("dative", "modal_remoteness")
BAND_MAX, DIRECT_MIN, INSTR_TOL, K = 0.50, 0.70, 0.15, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_seven_band_readers_v105",
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")},
               "modal_remoteness": m_modal}
    sets = {n: s["units"] for n, s in json.loads(V80.read_text())["sets"].items()}
    sets["modal_remoteness"] = json.loads(V97.read_text())["final"]
    v103 = json.loads(V103.read_text())["sets"]
    heads_all = [f"attn:{L:02d}:head:{h:02d}" for L in range(g.N_LAYERS) for h in range(g.N_HEADS)]
    mlps = lambda lo, hi: [f"mlp:{L:02d}" for L in range(lo, hi + 1)]

    def effect(O, set_units, clamp):
        merged = dict(O.base_cache)
        for rid in O.base_batch.row_ids:
            for u in set_units:
                merged[(rid, u)] = O.donor_cache[(rid, u)]
            for u in clamp:
                assert (rid, u) in O.base_cache, u
        out = g.forward_units(backend, O.base_batch, units=list(set_units) + list(clamp), donor_cache=merged, base_cache=O.base_cache)
        return sum(-(float(a) - float(f)) - b for (a, f), b in zip(out.tolist(), O.base_axis))

    report = {}
    for n, units in sets.items():
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        exact = effect(O, units, [])
        ref = sum(e - b for e, b in zip(g.patched_axis(backend, O, units), O.base_axis))
        assert abs(exact - ref) <= 1e-3 * max(abs(ref), 1.0), (exact, ref)
        others = [u for u in heads_all if u not in units]
        groups = {"mlp_band": mlps(7, 10), "mlp_late": mlps(11, 17), "mlp_early": mlps(0, 6), "attn_other": others,
                  "all": mlps(0, 17) + others}
        surv = {k: round(effect(O, units, c) / exact, 3) for k, c in groups.items()}
        report[n] = {"units": units, "exact": round(exact, 3), "survival": surv, "v103_direct_share": v103[n]["direct_share"]}
        print(n, report[n]["exact"], surv, "v103", v103[n]["direct_share"], flush=True)

    S = {n: report[n]["survival"] for n in report}
    predictions = {
        'pred_a_band_carries_indirect': all(S[n]["mlp_band"] <= BAND_MAX for n in INDIRECT),
        'pred_b_direct_sets_spared': all(S[n]["mlp_band"] >= DIRECT_MIN for n in DIRECT),
        'pred_c_instrument': sum(abs(S[n]["all"] - report[n]["v103_direct_share"]) <= INSTR_TOL for n in S) >= K,
        'pred_d_band_over_late': sum((1 - S[n]["mlp_band"]) > (1 - S[n]["mlp_late"]) for n in S) >= K,
        'pred_e_band_over_attention': sum((1 - S[n]["mlp_band"]) > (1 - S[n]["attn_other"]) for n in S) >= K,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_band_readers_result_v1", "candidate_id": "corpus.unit_seven_band_readers_v105",
              "summary": {n: {**S[n], "v103_direct_share": report[n]["v103_direct_share"]} for n in S}, "sets": report,
              "bars": {"band_max": BAND_MAX, "direct_min": DIRECT_MIN, "instr_tol": INSTR_TOL, "k": K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": result["summary"], "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
