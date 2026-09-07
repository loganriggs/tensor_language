#!/usr/bin/env python3
# BQGATE: five frozen predictions; set, budget, selection rows and bars fixed before the run.
"""v83: voice_frame is set-limited at hub+8 (exact-set extraction 0.809) -- does the removal-greedy, continued to hub+16, lift it?

v67's removal-greedy on voice had not plateaued at step 8 (even-row gains 0.02-0.033 per head, final odd removal 0.322 nat);
v68 measured exact-set extraction 0.809 on A1 odd and v81 0.752 for the rank-1 direction. Every other set's row 2 is
direction- or objective-limited; voice's is the set. Here the identical greedy (v51 removal damage with per-block
diff-in-means refit per candidate set, selection on A1 EVEN rows) continues from v67's final set for 8 more additions.
Measured on ODD rows at hub+8 (reproduction) and hub+16: exact-set extraction on A1 and A2 (v52/v68 procedure: all
other heads mean-ablated, set heads live), diff-in-means removal on A1 / A2 / own C, cross-collateral on the other five
behaviours' A1 rows. No fitted direction here -- that is the next screen if the set improves.

REGISTERED BEFORE THE RUN
    pred_a_reproduce      hub+8 exact-set extraction on A1 odd within 0.02 of v68 (0.809) and hub+8 dim removal on A1 odd within
                          0.02 of v67 (0.322). Worked: 0.805 / 0.320 True; 0.78 False.
    pred_b_not_plateaued  total even-row removal gain of the 8 new additions >= 0.10 nat. Worked: +0.15 True; +0.06 False.
    pred_c_row2_exact     hub+16 exact-set extraction on A1 odd >= 0.85 with lb95 >= 0.75. Worked: 0.88 (0.80) True; 0.83 False.
    pred_d_transfer       hub+16 dim removal on A1 odd >= hub+8 + 0.50 x the even gain, AND A2 odd extraction (exact) at hub+16 >= hub+8 - 0.02.
                          Worked: 0.322+0.075 -> 0.41 True; 0.34 False.
    pred_e_cross_clean    hub+16 dim direction: CE damage on each of the other five A1 odd families <= 0.05 nat AND own C odd <= 0.05.
                          Worked: max 0.03, C 0.01 True; 0.07 False.
    Prior: a True; b True (curve still rising); c is the hypothesis and I am unsure -- 0.809 -> 0.85 needs the missing
    content to be in heads (an MLP-carried remainder would not move); d True if c; e unsure (voice's greedy pulled from
    layers 0-2, which are behaviour-general). c False with b True: the set is not the limit either -- report as
    "extraction limited by non-head components", no MLP will be added without its own registered screen.
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
import run_unit_selective_removal_four_sets_v51 as v51
import run_unit_extraction_four_sets_v52 as v52

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_voice_greedy_continuation_v83_result.json"
SRC = ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"
V68 = ROOT / "circuits/followups/unit_greedy_sets_extraction_v68_result.json"
NAME, MAX_ADD = "voice_frame", 8
REPRO_EXT, REPRO_REM, GAIN_MIN, ROW2, ROW2_LB, HOLD, A2_SLACK, CROSS_MAX = 0.02, 0.02, 0.10, 0.85, 0.75, 0.50, 0.02, 0.05
V68_EXACT, V67_ODD = 0.809, 0.322


def _plan():
    return {"candidate_id": "corpus.unit_voice_greedy_continuation_v83", "set": NAME, "max_add": MAX_ADD,
            "model_forwards_max": 4000, "example_evaluations_max": 60000, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    all_heads = g.all_head_units()
    module = v23.SETS[NAME][0]
    hub8 = json.loads(SRC.read_text())["sets"][NAME]["final"]
    v68_ref = json.loads(V68.read_text())
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    rows = g.rows_of(module, "A1")
    even, odd = g.prepare(backend, rows[0::2]), g.prepare(backend, rows[1::2])
    odd_a2, odd_c = g.prepare(backend, g.rows_of(module, "A2")[1::2]), g.prepare(backend, g.rows_of(module, "C")[1::2])
    cross = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items() if n != NAME}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    def damage(fit, ev, units):
        q = g.block_diff_in_means(backend, fit, units)
        return v51.summary(torch, v51.removal(backend, ev, units, q, mu_of(fit, units)))

    def exact_extraction(prep, units):
        mu_all = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in all_heads}
        order = v52.ordered_units(units)
        bq = v52.block_q(torch, backend.device, units, g.block_identity(backend, units))
        b0 = v52.block_q(torch, backend.device, units, None)
        nat, null, arm = [], [], []
        for side in ("base", "donor"):
            batch = prep.base_batch if side == "base" else prep.donor_batch
            cache = prep.base_cache if side == "base" else prep.donor_cache
            bg = dict(cache)
            for rid in batch.row_ids:
                for u in all_heads:
                    bg[(rid, u)] = mu_all[u]
            af_n = g.forward_units(backend, batch, units=[])
            af_0 = g.forward_units(backend, batch, units=order, donor_cache=bg, base_cache=cache, q=b0, complement=True)
            af_a = g.forward_units(backend, batch, units=order, donor_cache=bg, base_cache=cache, q=bq, complement=True)
            nat += (af_n[:, 0] - af_n[:, 1]).tolist(); null += (af_0[:, 0] - af_0[:, 1]).tolist(); arm += (af_a[:, 0] - af_a[:, 1]).tolist()
        return v52._boot_ratio(torch, [a - z for a, z in zip(arm, null)], [n - z for n, z in zip(nat, null)])

    chosen, curve = list(hub8), []
    cur = damage(even, even, chosen)["ce_damage"]
    base_even, base_odd = cur, damage(even, odd, chosen)["ce_damage"]
    pool = [u for u in all_heads if u not in chosen]
    for step in range(MAX_ADD):
        scores = {u: damage(even, even, chosen + [u])["ce_damage"] for u in pool}
        best = max(scores, key=scores.get)
        gain = scores[best] - cur
        chosen.append(best); pool.remove(best); cur = scores[best]
        o = damage(even, odd, chosen)
        curve.append({"step": len(hub8) + step + 1, "added": best, "even_damage": cur, "gain_even": gain, "odd_damage": o["ce_damage"], "odd_lb": o["ce_lb975"],
                      "runner_up": sorted(scores.items(), key=lambda kv: -kv[1])[1:3]})
        print("step", curve[-1]["step"], best, "even", round(cur, 3), "gain", round(gain, 3), "odd", round(o["ce_damage"], 3), round(time.perf_counter() - t0), "s", flush=True)

    S = {}
    for label, units in (("hub8", hub8), ("hub16", chosen)):
        q = g.block_diff_in_means(backend, even, units)
        mu = mu_of(even, units)
        S[label] = {"units": units, "extraction": {"A1": exact_extraction(odd, units), "A2": exact_extraction(odd_a2, units)},
                    "removal": {"A1": v51.summary(torch, v51.removal(backend, odd, units, q, mu)), "A2": v51.summary(torch, v51.removal(backend, odd_a2, units, q, mu)),
                                "C": v51.summary(torch, v51.removal(backend, odd_c, units, q, mu)),
                                "cross": {n: v51.summary(torch, v51.removal(backend, p, units, q, mu))["ce_damage"] for n, p in cross.items()}}}
        print(label, "ext", {k: round(v["point"], 3) for k, v in S[label]["extraction"].items()}, "rem", {k: round(S[label]["removal"][k]["ce_damage"], 3) for k in ("A1", "A2", "C")},
              "cross", {n: round(v, 3) for n, v in S[label]["removal"]["cross"].items()}, flush=True)

    gain_even = cur - base_even
    e8, e16 = S["hub8"]["extraction"], S["hub16"]["extraction"]
    r8, r16 = S["hub8"]["removal"], S["hub16"]["removal"]
    predictions = {
        'pred_a_reproduce': abs(e8["A1"]["point"] - V68_EXACT) <= REPRO_EXT and abs(r8["A1"]["ce_damage"] - V67_ODD) <= REPRO_REM,
        'pred_b_not_plateaued': gain_even >= GAIN_MIN,
        'pred_c_row2_exact': e16["A1"]["point"] >= ROW2 and e16["A1"]["lb95"] >= ROW2_LB,
        'pred_d_transfer': r16["A1"]["ce_damage"] >= r8["A1"]["ce_damage"] + HOLD * gain_even and e16["A2"]["point"] >= e8["A2"]["point"] - A2_SLACK,
        'pred_e_cross_clean': all(v <= CROSS_MAX for v in r16["cross"].values()) and r16["C"]["ce_damage"] <= CROSS_MAX,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_voice_greedy_continuation_result_v1", "candidate_id": "corpus.unit_voice_greedy_continuation_v83",
              "hub8": hub8, "hub16": chosen, "curve": curve, "base_even": base_even, "base_odd": base_odd, "gain_even_total": gain_even,
              "bars": {"repro_ext": REPRO_EXT, "repro_rem": REPRO_REM, "gain_min": GAIN_MIN, "row2": ROW2, "row2_lb": ROW2_LB, "hold": HOLD, "a2_slack": A2_SLACK, "cross_max": CROSS_MAX,
                       "v68_exact": V68_EXACT, "v67_odd": V67_ODD},
              "sets": S, "v68_reference_keys": list(v68_ref.keys())[:5],
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "gain_even": round(gain_even, 3), "hub16_added": chosen[len(hub8):],
                      "ext": {l: {k: round(v["point"], 3) for k, v in S[l]["extraction"].items()} for l in S}, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
