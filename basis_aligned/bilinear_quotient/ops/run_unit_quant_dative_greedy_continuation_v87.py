#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, split and bars fixed before the run.
"""v87: greedy continuation (+8 heads) for quantifier_number and dative -- the two sets whose tier rows are set-limited.

v66/v67 stopped removal-greedy at hub+8 while every set was still gaining 0.018-0.037 nat per head; v83 continued voice and
found +0.270 nat even / +0.204 odd and exact extraction 0.809 -> 0.857, which then let one direction meet rows 2/3/4/5 (v85).
quantifier_number has row 2 open at the set level (exact 0.816, direction 0.757) and dative's A2 extraction is 0.45 against
0.85 for A1. Same protocol as v83: removal-greedy on EVEN A1 with the diff-in-means direction, eight additions, then exact-set
extraction (A1, A2) and dim removal (A1, A2, C, other five) on ODD rows at hub8 and hub16.

REGISTERED BEFORE THE RUN (extraction = fraction of full-ablation margin restored; removal = CE damage in nat; ODD rows)
    pred_a_reproduce      hub8 exact extraction within 0.02 of v68 (quantifier 0.816, dative 0.877) and hub8 odd removal within 0.02 of v67 (0.720, 0.512).
    pred_b_not_plateaued  even removal gain over the eight additions >= 0.10 on BOTH sets. Worked: 0.15, 0.12 True; 0.15, 0.08 False.
    pred_c_row2_quant     quantifier hub16 exact extraction A1 >= 0.85 with lb95 >= 0.75. Worked: 0.86 / 0.80 True; 0.84 False.
    pred_d_transfer       odd removal gain >= 0.50 x even gain on both AND hub16 A2 exact extraction >= hub8 A2 - 0.02 on both.
    pred_e_cross_clean    hub16 dim removal on the other five behaviours <= 0.05 (signed, as v83) and own C <= 0.05 on both.
    Prior: a True; b quantifier ~90% (last gain 0.037), dative ~55% (last gains 0.018 -> eight of them is 0.14 only if they hold);
    c ~60%; d ~70%; e ~75% (quantifier hub8 dim removal is already -0.11 signed on dative/polarity, which the signed bar passes;
    the absolute collateral is the constrained direction's job, not the set's).
    b False on dative: dative stays at hub+8 and the A2 gap is not a set-size problem.
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
OUT = ROOT / "circuits/followups/unit_quant_dative_greedy_continuation_v87_result.json"
SRC = ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"
V68 = ROOT / "circuits/followups/unit_greedy_sets_extraction_v68_result.json"
NAMES, MAX_ADD = ("quantifier_number", "dative"), 8
REPRO_EXT, REPRO_REM, GAIN_MIN, ROW2, ROW2_LB, HOLD, A2_SLACK, CROSS_MAX = 0.02, 0.02, 0.10, 0.85, 0.75, 0.50, 0.02, 0.05
V68_EXACT, V67_ODD = {"quantifier_number": 0.816, "dative": 0.877}, {"quantifier_number": 0.720, "dative": 0.512}


def _plan():
    return {"candidate_id": "corpus.unit_quant_dative_greedy_continuation_v87", "sets": list(NAMES), "max_add": MAX_ADD,
            "model_forwards_max": 4000, "example_evaluations_max": 60000, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    all_heads = g.all_head_units()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    ALL = {}
    for NAME in NAMES:
        module = modules[NAME]
        hub8 = json.loads(SRC.read_text())["sets"][NAME]["final"]
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

        ALL[NAME] = {"hub8": hub8, "hub16": chosen, "curve": curve, "base_even": base_even, "base_odd": base_odd, "gain_even": cur - base_even, "sets": S}
    def ok(f):
        return all(f(n, ALL[n]) for n in NAMES)
    ext = lambda A, l: A["sets"][l]["extraction"]
    rem = lambda A, l: A["sets"][l]["removal"]
    predictions = {
        'pred_a_reproduce': ok(lambda n, A: abs(ext(A, "hub8")["A1"]["point"] - V68_EXACT[n]) <= REPRO_EXT and abs(rem(A, "hub8")["A1"]["ce_damage"] - V67_ODD[n]) <= REPRO_REM),
        'pred_b_not_plateaued': ok(lambda n, A: A["gain_even"] >= GAIN_MIN),
        'pred_c_row2_quant': ext(ALL["quantifier_number"], "hub16")["A1"]["point"] >= ROW2 and ext(ALL["quantifier_number"], "hub16")["A1"]["lb95"] >= ROW2_LB,
        'pred_d_transfer': ok(lambda n, A: rem(A, "hub16")["A1"]["ce_damage"] >= rem(A, "hub8")["A1"]["ce_damage"] + HOLD * A["gain_even"]
                              and ext(A, "hub16")["A2"]["point"] >= ext(A, "hub8")["A2"]["point"] - A2_SLACK),
        'pred_e_cross_clean': ok(lambda n, A: all(v <= CROSS_MAX for v in rem(A, "hub16")["cross"].values()) and rem(A, "hub16")["C"]["ce_damage"] <= CROSS_MAX),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_greedy_continuation_two_sets_result_v1", "candidate_id": "corpus.unit_quant_dative_greedy_continuation_v87",
              "bars": {"repro_ext": REPRO_EXT, "repro_rem": REPRO_REM, "gain_min": GAIN_MIN, "row2": ROW2, "row2_lb": ROW2_LB, "hold": HOLD, "a2_slack": A2_SLACK, "cross_max": CROSS_MAX,
                       "v68_exact": V68_EXACT, "v67_odd": V67_ODD},
              "sets": ALL, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "gain_even": {n: round(ALL[n]["gain_even"], 3) for n in NAMES}, "added": {n: ALL[n]["hub16"][len(ALL[n]["hub8"]):] for n in NAMES},
                      "ext": {n: {l: {k: round(v["point"], 3) for k, v in ALL[n]["sets"][l]["extraction"].items()} for l in ALL[n]["sets"]} for n in NAMES},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
