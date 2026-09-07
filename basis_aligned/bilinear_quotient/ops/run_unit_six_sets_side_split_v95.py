#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets (v80 hub+8), recipe, split and bars fixed before the run.
"""v95: side-split removal for all six behaviours -- which rank-1 directions are one-sided markers with a default side?

v94 found the verb_complementizer direction is predominantly an interrogative marker: removing it costs ~2.0 nat on the
'whether' side and 0.03-0.4 on the 'that' side. Every removal number in the tier table is side-pooled ((base + donor) / 2), so a
one-sided direction reads as half its true effect on the marked side and its sibling C row 4 measures the DEFAULT side's leak.
This run fits the standard full-specificity direction on each hub+8 set (v80 recipe: rank 1 per block, pooled EVEN A1 + verb
variants, complement 1.0, own C EVEN + other five A1 EVEN as controls at 30 each, 120 steps, lr 0.05, seed 0; pred_a is an exact
reproduction of v80's pooled A1 odd removal) and splits the ODD A1 removal by side. In every family the ODD rows' base side is
the second answer and the donor side the first: quantifier was | were, dative to | for, polarity anything | something,
voice by | the, complementizer whether | that, preposition on | to (first | second).

REGISTERED BEFORE THE RUN (removal = CE damage in nat on the row's answer; ODD A1 rows; sides split; ratio = larger / smaller side)
    pred_a_reproduce      pooled A1 odd removal within 0.02 of v80's xctl value for all six sets.
    pred_b_complementizer whether side >= 3 x that side on the hub+8 set (v94 found 5x on hub+16). Worked: 1.9 / 0.4 True; 1.5 / 0.7 False.
    pred_c_polarity       'anything' side >= 3 x 'something' side (the NPI is the licensed, marked form; 'something' is the default).
    pred_d_quantifier     two-sided: ratio <= 2 (agreement needs the number on both sides). Worked: 0.9 / 0.7 True; 1.2 / 0.4 False.
    pred_e_voice          'by' side >= 2 x 'the' side (the passive agent marker; 'the' is the plain object continuation).
    Prior: a True; b ~85%; c ~55%; d ~65%; e ~55%. Dative (to | for) and preposition (on | to) are reported without a prediction:
    both alternations are lexical and I have no basis for a side.
    A one-sided verdict on a behaviour means its table removal numbers understate the marked side by ~2x and its row-4 sibling
    should be read as a default-side control; a two-sided verdict means the sibling is a full-strength control.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_six_sets_side_split_v95_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
REPRO, COMP_RATIO, POL_RATIO, QUANT_MAX, VOICE_RATIO = 0.02, 3.0, 3.0, 2.0, 2.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 900, 60000


def _plan():
    return {"candidate_id": "corpus.unit_six_sets_side_split_v95", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 6 * 2 * STEPS, "model_updates": 0, "fit_parameters": 6 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    v80 = json.loads(V80.read_text())["sets"]
    sets = {n: v80[n]["units"] for n in v80}
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    cross_even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}

    R = {}
    for name, units in sets.items():
        m = modules[name]
        a1 = g.rows_of(m, "A1")
        maps = v15.SETS[name][2] if name in v15.SETS else ()
        pool = g.prepare(backend, a1[0::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[0::2]])
        even_c = g.prepare(backend, g.rows_of(m, "C")[0::2])
        xc = (even_c,) + tuple(cross_even[n] for n in modules if n != name)
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache) for rid in pool.base_batch.row_ids]).mean(0) for u in units}
        q, _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu)

        def rem(rows):
            prep = g.prepare(backend, rows)
            d = v51.removal(backend, prep, units, q, mu)
            k = len(prep.base_batch.row_ids)
            s = v51.summary(torch, d)
            s["base_side"] = v51.summary(torch, {kk: v[:k] for kk, v in d.items()})
            s["donor_side"] = v51.summary(torch, {kk: v[k:] for kk, v in d.items()})
            return s

        odd = a1[1::2]
        sides = {"base": sorted(set(r["base_answer"] for r in odd)), "donor": sorted(set(r["donor_answer"] for r in odd))}
        a1_odd = rem(odd)
        c_odd = rem(g.rows_of(m, "C")[1::2])
        b, d = a1_odd["base_side"]["ce_damage"], a1_odd["donor_side"]["ce_damage"]
        R[name] = {"units": units, "answers": sides, "A1_odd": a1_odd, "C_odd": c_odd, "v80_pooled": v80[name]["damage"]["xctl"]["A1"]["ce_damage"],
                   "by_answer": {sides["base"][0].strip(): b, sides["donor"][0].strip(): d}, "ratio": max(b, d) / max(min(b, d), 1e-6), "larger_side": sides["base"][0].strip() if b >= d else sides["donor"][0].strip()}
        print(name, {sides["base"][0].strip(): round(b, 3), sides["donor"][0].strip(): round(d, 3)}, "pooled", round(a1_odd["ce_damage"], 3), "v80", round(R[name]["v80_pooled"], 3), "C sides", round(c_odd["base_side"]["ce_damage"], 3), round(c_odd["donor_side"]["ce_damage"], 3), round(time.perf_counter() - t0), "s", flush=True)

    ba = lambda n: R[n]["by_answer"]
    predictions = {
        'pred_a_reproduce': all(abs(R[n]["A1_odd"]["ce_damage"] - R[n]["v80_pooled"]) <= REPRO for n in R),
        'pred_b_complementizer': ba("verb_complementizer")["whether"] >= COMP_RATIO * ba("verb_complementizer")["that"],
        'pred_c_polarity': ba("polarity_licensing")["anything"] >= POL_RATIO * ba("polarity_licensing")["something"],
        'pred_d_quantifier': R["quantifier_number"]["ratio"] <= QUANT_MAX,
        'pred_e_voice': ba("voice_frame")["by"] >= VOICE_RATIO * ba("voice_frame")["the"],
    }
    summary = {n: {"sides": {k: round(v, 3) for k, v in R[n]["by_answer"].items()}, "ratio": round(R[n]["ratio"], 2), "larger": R[n]["larger_side"], "pooled": round(R[n]["A1_odd"]["ce_damage"], 3), "v80": round(R[n]["v80_pooled"], 3),
                   "C_sides": [round(R[n]["C_odd"]["base_side"]["ce_damage"], 3), round(R[n]["C_odd"]["donor_side"]["ce_damage"], 3)]} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_six_sets_side_split_result_v1", "candidate_id": "corpus.unit_six_sets_side_split_v95",
              "summary": summary, "sets": R, "bars": {"repro": REPRO, "comp_ratio": COMP_RATIO, "pol_ratio": POL_RATIO, "quant_max": QUANT_MAX, "voice_ratio": VOICE_RATIO},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
