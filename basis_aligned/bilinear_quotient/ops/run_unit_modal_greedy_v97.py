#!/usr/bin/env python3
# BQGATE: five frozen predictions; behaviour, pipeline, hub family and bars fixed before the run.
"""v97: a seventh behaviour through the greedy head-set pipeline -- modal remoteness (would / will).

The six greedy sets (quantifier, dative, polarity, voice, complementizer, preposition) are at rubric rows 2/3/4/5 or 2/3/5
with explained residuals. `modal_remoteness.would_vs_will` has a terminal resid:18 record (A1/A2 1.000, P 0.084, C 0.158)
but no head-level set. Cues: 'If the window closed early, the short route' -> would; 'When the window closes early, the
short route' -> will (two agreeing cues, if/when and the verb tense, five to six tokens before the answer). Pipeline as
for the others: (1) 162-head interchange sweep on EVEN A1 rows and forward selection to a hub (<= 6 heads, target 0.50
recovery, min gain 0.02); (2) removal-greedy +8 (v66 recipe: diff-in-means direction refit per candidate set, mean-ablation
CE damage, selection on EVEN, evaluation on ODD); (3) the final set's ODD exact extraction, ODD A1 / A2 / P removal, and
cross-collateral on the six v80 A1 families. HUB_FAMILY is v66's 18-head family, fixed here.

REGISTERED BEFORE THE RUN (recovery = fraction of the donor margin; removal = CE damage in nat on ODD rows)
    pred_a_localizable   the extraction-greedy hub (<= 6 heads) recovers >= 0.50 of the donor margin on ODD A1 rows.
                         Worked: 0.62 True; 0.41 False.
    pred_b_hub_family    >= 2 of the final set's heads are in HUB_FAMILY (hub heads multiplex across behaviours).
    pred_c_removal       the hub+8 set removes >= 0.60 nat on ODD A1 and the +8 adds >= 0.15 over the hub. Worked: hub 0.45,
                         final 0.72 True; hub 0.50, final 0.62 False (gain 0.12).
    pred_d_cross_clean   the final set's diff-in-means direction costs <= 0.10 x its own ODD A1 removal on every one of the
                         six v80 A1 families. Worked: 0.05 vs 0.70 True; 0.09 vs 0.70 False.
    pred_e_specific      ODD A2 removal >= 0.50 x ODD A1 removal AND ODD P removal <= 0.10 x ODD A1 removal.
    Prior: a ~70%; b ~80%; c ~60%; d ~60% (temporal directions may share heads with quantifier's number axis); e ~65%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
NAME = "modal_remoteness"
HUB_FAMILY = {"attn:07:head:08", "attn:11:head:03", "attn:06:head:03", "attn:13:head:08", "attn:14:head:08", "attn:08:head:01",
              "attn:04:head:07", "attn:03:head:00", "attn:01:head:05", "attn:00:head:03", "attn:04:head:01", "attn:05:head:03",
              "attn:13:head:01", "attn:09:head:07", "attn:14:head:03", "attn:10:head:05", "attn:02:head:02", "attn:08:head:08"}
LOCAL, FAMILY_MIN, CEILING, GAIN, CROSS_FRAC, A2_FRAC, P_FRAC, MAX_ADD = 0.50, 2, 0.60, 0.15, 0.10, 0.50, 0.10, 8
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3600, 160000


def _plan():
    return {"candidate_id": "corpus.unit_modal_greedy_v97", "max_add": MAX_ADD, "hub_max": 6,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    cross = {k: g.prepare(backend, g.rows_of(m, "A1")) for k, m in modules.items()}

    a1 = g.rows_of(m_modal, "A1")
    even, odd = g.prepare(backend, a1[0::2]), g.prepare(backend, a1[1::2])
    a2 = g.prepare(backend, g.rows_of(m_modal, "A2")[1::2])
    p = g.prepare(backend, g.rows_of(m_modal, "P")[1::2])

    singles, ranked, hub = g.greedy_heads(backend, even, pool=12, target=LOCAL, min_gain=0.02, max_units=6)
    hub = list(hub["chosen"])
    hub_odd_ext = g.recovery(odd, g.patched_axis(backend, odd, hub))
    print("hub", hub, "even ext", round(g.recovery(even, g.patched_axis(backend, even, hub)), 3), "odd ext", round(hub_odd_ext, 3), round(time.perf_counter() - t0), "s", flush=True)

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    def damage(fit, ev, units):
        q = g.block_diff_in_means(backend, fit, units)
        return v51.summary(torch, v51.removal(backend, ev, units, q, mu_of(fit, units)))

    chosen, curve = list(hub), []
    cur = damage(even, even, chosen)["ce_damage"]
    base_even, base_odd = cur, damage(even, odd, chosen)["ce_damage"]
    pool = [u for u in g.all_head_units() if u not in hub]
    for step in range(MAX_ADD):
        scores = {u: damage(even, even, chosen + [u])["ce_damage"] for u in pool}
        best = max(scores, key=scores.get)
        gain = scores[best] - cur
        chosen.append(best); pool.remove(best); cur = scores[best]
        o = damage(even, odd, chosen)
        curve.append({"step": step + 1, "added": best, "in_family": best in HUB_FAMILY, "even_damage": cur, "gain_even": gain,
                      "odd_damage": o["ce_damage"], "odd_lb": o["ce_lb975"]})
        print("step", step + 1, best, round(gain, 3), "odd", round(o["ce_damage"], 3), round(time.perf_counter() - t0), "s", flush=True)

    q = g.block_diff_in_means(backend, even, chosen)
    mu = mu_of(even, chosen)
    final = {"A1": v51.summary(torch, v51.removal(backend, odd, chosen, q, mu)), "A2": v51.summary(torch, v51.removal(backend, a2, chosen, q, mu)),
             "P": v51.summary(torch, v51.removal(backend, p, chosen, q, mu))}
    xc = {k: v51.summary(torch, v51.removal(backend, pr, chosen, q, mu))["ce_damage"] for k, pr in cross.items()}
    final_ext = g.recovery(odd, g.patched_axis(backend, odd, chosen))
    a1r = final["A1"]["ce_damage"]
    in_fam = [u for u in chosen if u in HUB_FAMILY]
    predictions = {
        'pred_a_localizable': hub_odd_ext >= LOCAL,
        'pred_b_hub_family': len(in_fam) >= FAMILY_MIN,
        'pred_c_removal': a1r >= CEILING and (a1r - base_odd) >= GAIN,
        'pred_d_cross_clean': all(x <= CROSS_FRAC * a1r for x in xc.values()),
        'pred_e_specific': final["A2"]["ce_damage"] >= A2_FRAC * a1r and final["P"]["ce_damage"] <= P_FRAC * a1r,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_new_behaviour_greedy_result_v1", "candidate_id": "corpus.unit_modal_greedy_v97", "name": NAME,
              "singles_top12": {u: singles[u] for u in ranked[:12]}, "hub": hub, "hub_odd_extraction": hub_odd_ext, "hub_even_removal": base_even, "hub_odd_removal": base_odd,
              "curve": curve, "final": chosen, "final_in_family": in_fam, "final_odd_extraction": final_ext, "final_removal": final, "cross": xc,
              "bars": {"local": LOCAL, "family_min": FAMILY_MIN, "ceiling": CEILING, "gain": GAIN, "cross_frac": CROSS_FRAC, "a2_frac": A2_FRAC, "p_frac": P_FRAC, "max_add": MAX_ADD},
              "hub_family": sorted(HUB_FAMILY), "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "hub": hub, "hub_odd_extraction": round(hub_odd_ext, 3), "final": chosen, "in_family": in_fam,
                      "final_odd_extraction": round(final_ext, 3), "removal": {k: round(v["ce_damage"], 3) for k, v in final.items()}, "hub_odd_removal": round(base_odd, 3),
                      "cross": {k: round(v, 3) for k, v in xc.items()}, "seconds": result["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
