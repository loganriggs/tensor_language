#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, budget, hub family and bars fixed before the run.
"""v66: where does the removal-greedy saturate, and does it stay inside the hub family?

v64 (verb_complementizer, +3 heads: A2 0.35->0.60) and v65 (verb_preposition, +3: A1 0.20->0.48) both stopped at the
registered budget with the marginal even-row gain still ~0.07-0.10, so neither set is saturated. Same greedy (v51
removal damage, diff-in-means direction refit per candidate set, selection on A1 EVEN rows, evaluation on A1 ODD rows)
run to 8 additions on both sets. HUB_FAMILY = the union of the six v23/v15 sets and the v54 enlargements (18 heads),
fixed here before the run.

REGISTERED BEFORE THE RUN (A1 odd-row removal damage in nat)
    pred_a_plateau        for both sets the even-row marginal gain of the 8th addition is < 0.03.
                          Worked: gains 0.10,0.09,0.08,0.05,0.04,0.03,0.02,0.02 True; 8th gain 0.05 False.
    pred_b_ceiling        both final (hub+8) sets remove >= 0.60 on A1 odd rows (above every 3-5 head set so far).
                          Worked: 0.65 True; 0.52 False.
    pred_c_hub_family     >= 50% of the 16 added heads are in HUB_FAMILY. Worked: 10/16 True; 6/16 False.
    pred_d_cross_clean    each final set's cross-collateral CE (A1-fit direction) on every v23 A1 family <= 0.10 x its own
                          A1 odd removal. Worked: 0.03 vs 0.65 True; 0.08 vs 0.65 False.
    pred_e_transfer       total even-row gain transfers to odd rows at >= 0.50x for both sets.
                          Worked: even +0.45, odd +0.30 True; even +0.45, odd +0.20 False.
    Prior: a unsure (v65's gains were flat), b unsure, c True (hubs multiplex), d unsure (8 additions is where a
    generic-damage head could enter), e True.
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
OUT = ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json"
NAMES = ("verb_complementizer", "verb_preposition")
HUB_FAMILY = {"attn:07:head:08", "attn:11:head:03", "attn:06:head:03", "attn:13:head:08", "attn:14:head:08", "attn:08:head:01",
              "attn:04:head:07", "attn:03:head:00", "attn:01:head:05", "attn:00:head:03", "attn:04:head:01", "attn:05:head:03",
              "attn:13:head:01", "attn:09:head:07", "attn:14:head:03", "attn:10:head:05", "attn:02:head:02", "attn:08:head:08"}
PLATEAU, CEILING, FAMILY_FRAC, CROSS_FRAC, HOLD_FRAC, MAX_ADD = 0.03, 0.60, 0.50, 0.10, 0.50, 8
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 5400, 250000


def _plan():
    return {"candidate_id": "corpus.unit_verb_greedy_saturation_v66", "sets": {n: v15.SETS[n][1] for n in NAMES}, "max_add": MAX_ADD,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cross = {k: g.prepare(backend, g.rows_of(m, "A1")) for k, (m, _) in v23.SETS.items()}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    def damage(fit, ev, units):
        q = g.block_diff_in_means(backend, fit, units)
        return v51.summary(torch, v51.removal(backend, ev, units, q, mu_of(fit, units)))

    R = {}
    for name in NAMES:
        module, hub = v15.SETS[name][:2]
        hub = list(hub)
        rows = g.rows_of(module, "A1")
        even, odd = g.prepare(backend, rows[0::2]), g.prepare(backend, rows[1::2])
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
                          "odd_damage": o["ce_damage"], "odd_lb": o["ce_lb975"],
                          "runner_up": sorted(scores.items(), key=lambda kv: -kv[1])[1:3]})
        q = g.block_diff_in_means(backend, even, chosen)
        mu = mu_of(even, chosen)
        xc = {k: v51.summary(torch, v51.removal(backend, p, chosen, q, mu))["ce_damage"] for k, p in cross.items()}
        R[name] = {"hub": hub, "final": chosen, "base_even": base_even, "base_odd": base_odd, "curve": curve, "cross": xc,
                   "final_odd": curve[-1]["odd_damage"], "gain_even_total": cur - base_even, "gain_odd_total": curve[-1]["odd_damage"] - base_odd}

    added = [c["added"] for r in R.values() for c in r["curve"]]
    fam = sum(u in HUB_FAMILY for u in added)
    predictions = {
        'pred_a_plateau': all(r["curve"][-1]["gain_even"] < PLATEAU for r in R.values()),
        'pred_b_ceiling': all(r["final_odd"] >= CEILING for r in R.values()),
        'pred_c_hub_family': fam >= FAMILY_FRAC * len(added),
        'pred_d_cross_clean': all(x <= CROSS_FRAC * r["final_odd"] for r in R.values() for x in r["cross"].values()),
        'pred_e_transfer': all(r["gain_odd_total"] >= HOLD_FRAC * r["gain_even_total"] for r in R.values()),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_verb_greedy_saturation_result_v1", "candidate_id": "corpus.unit_verb_greedy_saturation_v66",
              "bars": {"plateau": PLATEAU, "ceiling": CEILING, "family_frac": FAMILY_FRAC, "cross_frac": CROSS_FRAC, "hold_frac": HOLD_FRAC, "max_add": MAX_ADD},
              "hub_family": sorted(HUB_FAMILY), "added_in_family": fam, "added_total": len(added), "sets": R,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "added_in_family": f"{fam}/{len(added)}",
                      "curves": {n: [(c["added"], round(c["gain_even"], 3), round(c["odd_damage"], 3)) for c in r["curve"]] for n, r in R.items()},
                      "base_odd": {n: round(r["base_odd"], 3) for n, r in R.items()},
                      "cross": {n: {k: round(v, 3) for k, v in r["cross"].items()} for n, r in R.items()},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
