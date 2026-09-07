#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, budget, hub family, v59 reference values and bars fixed before the run.
"""v67: the v66 removal-greedy saturation curve on the four v23 sets (quantifier, dative, polarity, voice).

v66: on the verb sets the removal-greedy reached 0.72 (plateau) and 1.12 (no plateau) nat at hub+8 with cross <= 0.014
and 10/16 additions inside the hub family. The four v23 sets were enlarged by INTERCHANGE-greedy (v54) and those
enlargements remove 0.592 / 0.365 / 0.361 / 0.208 (v59). Same greedy here (v51 damage, diff-in-means refit per
candidate set, A1 EVEN selection, A1 ODD evaluation, 8 additions), cross-collateral on the OTHER three v23 A1 families.

REGISTERED BEFORE THE RUN (A1 odd-row removal damage in nat)
    pred_a_plateau        for all four sets the even-row marginal gain of the 8th addition is < 0.03.
                          Worked: 8th gains 0.02,0.01,0.02,0.02 True; one set at 0.05 False.
    pred_b_beats_v54      each hub+8 set removes >= 1.25 x its v59 enlarged-set damage (0.74 / 0.46 / 0.45 / 0.26).
                          Worked: quantifier 0.80 vs 0.74 True; 0.70 False.
    pred_c_hub_family     >= 50% of the 32 added heads are in HUB_FAMILY. Worked: 18/32 True; 12/32 False.
    pred_d_cross_clean    each final set's cross-collateral CE on the other three v23 A1 families <= 0.10 x its own A1 odd
                          removal. Worked: 0.04 vs 0.74 True; 0.09 vs 0.74 False.
    pred_e_transfer       total even-row gain transfers to odd rows at >= 0.50x for all four sets.
                          Worked: even +0.40, odd +0.25 True; +0.40 / +0.15 False.
    Prior: a unsure, b True for quantifier/polarity (removal-greedy optimises the measured quantity), unsure for voice
    (weakest set, 0.208), c True, d unsure (quantifier and dative share the number axis), e True.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier2_characterization_v23 as v23
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"
NAMES = tuple(v23.SETS)
V59 = {"quantifier_number": 0.592, "dative": 0.365, "polarity_licensing": 0.361, "voice_frame": 0.208}
HUB_FAMILY = {"attn:07:head:08", "attn:11:head:03", "attn:06:head:03", "attn:13:head:08", "attn:14:head:08", "attn:08:head:01",
              "attn:04:head:07", "attn:03:head:00", "attn:01:head:05", "attn:00:head:03", "attn:04:head:01", "attn:05:head:03",
              "attn:13:head:01", "attn:09:head:07", "attn:14:head:03", "attn:10:head:05", "attn:02:head:02", "attn:08:head:08"}
PLATEAU, BEAT, FAMILY_FRAC, CROSS_FRAC, HOLD_FRAC, MAX_ADD = 0.03, 1.25, 0.50, 0.10, 0.50, 8
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 11000, 500000


def _plan():
    return {"candidate_id": "corpus.unit_four_sets_greedy_saturation_v67", "sets": {n: list(v23.SETS[n][1]) for n in NAMES}, "max_add": MAX_ADD,
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
        module, hub = v23.SETS[name][:2]
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
        xc = {k: v51.summary(torch, v51.removal(backend, p, chosen, q, mu))["ce_damage"] for k, p in cross.items() if k != name}
        R[name] = {"hub": hub, "final": chosen, "base_even": base_even, "base_odd": base_odd, "curve": curve, "cross": xc,
                   "final_odd": curve[-1]["odd_damage"], "gain_even_total": cur - base_even, "gain_odd_total": curve[-1]["odd_damage"] - base_odd}

    added = [c["added"] for r in R.values() for c in r["curve"]]
    fam = sum(u in HUB_FAMILY for u in added)
    predictions = {
        'pred_a_plateau': all(r["curve"][-1]["gain_even"] < PLATEAU for r in R.values()),
        'pred_b_beats_v54': all(r["final_odd"] >= BEAT * V59[n] for n, r in R.items()),
        'pred_c_hub_family': fam >= FAMILY_FRAC * len(added),
        'pred_d_cross_clean': all(x <= CROSS_FRAC * r["final_odd"] for r in R.values() for x in r["cross"].values()),
        'pred_e_transfer': all(r["gain_odd_total"] >= HOLD_FRAC * r["gain_even_total"] for r in R.values()),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_four_sets_greedy_saturation_result_v1", "candidate_id": "corpus.unit_four_sets_greedy_saturation_v67",
              "bars": {"plateau": PLATEAU, "beat": BEAT, "v59": V59, "family_frac": FAMILY_FRAC, "cross_frac": CROSS_FRAC, "hold_frac": HOLD_FRAC, "max_add": MAX_ADD},
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
