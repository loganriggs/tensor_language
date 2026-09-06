#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9), families, reference distribution and control fixed before the run.
"""v24: Tier 2 off-target, referenced to the DONOR, for the four v23 head sets.

v23 measured off-target as KL(base || patched) over the vocabulary minus the answer pair: 0.12-1.71 nat.
That predicate was mis-referenced: on answer-changing rows the donor sentence differs lexically, so the
donor's own off-target distribution differs from the base's, and a set that carries the donor's head
outputs is SUPPOSED to move the rest of the distribution toward the donor's. voice's worst-row gainers
were object names after "promoted" -- active-voice objects, i.e. toward the donor. The question is
whether the patch moves the off-target mass ALONG the base->donor path (on-manifold) or elsewhere.

Per behaviour x family (A1, A2), exact block-live set S and a size-matched random head set R (same
number of heads drawn from layers 0-14, seed 1, excluding S), off-target vocabulary V' = V minus the
answer pair, distributions renormalised on V':
    toward   KL(donor || patched) / KL(donor || base)   on V'  (< 1: moved toward the donor's tail)
    overshoot KL(base || patched) / KL(base || donor)   on V'  (> 1: moved MORE than the donor differs)
    native argmax  fraction of rows whose native-donor argmax is the donor answer, and whose native-base
                   argmax is the base answer
    competitor     log-prob shift of v23's top competitor token (' are' for quantifier, ' that' for
                   polarity, ' to' for dative/voice) under the patch
    control        the same ratios for R, and R's median recovery

REGISTERED BEFORE THE RUN
    pred_a_toward_donor       median toward <= 0.70 on A1 for every behaviour (the off-target mass moves at
                              least 30% of the way to the donor's). Worked: 0.45 -> True; 0.92 -> False.
    pred_b_no_overshoot       median overshoot <= 1.20 on A1 for every behaviour.
                              Worked: 0.80 -> True; 1.50 -> False.
    pred_c_tail_contrasts     native-donor argmax is the donor answer on <= 0.10 of A1 rows for polarity and
                              quantifier and >= 0.50 for dative and voice (v23: base-side 0.00/0.06/0.78/0.66).
                              Worked: 0.00, 0.03, 0.75, 0.62 -> True.
    pred_d_competitor_stable  median |log-prob shift| of the top competitor <= 0.50 nat on A1 for
                              quantifier and polarity (the set moves the answer axis, not the third token).
                              Worked: 0.21 -> True; 1.1 -> False.
    pred_e_random_calibration size-matched random set: median A1 recovery <= 0.10 AND median
                              KL(base||patched) on V' <= 0.25 x the set's, every behaviour.
                              Worked: rec 0.02, KL 0.03 vs 0.45 -> True.
    Reading rule. a and b True: the v23 off-target numbers are on-manifold movement toward the donor and the
    Tier 2 off-target field is `toward`/`overshoot`, not the raw KL; the four sets' Tier 2 is supported on
    direction, magnitude, negatives and off-target, with the tail-contrast caveat (c) recorded for polarity
    and quantifier. b False on a behaviour: the set writes more than the donor difference -- a feature
    writer broader than the behaviour; record and do not claim Tier 2 there.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier2_off_target_v24_result.json"
SETS = v23.SETS
ENC = v23.ENC
FAMILIES = ("A1", "A2")
COMPETITOR = {"quantifier_number": " are", "polarity_licensing": " that", "dative": " to", "voice_frame": " to"}
TOWARD_BAR, OVER_BAR, TAIL_LO, TAIL_HI, COMP_BAR, RAND_REC, RAND_KL = 0.70, 1.20, 0.10, 0.50, 0.50, 0.10, 0.25
TAIL = ("polarity_licensing", "quantifier_number")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 2000


def _plan():
    return {"candidate_id": "corpus.unit_tier2_off_target_v24", "sets": {k: v[1] for k, v in SETS.items()},
            "families": list(FAMILIES), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def _random_set(units, seed):
    rng = random.Random(seed)
    pool = [u for u in g.all_head_units() if g.unit_layer(u) <= 14 and u not in units]
    return sorted(rng.sample(pool, len(units)))


def _kl(lp_p, lp_q, keep):
    return (lp_p.exp() * (lp_p - lp_q)).masked_fill(~keep, 0.0).sum(-1)


def _family(backend, module, units, rand_units, family, comp_id):
    torch = backend.torch
    prep = g.prepare(backend, g.rows_of(module, family), valid_only=True)
    rows = prep.rows
    n = len(rows)
    _, base_logits = g.forward_units(backend, prep.base_batch, return_logits=True)
    _, donor_logits = g.forward_units(backend, prep.donor_batch, return_logits=True)
    af, pat_logits = g.forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                                     base_cache=prep.base_cache, return_logits=True)
    af_r, rand_logits = g.forward_units(backend, prep.base_batch, units=rand_units, donor_cache=prep.donor_cache,
                                        base_cache=prep.base_cache, return_logits=True)
    idx = torch.arange(n, device=base_logits.device)
    d_ans = torch.tensor([r["donor_answer_id"] for r in rows], device=idx.device)
    b_ans = torch.tensor([r["base_answer_id"] for r in rows], device=idx.device)
    keep = torch.ones_like(base_logits, dtype=torch.bool); keep[idx, d_ans] = False; keep[idx, b_ans] = False
    lp = {k: torch.log_softmax(v.masked_fill(~keep, -1e9), -1) for k, v in
          (("base", base_logits), ("donor", donor_logits), ("pat", pat_logits), ("rand", rand_logits))}
    kl_donor_base = _kl(lp["donor"], lp["base"], keep)
    kl_base_donor = _kl(lp["base"], lp["donor"], keep)
    toward = _kl(lp["donor"], lp["pat"], keep) / kl_donor_base
    over = _kl(lp["base"], lp["pat"], keep) / kl_base_donor
    toward_r = _kl(lp["donor"], lp["rand"], keep) / kl_donor_base
    kl_set = _kl(lp["base"], lp["pat"], keep); kl_rand = _kl(lp["base"], lp["rand"], keep)
    full_base, full_pat = torch.log_softmax(base_logits, -1), torch.log_softmax(pat_logits, -1)
    pat_axis = [-(float(a) - float(f)) for a, f in af.tolist()]
    rand_axis = [-(float(a) - float(f)) for a, f in af_r.tolist()]
    rec = [kernel.signed_pairwise_donor_recovery(b, d, p) for b, d, p in zip(prep.base_axis, prep.donor_axis, pat_axis)]
    rec_r = [kernel.signed_pairwise_donor_recovery(b, d, p) for b, d, p in zip(prep.base_axis, prep.donor_axis, rand_axis)]
    med = lambda t: float(t.median())
    return {"rows": n, "dropped": prep.dropped, "units": list(units), "random_units": list(rand_units),
            "median_toward": med(toward), "median_overshoot": med(over), "max_overshoot": float(over.max()),
            "median_kl_donor_base_off": med(kl_donor_base), "median_kl_base_patched_off": med(kl_set),
            "random_median_toward": med(toward_r), "random_median_kl_base_patched_off": med(kl_rand),
            "random_kl_ratio": med(kl_rand) / med(kl_set) if med(kl_set) > 0 else None,
            "median_recovery": float(statistics.median(rec)), "random_median_recovery": float(statistics.median(rec_r)),
            "native_donor_argmax_is_donor_answer": float((donor_logits.argmax(-1) == d_ans).float().mean()),
            "native_base_argmax_is_base_answer": float((base_logits.argmax(-1) == b_ans).float().mean()),
            "native_donor_top_tokens": [t for t, _ in __import__("collections").Counter(ENC.decode([int(t)]) for t in donor_logits.argmax(-1).tolist()).most_common(3)],
            "competitor_token": ENC.decode([comp_id]),
            "competitor_logprob_shift_median": med(full_pat[idx, comp_id] - full_base[idx, comp_id]),
            "competitor_abs_logprob_shift_median": med((full_pat[idx, comp_id] - full_base[idx, comp_id]).abs()),
            "competitor_native_donor_minus_base_median": med(torch.log_softmax(donor_logits, -1)[idx, comp_id] - full_base[idx, comp_id])}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        rand_units = _random_set(units, 1)
        comp_ids = ENC.encode(COMPETITOR[name]); assert len(comp_ids) == 1, (name, comp_ids)
        report[name] = {fam: _family(backend, module, units, rand_units, fam, comp_ids[0]) for fam in FAMILIES}
        a = report[name]["A1"]
        print(name, "toward %.2f over %.2f | rand toward %.2f klratio %.2f rec %.2f | donor argmax %.2f comp |d| %.2f (%s)" % (
            a["median_toward"], a["median_overshoot"], a["random_median_toward"], a["random_kl_ratio"] or -1,
            a["random_median_recovery"], a["native_donor_argmax_is_donor_answer"], a["competitor_abs_logprob_shift_median"],
            a["native_donor_top_tokens"]), flush=True)

    a1 = {k: v["A1"] for k, v in report.items()}
    predictions = {
        'pred_a_toward_donor': all(v["median_toward"] <= TOWARD_BAR for v in a1.values()),
        'pred_b_no_overshoot': all(v["median_overshoot"] <= OVER_BAR for v in a1.values()),
        'pred_c_tail_contrasts': all((v["native_donor_argmax_is_donor_answer"] <= TAIL_LO) if k in TAIL
                                     else (v["native_donor_argmax_is_donor_answer"] >= TAIL_HI) for k, v in a1.items()),
        'pred_d_competitor_stable': all(a1[k]["competitor_abs_logprob_shift_median"] <= COMP_BAR for k in TAIL),
        'pred_e_random_calibration': all(v["random_median_recovery"] <= RAND_REC and v["random_kl_ratio"] is not None
                                         and v["random_kl_ratio"] <= RAND_KL for v in a1.values()),
    }
    tier2_off_target = {k: bool(v["median_toward"] <= TOWARD_BAR and v["median_overshoot"] <= OVER_BAR) for k, v in a1.items()}
    result = {"predictions": predictions, "tier2_off_target_supported": tier2_off_target,
              "schema": "circuit_unit_tier2_off_target_result_v1", "candidate_id": "corpus.unit_tier2_off_target_v24",
              "semantics": "block_live_exact_set", "bars": {"toward": TOWARD_BAR, "overshoot": OVER_BAR, "tail": [TAIL_LO, TAIL_HI],
              "competitor": COMP_BAR, "random_recovery": RAND_REC, "random_kl_ratio": RAND_KL},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "tier2_off_target_supported": tier2_off_target, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
