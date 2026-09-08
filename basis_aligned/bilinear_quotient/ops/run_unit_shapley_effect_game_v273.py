#!/usr/bin/env python3
# BQGATE: five frozen predictions; cases, sources, downstream sets and bars fixed before the run.
"""v273: does v271's headline generalise, or was it about the corpus's one anomalous cell? A SECOND effect game on
TWO cleanly-separable behaviours, one per family. Exact enumeration again: 2 x 2^8 forwards per case, no fit, no surrogate.

WHAT v271 FOUND, ON ONE BEHAVIOUR. Allocating attn:06:head:03's causal effect on verb_preposition_of_over across the
nine other units of its own greedy set: all three instrument checks passed (Shapley efficiency residual -2.08e-17,
the no-op through the new path reproduced the parent battery's extraction_held to 0.00023, and a layer-4 unit EARLIER
than the layer-6 source scored psi 0.0070 against a 0.02 bar), and then E_a(empty) = 0.204 against E_a(full) = 0.252 --
81% of the source's effect survives with NONE of its nine downstream set-mates patched. All 36 pair interactions were
<= 0, the largest sub-additivity -0.027, none complementary.
That is a claim about how our greedy sets work -- they would be a mediation chain if the source's effect needed them --
but it was measured on of_over, which is the ONE cell in the corpus that fuses with two others and the cell whose
anomaly started the whole cue-lexeme line. A finding measured only on the anomaly is not a finding about the corpus.

THE TWO CASES, both cleanly separable on every rung they have appeared in, one from each preposition family:
  verb_preposition_to_at (listened/looked -> to/at), 8 units, separable at v244/v246/v248/v250/v252/v254 (worst 0.024)
  adjective_preposition_of_within (fond/enclosed -> of/within), 17 units, separable at v260 (sib 0.015)
Source in both is attn:06:head:03, the earliest hub unit and the same source as v271, so the comparison is on the
behaviour and not on the source. Downstream is the next eight units of each cell's own greedy set, with the LAST entry
in each list an EARLIER-layer unit (attn:04:head:01 for to_at, attn:05:head:03 for of_within) retained as the backward
control: a unit that precedes the source cannot mediate its write and must score ~0.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_identity_efficiency  -> |E_a(D) - E_a(empty) - sum psi| <= 1e-6 in BOTH cases. Shapley efficiency is an
      identity, so this is an INSTRUMENT CHECK, not a result; a failure voids d and e.               prior 99%
  pred_b_noop_through_new_path -> F(1, D) is within 0.02 of each case's parent battery extraction_held, in BOTH cases.
      The no-op runs through the NEW code path (v271 hit 0.00023).                                    prior 90%
  pred_c_backward_control_inert -> |psi| for the pre-source unit is <= 0.02 in BOTH cases.            prior 85%
  pred_d_direct_share_generalises -> E_a(empty)/E_a(full) >= 0.50 in BOTH cases. This is the v271 headline, restated as
      a bar on two DIFFERENT behaviours. v271 measured 0.81 on of_over; 0.50 is deliberately well below it so the
      prediction is that the effect is mostly-direct in general, not that the number reproduces.      prior 60%
  pred_e_no_complementary_pair -> every pair interaction is <= 0.0 in BOTH cases (v271: all 36 were, max -0.0000).
      A single positive pair refutes it, so this is the easiest predicate here to break.              prior 55%
WHAT EACH OUTCOME MEANS, fixed now: d true in both -> the greedy sets are not mediation chains and the direct route
dominates generally, which reframes what a "unit set" is in this corpus. d false in both -> v271's 81% was about
of_over specifically and I retract the general reading. d true in one -> family-dependent, reported as such and not
averaged. e false anywhere -> there IS complementarity somewhere and the uniform sub-additivity of v271 was local.
COST AND SCOPE: 2 x 2 x 2^8 = 1024 exact block-live forwards, ~20 s. This rung CANNOT change the circuit count: no new
cells, no new capability checks, every unit frozen from a landed receipt. As at v271, a large psi says the source's
effect DEPENDS on that unit being patched -- mediation OR common cause through the shared residual -- and does not
establish a directed edge; edge-specific interventions remain separate.
Smoke: V273_SMOKE=<out.json> (CPU, V273_SMOKE_ROWS=4, V273_SMOKE_D=3 -> the first case only, 2 x 2^3 forwards).
BARS = {"identity_tol": 1e-6, "noop_tol": 0.02, "backward_max": 0.02, "direct_share_min": 0.50, "all_pairs_at_most": 0.0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 128000
"""
from __future__ import annotations

import importlib
import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_shapley_effect_game_v273_result.json"

CASES = (
    # (behaviour, spec module, source, downstream (LAST entry is the backward control), parent receipt)
    ("verb_preposition_to_at", "circuit_fast_screen_candidate_verb_preposition_to_at", "attn:06:head:03",
     ("attn:07:head:08", "attn:08:head:08", "attn:11:head:03", "attn:13:head:08", "attn:08:head:01",
      "attn:14:head:08", "attn:16:head:08", "attn:04:head:01"),
     "unit_tier3_batch_amended_spec8w_v247_result.json"),
    ("adjective_preposition_of_within", "circuit_fast_screen_candidate_adjective_preposition_of_within",
     "attn:06:head:03",
     ("attn:07:head:08", "attn:08:head:08", "attn:13:head:08", "attn:08:head:01", "attn:11:head:03",
      "attn:14:head:08", "attn:16:head:08", "attn:05:head:03"),
     "unit_tier3_batch_amended_spec9h_v269_result.json"),
)
PRED_NAMES = ("pred_a_identity_efficiency", "pred_b_noop_through_new_path",
              "pred_c_backward_control_inert", "pred_d_direct_share_generalises",
              "pred_e_no_complementary_pair")
BARS = {"identity_tol": 1e-6, "noop_tol": 0.02, "backward_max": 0.02,
        "direct_share_min": 0.50, "all_pairs_at_most": 0.0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 128000


def _plan():
    return {"candidate_id": "corpus.unit_shapley_effect_game_v273",
            "behaviours": [c[0] for c in CASES],
            "coalitions": sum(2 ** len(c[3]) for c in CASES),
            "forwards": sum(2 * 2 ** len(c[3]) for c in CASES),
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def shapley_from_game(E, D):
    """Exact Shapley values of the game E (dict: frozenset -> float) over players D."""
    n = len(D)
    fact = [math.factorial(k) for k in range(n + 1)]
    psi = {}
    for j in D:
        rest = [d for d in D if d != j]
        total = 0.0
        for r in range(len(rest) + 1):
            w = fact[r] * fact[n - r - 1] / fact[n]
            for S in itertools.combinations(rest, r):
                fs = frozenset(S)
                total += w * (E[fs | {j}] - E[fs])
        psi[j] = total
    return psi


def pair_interactions(E, D):
    """Unweighted mean over coalitions of the second difference (sign is what we read)."""
    out = {}
    for j, k in itertools.combinations(D, 2):
        rest = [d for d in D if d not in (j, k)]
        vals = []
        for r in range(len(rest) + 1):
            for S in itertools.combinations(rest, r):
                fs = frozenset(S)
                vals.append(E[fs | {j, k}] - E[fs | {j}] - E[fs | {k}] + E[fs])
        out[f"{j}|{k}"] = sum(vals) / len(vals)
    return out


def PREDS(R):
    B, C = BARS, R.get("cases", {})
    ok = len(C) == len(CASES)
    a = ok and all(abs(c["efficiency_residual"]) <= B["identity_tol"] for c in C.values())
    b = ok and all(c["noop_abs_diff"] is not None and c["noop_abs_diff"] <= B["noop_tol"] for c in C.values())
    cc = ok and all(abs(c["shapley"][c["backward_control"]]) <= B["backward_max"] for c in C.values())
    d = ok and all(c["direct_share"] is not None and c["direct_share"] >= B["direct_share_min"] for c in C.values())
    e = ok and all(c["max_pair_interaction"] <= B["all_pairs_at_most"] for c in C.values())
    return dict(zip(PRED_NAMES, (bool(a), bool(b), bool(cc), bool(d), bool(e))))


def one_case(backend, behaviour, modname, source, down, parent_file, smoke):
    cell = importlib.import_module(modname)
    backward = down[-1]
    if smoke:
        k = int(os.environ.get("V273_SMOKE_D", "3"))
        D = list(down[:-1])[: k - 1] + [backward]
    else:
        D = list(down)
    a1 = [r for r in cell.build_rows() if r["family"] == "A1"]
    rows = a1[2::4] + a1[3::4]          # the parent battery's HELD half
    if smoke:
        rows = rows[: int(os.environ.get("V273_SMOKE_ROWS", "4"))]
    prep = g.prepare(backend, rows, valid_only=True)

    F = {}
    for b in (0, 1):
        for r in range(len(D) + 1):
            for S in itertools.combinations(D, r):
                units = tuple(([source] if b else []) + list(S))
                F[(b, frozenset(S))] = g.recovery(prep, g.patched_axis(backend, prep, units))
    E = {S: F[(1, S)] - F[(0, S)] for (_, S) in {k for k in F if k[0] == 1}}

    psi = shapley_from_game(E, D)
    inter = pair_interactions(E, D)
    full, empty = frozenset(D), frozenset()
    noop = None
    pf = ROOT / "circuits/followups" / parent_file
    if pf.exists():
        ex = json.loads(pf.read_text())["behaviours"].get(behaviour, {}).get("extraction_held")
        if ex is not None:
            noop = abs(F[(1, full)] - float(ex))
    return {"source": source, "downstream": D, "backward_control": backward,
            "n_rows": len(prep.base_axis), "n_dropped": prep.dropped,
            "E_empty": E[empty], "E_full": E[full],
            "direct_share": (E[empty] / E[full]) if abs(E[full]) > 1e-9 else None,
            "efficiency_residual": E[full] - E[empty] - sum(psi.values()),
            "shapley": psi, "pair_interactions": inter,
            "single_deletion": {j: E[frozenset({j})] - E[empty] for j in D},
            "noop_abs_diff": noop, "F_full_with_source": F[(1, full)],
            "max_pair_interaction": max(inter.values()), "min_pair_interaction": min(inter.values()),
            "forwards": len(F)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V273_SMOKE")
    t0 = time.time()
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    cases = CASES[:1] if smoke else CASES
    R = {"cases": {c[0]: one_case(backend, *c, smoke) for c in cases},
         "seconds": None}
    R["seconds"] = round(time.time() - t0, 1)
    R["predictions"] = PREDS(R)
    R["bars"] = BARS
    R["instrument_checks"] = {k: R["predictions"][k] for k in PRED_NAMES[:3]}
    R["findings_valid"] = all(R["instrument_checks"].values())
    R["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": R["predictions"], "findings_valid": R["findings_valid"],
                      "direct_share": {k: (round(v["direct_share"], 3) if v["direct_share"] is not None else None)
                                       for k, v in R["cases"].items()},
                      "seconds": R["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
