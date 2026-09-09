#!/usr/bin/env python3
# BQGATE: five frozen predictions; cases, sources, downstream sets and bars fixed before the run.
"""v307: does DIRECT SHARE track SEPARABILITY? Two fused cells and two separable ones, matched on source and set size.

TWO INSTRUMENTS THAT HAVE NEVER BEEN COMPARED. Separability asks whether a cell's rank-1 direction is inert on its
family siblings; the effect game asks what fraction of a unit's causal effect survives with none of its set-mates
patched. Both are computed on the same cells and neither has ever been regressed on the other.
v291 gave the first two points and they point the WRONG way for any "fusion means shared routing" intuition:
verb_preposition_by_with, which FUSES (it shares `abided` with of_by and `tinkered` with through_with), had direct
share 0.832, while verb_preposition_on_across, which is SEPARABLE, had 0.677. n=2, one of each, so it is not a result;
it is a reason to run four more.
THE FOUR CASES, all with the SAME source unit (attn:06:head:03) and unit sets of 6-7 downstream members, so the
comparison is on the cell rather than on set size or source:
  FUSED:     verb_preposition_insisted (7 units, fused at v289 into on_toward/on_within, shared cue `insisted`)
             verb_preposition_to_at    (8 units, fused at v297 into at_under, shared cue `looked`)
  SEPARABLE: adjective_preposition_fond (7 units, separable at v299)
             verb_preposition_at_to     (8 units, separable at v289)
Every case's D u {source} is EXACTLY the parent's unit set, so the no-op control is exact by construction -- the check
v273 failed and v275 fixed.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_identity_efficiency  -> |E_a(D) - E_a(empty) - sum psi| <= 1e-6 in ALL FOUR. Identity; an INSTRUMENT CHECK.
                                                                                                       prior 99%
  pred_b_noop_through_new_path -> |F(1, D) - parent extraction_held| <= 0.02 in ALL FOUR.              prior 85%
  pred_c_full_set_matches_parent -> the same comparison stated on F_full_with_source.                  prior 85%
  pred_d_direct_share_generalises -> E_a(empty)/E_a(full) >= 0.50 in ALL FOUR. This is the standing finding (v271 0.81,
      v275 0.95/0.89/1.00, v291 0.83/0.68) restated on four more cells; a case below 0.50 would be the first.
                                                                                                       prior 75%
  pred_e_no_complementary_pair -> every pair interaction <= 0.0 in ALL FOUR. It has failed at v275 (+0.0004) and v291
      (+0.0034) on tiny positive pairs, so I expect it to fail again and it is registered to be failed rather than
      quietly dropped.                                                                                 prior 25%
THE COMPARISON IS NOT A REGISTERED PREDICATE, and that is deliberate: with two cells per group I cannot set a bar on a
difference of means without inventing a threshold. The receipt records direct_share per case and the ledger will state
the four numbers next to their separability verdicts. If fused and separable cells separate cleanly on this measure
across six cells (v291's two plus these four), that earns a REGISTERED test on a larger sample, not a claim here.
COST: 2 x 2^6 + 2 x 2^7 = 512 exact block-live forwards, seconds. No new cells, no capability checks, no family growth
-- which is why this rung is authored now: separability on verb_preposition is halted by the memory wall and every
battery I add pushes a family closer to it, while this touches neither.
Smoke: V307_SMOKE=<out.json> (CPU, V307_SMOKE_ROWS=4, V307_SMOKE_D=3 -> the first case only).
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
OUT = ROOT / "circuits/followups/unit_shapley_effect_game_v307_result.json"

CASES = (
    # (behaviour, spec module, source, downstream = THE REST OF THE PARENT'S OWN UNIT SET, parent receipt)
    # FUSED at their latest separability rung:
    ("verb_preposition_insisted", "circuit_fast_screen_candidate_verb_preposition_insisted", "attn:06:head:03",
     ("attn:13:head:08", "attn:08:head:08", "attn:07:head:08", "attn:07:head:07", "attn:14:head:08",
      "attn:14:head:03"),
     "unit_tier3_batch_amended_spec8k_v223_result.json"),
    ("verb_preposition_to_at", "circuit_fast_screen_candidate_verb_preposition_to_at", "attn:06:head:03",
     ("attn:13:head:08", "attn:07:head:08", "attn:08:head:08", "attn:11:head:03", "attn:08:head:01",
      "attn:14:head:08", "attn:04:head:03"),
     "unit_tier3_batch_amended_spec8w_v247_result.json"),
    # SEPARABLE at their latest separability rung:
    ("adjective_preposition_fond", "circuit_fast_screen_candidate_adjective_preposition_fond", "attn:06:head:03",
     ("attn:08:head:08", "attn:13:head:08", "attn:07:head:08", "attn:11:head:03", "attn:08:head:01",
      "attn:16:head:08"),
     "unit_tier3_batch_amended_spec8h_v217_result.json"),
    ("verb_preposition_at_to", "circuit_fast_screen_candidate_verb_preposition_at_to", "attn:06:head:03",
     ("attn:13:head:08", "attn:07:head:08", "attn:08:head:08", "attn:14:head:08", "attn:08:head:01",
      "attn:11:head:03", "attn:09:head:07"),
     "unit_tier3_batch_amended_spec9b_v257_result.json"),
)
PRED_NAMES = ("pred_a_identity_efficiency", "pred_b_noop_through_new_path",
              "pred_c_full_set_matches_parent", "pred_d_direct_share_generalises",
              "pred_e_no_complementary_pair")
BARS = {"identity_tol": 1e-6, "noop_tol": 0.02, "backward_max": 0.02,
        "direct_share_min": 0.50, "all_pairs_at_most": 0.0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 128000


def _plan():
    return {"candidate_id": "corpus.unit_shapley_effect_game_v307",
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
    cc = ok and all(c["parent_extraction_held"] is not None
                    and abs(c["F_full_with_source"] - c["parent_extraction_held"]) <= B["noop_tol"]
                    for c in C.values())
    d = ok and all(c["direct_share"] is not None and c["direct_share"] >= B["direct_share_min"] for c in C.values())
    e = ok and all(c["max_pair_interaction"] <= B["all_pairs_at_most"] for c in C.values())
    return dict(zip(PRED_NAMES, (bool(a), bool(b), bool(cc), bool(d), bool(e))))


def one_case(backend, behaviour, modname, source, down, parent_file, smoke):
    cell = importlib.import_module(modname)
    backward = down[-1]
    if smoke:
        k = int(os.environ.get("V307_SMOKE_D", "3"))
        D = list(down[:-1])[: k - 1] + [backward]
    else:
        D = list(down)
    a1 = [r for r in cell.build_rows() if r["family"] == "A1"]
    rows = a1[2::4] + a1[3::4]          # the parent battery's HELD half
    if smoke:
        rows = rows[: int(os.environ.get("V307_SMOKE_ROWS", "4"))]
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
    noop, ex = None, None
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
            "parent_extraction_held": (float(ex) if ex is not None else None),
            "max_pair_interaction": max(inter.values()), "min_pair_interaction": min(inter.values()),
            "forwards": len(F)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V307_SMOKE")
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
