#!/usr/bin/env python3
# BQGATE: five frozen predictions; cases, sources, downstream sets and bars fixed before the run.
"""v275: the effect game on THREE behaviours whose ENTIRE unit set is enumerable -- so the no-op control is exact.

WHY THIS EXISTS: v273 FAILED ITS OWN INSTRUMENT CHECK AND ITS FINDINGS ARE VOID. v273 ran the effect game on two
cleanly-separable cells and reported direct shares of 0.686 (to_at) and 0.699 (of_within), which would have confirmed
v271's headline. But pred_b_noop_through_new_path failed on of_within at 0.155, and by v273's own registered rule a
failed instrument check voids pred_d and pred_e. The cause is a mis-specified comparator, not a broken game: I compared
F(1, D) against the parent battery's extraction_held while D u {source} was a TRUNCATION of the parent's unit set
(of_within has 17 units; I enumerated 8 downstream). The two numbers patch different sets, so they were never going to
match, and to_at passing at 0.014 was luck rather than validation. The Shapley identity itself held at 1.4e-17 in both
cases and both backward controls were inert, so the machinery is fine; the CHECK was wrong.
The fix is not a looser bar. It is to choose behaviours where D u {source} IS the parent's whole set:
  lexical_number_pp          5 units, extraction_held 0.873  (v197)
  relative_clause_agreement  5 units, extraction_held 0.904  (v221)
  wh_adjunct_when_where      5 units, extraction_held 0.908  (v227)
Each is 2 x 2^4 = 32 forwards, 96 in total, a few seconds. These are also NOT preposition cells -- they are number,
agreement and wh behaviours from three different batteries -- so if the direct-share result holds here it holds
somewhere the cue/fusion line never touched.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_identity_efficiency  -> |E_a(D) - E_a(empty) - sum psi| <= 1e-6 in ALL THREE. Identity, so an INSTRUMENT
      CHECK; a failure voids d and e.                                                                  prior 99%
  pred_b_noop_through_new_path -> |F(1, D) - parent extraction_held| <= 0.02 in ALL THREE. NOW EXACT by construction:
      D u {source} is the parent's whole unit set, so this compares the same intervention computed by the enumeration
      against the number in the landed receipt. This is the check v273 got wrong and it is the reason this rung exists.
                                                                                                       prior 85%
  pred_c_full_set_matches_parent -> the same comparison stated on F_full_with_source directly, so a receipt that
      somehow reports a passing noop with a missing parent value cannot slip through.                  prior 85%
  pred_d_direct_share_generalises -> E_a(empty)/E_a(full) >= 0.50 in ALL THREE. v271 measured 0.81 on of_over; v273
      measured 0.69/0.69 but VOID. If this holds on three non-preposition behaviours with exact no-op controls, the
      claim "a unit's effect is mostly direct, not mediated by its own greedy set" is established for this corpus.
                                                                                                       prior 60%
  pred_e_no_complementary_pair -> every pair interaction <= 0.0 in ALL THREE (v271: all 36 were; v273: to_at had one
      at +0.0001, which is why its pred_e read false). A single positive pair breaks it.               prior 40%
WHAT EACH OUTCOME MEANS, fixed now: d true in all three with a and b true -> the direct-route dominance is a property
of the corpus, not of of_over, and our greedy unit sets are not mediation chains. d false anywhere -> it is
behaviour-specific and v271's 81% is not generalisable; report per behaviour, do not average. b false -> the fix did
not work and I stop building on the effect game until the comparator is right.
COST: 96 exact block-live forwards, seconds. No new cells, no capability checks, every unit frozen from a landed
receipt; this rung CANNOT change the circuit count. As at v271, a large psi says the source's effect DEPENDS on that
unit being patched (mediation OR common cause through the shared residual), not that there is a directed edge.
Smoke: V275_SMOKE=<out.json> (CPU, V275_SMOKE_ROWS=4, V275_SMOKE_D=3 -> the first case only).
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
OUT = ROOT / "circuits/followups/unit_shapley_effect_game_v275_result.json"

CASES = (
    # (behaviour, spec module, source, downstream = THE REST OF THE PARENT'S OWN UNIT SET, parent receipt)
    ("lexical_number_pp", "circuit_fast_screen_candidate_lexical_number_pp", "attn:11:head:03",
     ("attn:07:head:08", "attn:09:head:07", "attn:05:head:03", "attn:13:head:01"),
     "unit_tier3_batch_amended_all_v197_result.json"),
    ("relative_clause_agreement", "circuit_fast_screen_candidate_relative_clause_agreement", "attn:11:head:03",
     ("attn:07:head:08", "attn:09:head:07", "attn:09:head:01", "attn:05:head:03"),
     "unit_tier3_batch_amended_spec8j_v221_result.json"),
    ("wh_adjunct_when_where", "circuit_fast_screen_candidate_wh_adjunct_when_where", "attn:11:head:01",
     ("attn:11:head:06", "attn:10:head:03", "attn:08:head:01", "attn:09:head:07"),
     "unit_tier3_batch_amended_spec8m_v227_result.json"),
)
PRED_NAMES = ("pred_a_identity_efficiency", "pred_b_noop_through_new_path",
              "pred_c_full_set_matches_parent", "pred_d_direct_share_generalises",
              "pred_e_no_complementary_pair")
BARS = {"identity_tol": 1e-6, "noop_tol": 0.02, "backward_max": 0.02,
        "direct_share_min": 0.50, "all_pairs_at_most": 0.0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 128000


def _plan():
    return {"candidate_id": "corpus.unit_shapley_effect_game_v275",
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
        k = int(os.environ.get("V275_SMOKE_D", "3"))
        D = list(down[:-1])[: k - 1] + [backward]
    else:
        D = list(down)
    a1 = [r for r in cell.build_rows() if r["family"] == "A1"]
    rows = a1[2::4] + a1[3::4]          # the parent battery's HELD half
    if smoke:
        rows = rows[: int(os.environ.get("V275_SMOKE_ROWS", "4"))]
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
    smoke = os.environ.get("V275_SMOKE")
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
