#!/usr/bin/env python3
# BQGATE: six frozen predictions; source unit, downstream set, game and bars fixed before the run.
"""v271: an EXACT effect-game enumeration -- Shapley allocation of ONE hub unit's causal effect across the nine
downstream units that could carry it, plus all 36 pair interactions. No surrogate, no fit, no new mathematics:
2 x 2^9 = 1024 exact block-live interchange forwards, enumerated.

WHY THIS, AND WHY NOW. Logan passed a worked note on Shapley/tensor-network attribution (shap_tensor.md, 2026-09-08,
from arxiv 2606.01540). Most of it does not transfer to bilin18 as stated: the exact weight-based formula
phi_i = a_i + (1/2) sum_j a_ij needs a POLYNOMIAL subnetwork, and bilin18 is not one under live intervention
(rms_norm on the block input and final stream, QK-norm on q,k AND q2,k2), while the tensor-network speedup needs small
bond dimension in the COALITION tensor, which our reused 5-head hub is the worst case for. What DOES transfer needs
none of that machinery: the EFFECT GAME of its section 3, which is exactly our existing block-live forward evaluated
under a mask enumeration.
The question it answers is one our instruments cannot: our removal and DAS rungs give ONE number per site, and a site
whose single deletion does nothing is scored as carrying nothing. That is wrong whenever two downstream routes are
REDUNDANT -- either alone suffices, so deleting either alone shows zero. I have hit this twice: attention-column
knockouts where MORE knockouts LOWERED the loss, and Codex's repeated single-deletion bypass nulls.

THE GAME. Behaviour: verb_preposition_of_over (consisted/presided -> of/over), chosen because it is the corpus's
anomaly -- the only cell that leaks into TWO others (of_by +0.294, over_with +0.134) and the one whose fusion started
the cue-lexeme question that v256/v258 refuted as sufficient. If its effect turns out to be carried redundantly by
several downstream units, that is a candidate explanation for why its direction is not separable while 100+ others are.
  SOURCE a  = attn:06:head:03, the earliest unit in its greedy set (layer 6).
  DOWNSTREAM D = the other nine units of that set, all strictly later than layer 6:
     attn:07:head:08, attn:08:head:01, attn:08:head:08, attn:11:head:03, attn:13:head:08,
     attn:14:head:03, attn:14:head:08, attn:16:head:08, attn:04:head:01
     -- NOTE attn:04:head:01 is EARLIER than the source and is retained deliberately as a NEGATIVE CONTROL: a unit that
     cannot mediate a layer-6 write must receive ~0 Shapley credit. If it does not, the instrument is wrong.
  F(b, S) = recovery(prep, patched_axis(backend, prep, units)) with units = ({a} if b else {}) | S, i.e. the standard
     block-live interchange with exactly those units patched to donor and everything else LIVE.
  E_a(S) = F(1,S) - F(0,S): how much the source unit's patch is worth when those downstream units are also patched.
  psi_j  = the exact Shapley value of j in the game E_a over D (all 2^9 coalitions enumerated, no sampling).
  I_jk   = the exact pair interaction, mean over coalitions of
           E_a(S+j+k) - E_a(S+j) - E_a(S+k) + E_a(S).  Negative = redundant, positive = complementary.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_identity_efficiency  -> |E_a(D) - E_a(empty) - sum_j psi_j| <= 1e-6. This is an IDENTITY that must hold by
      construction (Shapley efficiency); it is an INSTRUMENT CHECK, not a result, and is labelled as such in the
      receipt. If it fails, the enumeration is wired wrong and every other number here is void.        prior 99%
  pred_b_noop_through_new_path -> F(1, D) computed by this runner equals the parent receipt's exact full-set recovery
      for of_over within 0.02. The no-op control runs through the NEW code path (standing lesson: a control that does
      not exercise the new path proves nothing).                                                       prior 95%
  pred_c_backward_control_inert -> |psi| for attn:04:head:01, the pre-source unit, is <= 0.02, i.e. below every other
      unit's |psi|. A unit earlier than the source cannot mediate its write; if this fails the instrument is measuring
      something other than mediation.                                                                  prior 90%
  pred_d_redundancy_exists -> at least one pair (j,k) has I_jk <= -0.05. THIS IS THE SCIENCE QUESTION and I am
      genuinely unsure: the hub is five heads that co-occur in every preposition set, which is what redundancy would
      look like, but they were selected greedily for MARGINAL gain, which selects AGAINST redundancy.  prior 45%
  pred_e_shapley_reorders -> the psi ranking differs from the single-deletion ranking (psi_j vs E_a({j}) - E_a(empty))
      on at least 3 of the 9 units, by rank position. If this fails, Shapley allocation buys nothing over the single
      ablations we already run, and I should stop here rather than build on it.                        prior 55%
  pred_f_direct_route -> E_a(empty) >= 0.10: the source has an effect even with no downstream unit patched, i.e. a
      route outside D (direct residual or an unpatched path).                                          prior 80%
pred_e is the load-bearing one for whether this instrument earns a place in the corpus, and pred_d is the one the note
was proposed to answer. pred_a/b/c are checks that the instrument is what I claim; they are reported separately from
the findings and a failure on any of them voids d/e/f.

COST AND SCOPE. 1024 forwards x 32 rows, one behaviour, ~2-4 GPU-min. No fit, no gradient, no surrogate, so the
TN-SHAP-G fidelity bound (|phi - phi_hat| <= 2 eps, uniform error) does not apply and is not claimed -- these are exact
values of the game as defined, and the ONLY approximation is the choice of D and of the ablation semantics.
This rung does NOT establish a directed edge from a to any j: a large psi_j says the source's effect DEPENDS on j being
patched, which is mediation OR common-cause through the shared residual. Edge-specific interventions remain separate.
Distinct from Codex's rung496 (exact Q1/K1/Q2/K2/V factor allocation WITHIN one head): this is a unit-level effect game
ACROSS layers, and the prior-art search over that vocabulary returned 0 events.
Smoke: V271_SMOKE=<out.json> (CPU, V271_SMOKE_ROWS=4, V271_SMOKE_D=4 -> 2 x 2^4 = 32 forwards).
BARS = {"identity_tol": 1e-6, "noop_tol": 0.02, "backward_max": 0.02, "redundant_at": -0.05, "reorder_min": 3, "direct_min": 0.10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 128000
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import circuit_fast_screen_candidate_verb_preposition_of_over as cell

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_shapley_effect_game_v271_result.json"
PARENT = ROOT / "circuits/followups/unit_tier3_batch_amended_spec9d_v261_result.json"

BEHAVIOUR = "verb_preposition_of_over"
SOURCE = "attn:06:head:03"
DOWNSTREAM = ("attn:07:head:08", "attn:08:head:01", "attn:08:head:08", "attn:11:head:03",
              "attn:13:head:08", "attn:14:head:03", "attn:14:head:08", "attn:16:head:08",
              "attn:04:head:01")
BACKWARD_CONTROL = "attn:04:head:01"
PRED_NAMES = ("pred_a_identity_efficiency", "pred_b_noop_through_new_path",
              "pred_c_backward_control_inert", "pred_d_redundancy_exists",
              "pred_e_shapley_reorders", "pred_f_direct_route")
BARS = {"identity_tol": 1e-6, "noop_tol": 0.02, "backward_max": 0.02, "redundant_at": -0.05,
        "reorder_min": 3, "direct_min": 0.10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 4000, 128000


def _plan():
    return {"candidate_id": "corpus.unit_shapley_effect_game_v271", "behaviour": BEHAVIOUR,
            "source": SOURCE, "downstream": list(DOWNSTREAM),
            "coalitions": 2 ** len(DOWNSTREAM), "forwards": 2 * 2 ** len(DOWNSTREAM),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
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
    B = BARS
    if "error" in R:
        return dict.fromkeys(PRED_NAMES, False)
    psi, inter = R["shapley"], R["pair_interactions"]
    a = abs(R["efficiency_residual"]) <= B["identity_tol"]
    b = R["noop_abs_diff"] is not None and R["noop_abs_diff"] <= B["noop_tol"]
    c = abs(psi[BACKWARD_CONTROL]) <= B["backward_max"]
    d = any(v <= B["redundant_at"] for v in inter.values())
    order_psi = [u for u, _ in sorted(psi.items(), key=lambda kv: -kv[1])]
    order_solo = [u for u, _ in sorted(R["single_deletion"].items(), key=lambda kv: -kv[1])]
    e = sum(1 for i, u in enumerate(order_psi) if order_solo.index(u) != i) >= B["reorder_min"]
    f = R["E_empty"] >= B["direct_min"]
    return dict(zip(PRED_NAMES, (bool(a), bool(b), bool(c), bool(d), bool(e), bool(f))))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    smoke = os.environ.get("V271_SMOKE")
    if smoke:  # keep the backward control in the smoke set so pred_c exercises the same code path
        k = int(os.environ.get("V271_SMOKE_D", "4"))
        D = [u for u in DOWNSTREAM if u != BACKWARD_CONTROL][: k - 1] + [BACKWARD_CONTROL]
    else:
        D = list(DOWNSTREAM)
    t0 = time.time()
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    a1 = [r for r in cell.build_rows() if r["family"] == "A1"]
    rows = a1[2::4] + a1[3::4]   # the parent battery's HELD half, so F(1, full) is comparable to its extraction_held
    if smoke:
        rows = rows[:int(os.environ.get("V271_SMOKE_ROWS", "4"))]
    prep = g.prepare(backend, rows, valid_only=True)

    F = {}
    for b in (0, 1):
        for r in range(len(D) + 1):
            for S in itertools.combinations(D, r):
                units = tuple(([SOURCE] if b else []) + list(S))
                F[(b, frozenset(S))] = g.recovery(prep, g.patched_axis(backend, prep, units))
    E = {S: F[(1, S)] - F[(0, S)] for (_, S) in {k for k in F if k[0] == 1}}

    psi = shapley_from_game(E, D)
    inter = pair_interactions(E, D)
    full = frozenset(D)
    solo = {j: E[frozenset({j})] - E[frozenset()] for j in D}

    noop_diff = None
    if PARENT.exists():
        parent = json.loads(PARENT.read_text())["behaviours"].get(BEHAVIOUR, {})
        exact = parent.get("extraction_held")
        if exact is not None:
            noop_diff = abs(F[(1, full)] - float(exact))

    R = {"behaviour": BEHAVIOUR, "source": SOURCE, "downstream": D,
         "n_rows": len(prep.base_axis), "n_dropped": prep.dropped,
         "E_empty": E[frozenset()], "E_full": E[full],
         "efficiency_residual": E[full] - E[frozenset()] - sum(psi.values()),
         "shapley": psi, "pair_interactions": inter, "single_deletion": solo,
         "F_source_only": F[(1, frozenset())], "F_full_with_source": F[(1, full)],
         "F_full_without_source": F[(0, full)], "noop_abs_diff": noop_diff,
         "forwards": len(F), "seconds": round(time.time() - t0, 1)}
    R["predictions"] = PREDS(R)
    R["bars"] = BARS
    R["instrument_checks"] = {k: R["predictions"][k] for k in PRED_NAMES[:3]}
    R["findings_valid"] = all(R["instrument_checks"].values())
    R["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": R["predictions"], "findings_valid": R["findings_valid"],
                      "E_empty": R["E_empty"], "E_full": R["E_full"],
                      "efficiency_residual": R["efficiency_residual"],
                      "seconds": R["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
