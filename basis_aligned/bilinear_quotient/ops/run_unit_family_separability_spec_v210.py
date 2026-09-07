#!/usr/bin/env python3
# BQGATE: six frozen predictions; families, units (v197 + v201 + v204 + v205 + v207 + v209 receipts), recipe and bars fixed before the run.
"""v210: within-family separability for the v209 spec-authored behaviours -- and the first test of negative_inversion's circuit count.

v209 puts the fourth eight spec-authored behaviours through the four-row battery; every one of them has a sibling in the standing,
so none counts before this pass (rule since v114/v198). Recipe as v198/v202/v206/v208: per member, arm `own` (own C_fit control)
and arm `fam` (own C_fit + every sibling's A1 FIT prep, control_weight 30 x len(controls)); evaluated on the member's held A1
(extraction), each sibling's held A1 (|CE| leak) and held C. Units frozen from the v197/v201/v204/v205/v207/v209 receipts.
    separable(member) := fam-arm sibling max|CE| <= 0.05 AND fam-arm own extraction >= 0.80 x own-arm own extraction
                         AND own-arm extraction >= 0.50 (v206 floor).
Families (FULL within-family control sets, per proposal (iv)): inversion = negative_inversion (v207, counted provisionally as
circuit 21 with no sibling), so_inversion, neither_inversion (v209; the same did/the readout under `so` / `neither`);
person_readout = person_possessive, person_possessive_plural, reflexive_person, object_pronoun_person (v208's four) +
reflexive_person_plural, object_pronoun_person_plural (v209); complement_type = causative_complement, rather_prefer,
finiteness_selection (v208's three) + gerund_selection, adjective_complement, perception_complement (v209; four distinct answer
pairs across the six); number_agreement = partitive_agreement (v209) + perfect_number, lexical_number_pp (v197, one fused
number-agreement circuit in the standing). 18 members, ~40-120 s each (~25 min). A v209 member that missed a row is reported
here but not counted either way.

REGISTERED BEFORE THE RUN (each coded predicate is the sentence here):
    pred_a_inversion_own         negative_inversion is separable from so_inversion and neither_inversion (if it is NOT, the
                                 inversion trio is one circuit and the count 21 falls back to 20 + 1 fused).          prior 45%
    pred_b_person_family         separable on >= 4 of the 6 person readouts (v208 registers >= 3 of 4 at 40%).       prior 40%
    pred_c_complement_family     separable on >= 3 of the 6 complement-type members.                               prior 45%
    pred_d_partitive_fused       partitive_agreement is NOT separable from perfect_number / lexical_number_pp (the head-numeral
                                 number should ride the same number-agreement readout).                            prior 60%
    pred_e_row4_kept             fam-arm own-C UB975 <= 0.01 on >= 9 of 18 members (v206: 10 of 15).                prior 55%
    pred_f_instrument            person_possessive own-arm extraction within +-0.03 of its v208 own-arm value (same units,
                                 seed, split and code path: a determinism check, NOT an independent run outcome).   prior 90%
Smoke: V210_SMOKE=<out.json> (CPU, V210_SMOKE_ROWS=4, V210_SMOKE_NAMES=so_inversion,neither_inversion; steps 5).
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51
import run_unit_tier3_batch_amended_all_v197 as v197
import run_unit_tier3_batch_amended_spec8_v204 as v204
import run_unit_tier3_batch_amended_spec8b_v205 as v205
import run_unit_tier3_batch_amended_spec8c_v207 as v207
import run_unit_tier3_batch_amended_spec8d_v209 as v209
import run_unit_tier3_batch_amended_new9_v201 as v201

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_family_separability_spec_v210_result.json"
V197 = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
V201 = ROOT / "circuits/followups/unit_tier3_batch_amended_new9_v201_result.json"
V204 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8_v204_result.json"
V205 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8b_v205_result.json"
V207 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8c_v207_result.json"
V209 = ROOT / "circuits/followups/unit_tier3_batch_amended_spec8d_v209_result.json"
V208 = ROOT / "circuits/followups/unit_family_separability_spec_v210_result.json"
FAMILIES = {
    "inversion": ["negative_inversion", "so_inversion", "neither_inversion"],
    "person_readout": ["person_possessive", "person_possessive_plural", "reflexive_person", "object_pronoun_person",
                       "reflexive_person_plural", "object_pronoun_person_plural"],
    "complement_type": ["causative_complement", "rather_prefer", "finiteness_selection", "gerund_selection",
                        "adjective_complement", "perception_complement"],
    "number_agreement": ["partitive_agreement", "perfect_number", "lexical_number_pp"],
}
NAMES = {**v197.NAMES, **v201.NAMES, **v204.NAMES, **v205.NAMES, **v207.NAMES, **v209.NAMES}
INSTRUMENT = "person_possessive"


def _v208_own():
    return float(json.loads(V208.read_text())["members"][INSTRUMENT]["arms"]["own"]["extraction"])
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"cross_max": 0.05, "keep": 0.80, "ext_floor": 0.50, "c_ub_max": 0.01, "k_e": 9, "k_b": 4, "k_c": 3, "instr_tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    n = sum(len(v) for v in FAMILIES.values())
    return {"candidate_id": "corpus.unit_family_separability_spec_v210", "members": n, "families": list(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * n * STEPS, "model_updates": 0, "fit_parameters": 2 * n * 30 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "separable" in r and "error" not in r}
    ok = bool(good)
    sep = lambda n: n in good and good[n]["separable"]
    a = ok and sep("negative_inversion")
    b = ok and sum(1 for n in FAMILIES["person_readout"] if sep(n)) >= B["k_b"]
    c = ok and sum(1 for n in FAMILIES["complement_type"] if sep(n)) >= B["k_c"]
    d = ok and "partitive_agreement" in good and not good["partitive_agreement"]["separable"]
    e = ok and sum(1 for n in good if good[n]["arms"]["fam"]["C"]["ce_ub975"] <= B["c_ub_max"]) >= B["k_e"]
    f = ok and INSTRUMENT in good and abs(good[INSTRUMENT]["arms"]["own"]["extraction"] - _v208_own()) <= B["instr_tol"]
    return {"pred_a_inversion_own": bool(a), "pred_b_person_family": bool(b), "pred_c_complement_family": bool(c),
            "pred_d_partitive_fused": bool(d), "pred_e_row4_kept": bool(e), "pred_f_instrument": bool(f)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V210_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V210_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    steps = 5 if smoke else STEPS
    members = [m for f in FAMILIES.values() for m in f]
    which = [n for n in members if not smoke or n in os.environ.get("V210_SMOKE_NAMES", "so_inversion,neither_inversion").split(",")]
    prior = {**json.loads(V197.read_text())["behaviours"], **json.loads(V201.read_text())["behaviours"], **json.loads(V204.read_text())["behaviours"], **json.loads(V205.read_text())["behaviours"], **json.loads(V207.read_text())["behaviours"], **json.loads(V209.read_text())["behaviours"]}
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]
    prep = {}
    for n in which:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[n]}")
        a1, c = g.rows_of(m, "A1"), g.rows_of(m, "C")
        prep[n] = {"fit": g.prepare(backend, cut(fit_half(a1)), valid_only=True), "held": g.prepare(backend, cut(held_half(a1)), valid_only=True),
                   "C_fit": g.prepare(backend, cut(fit_half(c))), "C_held": g.prepare(backend, cut(held_half(c)))}

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def ce(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975")}

    R = {}
    for fam, mem in FAMILIES.items():
        for n in mem:
            if n not in which:
                continue
            t1 = time.perf_counter()
            try:
                if "error" in prior.get(n, {"error": "missing"}):
                    raise RuntimeError(f"v197/v201/v204/v205/v207 have no clean receipt for {n}")
                units = list(prior[n]["units"])
                P = prep[n]
                mu = mu_of(P["fit"], units)
                exact = g.recovery(P["held"], g.patched_axis(backend, P["held"], units))
                sibs = [s for s in mem if s != n and s in prep]
                arms = {}
                for arm, controls in (("own", (P["C_fit"],)), ("fam", (P["C_fit"],) + tuple(prep[s]["fit"] for s in sibs))):
                    q, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
                                                               controls=controls, control_weight=LAM * len(controls), mu=mu)
                    a = {"extraction": round(g.recovery(P["held"], g.patched_axis(backend, P["held"], units, q=q)) / exact, 3) if abs(exact) > 1e-6 else None,
                         "A1": ce(P["held"], units, q, mu), "C": ce(P["C_held"], units, q, mu),
                         "siblings": {s: ce(prep[s]["held"], units, q, mu)["ce_damage"] for s in sibs}}
                    a["sib_abs_max"] = round(max([abs(v) for v in a["siblings"].values()] or [0.0]), 4)
                    arms[arm] = a
                sep = arms["fam"]["sib_abs_max"] <= BARS["cross_max"] and arms["fam"]["extraction"] is not None and arms["own"]["extraction"] \
                    and arms["fam"]["extraction"] >= BARS["keep"] * arms["own"]["extraction"] and arms["own"]["extraction"] >= BARS["ext_floor"]
                R[n] = {"family": fam, "units": units, "n_units": len(units), "exact_held": round(exact, 3), "arms": arms, "separable": bool(sep),
                        "parent_rows": prior[n]["rows"], "n_dropped": {k: P[k].dropped for k in P}, "seconds": round(time.perf_counter() - t1, 1)}
            except Exception as exc:  # noqa: BLE001 - one member must not lose the batch
                R[n] = {"family": fam, "error": f"{type(exc).__name__}: {exc}", "separable": False, "seconds": round(time.perf_counter() - t1, 1)}
            print(fam, n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "arms")}),
                  "own", R[n].get("arms", {}).get("own", {}).get("extraction"), "fam", R[n].get("arms", {}).get("fam", {}).get("extraction"),
                  R[n].get("arms", {}).get("fam", {}).get("sib_abs_max"), round(time.perf_counter() - t0), "s", flush=True)

    predictions = PREDS(R)
    nsep = {fam: sum(1 for n in mem if n in R and R[n]["separable"]) for fam, mem in FAMILIES.items()}
    summary = {n: {"family": R[n]["family"], "separable": R[n]["separable"], "n_units": R[n].get("n_units"),
                   "own": (R[n]["arms"]["own"]["extraction"], R[n]["arms"]["own"]["sib_abs_max"], R[n]["arms"]["own"]["C"]["ce_ub975"]) if "arms" in R[n] else None,
                   "fam": (R[n]["arms"]["fam"]["extraction"], R[n]["arms"]["fam"]["sib_abs_max"], R[n]["arms"]["fam"]["C"]["ce_ub975"]) if "arms" in R[n] else None} for n in R}
    result = {"predictions": predictions, "schema": "unit_family_separability_spec_v210", "candidate_id": "corpus.unit_family_separability_spec_v210",
              "separable_counts": nsep, "families": FAMILIES, "summary": summary, "members": R, "bars": BARS,
              "recipe": {"lambda": LAM, "steps": steps, "lr": LR, "complement_weight": CW, "rank": 1, "seed": 0, "units_source": "v197 + v201 + v204 + v205 + v207 + v209 receipts (later receipts override earlier ones)",
                         "split": "within-direction: fit rows[0::4]+rows[1::4] (fits and controls), held rows[2::4]+rows[3::4] (evaluation)"},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "separable_counts": nsep, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
