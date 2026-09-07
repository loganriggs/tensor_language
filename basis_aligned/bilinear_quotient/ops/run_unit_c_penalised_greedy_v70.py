#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, objective (lambda = 2), budget and bars fixed before the run.
"""v70: C-penalised removal-greedy for the three hub+8 sets that fail row 4 (verb_complementizer, polarity, voice).

v69: the A1-only removal-greedy (v66/v67) gives sets that damage their own C family -- verb_complementizer 0.387 nat
(UB 0.483), polarity UB 0.041, voice UB 0.030 against the rubric's 0.01. This is a one-sided capability failure of
the selection objective, so the licensed repair is one change: score each candidate on EVEN rows by
    A1 damage - LAMBDA * max(C damage, 0),  LAMBDA = 2
(a head that costs 0.05 on C must bring 0.10 on A1; negative C damage is not rewarded). To keep the cost within one
run, C damage is evaluated only for the TOP_K = 20 candidates by A1 damage at each step. Everything else is as in
v66/v67: diff-in-means refit per candidate set, 8 additions, evaluation on ODD rows (A1, C, A2, extraction as in v68).
The per-step C curve of the ORIGINAL path is also recorded to name the heads that brought the C damage.

REGISTERED BEFORE THE RUN (odd rows; CE damage in nat, 97.5% document bootstrap; extraction as v68)
    pred_a_row4_repaired   own-C CE UB <= 0.01 for all three penalised hub+8 sets. Worked: UB 0.005 True; UB 0.03 False.
    pred_b_removal_kept    penalised A1 odd removal >= 0.80 x the unpenalised hub+8 (1.119 / 0.567 / 0.322).
                           Worked: complementizer 0.95 True; 0.80 False.
    pred_c_extraction_kept keep_exact extraction on odd rows >= 0.80 for all three (row 2 survives). Worked: 0.85 True; 0.75 False.
    pred_d_few_swaps       for each set at most 3 of the 8 additions differ from the unpenalised path (the C damage
                           comes from a few heads). Worked: 2 differ True; 5 differ False.
    pred_e_row5_kept       A1-fit direction on A2 odd: LB > 0 and >= 0.50 x A1 odd, all three. Worked: 0.60 vs 0.90 True; 0.40 False.
    Prior: a unsure (16-document C halves give UB ~ point + 0.03 even at zero mean -- the rubric bar may be unreachable
    on odd rows for any set, which would be a finding about the bar), b True, c True, d True for complementizer, e True.
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
import run_unit_extraction_four_sets_v52 as v52

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_c_penalised_greedy_v70_result.json"
SRC = [ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json",
       ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"]
NAMES = ("verb_complementizer", "polarity_licensing", "voice_frame")
LAMBDA, TOP_K, MAX_ADD = 2.0, 20, 8
C_UB, KEEP_FRAC, ROW2, MAX_SWAPS, OOD_FRAC = 0.01, 0.80, 0.80, 3, 0.50
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 9500, 400000


def _plan():
    return {"candidate_id": "corpus.unit_c_penalised_greedy_v70", "names": NAMES, "lambda": LAMBDA, "top_k": TOP_K,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    all_heads = g.all_head_units()
    old = {n: r for p in SRC for n, r in json.loads(p.read_text())["sets"].items() if n in NAMES}
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}

    def mu_of(prep, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (prep.base_cache, prep.donor_cache) for rid in prep.base_batch.row_ids]).mean(0) for u in units}

    def damage(fit, ev, units, q=None, mu=None):
        q = q if q is not None else g.block_diff_in_means(backend, fit, units)
        mu = mu if mu is not None else mu_of(fit, units)
        return v51.summary(torch, v51.removal(backend, ev, units, q, mu))["ce_damage"]

    R = {}
    for n in NAMES:
        hub = list(old[n]["hub"])
        P = {}
        for fam in ("A1", "A2", "C"):
            rows = g.rows_of(modules[n], fam)
            P[(fam, "even")], P[(fam, "odd")] = g.prepare(backend, rows[0::2]), g.prepare(backend, rows[1::2])
        A1e, Ce = P[("A1", "even")], P[("C", "even")]

        # per-step C curve of the ORIGINAL (A1-only) path
        old_path = old[n]["final"]
        old_c = [{"k": k, "set_tail": old_path[len(hub):len(hub) + k], "C_odd": damage(A1e, P[("C", "odd")], old_path[:len(hub) + k])} for k in range(MAX_ADD + 1)]

        # penalised greedy on even rows
        chosen, curve = list(hub), []
        pool = [u for u in all_heads if u not in hub]
        for step in range(MAX_ADD):
            a1 = {u: damage(A1e, A1e, chosen + [u]) for u in pool}
            top = sorted(a1, key=a1.get, reverse=True)[:TOP_K]
            sc = {}
            for u in top:
                c = damage(A1e, Ce, chosen + [u])
                sc[u] = (a1[u] - LAMBDA * max(c, 0.0), a1[u], c)
            best = max(sc, key=lambda u: sc[u][0])
            chosen.append(best); pool.remove(best)
            curve.append({"step": step + 1, "added": best, "score": sc[best][0], "A1_even": sc[best][1], "C_even": sc[best][2],
                          "a1_rank_of_pick": top.index(best) + 1, "in_old_path": best in old_path})

        # evaluation on odd rows
        q, mu = g.block_diff_in_means(backend, A1e, chosen), mu_of(A1e, chosen)
        ev = {k: v51.summary(torch, v51.removal(backend, P[(f, "odd")], chosen, q, mu)) for k, f in (("A1", "A1"), ("C", "C"), ("A2", "A2"))}
        # extraction (v68 recipe) on A1 odd
        odd = P[("A1", "odd")]
        mu_all = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (odd.base_cache, odd.donor_cache) for rid in odd.base_batch.row_ids]).mean(0) for u in all_heads}

        def margins(set_units, qs):
            order = v52.ordered_units(set_units)
            bq = v52.block_q(torch, backend.device, set_units, qs)
            nat, arm = [], []
            for side in ("base", "donor"):
                batch = odd.base_batch if side == "base" else odd.donor_batch
                cache = odd.base_cache if side == "base" else odd.donor_cache
                bg = dict(cache)
                for rid in batch.row_ids:
                    for u in all_heads:
                        bg[(rid, u)] = mu_all[u]
                af_n = g.forward_units(backend, batch, units=[])
                af_a = g.forward_units(backend, batch, units=order, donor_cache=bg, base_cache=cache, q=bq, complement=True)
                nat += (af_n[:, 0] - af_n[:, 1]).tolist(); arm += (af_a[:, 0] - af_a[:, 1]).tolist()
            return nat, arm
        nat, m_null = margins(chosen, None)
        den = [a - b for a, b in zip(nat, m_null)]
        _, m_arm = margins(chosen, g.block_identity(backend, chosen))
        ext = v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)
        swaps = len([u for u in chosen[len(hub):] if u not in old_path])
        R[n] = {"hub": hub, "final": chosen, "old_final": old_path, "curve": curve, "old_path_C_curve": old_c, "swaps": swaps,
                "eval_odd": ev, "extraction_odd": ext, "old_A1_odd": old[n]["final_odd"]}
        print(n, json.dumps({"A1": round(ev["A1"]["ce_damage"], 3), "old_A1": round(old[n]["final_odd"], 3), "C": round(ev["C"]["ce_damage"], 3), "C_ub": round(ev["C"]["ce_ub975"], 3),
                             "A2": round(ev["A2"]["ce_damage"], 3), "ext": round(ext["point"], 3), "swaps": swaps,
                             "added": [c["added"][5:] for c in curve], "old_C_curve": [round(x["C_odd"], 3) for x in old_c]}))

    predictions = {
        'pred_a_row4_repaired': all(R[n]["eval_odd"]["C"]["ce_ub975"] <= C_UB for n in R),
        'pred_b_removal_kept': all(R[n]["eval_odd"]["A1"]["ce_damage"] >= KEEP_FRAC * R[n]["old_A1_odd"] for n in R),
        'pred_c_extraction_kept': all(R[n]["extraction_odd"]["point"] >= ROW2 for n in R),
        'pred_d_few_swaps': all(R[n]["swaps"] <= MAX_SWAPS for n in R),
        'pred_e_row5_kept': all(R[n]["eval_odd"]["A2"]["ce_lb975"] > 0 and R[n]["eval_odd"]["A2"]["ce_damage"] >= OOD_FRAC * R[n]["eval_odd"]["A1"]["ce_damage"] for n in R),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_c_penalised_greedy_result_v1", "candidate_id": "corpus.unit_c_penalised_greedy_v70",
              "bars": {"lambda": LAMBDA, "top_k": TOP_K, "c_ub": C_UB, "keep_frac": KEEP_FRAC, "row2": ROW2, "max_swaps": MAX_SWAPS, "ood_frac": OOD_FRAC},
              "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
