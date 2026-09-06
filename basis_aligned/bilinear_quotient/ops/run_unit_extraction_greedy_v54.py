"""v54: how many heads does a 0.80-EXTRACTION circuit need? Greedy enlargement with keep-only as the objective.

v52/v53: the four Tier-2 head sets extract 0.51-0.60 of their behaviour when every other head is mean-ablated at the
prediction position -- what their interchange said, and below the rubric's 0.80 row-2 bar. The terminal axis therefore asks
for the writers of the missing 40%, chosen by the extraction objective itself. FIT/SELECT on the even A1 rows (16 pairs, 32
documents); FINAL one-shot on the odd A1 rows and on A2 (frozen). Candidates: the other heads (158-160); at each step the head
whose addition raises keep-only recovery most is added if the gain is >= 0.02; stop at 0.80 or after 4 additions.
Background for every arm = per-head mean over the FIT rows' base+donor sentences (so FINAL rows never enter the background).
Extraction recovery = sum(m_keep - m_null) / sum(m_native - m_null) (v52), bootstrap 2000 x seed 0 on FINAL rows.

REGISTERED BEFORE THE RUN
    pred_a_bar_on_fit         every set reaches keep-only >= 0.80 on the FIT rows within 4 added heads. Worked: 0.60 -> 0.83 at 3 True.
    pred_b_holds_heldout      every set: enlarged-set keep-only on the odd A1 rows point >= 0.80 and 95% LB >= 0.60 (rubric row 2).
                              Worked: 0.83 [0.72] True; 0.74 False.
    pred_c_a2_transfer        every set: A2 keep-only of the enlarged set >= 0.50 x its odd-A1 point with LB > 0. Worked 0.60/0.83 True.
    pred_d_gain_concentrated  every set: the first added head gives >= 0.50 of the total greedy gain (one missing writer, not a
                              diffuse tail). Worked: gains 0.12/0.05/0.03 -> 0.60 True; 0.06/0.06/0.05 -> 0.35 False.
    pred_e_direction_keeps    every set: the enlarged set's block diff-in-means direction (rank 1 per block, fit on FIT rows) keeps
                              >= 0.80 of the enlarged keep-only on the odd A1 rows. Worked: v53 0.96 True.
    Reading rule. a/b True: the row-2 circuit for the behaviour is the enlarged set; report its size. a False: attention alone at the
    prediction position does not reach 0.80 with <= 4 more heads -- the rest is not a few more heads; do not raise the head budget.
    Held-out and A2 are one-shot: no re-selection after seeing them.
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
import run_unit_extraction_four_sets_v52 as v52

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_extraction_greedy_v54_result.json"
SETS = v23.SETS
BAR, LB_MIN, MIN_GAIN, MAX_ADD, A2_FRAC, CONC_MIN, DIR_FRAC = 0.80, 0.60, 0.02, 4, 0.50, 0.50, 0.80
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 5400, 180000


def _plan():
    return {"candidate_id": "corpus.unit_extraction_greedy_v54", "sets": {k: v[1] for k, v in SETS.items()},
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
    report = {}
    for n, (m, units) in SETS.items():
        units = list(units)
        rows = g.rows_of(m, "A1")
        fit = g.prepare(backend, rows[0::2]); held = g.prepare(backend, rows[1::2]); a2 = g.prepare(backend, g.rows_of(m, "A2"))
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (fit.base_cache, fit.donor_cache) for rid in fit.base_batch.row_ids]).mean(0)
              for u in all_heads}

        def margins(prep, set_units, qs):
            order = v52.ordered_units(set_units)
            bq = v52.block_q(torch, backend.device, set_units, qs)
            nat, arm = [], []
            for side in ("base", "donor"):
                batch = prep.base_batch if side == "base" else prep.donor_batch
                cache = prep.base_cache if side == "base" else prep.donor_cache
                bg = dict(cache)
                for rid in batch.row_ids:
                    for u in all_heads:
                        bg[(rid, u)] = mu[u]
                af_n = g.forward_units(backend, batch, units=[])
                af_a = g.forward_units(backend, batch, units=order, donor_cache=bg, base_cache=cache, q=bq, complement=True)
                nat += (af_n[:, 0] - af_n[:, 1]).tolist(); arm += (af_a[:, 0] - af_a[:, 1]).tolist()
            return nat, arm

        nulls = {}
        for key, prep in (("fit", fit), ("held", held), ("a2", a2)):
            nat, m_null = margins(prep, units, None)
            nulls[key] = (nat, m_null, [a - b for a, b in zip(nat, m_null)])

        def keep(prep_key, prep, set_units, qs=None):
            _, m_arm = margins(prep, set_units, qs if qs is not None else g.block_identity(backend, set_units))
            nat, m_null, den = nulls[prep_key]
            return v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)

        cur = list(units)
        path = [{"added": None, "fit_keep": keep("fit", fit, cur)["point"]}]
        for step in range(MAX_ADD):
            if path[-1]["fit_keep"] >= BAR:
                break
            best, best_val = None, -1e9
            for h in all_heads:
                if h in cur:
                    continue
                val = keep("fit", fit, cur + [h])["point"]
                if val > best_val:
                    best, best_val = h, val
            if best_val - path[-1]["fit_keep"] < MIN_GAIN:
                path.append({"added": None, "fit_keep": path[-1]["fit_keep"], "stopped": f"best gain {best_val - path[-1]['fit_keep']:.3f} < {MIN_GAIN} ({best})"})
                break
            cur.append(best)
            path.append({"added": best, "fit_keep": best_val})
        gains = [p["fit_keep"] - q["fit_keep"] for q, p in zip(path, path[1:]) if p.get("added")]
        q_dir = g.block_diff_in_means(backend, fit, cur)
        r = {"units_start": units, "units_final": cur, "added": [p["added"] for p in path[1:] if p.get("added")], "path": path,
             "held_keep_start": keep("held", held, units), "held_keep_final": keep("held", held, cur), "held_keep_direction": keep("held", held, cur, q_dir),
             "a2_keep_start": keep("a2", a2, units), "a2_keep_final": keep("a2", a2, cur),
             "held_interchange_start": g.recovery(held, g.patched_axis(backend, held, units)), "held_interchange_final": g.recovery(held, g.patched_axis(backend, held, cur)),
             "first_gain_share": (gains[0] / sum(gains)) if gains and sum(gains) > 0 else None}
        report[n] = r
        print(n, "fit %.3f -> %.3f via %s | held %.3f -> %.3f [%.3f] dir %.3f | A2 %.3f -> %.3f | interchange %.3f -> %.3f" % (
            path[0]["fit_keep"], path[-1]["fit_keep"], r["added"], r["held_keep_start"]["point"], r["held_keep_final"]["point"], r["held_keep_final"]["lb95"],
            r["held_keep_direction"]["point"], r["a2_keep_start"]["point"], r["a2_keep_final"]["point"], r["held_interchange_start"], r["held_interchange_final"]), flush=True)
    predictions = {
        'pred_a_bar_on_fit': all(report[n]["path"][-1]["fit_keep"] >= BAR for n in SETS),
        'pred_b_holds_heldout': all(report[n]["held_keep_final"]["point"] >= BAR and report[n]["held_keep_final"]["lb95"] >= LB_MIN for n in SETS),
        'pred_c_a2_transfer': all(report[n]["a2_keep_final"]["point"] >= A2_FRAC * report[n]["held_keep_final"]["point"] and report[n]["a2_keep_final"]["lb95"] > 0 for n in SETS),
        'pred_d_gain_concentrated': all(report[n]["first_gain_share"] is not None and report[n]["first_gain_share"] >= CONC_MIN for n in SETS),
        'pred_e_direction_keeps': all(report[n]["held_keep_direction"]["point"] >= DIR_FRAC * report[n]["held_keep_final"]["point"] for n in SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_extraction_greedy_result_v1", "candidate_id": "corpus.unit_extraction_greedy_v54",
              "bars": {"bar": BAR, "lb_min": LB_MIN, "min_gain": MIN_GAIN, "max_add": MAX_ADD, "a2_frac": A2_FRAC, "conc_min": CONC_MIN, "dir_frac": DIR_FRAC},
              "split": "A1 even rows FIT/SELECT; A1 odd rows and A2 FINAL one-shot; background from FIT rows",
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
