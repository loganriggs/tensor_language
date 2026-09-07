#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets read from the v66/v67 receipts, bars fixed before the run.
"""v68: does a removal-greedy set also EXTRACT the behaviour? Row 2 for the hub+3 / hub+8 sets of v66/v67.

Row 2 of the tier rubric (keep the set, mean-ablate every other head, recover >= 0.80 of the natural margin) is unmet
by all six 3-5 head sets (0.51-0.60, v52/v57). v66/v67 selected 8 extra heads per behaviour by REMOVAL damage. Removal
and extraction are different quantities (a set can be necessary without being sufficient), so this is a real test:
v52 extraction on the A1 ODD rows (the rows the greedy never saw) for hub, hub+3 and hub+8 of each of the six sets,
with keep_dim (rank-1 diff-in-means per block, fit on even rows) and a random head set of equal size (seed 1).

REGISTERED BEFORE THE RUN (extraction = (margin(keep) - margin(null)) / (natural - null), A1 odd rows)
    pred_a_row2_reached   at least one hub+8 set reaches keep_exact >= 0.80. Worked: complementizer 0.85 True; max 0.72 False.
    pred_b_grows          keep_exact(hub+8) >= keep_exact(hub) + 0.10 for every behaviour. Worked: 0.58 -> 0.75 True; 0.58 -> 0.62 False.
    pred_c_random_set     a random 11-head set extracts <= 0.50 x the hub+8 set, every behaviour. Worked: 0.25 vs 0.75 True; 0.45 vs 0.75 False.
    pred_d_dim_tracks     keep_dim(hub+8) >= 0.80 x keep_exact(hub+8) for every behaviour. Worked: 0.65 / 0.75 True; 0.50 / 0.75 False.
    pred_e_floor          keep_exact(hub+8) >= 0.70 for all six behaviours. Worked: min 0.72 True; min 0.65 False.
    Prior: b True, c True, d unsure (11 blocks x rank 1 may miss the extra heads' content), a and e unsure -- this is the
    question.
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
import run_unit_extraction_four_sets_v52 as v52

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_greedy_sets_extraction_v68_result.json"
SRC = {"verb": ROOT / "circuits/followups/unit_verb_greedy_saturation_v66_result.json",
       "four": ROOT / "circuits/followups/unit_four_sets_greedy_saturation_v67_result.json"}
ROW2, GROW, RAND_FRAC, DIM_FRAC, FLOOR, N_ADD = 0.80, 0.10, 0.50, 0.80, 0.70, 3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 12000


def _plan():
    return {"candidate_id": "corpus.unit_greedy_sets_extraction_v68", "sources": [str(p) for p in SRC.values()],
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
    sets = {}
    for p in SRC.values():
        for n, r in json.loads(p.read_text())["sets"].items():
            sets[n] = (r["hub"], r["final"])
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    R = {}
    for n, (hub, final) in sets.items():
        rows = g.rows_of(modules[n], "A1")
        even, odd = g.prepare(backend, rows[0::2]), g.prepare(backend, rows[1::2])
        mu_all = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (odd.base_cache, odd.donor_cache) for rid in odd.base_batch.row_ids]).mean(0) for u in all_heads}
        gen = torch.Generator().manual_seed(1)
        pool = [u for u in all_heads if u not in final]
        rand_set = [pool[i] for i in torch.randperm(len(pool), generator=gen)[:len(final)].tolist()]

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
        nat, m_null = margins(final, None)
        den = [a - b for a, b in zip(nat, m_null)]
        arms = {"hub": (hub, g.block_identity(backend, hub)),
                "hub3": (final[:len(hub) + N_ADD], g.block_identity(backend, final[:len(hub) + N_ADD])),
                "hub8": (final, g.block_identity(backend, final)),
                "hub8_dim": (final, g.block_diff_in_means(backend, even, final)),
                "rand_set": (rand_set, g.block_identity(backend, rand_set))}
        ext = {}
        for k, (su, qs) in arms.items():
            _, m_arm = margins(su, qs)
            ext[k] = v52._boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)
        R[n] = {"hub": hub, "final": final, "rand_set": rand_set, "extraction": ext, "rows_odd": len(odd.rows),
                "interchange_recovery_hub8": g.recovery(odd, g.patched_axis(backend, odd, final))}
        print(n, json.dumps({k: round(v["point"], 3) if isinstance(v, dict) else v for k, v in ext.items()}))

    e = lambda n, k: R[n]["extraction"][k]["point"]
    predictions = {
        'pred_a_row2_reached': any(e(n, "hub8") >= ROW2 for n in R),
        'pred_b_grows': all(e(n, "hub8") >= e(n, "hub") + GROW for n in R),
        'pred_c_random_set': all(e(n, "rand_set") <= RAND_FRAC * e(n, "hub8") for n in R),
        'pred_d_dim_tracks': all(e(n, "hub8_dim") >= DIM_FRAC * e(n, "hub8") for n in R),
        'pred_e_floor': all(e(n, "hub8") >= FLOOR for n in R),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_greedy_sets_extraction_result_v1", "candidate_id": "corpus.unit_greedy_sets_extraction_v68",
              "bars": {"row2": ROW2, "grow": GROW, "rand_frac": RAND_FRAC, "dim_frac": DIM_FRAC, "floor": FLOOR, "n_add": N_ADD},
              "sets": R, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions,
                      "extraction": {n: {k: round(v["point"], 3) for k, v in r["extraction"].items()} for n, r in R.items()},
                      "lb": {n: {k: round(v["lb95"], 3) for k, v in r["extraction"].items()} for n, r in R.items()},
                      "recovery_hub8": {n: round(r["interchange_recovery_hub8"], 3) for n, r in R.items()},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
