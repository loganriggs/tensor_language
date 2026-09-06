#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9), downstream module list, freeze semantics and atlas scores fixed before the run.
"""v25: Tier 3 first-order readers of the quantifier and dative head sets -- direct path vs downstream modules.

TIER_RUBRIC Tier 3 = Tier 2 plus controlled identification of the upstream writers or readers that feed
the component. The sets act at the final input position, so everything between them and the logits is
a final-position downstream module (attention heads and MLPs at later layers) or the residual skip
itself. With the set S exactly patched (block-live, donor values), freezing one downstream module M at
its BASE value at the same position removes whatever part of S's effect M mediates:
    rec(S)            mean signed recovery of the exact set on A1 (v9/v23 numbers)
    drop_M            rec(S) - rec(S | M frozen to base)     first-order mediation by M
    direct            rec(S | EVERY downstream module frozen) what the residual skip alone carries
Downstream = all MLPs at layers >= the set's last layer and all heads at layers > it (quantifier: mlp 11-17,
heads 12-17, 61 modules; dative: mlp 14-17, heads 15-17, 31 modules). Modules between set members are
not readers of the whole set and are left live (recorded).
Weight atlas (Codex's `subspace_weight_atlas`, read-only): each block's diff-in-means axis is mapped through
c_proj into the residual; each downstream head is scored by the Frobenius norm of its (q, k, v) read of that
residual axis, each MLP by its (Left, Right) read; a module's score is the max over the set's axes.
Prospective test: Spearman rank correlation between weight read score and drop_M over the downstream list.

REGISTERED BEFORE THE RUN
    pred_a_direct_path_dominant   direct / rec(S) >= 0.50 for quantifier (the residual skip carries at least
                                  half; cf. Codex's is_was finding). Worked: 0.40/0.64 = 0.62 -> True; 0.31 -> False.
    pred_b_no_necessary_module    max_M drop_M / rec(S) <= 0.20 for quantifier. Worked: 0.08/0.64 -> True.
    pred_c_mediation_additive     |sum_M drop_M - (rec(S) - direct)| <= 0.30 * rec(S) for quantifier.
                                  Worked: rec 0.64, direct 0.40, sum 0.30 -> |0.06| <= 0.19 -> True; sum 0.60 -> False.
    pred_d_atlas_ranks_readers    Spearman(weight score, drop_M) >= 0.30 over quantifier's 61 downstream
                                  modules. Worked: 0.45 -> True; 0.10 -> False.
    pred_e_dative_direct_too      direct / rec(S) >= 0.50 for dative. Worked: 0.35/0.57 = 0.61 -> True.
    Reading rule. a True: the set's Tier 3 reader is the unembedding via the residual skip, and the module
    list with drops IS the first-order attribution. a False: the effect is converted downstream; the top
    mediators by drop are the readers to characterize next (and d says whether weights could have predicted
    them). b False: name the necessary module; it joins the circuit at Tier 3.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import subspace_weight_atlas as atlas

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier3_readers_v25_result.json"
SETS = {"quantifier_number": (v15.SETS["quantifier_number"][0], v15.SETS["quantifier_number"][1]),
        "dative": (v15.SETS["dative"][0], v15.SETS["dative"][1])}
DIRECT_BAR, MAX_DROP_BAR, ADD_BAR, RHO_BAR = 0.50, 0.20, 0.30, 0.30
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 4000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_readers_v25", "sets": {k: v[1] for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _downstream(units):
    last = max(g.unit_layer(u) for u in units)
    return [f"mlp:{l:02d}" for l in range(last, g.N_LAYERS)] + \
           [f"attn:{l:02d}:head:{h:02d}" for l in range(last + 1, g.N_LAYERS) for h in range(g.N_HEADS)]


def _rec(prep, patched):
    return sum(kernel.signed_pairwise_donor_recovery(b, d, p)
               for b, d, p in zip(prep.base_axis, prep.donor_axis, patched)) / len(patched)


def _merged_cache(prep, set_units):
    cache = dict(prep.base_cache)
    for rid in prep.base_batch.row_ids:
        for u in set_units:
            cache[(rid, u)] = prep.donor_cache[(rid, u)]
    return cache


def _spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys); n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _atlas_scores(backend, prep, set_units, downstream):
    torch, model = backend.torch, backend.model
    axes = g.block_diff_in_means(backend, prep, set_units)          # (layer, kind) -> (D_blk, 1)
    resid_axes = []
    for (layer, kind), q in axes.items():
        heads = [int(u.rsplit(":", 1)[1]) for u in set_units if g.unit_layer(u) == layer and kind == "heads"]
        basis, _sv = atlas.map_head_bank_subspace_to_residual(model.transformer.h[layer].attn, heads, q.detach().float())
        resid_axes.append(basis)
    scores = {}
    for m in downstream:
        layer = g.unit_layer(m)
        best = 0.0
        for basis in resid_axes:
            if m.startswith("mlp"):
                s = atlas.mlp_subspace_tensor(model.transformer.h[layer].mlp, basis)["scores"]
                val = (s["left"] ** 2 + s["right"] ** 2) ** 0.5
            else:
                h = int(m.rsplit(":", 1)[1])
                s = atlas.attention_subspace_factors(model.transformer.h[layer].attn, basis)[h]["scores"]
                val = (s["q"] ** 2 + s["k"] ** 2 + s["v"] ** 2) ** 0.5
            best = max(best, val)
        scores[m] = best
    return scores


def _behaviour(backend, module, set_units):
    prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
    downstream = _downstream(set_units)
    for u in set_units + downstream:
        assert (prep.base_batch.row_ids[0], u) in prep.base_cache, u
    cache = _merged_cache(prep, set_units)
    rec_s = _rec(prep, g.patched_axis(backend, prep, set_units))
    def frozen(mods):
        out = g.forward_units(backend, prep.base_batch, units=list(set_units) + list(mods), donor_cache=cache, base_cache=prep.base_cache)
        return _rec(prep, [-(float(a) - float(f)) for a, f in out.tolist()])
    drops = {m: rec_s - frozen([m]) for m in downstream}
    direct = frozen(downstream)
    scores = _atlas_scores(backend, prep, set_units, downstream)
    top = sorted(downstream, key=lambda m: -drops[m])[:5]
    return {"rows": len(prep.rows), "units": list(set_units), "downstream_count": len(downstream),
            "rec_set": rec_s, "direct": direct, "direct_ratio": direct / rec_s if rec_s else None,
            "sum_drops": sum(drops.values()), "max_drop": max(drops.values()), "max_drop_module": max(drops, key=drops.get),
            "min_drop": min(drops.values()), "min_drop_module": min(drops, key=drops.get),
            "top_mediators": [(m, drops[m], scores[m]) for m in top],
            "top_by_weight": [(m, scores[m], drops[m]) for m in sorted(downstream, key=lambda m: -scores[m])[:5]],
            "spearman_weight_vs_drop": _spearman([scores[m] for m in downstream], [drops[m] for m in downstream]),
            "drops": drops, "weight_scores": scores}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        report[name] = r = _behaviour(backend, module, units)
        print(name, "rec %.3f direct %.3f (%.2f) sum drops %.3f max %.3f @%s min %.3f @%s rho %.2f" % (
            r["rec_set"], r["direct"], r["direct_ratio"], r["sum_drops"], r["max_drop"], r["max_drop_module"],
            r["min_drop"], r["min_drop_module"], r["spearman_weight_vs_drop"]), flush=True)
        print("  top mediators", [(m, round(d, 3), round(s, 2)) for m, d, s in r["top_mediators"]], flush=True)
        print("  top by weight", [(m, round(s, 2), round(d, 3)) for m, s, d in r["top_by_weight"]], flush=True)
    q, d = report["quantifier_number"], report["dative"]
    predictions = {
        'pred_a_direct_path_dominant': q["direct_ratio"] >= DIRECT_BAR,
        'pred_b_no_necessary_module': q["max_drop"] / q["rec_set"] <= MAX_DROP_BAR,
        'pred_c_mediation_additive': abs(q["sum_drops"] - (q["rec_set"] - q["direct"])) <= ADD_BAR * q["rec_set"],
        'pred_d_atlas_ranks_readers': q["spearman_weight_vs_drop"] >= RHO_BAR,
        'pred_e_dative_direct_too': d["direct_ratio"] >= DIRECT_BAR,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_tier3_readers_result_v1",
              "candidate_id": "corpus.unit_tier3_readers_v25", "semantics": "block_live_exact_set_plus_base_freeze",
              "bars": {"direct": DIRECT_BAR, "max_drop": MAX_DROP_BAR, "additive": ADD_BAR, "rho": RHO_BAR},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
