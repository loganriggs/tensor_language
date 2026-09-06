#!/usr/bin/env python3
# BQGATE: frozen predictions; sets (v9), downstream lists, z-score baseline (64 random axes, seed 0) and freeze semantics fixed before the run.
"""v26: Tier 3 readers for all four sets, z-scored weight atlas, and dative's damping MLPs.

v25: quantifier's set reaches the logits 73% through the residual skip with mlp 11-13 as small positive
mediators; dative's set writes 170% of its observed effect and mlp 15-17 push back. v25's prospective
weight test failed at rho 0.296 because a raw read norm is not discriminative within a module type (every
MLP 20-24, every head 3-6 for any axis). Three things here, one runner:
  1. reader decomposition (v25 `_behaviour`: freeze each downstream module to base, and all of them) for
     polarity (downstream mlp 8-17, heads 9-17) and voice (mlp 7-17, heads 8-17), quantifier and dative
     recomputed for one consistent table;
  2. z-scored atlas: each module's read norm of the set's residual-mapped axis, minus the mean and over the
     std of its read norms of 64 random unit axes (seed 0); prospective Spearman against drop / rec(S),
     POOLED over the four behaviours, MLPs and heads separately;
  3. dative's late MLPs: with the set patched, freeze mlp 15-17 to their DONOR values (is the damping a
     base-context computation?), and freeze them to donor WITHOUT the set (do they carry the decision?).

REGISTERED BEFORE THE RUN
    pred_a_zscore_ranks_mlps   pooled-MLP Spearman(z-score, drop / rec) >= 0.30 over the (behaviour, MLP)
                               pairs (7 + 4 + 10 + 11 = 32). Worked: 0.42 -> True; 0.12 -> False.
    pred_b_direct_generalises  direct / rec(S) >= 0.50 for polarity AND voice. Worked: 0.70, 0.85 -> True.
    pred_c_heads_not_readers   no downstream head has |drop| > 0.05 * rec(S) on polarity or voice.
                               Worked: max 0.012 / 0.59 = 0.02 -> True; 0.06 / 0.59 = 0.10 -> False.
    pred_d_damping_is_context  dative: rec(S | mlp 15-17 := donor) >= 0.90. Worked: 0.95 -> True; 0.62 -> False.
    pred_e_late_mlps_not_carriers  dative: rec(mlp 15-17 := donor, no set) <= 0.30. Worked: 0.12 -> True.
    Reading rule. a True: the z-scored atlas predicts which MLPs read a set's axis -- a prospective Tier 3
    instrument. a False: weights do not rank readers even after normalisation; readers must be found
    causally. d True and e True: dative's late MLPs are context-conditioned dampers, not carriers -- the
    set's Tier 3 reader is the unembedding, and the damping belongs to the base sentence's own evidence.
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
import run_unit_tier3_readers_v25 as v25
import subspace_weight_atlas as atlas

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier3_readers_zscore_v26_result.json"
SETS = v23.SETS
N_RANDOM, RANDOM_SEED = 64, 0
RHO_BAR, DIRECT_BAR, HEAD_BAR, DAMP_BAR, CARRY_BAR = 0.30, 0.50, 0.05, 0.90, 0.30
DATIVE_LATE = ["mlp:15", "mlp:16", "mlp:17"]
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 10000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_readers_zscore_v26", "sets": {k: v[1] for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _read_norm(model, m, basis):
    layer = g.unit_layer(m)
    if m.startswith("mlp"):
        s = atlas.mlp_subspace_tensor(model.transformer.h[layer].mlp, basis)["scores"]
        return (s["left"] ** 2 + s["right"] ** 2) ** 0.5
    h = int(m.rsplit(":", 1)[1])
    s = atlas.attention_subspace_factors(model.transformer.h[layer].attn, basis)[h]["scores"]
    return (s["q"] ** 2 + s["k"] ** 2 + s["v"] ** 2) ** 0.5


def _baselines(backend, modules):
    torch, model = backend.torch, backend.model
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    axes = [torch.nn.functional.normalize(torch.randn(g.N_EMBD, 1, generator=gen), dim=0).to(backend.device) for _ in range(N_RANDOM)]
    stats = {}
    for m in modules:
        vals = torch.tensor([_read_norm(model, m, a) for a in axes])
        stats[m] = (float(vals.mean()), float(vals.std()))
    return stats


def _resid_axes(backend, prep, set_units):
    model = backend.model
    out = []
    for (layer, kind), q in g.block_diff_in_means(backend, prep, set_units).items():
        heads = [int(u.rsplit(":", 1)[1]) for u in set_units if g.unit_layer(u) == layer]
        basis, _ = atlas.map_head_bank_subspace_to_residual(model.transformer.h[layer].attn, heads, q.detach().float())
        out.append(basis)
    return out


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report, pooled = {}, {"mlp": [], "attn": []}
    all_modules = sorted({m for _, units in SETS.values() for m in v25._downstream(units)})
    baselines = _baselines(backend, all_modules)
    for name, (module, units) in SETS.items():
        r = v25._behaviour(backend, module, units)
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        axes = _resid_axes(backend, prep, units)
        z = {}
        for m in r["drops"]:
            raw = max(_read_norm(backend.model, m, a) for a in axes)
            mu, sd = baselines[m]
            z[m] = (raw - mu) / sd if sd > 0 else 0.0
            pooled["mlp" if m.startswith("mlp") else "attn"].append((z[m], r["drops"][m] / r["rec_set"], f"{name}:{m}"))
        r["zscores"] = z
        r["top_by_z"] = [(m, round(z[m], 2), round(r["drops"][m], 3)) for m in sorted(z, key=lambda m: -z[m])[:5]]
        heads = [m for m in r["drops"] if m.startswith("attn")]
        r["max_head_abs_drop_ratio"] = max(abs(r["drops"][m]) for m in heads) / r["rec_set"]
        del r["weight_scores"]
        report[name] = r
        print(name, "rec %.3f direct %.3f (%.2f) max head |drop|/rec %.3f" % (r["rec_set"], r["direct"], r["direct_ratio"], r["max_head_abs_drop_ratio"]), flush=True)
        print("  top by z", r["top_by_z"], flush=True)
        print("  top mediators", [(m, round(d, 3), round(z[m], 2)) for m, d, _ in r["top_mediators"]], flush=True)
    rho = {k: v25._spearman([a for a, _, _ in v], [b for _, b, _ in v]) for k, v in pooled.items()}
    print("pooled rho", rho, {k: len(v) for k, v in pooled.items()}, flush=True)

    # dative's late MLPs
    module, units = SETS["dative"]
    prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
    cache_set = v25._merged_cache(prep, units)                       # set := donor, everything else base
    # freeze late MLPs to DONOR under the set patch: donor cache for set + late MLPs, base elsewhere
    cache_both = v25._merged_cache(prep, list(units) + DATIVE_LATE)
    def rec_with(cache, us):
        out = g.forward_units(backend, prep.base_batch, units=us, donor_cache=cache, base_cache=prep.base_cache)
        return v25._rec(prep, [-(float(a) - float(f)) for a, f in out.tolist()])
    dative = {"rec_set": rec_with(cache_set, list(units)),
              "rec_set_late_frozen_base": rec_with(cache_set, list(units) + DATIVE_LATE),
              "rec_set_late_frozen_donor": rec_with(cache_both, list(units) + DATIVE_LATE),
              "rec_late_donor_alone": rec_with(cache_both, DATIVE_LATE),
              "rec_each_late_donor_alone": {m: rec_with(cache_both, [m]) for m in DATIVE_LATE}}
    print("dative late", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dative.items()}, flush=True)

    predictions = {
        'pred_a_zscore_ranks_mlps': rho["mlp"] >= RHO_BAR,
        'pred_b_direct_generalises': all(report[k]["direct_ratio"] >= DIRECT_BAR for k in ("polarity_licensing", "voice_frame")),
        'pred_c_heads_not_readers': all(report[k]["max_head_abs_drop_ratio"] <= HEAD_BAR for k in ("polarity_licensing", "voice_frame")),
        'pred_d_damping_is_context': dative["rec_set_late_frozen_donor"] >= DAMP_BAR,
        'pred_e_late_mlps_not_carriers': dative["rec_late_donor_alone"] <= CARRY_BAR,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_tier3_readers_zscore_result_v1",
              "candidate_id": "corpus.unit_tier3_readers_zscore_v26", "semantics": "block_live_exact_set_plus_freeze",
              "bars": {"rho": RHO_BAR, "direct": DIRECT_BAR, "head": HEAD_BAR, "damp": DAMP_BAR, "carry": CARRY_BAR,
                       "random_axes": N_RANDOM, "random_seed": RANDOM_SEED},
              "pooled_spearman": rho, "pooled_points": {k: len(v) for k, v in pooled.items()},
              "pooled_mlp_table": sorted(pooled["mlp"], key=lambda t: -t[0]),
              "behaviours": report, "dative_late_mlps": dative,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "pooled_spearman": rho, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
