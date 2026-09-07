#!/usr/bin/env python3
# BQGATE: five frozen predictions; set partition read from the v118 receipt by a fixed rule; arms, K and bars fixed before the run.
"""v119: is the non-MLP route of the head sets LITERAL (write -> final norm -> unembedding)? plus the two v118 arm repairs.
v118 (14 sets): the bilinear identity holds and the cross term alone gives >= 0.87 of exact on 14/14 -- but the DIRECT arm
(every downstream MLP output base-clamped at the read position, downstream attention LIVE) kept > 0.50 on 10 sets
(DIRECT sets: additive_scope, correlative, degree_frame, lexical_number_pp, narrative_tense, possessive x5) and <= 0.50 on
4 (CONVERSION sets: interrogative_licensing, perfect_number, preposition_selection, quantifier_number). Two v118 arms
were mis-designed and are re-registered here: the self arm was baseline + self (measures the direct route, not the
term); the sparse arm ranked neurons by SIGNED contribution and overshot (1.54) by keeping the aligned half of
cancelling pairs. Arms (heads of the set patched exactly; clamps at the read position only; `forward_units`):
    full_clamp      downstream MLPs clamped to out^P AND downstream heads clamped to their exact-run values (instrument)
    direct          downstream MLPs base-clamped, downstream attention live            (= v118 direct, re-measured)
    literal         downstream MLPs base-clamped AND downstream heads base-clamped     (only the set's own writes reach the logit)
    self_minus      (out^B + Down(self_h)) clamp minus the direct arm                  (the second-order term as a DIFFERENCE)
    sparse_K        out^B + Down(cross_h * mask_K), mask_K = top K by |mean first-order margin contribution| on EVEN,
                    K in {16, 64, 256, 1024} of 4608 per layer; read against the cross arm as |sparse_K - cross|
    neuron sharing  Jaccard of the top-64 |contribution| neuron sets at the shared downstream layers between pairs of
                    number-family sets (lexical_number_pp, perfect_number, quantifier_number)
REGISTERED BEFORE THE RUN (recoveries as fractions of the exact set interchange on ODD; rankings on EVEN)
    pred_a_instrument      full_clamp within 0.02 of exact on 14/14.                     Worked: |1.003-1| True; 0.03 False.
    pred_b_literal_route   literal >= 0.50 on >= 7 of the 10 DIRECT sets.                 Worked: 0.62 True; 0.35 False.
    pred_c_self_is_small   |self_minus| <= 0.30 on >= 12 of 14.                          Worked: -0.21 True; 0.41 False.
    pred_d_sparse_repaired |sparse_256 - cross| <= 0.20 on >= 3 of the 4 CONVERSION sets. Worked: 0.12 True; 0.31 False.
    pred_e_number_sharing  Jaccard(lexical_number_pp, perfect_number) >= 0.30 AND it exceeds Jaccard(lexical_number_pp,
                           quantifier_number).                                            Worked: 0.41 > 0.18 True; 0.22 False.
    Prior: a 85%; b 50%; c 80%; d 40%; e 55%.
    Reading: b True names the Tier-4 terminal stage of the DIRECT sets ("the set's write is read by the unembedding, no
    conversion"); b False says downstream attention at the read position converts it (name the layers from the per-layer
    receipt, no new arm). d is read at K=256 only; the K curve is reported, K is not raised.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import importlib
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier4_expansion_batch_v115 as v115
import run_unit_tier4_mlp_products_v118 as v118

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier4_direct_route_v119_result.json"
V118 = ROOT / "circuits/followups/unit_tier4_mlp_products_v118_result.json"
INSTR_TOL, LITERAL_MIN, SELF_MAX, SPARSE_TOL, JACCARD_MIN, K_B, K_C, K_D = 0.02, 0.50, 0.30, 0.20, 0.30, 7, 12, 3
KS, K_SHARE, K_READ, DIRECT_SPLIT = (16, 64, 256, 1024), 64, 256, 0.50
NUMBER = ("lexical_number_pp", "perfect_number", "quantifier_number")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_tier4_direct_route_v119", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def downstream_heads(units, layers):
    own = set(units)
    return [u for l in layers for u in (f"attn:{l:02d}:head:{h:02d}" for h in range(g.N_HEADS)) if u not in own]


def clamp_run(backend, prep, units, layers, terms, value, heads_from):
    """MLPs clamped to out_B + value(...); downstream heads clamped from `heads_from` (a cache) or live (None)."""
    torch = backend.torch
    cache = {k: v for k, v in prep.donor_cache.items() if k[1] in units}
    extra = []
    if heads_from is not None:
        extra = downstream_heads(units, layers)
        for u in extra:
            for rid in prep.base_batch.row_ids:
                cache[(rid, u)] = heads_from[(rid, u)]
    with torch.no_grad():
        for l in layers:
            mlp = backend.model.transformer.h[l].mlp
            for rid, (oB, cross_h, self_h, _) in terms[l].items():
                cache[(rid, f"mlp:{l:02d}")] = (oB + value(mlp, l, cross_h, self_h)).detach().cpu()
    out = g.forward_units(backend, prep.base_batch, units=list(units) + extra + [f"mlp:{l:02d}" for l in layers], donor_cache=cache)
    return g.recovery(prep, [-(float(a) - float(f)) for a, f in out.tolist()])


def exact_head_cache(backend, prep, units):
    """Every head's c_proj-input slice at the read position under the EXACT set interchange (for the instrument clamp)."""
    torch = backend.torch
    got = {}
    handles = []
    positions = list(prep.base_batch.semantic_positions)
    for l, block in enumerate(backend.model.transformer.h):
        def pre(_m, args, l=l):
            v = args[0]
            for i, rid in enumerate(prep.base_batch.row_ids):
                for h in range(g.N_HEADS):
                    got[(rid, f"attn:{l:02d}:head:{h:02d}")] = v[i, positions[i], h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM].detach().cpu().clone()
        handles.append(block.attn.c_proj.register_forward_pre_hook(pre))
    try:
        g.forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache)
    finally:
        for h in handles:
            h.remove()
    return got


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V119_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    prior = json.loads(V118.read_text())["summary"]
    direct_sets = sorted(n for n, v in prior.items() if v[4] is not None and v[4] > DIRECT_SPLIT)
    conv_sets = sorted(n for n in prior if n not in direct_sets)
    if smoke:
        S = dict(list(S.items())[:1])
    R, top_sets = {}, {}
    for n, units in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        if smoke:
            a1 = a1[:8]
        prep = {"even": g.prepare(backend, a1[0::2]), "odd": g.prepare(backend, a1[1::2])}
        layers = list(range(min(g.unit_layer(u) for u in units), g.N_LAYERS))
        T = {sp: v118.mlp_terms(backend, prep[sp], units, layers) for sp in prep}
        # EVEN ranking by |mean signed first-order margin contribution|
        W = v118.margin_dirs(backend, prep["even"].base_batch)
        score = {}
        with torch.no_grad():
            for l in layers:
                mlp = backend.model.transformer.h[l].mlp
                proj = mlp.Down.weight.float().T @ torch.stack([W[rid] for rid in T["even"][l]]).T
                score[l] = (torch.stack([T["even"][l][rid][1] for rid in T["even"][l]]).T * proj).mean(1).abs()
        masks = {}
        for K in KS:
            masks[K] = {}
            for l in layers:
                mk = torch.zeros(score[l].shape[0], device=backend.device)
                mk[score[l].topk(K).indices] = 1.0
                masks[K][l] = mk
        top_sets[n] = {l: set(score[l].topk(K_SHARE).indices.tolist()) for l in layers}
        exact = g.recovery(prep["odd"], g.patched_axis(backend, prep["odd"], units))
        frac = lambda v: round(v / exact, 3) if abs(exact) > 1e-6 else None
        P = prep["odd"]
        hx = exact_head_cache(backend, P, units)
        run = lambda value, heads_from=None: clamp_run(backend, P, units, layers, T["odd"], value, heads_from)
        full = run(lambda mlp, l, c, s: mlp.Down(c + s), hx)
        direct = run(lambda mlp, l, c, s: 0.0 * mlp.Down(c))
        literal = run(lambda mlp, l, c, s: 0.0 * mlp.Down(c), P.base_cache)
        selfa = run(lambda mlp, l, c, s: mlp.Down(s))
        cross = run(lambda mlp, l, c, s: mlp.Down(c))
        sparse = {K: frac(run(lambda mlp, l, c, s, K=K: mlp.Down(c * masks[K][l]))) for K in KS}
        arms = {"full_clamp": frac(full), "direct": frac(direct), "literal": frac(literal), "cross": frac(cross),
                "self_plus_direct": frac(selfa), "self_minus": (round((selfa - direct) / exact, 3) if abs(exact) > 1e-6 else None),
                "sparse": {str(K): v for K, v in sparse.items()},
                "sparse_minus_cross": {str(K): (round(v - frac(cross), 3) if v is not None else None) for K, v in sparse.items()}}
        R[n] = {"units": units, "layers": [layers[0], layers[-1]], "exact_odd": round(exact, 3), "arms": arms,
                "partition": "direct" if n in direct_sets else "conversion", "seconds": round(time.perf_counter() - t1, 1)}
        print(n, "exact", round(exact, 3), arms, round(time.perf_counter() - t0), "s", flush=True)

    def jaccard(a, b):
        ls = sorted(set(top_sets[a]) & set(top_sets[b]))
        if not ls:
            return None
        A = {(l, i) for l in ls for i in top_sets[a][l]}; B = {(l, i) for l in ls for i in top_sets[b][l]}
        return round(len(A & B) / len(A | B), 3)
    share = {f"{a}|{b}": jaccard(a, b) for i, a in enumerate(NUMBER) for b in NUMBER[i + 1:] if a in top_sets and b in top_sets}
    ok = lambda x: x is not None
    inst = [n for n, v in R.items() if ok(v["arms"]["full_clamp"]) and abs(v["arms"]["full_clamp"] - 1.0) <= INSTR_TOL]
    lit = [n for n in direct_sets if n in R and ok(R[n]["arms"]["literal"]) and R[n]["arms"]["literal"] >= LITERAL_MIN]
    selfs = [n for n, v in R.items() if ok(v["arms"]["self_minus"]) and abs(v["arms"]["self_minus"]) <= SELF_MAX]
    spars = [n for n in conv_sets if n in R and ok(R[n]["arms"]["sparse_minus_cross"][str(K_READ)])
             and abs(R[n]["arms"]["sparse_minus_cross"][str(K_READ)]) <= SPARSE_TOL]
    jl, jq = share.get("lexical_number_pp|perfect_number"), share.get("lexical_number_pp|quantifier_number")
    predictions = {
        "pred_a_instrument": len(inst) == len(R),
        "pred_b_literal_route": len(lit) >= K_B,
        "pred_c_self_is_small": len(selfs) >= K_C,
        "pred_d_sparse_repaired": len(spars) >= K_D,
        "pred_e_number_sharing": bool(ok(jl) and ok(jq) and jl >= JACCARD_MIN and jl > jq),
    }
    result = {"predictions": predictions, "schema": "unit_tier4_direct_route_v119",
              "candidate_id": "corpus.unit_tier4_direct_route_v119",
              "bars": {"instr_tol": INSTR_TOL, "literal_min": LITERAL_MIN, "self_max": SELF_MAX, "sparse_tol": SPARSE_TOL,
                       "jaccard_min": JACCARD_MIN, "K": [K_B, K_C, K_D], "ks": list(KS), "k_share": K_SHARE, "k_read": K_READ},
              "partition": {"direct": direct_sets, "conversion": conv_sets},
              "counts": {"instrument": len(inst), "literal": len(lit), "self_small": len(selfs), "sparse_ok": len(spars), "n": len(R)},
              "literal_sets": lit, "number_sharing_jaccard": share,
              "summary": {n: [v["exact_odd"], v["arms"]["full_clamp"], v["arms"]["direct"], v["arms"]["literal"], v["arms"]["cross"],
                              v["arms"]["self_minus"], v["arms"]["sparse"][str(K_READ)]] for n, v in R.items()},
              "summary_columns": ["exact_odd", "full_clamp", "direct", "literal", "cross", "self_minus", "sparse_256"],
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "jaccard": share, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
