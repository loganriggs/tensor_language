#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets (v115), arms, K and bars fixed before the run; ranking on EVEN, replay on ODD.
"""v118: the DOWNSTREAM half of the Tier-4 expansion -- every downstream bilinear MLP's delta as exact products.
v115-v117 expanded each chosen head's WRITE into exact (offset, upstream-writer) products and replayed them; the
conversion of those writes into the answer margin by the MLPs after the heads was left unexpanded. bilin18's MLP is
`Bilinear` (hidden = Left(u) * Right(u), 4608 product terms, out = Down(hidden) + bias, no gate). For a downstream
layer l at the read position, with u^B = rms_norm(x^B), u^P = rms_norm(x^P) (P = the EXACT interchange of the set) and
s = u^P - u^B, the identity is exact (v42, one set / one layer; here every downstream layer of 14 sets):
    hidden^P - hidden^B = [L(u^B) R(s) + L(s) R(u^B)]  +  L(s) R(s)      =: cross_h + self_h
    out^P - out^B       = Down(cross_h) + Down(self_h)
cross is first order in the interchange delta (the base input multiplied by the delta), self is second order.
REPLAY (executable sufficiency): `forward_units` with the set's heads patched exactly AND every downstream MLP output at
the read position CLAMPED to a synthetic value (unit `mlp:LL`, layers l >= min head layer), later attention live:
    full    out^B + Down(cross_h) + Down(self_h)   (= out^P; instrument, must reproduce exact)
    cross   out^B + Down(cross_h)                  (the products linear in the delta)
    self    out^B + Down(self_h)                   (the products quadratic in the delta)
    direct  out^B                                  (no downstream MLP conversion at the read position at all)
    sparse  out^B + Down(cross_h * mask_l)          mask_l = top K=64 of 4608 neurons per layer, ranked on EVEN by the
            mean signed first-order margin contribution  <Down[:, i] cross_h_i, W_U[donor answer] - W_U[base answer]>
s is measured from the exact run (the clamp program is fully determined; no term is re-derived from a live input).
REGISTERED BEFORE THE RUN (14 behaviours; recoveries as fractions of the exact set interchange on ODD)
    pred_a_identity_instrument   max over downstream layers of |Delta out - Down(cross_h + self_h)| / |Delta out| <= 1e-3 on
                                 14/14 AND the full-clamp replay within 0.02 of exact on 14/14. Worked: 3e-7, |1.001-1.000|
                                 True; 0.02, 0.03 False.
    pred_b_cross_sufficient      cross arm >= 0.80 on >= 10 of 14.          Worked: 0.91 True; 0.55 False.
    pred_c_conversion_needed     |direct arm| <= 0.50 on >= 10 of 14.       Worked: 0.12 True; 0.63 False.
    pred_d_sparse_neurons        sparse arm (K=64 per layer, EVEN-ranked) >= 0.80 on >= 8 of 14. Worked: 0.84 True; 0.71 False.
    pred_e_self_small            |self arm| <= 0.30 on >= 10 of 14.         Worked: 0.08 True; 0.41 False.
    Prior: a 85%; b 60%; c 60%; d 40%; e 60%.
    Reading: a+b True on a set = the set's Tier-4 statement extends through the MLPs: the margin is produced by the base
    input's products with the head-write delta; c False on a set says the heads' writes reach the logit without MLP
    conversion (the direct route), which is reported, not repaired. A miss on d is reported with the K curve; K is not raised.
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
import run_unit_tier3_batch_v112 as v112
import run_unit_tier4_expansion_batch_v115 as v115

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier4_mlp_products_v118_result.json"
IDENT_TOL, INSTR_TOL, CROSS_MIN, DIRECT_MAX, SPARSE_MIN, SELF_MAX = 1e-3, 0.02, 0.80, 0.50, 0.80, 0.30
K_B, K_C, K_D, K_E, K_NEURONS = 10, 10, 8, 10, 64
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_tier4_mlp_products_v118", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def mlp_terms(backend, prep, units, layers):
    """Exact run (set patched) and base run with the MLP input residual + hidden captured at the read position.
    Returns per layer: dict rid -> (out_B, cross_h, self_h, ident_rel) and the unembedding margin vectors per row."""
    torch, F, model = backend.torch, backend.F, backend.model
    resP, hidP, resB, hidB = {}, {}, {}, {}
    g.forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache, capture_resid=resP, capture_hidden=hidP)
    g.forward_units(backend, prep.base_batch, capture_resid=resB, capture_hidden=hidB)
    out = {}
    with torch.no_grad():
        for l in layers:
            mlp = model.transformer.h[l].mlp
            per = {}
            for rid in prep.base_batch.row_ids:
                uB = F.rms_norm(resB[(rid, l)].to(backend.device), (g.N_EMBD,))
                uP = F.rms_norm(resP[(rid, l)].to(backend.device), (g.N_EMBD,))
                s = uP - uB
                LB, RB, Ls, Rs = mlp.Left(uB), mlp.Right(uB), mlp.Left(s), mlp.Right(s)
                cross_h, self_h = LB * Rs + Ls * RB, Ls * Rs
                hB = torch.as_tensor(hidB[(rid, g.hidden_key(l))]).to(backend.device).float()
                hP = torch.as_tensor(hidP[(rid, g.hidden_key(l))]).to(backend.device).float()
                d_out = mlp.Down(hP - hB)
                ident = float((d_out - mlp.Down(cross_h + self_h)).norm() / max(float(d_out.norm()), 1e-12))
                per[rid] = (mlp.Down(hB) + mlp.Down_bias, cross_h, self_h, ident)
            out[l] = per
    return out


def margin_dirs(backend, batch):
    W = backend.model.lm_head.weight
    return {rid: (W[f] - W[a]).float() for rid, a, f in zip(batch.row_ids, batch.answer_ids, batch.foil_ids)}


def replay(backend, prep, units, layers, terms, value):
    """Clamp every downstream MLP output at the read position to out_B + value(row terms); heads patched exactly."""
    torch = backend.torch
    cache = {k: v for k, v in prep.donor_cache.items() if k[1] in units}
    with torch.no_grad():
        for l in layers:
            mlp = backend.model.transformer.h[l].mlp
            for rid, (oB, cross_h, self_h, _) in terms[l].items():
                cache[(rid, f"mlp:{l:02d}")] = (oB + value(mlp, l, cross_h, self_h)).detach().cpu()
    out = g.forward_units(backend, prep.base_batch, units=list(units) + [f"mlp:{l:02d}" for l in layers], donor_cache=cache)
    return g.recovery(prep, [-(float(a) - float(f)) for a, f in out.tolist()])


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V118_SMOKE")          # local CPU smoke test of the code path only; never the science run
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    if smoke:
        S = dict(list(S.items())[:1])
    R = {}
    for n, units in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        if smoke:
            a1 = a1[:8]
        prep = {"even": g.prepare(backend, a1[0::2]), "odd": g.prepare(backend, a1[1::2])}
        layers = list(range(min(g.unit_layer(u) for u in units), g.N_LAYERS))
        T = {sp: mlp_terms(backend, prep[sp], units, layers) for sp in prep}
        ident = max(v[3] for l in layers for v in T["odd"][l].values())
        # EVEN ranking: mean signed first-order margin contribution per neuron, per layer
        W = margin_dirs(backend, prep["even"].base_batch)
        mask = {}
        with torch.no_grad():
            for l in layers:
                mlp = backend.model.transformer.h[l].mlp
                proj = mlp.Down.weight.float().T @ torch.stack([W[rid] for rid in T["even"][l]]).T      # (4608, rows)
                score = torch.stack([T["even"][l][rid][1] for rid in T["even"][l]]).T * proj            # (4608, rows)
                mk = torch.zeros(mlp.Down.weight.shape[1], device=backend.device)
                mk[score.mean(1).topk(K_NEURONS).indices] = 1.0
                mask[l] = mk
        exact = g.recovery(prep["odd"], g.patched_axis(backend, prep["odd"], units))
        frac = lambda v: round(v / exact, 3) if abs(exact) > 1e-6 else None
        arms = {
            "full": frac(replay(backend, prep["odd"], units, layers, T["odd"], lambda mlp, l, c, s: mlp.Down(c + s))),
            "cross": frac(replay(backend, prep["odd"], units, layers, T["odd"], lambda mlp, l, c, s: mlp.Down(c))),
            "self": frac(replay(backend, prep["odd"], units, layers, T["odd"], lambda mlp, l, c, s: mlp.Down(s))),
            "direct": frac(replay(backend, prep["odd"], units, layers, T["odd"], lambda mlp, l, c, s: 0.0 * mlp.Down(c))),
            "sparse": frac(replay(backend, prep["odd"], units, layers, T["odd"], lambda mlp, l, c, s: mlp.Down(c * mask[l]))),
        }
        # per-layer size of the delta and its cross/self split (ODD, mean over rows)
        with torch.no_grad():
            split = {}
            for l in layers:
                mlp = backend.model.transformer.h[l].mlp
                cs = [(float(mlp.Down(c).norm()), float(mlp.Down(s).norm())) for (_, c, s, _) in T["odd"][l].values()]
                split[f"{l:02d}"] = [round(sum(a for a, _ in cs) / len(cs), 3), round(sum(b for _, b in cs) / len(cs), 3)]
        R[n] = {"units": units, "layers": [layers[0], layers[-1]], "exact_odd": round(exact, 3), "identity_rel_max": ident,
                "arms": arms, "cross_self_norms_by_layer": split, "rows_even_odd": [len(a1[0::2]), len(a1[1::2])],
                "seconds": round(time.perf_counter() - t1, 1)}
        print(n, "exact", round(exact, 3), "ident", f"{ident:.1e}", arms, round(time.perf_counter() - t0), "s", flush=True)

    ok = lambda x: x is not None
    inst = [n for n, v in R.items() if v["identity_rel_max"] <= IDENT_TOL and ok(v["arms"]["full"]) and abs(v["arms"]["full"] - 1.0) <= INSTR_TOL]
    cross = [n for n, v in R.items() if ok(v["arms"]["cross"]) and v["arms"]["cross"] >= CROSS_MIN]
    direct = [n for n, v in R.items() if ok(v["arms"]["direct"]) and abs(v["arms"]["direct"]) <= DIRECT_MAX]
    sparse = [n for n, v in R.items() if ok(v["arms"]["sparse"]) and v["arms"]["sparse"] >= SPARSE_MIN]
    selfs = [n for n, v in R.items() if ok(v["arms"]["self"]) and abs(v["arms"]["self"]) <= SELF_MAX]
    predictions = {
        "pred_a_identity_instrument": len(inst) == len(R),
        "pred_b_cross_sufficient": len(cross) >= K_B,
        "pred_c_conversion_needed": len(direct) >= K_C,
        "pred_d_sparse_neurons": len(sparse) >= K_D,
        "pred_e_self_small": len(selfs) >= K_E,
    }
    result = {"predictions": predictions, "schema": "unit_tier4_mlp_products_v118",
              "candidate_id": "corpus.unit_tier4_mlp_products_v118",
              "bars": {"ident_tol": IDENT_TOL, "instr_tol": INSTR_TOL, "cross_min": CROSS_MIN, "direct_max": DIRECT_MAX,
                       "sparse_min": SPARSE_MIN, "self_max": SELF_MAX, "K": [K_B, K_C, K_D, K_E], "k_neurons": K_NEURONS},
              "counts": {"instrument": len(inst), "cross": len(cross), "direct_small": len(direct), "sparse": len(sparse),
                         "self_small": len(selfs), "n": len(R)},
              "tier4_mlp_sufficient": sorted(set(inst) & set(cross)),
              "summary": {n: [v["exact_odd"], v["arms"]["full"], v["arms"]["cross"], v["arms"]["self"], v["arms"]["direct"],
                              v["arms"]["sparse"]] for n, v in R.items()},
              "summary_columns": ["exact_odd", "full", "cross", "self", "direct", "sparse"],
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
