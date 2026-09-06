#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, the fixed vector, the exact expansion and the replay arms fixed before the run.
"""v42 (Tier 4 promotion test): exact replayed expansion of the downstream product Delta x v.

v39-v41: the readout of the converted vector v at the first stack layer is a + b*alpha in the residual write, exactly
bilinear, write- and vector-specific, replicated on A2 and under the real write, and carried by downstream MLPs
(polarity mlp 12-17 distributed; quantifier mlp:15). TIER_RUBRIC Tier 4 asks for an exact replayed algebraic
expansion of the relevant PRODUCTS of upstream contributions plus an executable sufficiency test. Four runs at
alpha = 1, iso design (other stack MLPs base-frozen): B base, D write only, V vector only, DV both. At every
downstream bilinear layer M(u) = Down[L(u) R(u)] with normalized input u, write s_D = u_D - u_B = p,
s_V = u_V - u_B = q, s_DV = p + q + iota (iota = interaction already in the input, propagated from earlier layers).
Exactly, with bil(a, c) = Down[L(a)R(c) + L(c)R(a)], cross(s) = bil(u_B, s), self(s) = Down[L(s)R(s)]:
    I_l := M(u_DV) - M(u_D) - M(u_V) + M(u_B) = bil(p, q) + cross(iota) + bil(p + q, iota) + self(iota).
The SELECTED terms are the pairwise first-order products bil(p_l, q_l). Sufficiency replay: run DV with each
downstream MLP output at the answer position replaced by out_D,l + (out_V,l - out_B,l) + bil(p_l, q_l) (attention
live). Linear control: the same without the products.

REGISTERED BEFORE THE RUN
    pred_a_identity          max_l ||I_l - [bil(p,q) + cross(iota) + bil(p+q, iota) + self(iota)]|| / ||I_l|| <= 1e-3 on
                             all three (the captures and the algebra are exact). Worked: 3e-6 True.
    pred_b_products_suffice  |rec(selected replay) - rec(DV)| <= 0.20 b on polarity (b = the gated part from the D/DV
                             difference against the linear control). Worked: b 0.025, error 0.003 True; 0.010 False.
    pred_c_linear_control    |rec(linear control) - rec(D)| reproduces a within 0.20 a on all three -- i.e. the control
                             lands at a, not at a + b. Worked: a 0.020, error 0.002 True.
    pred_d_polarity_spread   the best single layer's bil(p,q) replayed alone gives <= 50% of b on polarity. Worked:
                             0.30 True; 0.7 False.
    pred_e_quantifier_local  mlp:15's bil(p,q) alone gives >= 60% of b on quantifier. Worked: 0.85 True; 0.3 False.
    Reading rule. a, b, c True: Tier 4 for the polarity conversion -- the write, its linear conversion by mlp 08-11
    and the pairwise product in mlp 12-17 are an exact, replayed, sufficient expansion; d/e say how the product is
    spread. b False & a True: the propagated iota terms carry the product (a cascade) -- replay them next.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_norm_gain_control_v38 as v38
import run_unit_write_gated_readout_v39 as v39

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_product_expansion_v42_result.json"
SETS, STACK, N_LAYERS = v35.SETS, v35.STACK, 18
ID_TOL, SUFF_TOL, LIN_TOL, SPREAD_MAX, LOCAL_MIN = 1e-3, 0.20, 0.20, 0.5, 0.6
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 4000


def _plan():
    return {"candidate_id": "corpus.unit_product_expansion_v42", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _bil(mlp, a, c):
    WL, WR, WD = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()
    return ((a @ WL.T) * (c @ WR.T) + (c @ WL.T) * (a @ WR.T)) @ WD.T


def _self(mlp, s):
    WL, WR, WD = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()
    return ((s @ WL.T) * (s @ WR.T)) @ WD.T


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        first, last = layers[0], layers[-1]
        down = list(range(last + 1, N_LAYERS))
        mlp_first = model.transformer.h[first].mlp
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        _, ins_p1, _ = v30._capture(backend, prep, layers, resid_add={first: delta1})
        v = v38._cross(mlp_first, ins_b[first], ins_p1[first] - ins_b[first])
        fz = v39._Freezer(torch, model, list(prep.base_batch.semantic_positions), backend.device)

        def cfg(write, vec):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (outs_b[l] + vec if (l == first and vec is not None) else outs_b[l])[i]
            kw = {"units": stack_mlps, "donor_cache": c, "base_cache": prep.base_cache}
            if write is not None:
                kw["resid_add"] = {first: write}
            return kw

        runs = {}
        for key, (w, vec) in {"B": (None, None), "D": (delta1, None), "V": (None, v), "DV": (delta1, v)}.items():
            rec, ins, outs = v30._capture(backend, prep, down, **cfg(w, vec))
            runs[key] = {"rec": rec, "ins": ins, "outs": outs}
        a_obs = runs["V"]["rec"] - runs["B"]["rec"]
        readout_dv = runs["DV"]["rec"] - runs["D"]["rec"]
        b_obs = readout_dv - a_obs
        # exact expansion per downstream layer
        ident, prod, lin, per_layer = [], {}, {}, {}
        for l in down:
            mlp = model.transformer.h[l].mlp
            uB, uD, uV, uDV = runs["B"]["ins"][l], runs["D"]["ins"][l], runs["V"]["ins"][l], runs["DV"]["ins"][l]
            p, q = uD - uB, uV - uB
            iota = uDV - uD - uV + uB
            I = runs["DV"]["outs"][l] - runs["D"]["outs"][l] - runs["V"]["outs"][l] + runs["B"]["outs"][l]
            pq = _bil(mlp, p, q)
            recon = pq + _bil(mlp, uB, iota) + _bil(mlp, p + q, iota) + _self(mlp, iota)
            rel = float(((I - recon).norm(dim=1) / I.norm(dim=1).clamp_min(1e-12)).max())
            ident.append(rel)
            prod[l] = pq
            lin[l] = runs["D"]["outs"][l] + (runs["V"]["outs"][l] - runs["B"]["outs"][l])
            per_layer[f"mlp:{l:02d}"] = {"identity_rel_err_max": rel, "pq_norm_over_I": float((pq.norm(dim=1) / I.norm(dim=1).clamp_min(1e-12)).mean()),
                                         "iota_norm_over_p": float((iota.norm(dim=1) / p.norm(dim=1).clamp_min(1e-12)).mean())}

        def replay(values):
            for l in down:
                fz.store[("mlp", l)] = values[l]
            with fz.hooks("replace", down, []):
                return v30._capture(backend, prep, down, **cfg(delta1, v))[0]

        rec_lin = replay(lin)
        rec_sel = replay({l: lin[l] + prod[l] for l in down})
        single = {f"mlp:{l:02d}": replay({k: lin[k] + (prod[k] if k == l else 0) for k in down}) - rec_lin for l in down}
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "downstream": down, "rows": len(prep.rows),
                        "rec": {k: runs[k]["rec"] for k in runs}, "a_obs": a_obs, "readout_dv": readout_dv, "b_obs": b_obs,
                        "rec_linear_control": rec_lin, "rec_selected": rec_sel,
                        "linear_control_minus_D": rec_lin - runs["D"]["rec"], "selected_error": rec_sel - runs["DV"]["rec"],
                        "single_layer_product_share_of_b": {k: (x / b_obs if b_obs else None) for k, x in single.items()},
                        "identity_rel_err_max": max(ident), "per_layer": per_layer}
        print(name, "a %.4f b %.4f | lin-D %.4f | selected err %.4f | ident %.1e" % (a_obs, b_obs, rec_lin - runs["D"]["rec"], rec_sel - runs["DV"]["rec"], max(ident)),
              {k: round(x, 2) for k, x in report[name]["single_layer_product_share_of_b"].items()}, flush=True)

    pol, qua = report["polarity_licensing"], report["quantifier_number"]
    predictions = {
        'pred_a_identity': all(report[n]["identity_rel_err_max"] <= ID_TOL for n in SETS),
        'pred_b_products_suffice': abs(pol["selected_error"]) <= SUFF_TOL * abs(pol["b_obs"]),
        'pred_c_linear_control': all(abs(report[n]["linear_control_minus_D"] - report[n]["a_obs"]) <= LIN_TOL * abs(report[n]["a_obs"]) for n in SETS),
        'pred_d_polarity_spread': all(x is not None and x <= SPREAD_MAX for x in pol["single_layer_product_share_of_b"].values()),
        'pred_e_quantifier_local': qua["single_layer_product_share_of_b"].get("mlp:15") is not None and qua["single_layer_product_share_of_b"]["mlp:15"] >= LOCAL_MIN,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_product_expansion_result_v1",
              "candidate_id": "corpus.unit_product_expansion_v42",
              "bars": {"id_tol": ID_TOL, "suff_tol": SUFF_TOL, "lin_tol": LIN_TOL, "spread_max": SPREAD_MAX, "local_min": LOCAL_MIN},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
