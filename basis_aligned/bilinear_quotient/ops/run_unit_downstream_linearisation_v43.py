#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, the fixed vector, the linearisation arms and the layer sets fixed before the run.
"""v43: WHERE is the write x converted-write product formed? Live linearisation of the downstream MLPs.

v42 showed the pairwise products bil(p_l, q_l) replayed at fixed captured inputs reproduce the gated readout b for
polarity and voice but not for quantifier (error 0.0048 of b = 0.0086), where the per-layer freezes cancel
(mlp:15 +0.45, 16 +0.34, 17 -0.44). Static replay cannot tell a product formed AT mlp:16/17 from the linear
propagation of a product formed at mlp:15. This run can: every downstream MLP (layers after the stack) is
LINEARISED live around its base input at the answer position, M(u) -> M(u_B) + bil(u_B, u - u_B) (exact first
order; drops every second-order term whatever its origin), except a chosen set K of layers that keep their full
bilinear map. Attention stays live throughout. Four runs per arm (B, D, V, DV; iso design as v42, alpha = 1),
interaction I_K = rec(DV) - rec(D) - rec(V) + rec(B); b_full = I under no linearisation (= v42's b_obs).
Shares s_K = I_K / b_full. Arms: K = all (full), K = {} (all linearised), K = {l} for each downstream l, K = {15,16,17}.

REGISTERED BEFORE THE RUN
    pred_a_linearised_removes    s_{} <= 0.50 on all three: with every downstream MLP first-order the gate is mostly
                                 gone (what remains is attention's own product; v39 said 0.29 for polarity).
                                 Worked: polarity 0.21 True; 0.70 False.
    pred_b_quantifier_15         quantifier s_{15} >= 0.60: the product is formed at mlp:15 and mlp:16/17's frozen
                                 contributions were propagation of it. Worked: 0.90 True; 0.45 False.
    pred_c_quantifier_top3       quantifier |s_{15,16,17} - 1| <= 0.20. Worked: 0.95 True; 0.6 False.
    pred_d_polarity_distributed  polarity max_l s_{l} <= 0.50. Worked: 0.30 True; 0.7 False.
    pred_e_products_additive     polarity and voice: sum_l s_{l} within [0.7, 1.3] x (1 - s_{}) -- single-layer
                                 products add up to the MLP-carried gate. Worked: 1.05 True; 1.6 False.
    Control (not a prediction, must hold): B under full linearisation equals live B to 1e-4 (the hook is the
    identity at the base input).
    Reading rule. a True: the gate is a second-order MLP effect. b True: quantifier's product is local to mlp:15
    (the v42 static replay failed because 16/17 propagate it); b False and c True: it is a genuine cascade.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_norm_gain_control_v38 as v38
import run_unit_product_expansion_v42 as v42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_downstream_linearisation_v43_result.json"
SETS, STACK, N_LAYERS = v35.SETS, v35.STACK, 18
LIN_MAX, LOCAL_MIN, TOP3_TOL, SPREAD_MAX, ADD_BAND, CTRL_TOL = 0.50, 0.60, 0.20, 0.50, (0.7, 1.3), 1e-4
TOP3 = (15, 16, 17)
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 160, 5200


def _plan():
    return {"candidate_id": "corpus.unit_downstream_linearisation_v43", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


@contextmanager
def _linearised(torch, model, layers, positions, device, u_base, out_base):
    """Replace mlp:l output at the row positions by M(u_B) + bil(u_B, u - u_B) with u the LIVE normalized input."""
    idx = torch.arange(len(positions), device=device)
    pos = torch.tensor(positions, device=device)
    handles = []

    def mk(l):
        mlp = model.transformer.h[l].mlp

        def hook(m, a, o):
            u = a[0][idx, pos].float()
            new = out_base[l] + v42._bil(mlp, u_base[l], u - u_base[l])
            y = o.clone()
            y[idx, pos] = new.to(y.dtype)
            return y
        return hook
    for l in layers:
        handles.append(model.transformer.h[l].mlp.register_forward_hook(mk(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        positions = list(prep.base_batch.semantic_positions)
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

        def cfg(write, vec):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (outs_b[l] + vec if (l == first and vec is not None) else outs_b[l])[i]
            kw = {"units": stack_mlps, "donor_cache": c, "base_cache": prep.base_cache}
            if write is not None:
                kw["resid_add"] = {first: write}
            return kw

        RUNS = {"B": (None, None), "D": (delta1, None), "V": (None, v), "DV": (delta1, v)}
        # live base captures of the downstream MLPs (inputs and outputs) for the linearisation
        rec_b_live, u_base, out_base = v30._capture(backend, prep, down, **cfg(None, None))

        def four(lin_layers):
            recs = {}
            for key, (w, vec) in RUNS.items():
                with _linearised(torch, model, lin_layers, positions, backend.device, u_base, out_base):
                    recs[key] = v30._capture(backend, prep, down, **cfg(w, vec))[0]
            return recs, recs["DV"] - recs["D"] - recs["V"] + recs["B"]

        full_recs, b_full = four([])
        lin_recs, i_none = four(down)
        ctrl_err = abs(lin_recs["B"] - rec_b_live)
        keep_one = {}
        for l in down:
            _, i_l = four([k for k in down if k != l])
            keep_one[f"mlp:{l:02d}"] = i_l / b_full if b_full else None
        _, i_top3 = four([k for k in down if k not in TOP3])
        s_none = i_none / b_full if b_full else None
        s_top3 = i_top3 / b_full if b_full else None
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "downstream": down, "rows": len(prep.rows),
                        "rec_full": full_recs, "rec_all_linearised": lin_recs, "b_full": b_full,
                        "a_full": full_recs["V"] - full_recs["B"], "a_all_linearised": lin_recs["V"] - lin_recs["B"],
                        "share_all_linearised": s_none, "share_keep_one": keep_one, "share_keep_top3": s_top3,
                        "sum_keep_one": sum(x for x in keep_one.values() if x is not None), "control_base_identity_err": ctrl_err}
        print(name, "b %.4f | s_none %.2f | top3 %.2f | ctrl %.1e" % (b_full, s_none, s_top3, ctrl_err),
              {k: round(x, 2) for k, x in keep_one.items()}, flush=True)

    pol, qua, voi = report["polarity_licensing"], report["quantifier_number"], report["voice_frame"]

    def additive(r):
        s = r["share_all_linearised"]
        target = 1 - s
        return target > 0 and ADD_BAND[0] * target <= r["sum_keep_one"] <= ADD_BAND[1] * target
    predictions = {
        'pred_a_linearised_removes': all(report[n]["share_all_linearised"] is not None and report[n]["share_all_linearised"] <= LIN_MAX for n in SETS),
        'pred_b_quantifier_15': qua["share_keep_one"].get("mlp:15") is not None and qua["share_keep_one"]["mlp:15"] >= LOCAL_MIN,
        'pred_c_quantifier_top3': qua["share_keep_top3"] is not None and abs(qua["share_keep_top3"] - 1) <= TOP3_TOL,
        'pred_d_polarity_distributed': all(x is not None and x <= SPREAD_MAX for x in pol["share_keep_one"].values()),
        'pred_e_products_additive': additive(pol) and additive(voi),
    }
    control_ok = all(report[n]["control_base_identity_err"] <= CTRL_TOL for n in SETS)
    result = {"predictions": predictions, "control_base_identity_ok": control_ok,
              "schema": "circuit_unit_downstream_linearisation_result_v1",
              "candidate_id": "corpus.unit_downstream_linearisation_v43",
              "bars": {"lin_max": LIN_MAX, "local_min": LOCAL_MIN, "top3_tol": TOP3_TOL, "spread_max": SPREAD_MAX, "add_band": ADD_BAND, "ctrl_tol": CTRL_TOL, "top3": TOP3},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "control_ok": control_ok, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
