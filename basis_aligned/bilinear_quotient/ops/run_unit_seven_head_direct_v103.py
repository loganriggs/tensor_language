#!/usr/bin/env python3
# BQGATE: five frozen predictions; the seven head sets and every bar are fixed before the run; no fitting.
"""v103: per-head MEASURED direct-path share for the seven head sets (readout heads vs feature heads).

v102 estimated (first-order DLA) that 5 of 7 sets act mostly by writing the answer logit directly, and that the residual
lambdas[0] (0.013 at layer 1, 0.064 at layer 5) nearly reset the stream, so layer <= 4 heads can only act through later
readers. This run MEASURES it per head with the producer's own forward (`g.forward_units(resid_add=...)`, which adds a
vector to the residual at the prediction position after a layer's attention):

  single_h   exact interchange of head h alone (c_proj-input patch), ODD A1, summed axis change over rows.
  own_h      resid_add at h's own layer L of W_L[:, h] (donor - base) -- must equal single_h (c_proj is linear, no bias).
  direct_h   resid_add at the LAST layer (17) of prod_{l>L} lambdas0[l] * W_L[:, h] (donor - base): the head's write carried
             to the output by the residual alone (it still passes MLP 17 and the final norm, so 'direct' = through no later
             attention or MLP < 17). share_h = direct_h / single_h, reported for heads whose |single_h| >= 5% of the set's
             exact-set effect (smaller heads are listed as 'small').
  set level  own-layer add of every head's write at once (instrument: must reproduce the exact-set patch) and final-layer
             add of the lambda-scaled sum (measured set direct share, to validate v102's first-order estimate).

REGISTERED BEFORE THE RUN (ODD A1 rows; sets = six v80 hub+8 + modal v97; cosines read from the v102 receipt)
    pred_a_instrument     own-layer add of all heads' writes / exact-set patch in [0.95, 1.05] on all seven sets. Worked: 0.99 True; 0.90 False.
    pred_b_first_order    measured set direct share within +-0.25 of v102's first-order share on >= 5 of 7 sets. Worked: 0.78 vs 0.70 True; 0.78 vs 0.30 False.
    pred_c_reset          every layer <= 4 head with |single| >= 5% has share <= 0.30 (needs >= 3 such heads pooled over sets, else UNTESTED). Worked: 0.05, 0.1, 0.2 True; one at 0.6 False.
    pred_d_readout_head   >= 5 of 7 sets have a head with |single| >= 5% and share >= 0.70. Worked: five sets with a 0.8 head True; three False.
    pred_e_cos_tracks     Spearman(|cos| from v102, share) >= 0.50 over the pooled >= 5% heads. Worked: 0.6 True; 0.2 False.
    Prior: a ~85%; b ~60%; c ~70%; d ~65%; e ~50%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_seven_head_direct_v103_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
V102 = ROOT / "circuits/followups/unit_seven_readout_audit_v102_result.json"
INSTR_LO, INSTR_HI, FO_TOL, MIN_SINGLE, RESET_MAX, READOUT_MIN, RHO_MIN, EARLY, MIN_EARLY, K = 0.95, 1.05, 0.25, 0.05, 0.30, 0.70, 0.50, 4, 3, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 12000


def _plan():
    return {"candidate_id": "corpus.unit_seven_head_direct_v103",
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def head_of(u):
    return int(u.rsplit(":", 1)[1])


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    rx, ry = ranks(xs), ranks(ys); n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")},
               "modal_remoteness": m_modal}
    sets = {n: s["units"] for n, s in json.loads(V80.read_text())["sets"].items()}
    sets["modal_remoteness"] = json.loads(V97.read_text())["final"]
    v102 = json.loads(V102.read_text())
    lam0 = [float(b.lambdas[0]) for b in model.transformer.h]
    W = [b.attn.c_proj.weight.detach().float() for b in model.transformer.h]
    LAST = g.N_LAYERS - 1

    def writes(O, u):
        """(n, 1152) residual write of head u's donor-base delta at the prediction position, per row."""
        L, h = g.unit_layer(u), head_of(u); sl = slice(h * g.HEAD_DIM, (h + 1) * g.HEAD_DIM)
        dv = torch.stack([torch.as_tensor(O.donor_cache[(rid, u)]).float() - torch.as_tensor(O.base_cache[(rid, u)]).float()
                          for rid in O.base_batch.row_ids]).to(backend.device)
        return dv @ W[L][:, sl].T

    def scale(L):
        s = 1.0
        for l in range(L + 1, g.N_LAYERS):
            s *= lam0[l]
        return s

    def axis_change(O, resid_add):
        out = g.forward_units(backend, O.base_batch, units=[], donor_cache=O.base_cache, base_cache=O.base_cache, resid_add=resid_add)
        return sum(-(float(a) - float(f)) - b for (a, f), b in zip(out.tolist(), O.base_axis))

    report = {}
    for n, units in sets.items():
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        exact = sum(e - b for e, b in zip(g.patched_axis(backend, O, units), O.base_axis))
        wr = {u: writes(O, u) for u in units}
        own = {}
        for u in units:
            own.setdefault(g.unit_layer(u), torch.zeros_like(wr[u]))
            own[g.unit_layer(u)] = own[g.unit_layer(u)] + wr[u]
        own_all = axis_change(O, own)
        direct_all = axis_change(O, {LAST: sum(scale(g.unit_layer(u)) * wr[u] for u in units)})
        heads = {}
        for u in units:
            single = sum(e - b for e, b in zip(g.patched_axis(backend, O, [u]), O.base_axis))
            own_u = axis_change(O, {g.unit_layer(u): wr[u]})
            direct_u = axis_change(O, {LAST: scale(g.unit_layer(u)) * wr[u]})
            big = abs(single) >= MIN_SINGLE * abs(exact)
            heads[u] = {"single": round(single, 3), "own": round(own_u, 3), "direct": round(direct_u, 3),
                        "share": round(direct_u / single, 3) if big else None, "big": big,
                        "cos": v102["sets"][n]["cos_per_head"][u], "layer": g.unit_layer(u), "lambda_product": round(scale(g.unit_layer(u)), 3)}
        report[n] = {"units": units, "exact_set": round(exact, 3), "own_all": round(own_all, 3), "instrument_ratio": round(own_all / exact, 3),
                     "direct_all": round(direct_all, 3), "direct_share": round(direct_all / exact, 3),
                     "first_order_share_v102": v102["sets"][n]["direct_share"], "heads": heads}
        print(n, {k: report[n][k] for k in ("exact_set", "instrument_ratio", "direct_share", "first_order_share_v102")},
              {u[5:]: h["share"] for u, h in heads.items() if h["big"]}, flush=True)

    big = [(n, u, h) for n, r in report.items() for u, h in r["heads"].items() if h["big"]]
    early = [h["share"] for _, _, h in big if h["layer"] <= EARLY]
    instr = {n: r["instrument_ratio"] for n, r in report.items()}
    fo = {n: (r["direct_share"], r["first_order_share_v102"]) for n, r in report.items()}
    readout_sets = [n for n, r in report.items() if any(h["big"] and h["share"] >= READOUT_MIN for h in r["heads"].values())]
    rho = spearman([h["cos"] for _, _, h in big], [h["share"] for _, _, h in big])
    predictions = {
        'pred_a_instrument': all(INSTR_LO <= v <= INSTR_HI for v in instr.values()),
        'pred_b_first_order': sum(abs(a - b) <= FO_TOL for a, b in fo.values()) >= K,
        'pred_c_reset': len(early) >= MIN_EARLY and all(s <= RESET_MAX for s in early),
        'pred_d_readout_head': len(readout_sets) >= K,
        'pred_e_cos_tracks': rho >= RHO_MIN,
    }
    summary = {"instrument_ratio": instr, "direct_share_measured": {n: v[0] for n, v in fo.items()}, "first_order_v102": {n: v[1] for n, v in fo.items()},
               "early_head_shares": [(n, u[5:], h["share"]) for n, u, h in big if h["layer"] <= EARLY], "early_tested": len(early) >= MIN_EARLY,
               "readout_sets": readout_sets, "spearman_cos_share": round(rho, 3), "n_big_heads": len(big),
               "lambda0": [round(v, 3) for v in lam0]}
    result = {"predictions": predictions, "schema": "circuit_unit_head_direct_result_v1", "candidate_id": "corpus.unit_seven_head_direct_v103",
              "summary": summary, "sets": report,
              "bars": {"instr": [INSTR_LO, INSTR_HI], "fo_tol": FO_TOL, "min_single": MIN_SINGLE, "reset_max": RESET_MAX, "readout_min": READOUT_MIN,
                       "rho_min": RHO_MIN, "early_layer": EARLY, "min_early": MIN_EARLY, "k_of_7": K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
