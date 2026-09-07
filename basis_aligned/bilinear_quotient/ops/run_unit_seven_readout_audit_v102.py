#!/usr/bin/env python3
# BQGATE: five frozen predictions; the seven head sets, the recipe and every bar are fixed before the run.
"""v102: are the seven rank-1 head directions READOUT directions (the answer logit pulled back through c_proj) or features?

v101 found the modal_remoteness direction is the modal-slot VALUE axis (would vs will) -- every sibling with a modal slot is
damaged. That is what a direction would look like if the heads simply wrote the answer margin into the residual and the
direction were the unembedding difference pulled back through the head's c_proj columns. This audit asks that for all
seven behaviours' sets (six v80 hub+8 sets, modal v97) with two instruments that need no new hypothesis-specific code:

  cosine    per head h at layer L: d_h = W_L[:, h-slice]^T (U[ans] - U[foil]) (c_proj columns of the head, unembedding rows
            of the two answer tokens); |cos(q_h, d_h)| for the fitted rank-1 direction's slice q_h, averaged over heads
            weighted by each head's share of the direction's norm. Random baseline: the same for a random rank-1 direction
            (expected ~0.07 in 128-d).
  direct share   first-order direct-path attribution of the EXACT-set patch on ODD A1 rows: for each row the head deltas
            (donor - base at the prediction position) are pushed through c_proj, scaled by the product of later residual
            lambdas[0], dotted with (U[foil] - U[ans]) of the base row (the patched axis), divided by the final residual's
            rms and multiplied by the soft-cap derivative at the base logits; summed over heads, over rows, divided by the
            actual exact-set axis change. ~1 = the heads write the margin directly; ~0 = the effect is routed through later
            blocks (features).
Direction recipe: v99/v80 full-specificity (rank 1 per block, pooled EVEN A1, own C EVEN + the other six A1 EVEN as inertness
controls at 30 each, complement 1.0, 120 steps, lr 0.05, seed 0, mu = pooled EVEN mean). Plain A1 pools for all seven
(no verb-variant maps), so v80 numbers are NOT claimed as reproduced; extraction on ODD is the instrument gate instead.

REGISTERED BEFORE THE RUN
    pred_a_readout_directions   >= 4 of 7 behaviours have norm-weighted mean |cos(q_h, d_h)| >= 0.50. Worked: cos 0.7,0.6,0.55,0.5 on four -> True; 0.3 everywhere -> False.
    pred_b_feature_directions   >= 4 of 7 have norm-weighted mean |cos| <= 0.20. Exclusive with a.
    pred_c_direct_share_high    >= 4 of 7 have direct share in [0.50, 1.50]. Worked: 0.8 on four -> True; 0.2 -> False.
    pred_d_direct_share_low     >= 4 of 7 have direct share <= 0.25. Exclusive with c.
    pred_e_instrument           every behaviour: ODD extraction >= 0.80 AND random-direction |cos| <= 0.15. Worked: 0.9/0.07 True; 0.7/0.07 False.
    Prior: a ~35%; b ~40%; c ~35%; d ~45%; e ~85%. Sets sit at layers 2-16 of 18; I expect a mix, with hub heads indirect.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_seven_readout_audit_v102_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
COS_HI, COS_LO, SHARE_LO, SHARE_HI, SHARE_FEAT, EXT_MIN, RAND_COS, K = 0.50, 0.20, 0.50, 1.50, 0.25, 0.80, 0.15, 4
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 120000


def _plan():
    return {"candidate_id": "corpus.unit_seven_readout_audit_v102", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 7 * 2 * STEPS, "model_updates": 0, "fit_parameters": 7 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def head_of(u):
    return int(u.rsplit(":", 1)[1])


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    F = torch.nn.functional
    model = backend.model
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")},
               "modal_remoteness": m_modal}
    sets = {n: s["units"] for n, s in json.loads(V80.read_text())["sets"].items()}
    sets["modal_remoteness"] = json.loads(V97.read_text())["final"]
    assert set(sets) == set(modules), (set(sets), set(modules))
    for us in sets.values():
        assert all(":head:" in u for u in us), us
    even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}
    odd = {n: g.prepare(backend, g.rows_of(m, "A1")[1::2]) for n, m in modules.items()}
    even_c = {n: g.prepare(backend, g.rows_of(m, "C")[0::2]) for n, m in modules.items()}
    U = model.lm_head.weight.detach().float()                                   # (vocab, 1152)
    lam0 = [float(b.lambdas[0]) for b in model.transformer.h]
    W = [b.attn.c_proj.weight.detach().float() for b in model.transformer.h]    # (1152 out, 1152 in)

    report = {}
    for n, units in sets.items():
        pool = even[n]
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache)
                              for rid in pool.base_batch.row_ids]).mean(0) for u in units}
        controls = (even_c[n],) + tuple(p for k, p in even.items() if k != n)
        q, hist = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0,
                                                   complement_weight=CW, controls=controls, control_weight=LAM * len(controls), mu=mu)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        O = odd[n]
        ex = g.patched_axis(backend, O, units)
        sub = g.patched_axis(backend, O, units, q=q)
        rec_ex = g.recovery(O, ex); rec_sub = g.recovery(O, sub)
        extraction = rec_sub / max(rec_ex, 1e-6)

        # answer-logit direction per head, in the head's 128-d output space
        r0 = O.rows[0]
        u_pair = U[r0["base_answer_id"]] - U[r0["base_foil_id"]]
        cos, cos_rand, shares = {}, {}, g.norm_shares(q, units)
        blocks = g.blocks_of(units)
        for key, us in blocks.items():
            L = key[0]; off = 0
            for u in us:
                h = head_of(u); sl = slice(h * g.HEAD_DIM, (h + 1) * g.HEAD_DIM)
                d = W[L][:, sl].T @ u_pair.to(W[L].device)
                qh = q[key][off:off + g.HEAD_DIM, 0].float(); qr = q_rand[key][off:off + g.HEAD_DIM, 0].float()
                cos[u] = float(F.cosine_similarity(qh, d, dim=0).abs())
                cos_rand[u] = float(F.cosine_similarity(qr, d, dim=0).abs())
                off += g.HEAD_DIM
        tot = sum(shares.values())
        wcos = sum(shares[u] * cos[u] for u in units) / tot
        wcos_rand = sum(shares[u] * cos_rand[u] for u in units) / tot

        # direct-path share of the exact-set patch on ODD A1 (base rows patched toward the donor)
        resid = {}
        _, logits = g.forward_units(backend, O.base_batch, units=[], donor_cache=O.base_cache, base_cache=O.base_cache,
                                    capture_resid=resid, return_logits=True)
        direct, actual = [], []
        for i, (rid, row) in enumerate(zip(O.base_batch.row_ids, O.rows)):
            axis = (U[row["base_foil_id"]] - U[row["base_answer_id"]]).to(backend.device)   # patched axis = -(ans - foil)
            xfin = resid[(rid, g.N_LAYERS - 1)].to(backend.device)
            rms = float(xfin.pow(2).mean().sqrt())
            zc = logits[i].float()
            dcap = 1.0 - (zc[row["base_foil_id"]] / 30.0) ** 2, 1.0 - (zc[row["base_answer_id"]] / 30.0) ** 2
            contrib = 0.0
            for u in units:
                L = g.unit_layer(u); h = head_of(u); sl = slice(h * g.HEAD_DIM, (h + 1) * g.HEAD_DIM)
                dv = (torch.as_tensor(O.donor_cache[(rid, u)]).float() - torch.as_tensor(O.base_cache[(rid, u)]).float()).to(backend.device)
                scale = 1.0
                for l in range(L + 1, g.N_LAYERS):
                    scale *= lam0[l]
                write = scale * (W[L][:, sl] @ dv) / rms
                contrib += float(dcap[0] * (U[row["base_foil_id"]].to(backend.device) @ write) - dcap[1] * (U[row["base_answer_id"]].to(backend.device) @ write))
            direct.append(contrib)
            actual.append(ex[i] - O.base_axis[i])
        share = sum(direct) / max(sum(actual), 1e-6)
        report[n] = {"units": units, "extraction_odd": round(extraction, 3), "weighted_cos": round(wcos, 3), "weighted_cos_random": round(wcos_rand, 3),
                     "cos_per_head": {u: round(v, 3) for u, v in cos.items()}, "norm_shares": {u: round(v / tot, 3) for u, v in shares.items()},
                     "direct_share": round(share, 3), "direct_sum": round(sum(direct), 3), "actual_sum": round(sum(actual), 3),
                     "direct_per_row": [round(v, 3) for v in direct], "actual_per_row": [round(v, 3) for v in actual],
                     "lambda_products": {str(L): round(float(torch.tensor(lam0[L + 1:]).prod()) if L + 1 < g.N_LAYERS else 1.0, 3) for L in sorted({g.unit_layer(u) for u in units})},
                     "final_match": hist[-1] if hist else None}
        print(n, {k: report[n][k] for k in ("extraction_odd", "weighted_cos", "weighted_cos_random", "direct_share")}, flush=True)

    wc = {n: r["weighted_cos"] for n, r in report.items()}
    sh = {n: r["direct_share"] for n, r in report.items()}
    predictions = {
        'pred_a_readout_directions': sum(v >= COS_HI for v in wc.values()) >= K,
        'pred_b_feature_directions': sum(v <= COS_LO for v in wc.values()) >= K,
        'pred_c_direct_share_high': sum(SHARE_LO <= v <= SHARE_HI for v in sh.values()) >= K,
        'pred_d_direct_share_low': sum(v <= SHARE_FEAT for v in sh.values()) >= K,
        'pred_e_instrument': all(r["extraction_odd"] >= EXT_MIN and r["weighted_cos_random"] <= RAND_COS for r in report.values()),
    }
    summary = {"weighted_cos": wc, "direct_share": sh, "extraction_odd": {n: r["extraction_odd"] for n, r in report.items()},
               "cos_random": {n: r["weighted_cos_random"] for n, r in report.items()}, "lambda0": [round(v, 3) for v in lam0]}
    result = {"predictions": predictions, "schema": "circuit_unit_readout_audit_result_v1", "candidate_id": "corpus.unit_seven_readout_audit_v102",
              "summary": summary, "sets": report,
              "bars": {"cos_hi": COS_HI, "cos_lo": COS_LO, "share_lo": SHARE_LO, "share_hi": SHARE_HI, "share_feat": SHARE_FEAT, "ext_min": EXT_MIN, "rand_cos": RAND_COS, "k_of_7": K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
