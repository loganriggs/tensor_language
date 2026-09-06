"""v52: rubric row 2 (EXTRACTION) for the four Tier-2 head sets -- keep only the set, mean-ablate every other head.

Selective removal (v50/v51) asks whether deleting the set's direction breaks the behaviour; extraction asks whether the set
ALONE carries it. Intervention at the prediction position: all 162 attention heads are mean-ablated to the deletion
background (per-head mean over the set's own A1 base+donor sentences) EXCEPT the kept part of the set:
    null        every head -> background                                (registered null background)
    keep_exact  the set's heads live, all others -> background
    keep_dim    the set's block diff-in-means direction kept (rank 1 per block), its complement and all others -> background
    keep_rand   a random direction of the same block ranks kept instead   (direction control)
    keep_set_r  a random head set of the same size kept exact             (matched-set control, seed 1, disjoint from the set)
    interchange the set's exact interchange (v23 recovery), same rows, for comparison
Recovery = sum_docs(m_arm - m_null) / sum_docs(m_native - m_null) over both sides' sentences (m = answer-foil log-odds on
the sentence's own correct axis); paired document bootstrap of this ratio-of-means, 2000 x, seed 0, 95% bounds.
The information distinguishing base from donor (never/often, each/all, sent/reserved, was/then) sits UPSTREAM of the final
token, so with every head ablated nothing but attention can carry it -- the null should be genuinely null (pred_e checks).
Implementation: one block-live q per layer over all nine heads (the set's heads first in the concatenation), complement=True
with donor = background: live + (I - q q^T)(mu - live); an empty q (rank 0) is full ablation, an identity block keeps it exact.

REGISTERED BEFORE THE RUN
    pred_a_rubric_row2          every set: keep_exact recovery point >= 0.80 and 95% LB >= 0.60 (campaign default bar).
                                Stated prior: FALSE for polarity/quantifier -- their interchange recovery is 0.585/0.635; registered
                                anyway because it is the rubric's bar, not mine. Worked: 0.85/0.70 True; 0.58 False.
    pred_b_tracks_interchange   every set: |keep_exact - interchange recovery| <= 0.15. Worked: 0.55 vs 0.585 True; 0.30 vs 0.585 False.
    pred_c_direction_suffices   every set: keep_dim >= 0.80 x keep_exact. Worked: 0.50/0.58 True; 0.30/0.58 False.
    pred_d_controls_fail        every set: keep_rand <= 0.25 x keep_exact AND keep_set_r <= 0.25 x keep_exact. Worked: 0.05/0.58 True.
    pred_e_null_is_null         every set: mean |m_null| <= 0.25 x mean |m_native|. Worked: 0.1/1.0 True; 0.4/1.0 False.
    Reading rule. b True: the set's contribution is separable from the rest of attention (swap-in and keep-alone agree); b False
    with keep_exact > interchange: other heads were suppressing the set in the native run -- report, do not enlarge the set.
    A2 arms (null / keep_exact / keep_dim) are frozen OOD, reported, no prediction (one-shot).
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_extraction_four_sets_v52_result.json"
SETS = v23.SETS
N_LAYERS, N_HEADS, HEAD_DIM = 18, 9, 128
REC_MIN, LB_MIN, TRACK_TOL, DIR_FRAC, CTRL_FRAC, NULL_FRAC, N_BOOT, SEED = 0.80, 0.60, 0.15, 0.80, 0.25, 0.25, 2000, 0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 3600


def _plan():
    return {"candidate_id": "corpus.unit_extraction_four_sets_v52", "sets": {k: v[1] for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _boot_ratio(torch, num, den, n=N_BOOT, seed=SEED):
    num, den = torch.as_tensor(num, dtype=torch.float64), torch.as_tensor(den, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(num), (n, len(num)), generator=gen)
    r = num[idx].sum(1) / den[idx].sum(1)
    return {"point": float(num.sum() / den.sum()), "lb95": float(torch.quantile(r, 0.025)), "ub95": float(torch.quantile(r, 0.975))}


def ordered_units(set_units):
    """All 162 heads, each layer's set heads first (the concatenation order the block q assumes)."""
    by_layer = g.blocks_of(set_units)
    out = []
    for l in range(N_LAYERS):
        first = by_layer.get((l, "heads"), [])
        out += first + [f"attn:{l:02d}:head:{h:02d}" for h in range(N_HEADS) if f"attn:{l:02d}:head:{h:02d}" not in first]
    return out


def block_q(torch, device, set_units, q_set):
    """{(layer,'heads'): (1152, r)} -- q_set's block placed on the set heads' leading dims; rank 0 elsewhere."""
    by_layer = g.blocks_of(set_units)
    out = {}
    for l in range(N_LAYERS):
        key = (l, "heads")
        if key in by_layer and q_set is not None:
            qs = q_set[key]
            qb = torch.zeros(N_HEADS * HEAD_DIM, qs.shape[1], device=device)
            qb[:qs.shape[0]] = qs
            out[key] = qb
        else:
            out[key] = torch.zeros(N_HEADS * HEAD_DIM, 0, device=device)
    return out


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
        a1, a2 = g.prepare(backend, g.rows_of(m, "A1")), g.prepare(backend, g.rows_of(m, "A2"))
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (a1.base_cache, a1.donor_cache) for rid in a1.base_batch.row_ids]).mean(0)
              for u in all_heads}
        q_dim = g.block_diff_in_means(backend, a1, units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        q_id = g.block_identity(backend, units)
        gen = torch.Generator().manual_seed(1)
        pool = [u for u in all_heads if u not in units]
        rand_set = [pool[i] for i in torch.randperm(len(pool), generator=gen)[:len(units)].tolist()]
        arms = {"null": (units, None), "keep_exact": (units, q_id), "keep_dim": (units, q_dim), "keep_rand": (units, q_rand),
                "keep_set_r": (rand_set, g.block_identity(backend, rand_set))}

        def margins(prep, set_units, qs):
            """Per-document (both sides) answer-foil log-odds on the sentence's own correct axis: native, arm."""
            order = ordered_units(set_units)
            bq = block_q(torch, backend.device, set_units, qs)
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
                nat += (af_n[:, 0] - af_n[:, 1]).tolist()
                arm += (af_a[:, 0] - af_a[:, 1]).tolist()
            return nat, arm

        def family(prep, names):
            nat, m_null = margins(prep, units, None)
            den = [a - b for a, b in zip(nat, m_null)]
            out = {"null_abs_over_native_abs": sum(abs(x) for x in m_null) / sum(abs(x) for x in nat), "documents": len(nat)}
            for a in names:
                su, qs = arms[a]
                _, m_arm = margins(prep, su, qs)
                out[a] = _boot_ratio(torch, [x - y for x, y in zip(m_arm, m_null)], den)
            return out
        r = {"units": units, "random_set": rand_set, "A1": family(a1, ("keep_exact", "keep_dim", "keep_rand", "keep_set_r")),
             "A2": family(a2, ("keep_exact", "keep_dim"))}
        r["interchange_recovery_A1"] = g.recovery(a1, g.patched_axis(backend, a1, units))
        report[n] = r
        print(n, "null|.| %.2f  exact %.3f [%.3f,%.3f]  dim %.3f  rand %.3f  set_r %.3f  interchange %.3f | A2 exact %.3f dim %.3f" % (
            r["A1"]["null_abs_over_native_abs"], r["A1"]["keep_exact"]["point"], r["A1"]["keep_exact"]["lb95"], r["A1"]["keep_exact"]["ub95"],
            r["A1"]["keep_dim"]["point"], r["A1"]["keep_rand"]["point"], r["A1"]["keep_set_r"]["point"], r["interchange_recovery_A1"],
            r["A2"]["keep_exact"]["point"], r["A2"]["keep_dim"]["point"]), flush=True)
    ex = {n: report[n]["A1"]["keep_exact"]["point"] for n in SETS}
    predictions = {
        'pred_a_rubric_row2': all(ex[n] >= REC_MIN and report[n]["A1"]["keep_exact"]["lb95"] >= LB_MIN for n in SETS),
        'pred_b_tracks_interchange': all(abs(ex[n] - report[n]["interchange_recovery_A1"]) <= TRACK_TOL for n in SETS),
        'pred_c_direction_suffices': all(report[n]["A1"]["keep_dim"]["point"] >= DIR_FRAC * ex[n] for n in SETS),
        'pred_d_controls_fail': all(report[n]["A1"]["keep_rand"]["point"] <= CTRL_FRAC * ex[n] and report[n]["A1"]["keep_set_r"]["point"] <= CTRL_FRAC * ex[n] for n in SETS),
        'pred_e_null_is_null': all(report[n]["A1"]["null_abs_over_native_abs"] <= NULL_FRAC for n in SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_extraction_result_v1", "candidate_id": "corpus.unit_extraction_four_sets_v52",
              "intervention": "all_heads_mean_ablated_at_prediction_position_except_kept_part_of_set",
              "bars": {"rec_min": REC_MIN, "lb_min": LB_MIN, "track_tol": TRACK_TOL, "dir_frac": DIR_FRAC, "ctrl_frac": CTRL_FRAC, "null_frac": NULL_FRAC, "n_boot": N_BOOT, "seed": SEED},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
