"""v51: v50's terminal axis for all four Tier-2 head sets, plus CROSS-BEHAVIOUR collateral.

Same intervention as v50 (mean-ablation along the block diff-in-means subspace, rank 1 per block, no fitting, background =
per-unit mean over the set's own A1 base+donor sentences, applied at the prediction position). New negative: each set's
direction is removed on the OTHER three behaviours' A1 sentences. The sets share hub heads (07:08 is in all four; 11:03 in
quantifier and dative, whose two heads are a subset of dative's five), so a shared direction would show up here -- the
hub-head multiplexing finding predicts near-orthogonal per-behaviour directions and therefore inert cross removal.
    sets       quantifier_number (2 heads), dative (5), polarity_licensing (4), voice_frame (4)   -- v23 SETS
    target     own A1 base+donor sentences (64 documents); negative C (64); cross: other sets' A1 (64 each); OOD A2 (64)
Damage = CE(correct answer | removed) - CE(native); paired document bootstrap 2000 x, seed 0, 97.5% bounds.

REGISTERED BEFORE THE RUN
    pred_a_damage_all      every set: target damage LB > 0 and point >= 0.05 nat. Worked: polarity 0.17 (LB 0.10) True; 0.03 False.
    pred_b_specific_all    every set: (target - C) LB > 0.
    pred_c_cross_inert     every ordered pair s != t: damage of s's direction on t's sentences <= 0.25 x s's own target damage.
                           Worked: 0.02 / 0.17 True; dative-on-quantifier 0.06 / 0.17 False.
    pred_d_ood_a2_all      every set: A2 damage LB > 0 and point >= 0.50 x own A1 point. Worked: polarity 0.29 / 0.17 True; 0.05 / 0.17 False.
    pred_e_random_inert    every set: random-subspace own damage <= 0.25 x own target damage. Worked: 0.003 / 0.17 True.
    Reading rule. c False on a pair sharing a head: the shared head carries a shared direction for those two behaviours -- name the
    pair, do not drop the head from either set. a False on a set: that set's direction is not load-bearing under mean-ablation
    at this rank; record it as a terminal null for the set (its Tier-2 claim stands on interchange, not removal).
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
import run_unit_polarity_selective_removal_v50 as v50

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_selective_removal_four_sets_v51_result.json"
SETS = v23.SETS
DAMAGE_MIN, CROSS_FRAC, OOD_FRAC, RAND_FRAC = 0.05, 0.25, 0.50, 0.25
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 130, 4200


def _plan():
    return {"candidate_id": "corpus.unit_selective_removal_four_sets_v51", "sets": {k: v[1] for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def removal(backend, prep, units, q, mu):
    """Per-document CE damage / margin damage / KL / top-1 change over both sides' sentences."""
    torch = backend.torch
    F = torch.nn.functional
    out = {"ce": [], "margin": [], "kl": [], "top1_change": []}
    for side in ("base", "donor"):
        batch = prep.base_batch if side == "base" else prep.donor_batch
        cache = prep.base_cache if side == "base" else prep.donor_cache
        rids = list(batch.row_ids)
        bg = dict(cache)
        for rid in rids:
            for u in units:
                bg[(rid, u)] = mu[u]
        ans = torch.tensor(batch.answer_ids, device=backend.device)
        foil = torch.tensor(batch.foil_ids, device=backend.device)
        _, nat = g.forward_units(backend, batch, units=[], return_logits=True)
        _, rem = g.forward_units(backend, batch, units=units, donor_cache=bg, base_cache=cache, q=q, return_logits=True)
        lp_n, lp_r = F.log_softmax(nat.float(), -1), F.log_softmax(rem.float(), -1)
        i = torch.arange(len(rids), device=backend.device)
        out["ce"] += (lp_n[i, ans] - lp_r[i, ans]).tolist()
        out["margin"] += ((lp_n[i, ans] - lp_n[i, foil]) - (lp_r[i, ans] - lp_r[i, foil])).tolist()
        out["kl"] += (lp_n.exp() * (lp_n - lp_r)).sum(-1).tolist()
        out["top1_change"] += (nat.argmax(-1) != rem.argmax(-1)).float().tolist()
    return out


def summary(torch, d):
    p, lb, ub = v50._boot(torch, d["ce"])
    return {"ce_damage": p, "ce_lb975": lb, "ce_ub975": ub, "margin_damage": v50._boot(torch, d["margin"])[0],
            "kl_mean": sum(d["kl"]) / len(d["kl"]), "top1_change_rate": sum(d["top1_change"]) / len(d["top1_change"]), "documents": len(d["ce"])}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    preps = {n: {fam: g.prepare(backend, g.rows_of(m, fam)) for fam in ("A1", "A2")} for n, (m, _) in SETS.items()}
    c_prep = g.prepare(backend, g.rows_of(SETS["polarity_licensing"][0], "C"))
    report = {}
    for n, (m, units) in SETS.items():
        units = list(units)
        a1 = preps[n]["A1"]
        q = g.block_diff_in_means(backend, a1, units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (a1.base_cache, a1.donor_cache) for rid in a1.base_batch.row_ids]).mean(0) for u in units}
        tgt = removal(backend, a1, units, q, mu)
        neg = removal(backend, c_prep, units, q, mu)
        r = {"units": units, "target_A1": summary(torch, tgt), "negative_C": summary(torch, neg),
             "random_subspace_A1": summary(torch, removal(backend, a1, units, q_rand, mu)),
             "ood_A2": summary(torch, removal(backend, preps[n]["A2"], units, q, mu)),
             "specificity_target_minus_C": dict(zip(("point", "lb975", "ub975"), v50._boot(torch, [a - b for a, b in zip(tgt["ce"], neg["ce"])]))),
             "cross": {}}
        for t in SETS:
            if t != n:
                r["cross"][t] = summary(torch, removal(backend, preps[t]["A1"], units, q, mu))
        report[n] = r
        print(n, "own %.3f [%.3f] C %.3f rand %.3f A2 %.3f [%.3f] cross %s" % (
            r["target_A1"]["ce_damage"], r["target_A1"]["ce_lb975"], r["negative_C"]["ce_damage"], r["random_subspace_A1"]["ce_damage"],
            r["ood_A2"]["ce_damage"], r["ood_A2"]["ce_lb975"], {t: round(v["ce_damage"], 3) for t, v in r["cross"].items()}), flush=True)
    own = {n: report[n]["target_A1"]["ce_damage"] for n in SETS}
    predictions = {
        'pred_a_damage_all': all(report[n]["target_A1"]["ce_lb975"] > 0 and own[n] >= DAMAGE_MIN for n in SETS),
        'pred_b_specific_all': all(report[n]["specificity_target_minus_C"]["lb975"] > 0 for n in SETS),
        'pred_c_cross_inert': all(v["ce_damage"] <= CROSS_FRAC * own[n] for n in SETS for v in report[n]["cross"].values()),
        'pred_d_ood_a2_all': all(report[n]["ood_A2"]["ce_lb975"] > 0 and report[n]["ood_A2"]["ce_damage"] >= OOD_FRAC * own[n] for n in SETS),
        'pred_e_random_inert': all(report[n]["random_subspace_A1"]["ce_damage"] <= RAND_FRAC * own[n] for n in SETS),
    }
    cross_fail = [(n, t) for n in SETS for t, v in report[n]["cross"].items() if v["ce_damage"] > CROSS_FRAC * own[n]]
    result = {"predictions": predictions, "schema": "circuit_unit_selective_removal_result_v2", "candidate_id": "corpus.unit_selective_removal_four_sets_v51",
              "intervention": "mean_ablation_block_diff_in_means_rank1_per_block_final_position",
              "bars": {"damage_min": DAMAGE_MIN, "cross_frac": CROSS_FRAC, "ood_frac": OOD_FRAC, "rand_frac": RAND_FRAC, "n_boot": v50.N_BOOT, "seed": v50.SEED},
              "cross_pairs_over_bar": cross_fail, "behaviours": report, "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "cross_pairs_over_bar": cross_fail, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
