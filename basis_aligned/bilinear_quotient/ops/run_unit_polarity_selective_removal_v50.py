"""v50: terminal-evidence axis for the polarity four-head set -- selective removal, paired document bootstrap, frozen OOD.

The mechanistic chain (v22-v49) says what the set does; the rubric's terminal axis asks whether REMOVING the set's
direction damages the behaviour, damages only it, and does so on a frozen construction it never saw. Intervention:
mean-ablation along the block diff-in-means subspace (rank 1 per (layer, kind) block over
['attn:07:head:08','attn:08:head:01','attn:04:head:07','attn:03:head:00'], no fitting): at each block
live <- live + q q^T (mu - live), mu = the unit's mean output over the A1 base+donor rows (the deletion background).
Applied at the final (prediction) position of each sentence; the caveat that this leaves earlier positions untouched
is stated -- collateral is therefore measured on matched-negative SENTENCES at the same position, not other positions.
    target     A1 base (never -> anything) and A1 donor (often -> something) sentences, 64 documents
    negative   C family (no polarity item; '... in the middle of the' -> night), 64 documents
    control    random orthonormal subspace of the same block ranks, same background, same rows
    OOD        A2 frame ('In the report the X had never/often admitted'), frozen, evaluated once;
               plus a lexical variant of A1 (noticed -> mentioned) if it tokenises (else reported skipped)
Damage per document = CE(correct answer | removed) - CE(correct answer | native) at the prediction position.
Bootstrap: 2000 paired resamples over documents, seed 0; "simultaneous 95%" = 97.5% one-sided bounds (Bonferroni, 2 tests).

REGISTERED BEFORE THE RUN (rubric defaults, not tightened)
    pred_a_target_damage   target damage 97.5% lower bound > 0 and point >= 0.05 nat. Worked: mean 0.30, LB 0.20 True; point 0.02 False.
    pred_b_specific        (target - negative) damage 97.5% lower bound > 0. Worked: 0.30 vs 0.01 True; 0.05 vs 0.04 with LB < 0 False.
    pred_c_offtarget       negative damage 97.5% upper bound <= 0.01 nat AND <= 0.10 x target point. Worked: UB 0.006 vs 0.03 True; UB 0.02 False.
    pred_d_random_inert    random-subspace target damage point <= 0.25 x diff-in-means target damage point. Worked: 0.02 / 0.30 True; 0.10 / 0.30 False.
    pred_e_ood_a2          A2 damage 97.5% lower bound > 0 and point >= 0.50 x A1 point. Worked: 0.20 / 0.30 True; 0.10 / 0.30 False.
    Reading rule. a-c True: terminal rows 3 and 4 of the rubric are met for this set at the prediction position; e True adds row 5
    (frozen OOD). c False with a True: the heads' outputs along the direction are load-bearing on non-polarity sentences too --
    report it as collateral, do not re-choose the background or the rank.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_pattern_freeze_v35 as v35

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_polarity_selective_removal_v50_result.json"
NAME = "polarity_licensing"
DAMAGE_MIN, OFF_MAX, OFF_FRAC, RAND_FRAC, OOD_FRAC, N_BOOT, SEED = 0.05, 0.01, 0.10, 0.25, 0.50, 2000, 0
VARIANT = {"noticed": "mentioned"}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40, 1400


def _plan():
    return {"candidate_id": "corpus.unit_polarity_selective_removal_v50", "units": v35.SETS[NAME][1],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _boot(torch, x, n=N_BOOT, seed=SEED):
    """Paired document bootstrap of a per-document statistic (already differenced); returns (point, lb97.5, ub97.5)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(x), (n, len(x)), generator=gen)
    means = x[idx].mean(1)
    return float(x.mean()), float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    F = torch.nn.functional
    t0 = time.perf_counter()
    module, units = v35.SETS[NAME]
    units = list(units)
    a1 = g.prepare(backend, g.rows_of(module, "A1"))
    q_dim = g.block_diff_in_means(backend, a1, units)
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
    # deletion background: per-unit mean over the A1 base and donor sentences
    mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (a1.base_cache, a1.donor_cache) for rid in a1.base_batch.row_ids]).mean(0)
          for u in units}

    def eval_side(prep, side, q):
        """Per-document CE damage, margin damage, KL(native || removed), top-1 change for one side's sentences."""
        batch = prep.base_batch if side == "base" else prep.donor_batch
        cache = prep.base_cache if side == "base" else prep.donor_cache
        rids = list(batch.row_ids)
        bg = dict(cache)
        for rid in rids:
            for u in units:
                bg[(rid, u)] = mu[u]
        ans = torch.tensor(batch.answer_ids, device=backend.device)
        _, nat = g.forward_units(backend, batch, units=[], return_logits=True)
        af, rem = g.forward_units(backend, batch, units=units, donor_cache=bg, base_cache=cache, q=q, return_logits=True)
        lp_n, lp_r = F.log_softmax(nat.float(), -1), F.log_softmax(rem.float(), -1)
        i = torch.arange(len(rids), device=backend.device)
        ce = (-(lp_r[i, ans]) + lp_n[i, ans]).tolist()
        foil = torch.tensor(batch.foil_ids, device=backend.device)
        margin = ((lp_n[i, ans] - lp_n[i, foil]) - (lp_r[i, ans] - lp_r[i, foil])).tolist()
        kl = (lp_n.exp() * (lp_n - lp_r)).sum(-1).tolist()
        top1 = (nat.argmax(-1) != rem.argmax(-1)).float().tolist()
        return {"ce": ce, "margin": margin, "kl": kl, "top1_change": top1}

    def both(prep, q):
        out = {"ce": [], "margin": [], "kl": [], "top1_change": []}
        for side in ("base", "donor"):
            r = eval_side(prep, side, q)
            for k in out:
                out[k] += r[k]
        return out

    def summary(d):
        p, lb, ub = _boot(torch, d["ce"])
        pm, _, _ = _boot(torch, d["margin"])
        return {"ce_damage": p, "ce_lb975": lb, "ce_ub975": ub, "margin_damage": pm, "kl_mean": sum(d["kl"]) / len(d["kl"]),
                "top1_change_rate": sum(d["top1_change"]) / len(d["top1_change"]), "documents": len(d["ce"])}

    c = g.prepare(backend, g.rows_of(module, "C"))
    a2 = g.prepare(backend, g.rows_of(module, "A2"))
    tgt, neg, rnd, ood = both(a1, q_dim), both(c, q_dim), both(a1, q_rand), both(a2, q_dim)
    report = {"target_A1": summary(tgt), "negative_C": summary(neg), "random_subspace_A1": summary(rnd), "ood_A2": summary(ood)}
    n = min(len(tgt["ce"]), len(neg["ce"]))
    spec = [t - m for t, m in zip(tgt["ce"][:n], neg["ce"][:n])]  # rows are matched by index (same subject list); paired resample
    report["specificity_target_minus_negative"] = dict(zip(("point", "lb975", "ub975"), _boot(torch, spec)))
    try:
        var_rows = g.lexical_variant(g.rows_of(module, "A1"), VARIANT)
        var = both(g.prepare(backend, var_rows), q_dim)
        report["ood_variant"] = summary(var); report["ood_variant"]["mapping"] = VARIANT
    except AssertionError as e:
        report["ood_variant"] = {"skipped": str(e)[:200], "mapping": VARIANT}
    tp = report["target_A1"]["ce_damage"]
    predictions = {
        'pred_a_target_damage': report["target_A1"]["ce_lb975"] > 0 and tp >= DAMAGE_MIN,
        'pred_b_specific': report["specificity_target_minus_negative"]["lb975"] > 0,
        'pred_c_offtarget': report["negative_C"]["ce_ub975"] <= OFF_MAX and report["negative_C"]["ce_ub975"] <= OFF_FRAC * tp,
        'pred_d_random_inert': report["random_subspace_A1"]["ce_damage"] <= RAND_FRAC * tp,
        'pred_e_ood_a2': report["ood_A2"]["ce_lb975"] > 0 and report["ood_A2"]["ce_damage"] >= OOD_FRAC * tp,
    }
    for k, v in report.items():
        print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}), flush=True)
    result = {"predictions": predictions, "schema": "circuit_unit_selective_removal_result_v1", "candidate_id": "corpus.unit_polarity_selective_removal_v50",
              "units": units, "intervention": "mean_ablation_block_diff_in_means_rank1_per_block_final_position",
              "bars": {"damage_min": DAMAGE_MIN, "off_max": OFF_MAX, "off_frac": OFF_FRAC, "rand_frac": RAND_FRAC, "ood_frac": OOD_FRAC, "n_boot": N_BOOT, "seed": SEED},
              "caveat": "intervention at the prediction position only; collateral measured on matched-negative sentences at that position",
              "report": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
