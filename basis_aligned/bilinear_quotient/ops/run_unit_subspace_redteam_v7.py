#!/usr/bin/env python3
# BQGATE: frozen diagnostics; sets, seeds, ranks, weights and bars fixed before the run.
"""v7: red team of the v4 direction results (user: "Could you red team the DAS results? Like the
concatenation cross layer stuff and the complement optimization not working?").

Four attacks, each with a registered outcome:

  1. CROSS-LAYER SEMANTICS. v2-v6 patched a joint direction q over the CONCATENATED unit space
     with `live + qq^T(donor - base)_cached` at every unit. For a set spanning several layers the
     later block's live value already carries the earlier patch, so adding a CACHED delta there is
     activation ADDITION (steering), not DAS (`x_base + QQ^T(x_donor - x_base)` at the live value).
     Test: the full-rank cached patch versus the exact set (must be equal if the semantics were
     right; it is not). Fix: BLOCK-LIVE mode, one subspace per (layer, kind) block, applied to the
     live value; its full-rank control must equal the exact set to float precision.
  2. LINEARITY. A direction that carries the variable linearly gives subspace + complement =
     exact (S + C = 1). A fitted direction that exploits the bilinear MLP or the softmax gives
     S + C > 1, and then "complement inert" and "subspace matches" are not the same test. Report
     S + C for every direction; it is the sharper trust criterion.
  3. NON-UNIQUENESS / OVERPARAMETERISATION. 16 fit rows against 384-1536 parameters. Refit DAS
     from three seeds and report the pairwise cosines per block and the cosine to diff-in-means:
     margin agreement across seeds with direction disagreement means the margin is a 1-d readout
     that many directions satisfy.
  4. THE COMPLEMENT TERM "NOT WORKING". v4 used rank 1. If the variable needs more than one
     dimension the constrained loss cannot be satisfied at rank 1. Sweep rank {1, 2, 4} per block
     for cdas on both MLP-containing sets.
  Plus: DIRECTION PURITY. The screens alternate direction row by row (ABAB...). v4-v6 fitted on
     even rows (direction-pure) and evaluated on odd rows (the REVERSE direction on fresh
     sentences). Unsigned diff-in-means on MIXED rows cancels; the library now sign-aligns each
     row's delta with row 0's (geometric, not by label: the spec-authored list candidate labels
     DUPLICATE rows with opposite `direction_id`, so labels would cancel exact duplicates).
     Control: unsigned vs signed dim fitted on the mixed first half of A1 and evaluated on the
     mixed second half.

SETS  the seven v4 sets (5 head sets, 2 sets with an MLP output).
FIT   even A1 rows; EVAL odd A1 rows (held-out) and all A2; block mode rank 1 per block unless
      stated; 120 Adam steps, lr 0.05; seeds 0, 1, 2; lambda 1 for cdas; random rank 1 per block,
      seed 1.

REGISTERED BEFORE THE RUN (band [0.50, 1.20], complement bar 0.30, linear band [0.85, 1.15])
    pred_a_block_full_rank_equals_exact     all 7 sets: |block identity - exact| <= 1e-3 on held-out
                                            (control on the new path; must hold or nothing else counts)
    pred_b_cached_cross_layer_bias_bounded  all multi-layer sets: |cached full-rank - exact| / |exact|
                                            <= 0.10 on held-out (the old semantics was wrong in
                                            principle but small in effect)
    pred_c_dim_is_linear                    all 7 sets: block dim S + C on held-out in [0.85, 1.15]
    pred_d_fitted_directions_exploit_nonlin at least one set: block das (seed 0) S + C on held-out
                                            > 1.15
    pred_e_das_directions_not_unique        mean over sets of the mean pairwise |cos| between the
                                            three seeds' block directions < 0.90, while every seed's
                                            held-out fraction is in band on >= 5 sets (margin agrees,
                                            direction does not)
    pred_f_cdas_rank_repairs_mlp_sets       both MLP sets: at some rank in {1, 2, 4} cdas has
                                            held-out and A2 in band AND complement <= 0.30
    pred_g_signed_dim_survives_mixed_rows   all 7 sets: signed dim on mixed rows in band on the
                                            mixed second half, and |unsigned| < 0.5 * |signed|

    Priors. a: must hold (verified on 2 sets by hand). b: expected (2% seen on 2 sets). c: expected
    (v4 joint dim gave 1.00-1.18; the MLP sets are the risk). d: expected (v4 das sums 1.5-4.8).
    e: expected. f: UNSURE -- this is the user's question; a rank-2/4 fix would mean "not working"
    was "rank too low". g: expected.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_control_choice as m_list
import circuit_fast_screen_candidate_correlative_pair as m_corr
import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_candidate_possessive_adjacent as m_poss
import circuit_fast_screen_candidate_polarity_state as m_pol

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_subspace_redteam_v7_result.json"
STEPS, LR, LAMBDA, SEEDS, CDAS_RANKS = 120, 0.05, 1.0, (0, 1, 2), (1, 2, 4)
LO, HI, COMP_BAR, LIN_LO, LIN_HI, BIAS_BAR, COS_BAR = 0.50, 1.20, 0.30, 0.85, 1.15, 0.10, 0.90
SETS = {
    "correlative_pair.both_vs_neither": (m_corr, "heads",
        ["attn:08:head:01", "attn:07:head:08", "attn:14:head:08"]),
    "modal_remoteness.would_vs_will": (m_modal, "heads", ["attn:09:head:04", "attn:11:head:03"]),
    "numbered_list.control_choice_discriminator": (m_list, "heads",
        ["attn:08:head:03", "attn:08:head:07"]),
    "polarity_state.negative_vs_positive.heads": (m_pol, "heads",
        ["attn:07:head:08", "attn:08:head:01", "attn:04:head:07", "attn:05:head:08"]),
    "possessive_number.adjacent_antecedent.heads": (m_poss, "heads",
        ["attn:04:head:05", "attn:03:head:04", "attn:09:head:06", "attn:10:head:05"]),
    "polarity_state.negative_vs_positive.with_mlp04": (m_pol, "mlp",
        ["attn:07:head:08", "attn:08:head:01", "mlp:04", "attn:10:head:05"]),
    "possessive_number.adjacent_antecedent.with_mlp08": (m_poss, "mlp",
        ["attn:04:head:05", "mlp:08", "attn:09:head:06", "attn:10:head:05"]),
}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 1500000


def _plan():
    fits = len(SETS) * (len(SEEDS) + 2) + 2 * (len(CDAS_RANKS) - 1)
    return {"candidate_id": "corpus.unit_subspace_redteam_v7",
            "sets": {k: v[2] for k, v in SETS.items()}, "seeds": SEEDS, "cdas_ranks": CDAS_RANKS,
            "das_steps": STEPS, "lambda": LAMBDA,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * STEPS * fits, "model_updates": 0,
            "fit_parameters": 1536 * 4, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _frac(e, v):
    return (v / e) if abs(e) > 1e-6 else None


def _battery(backend, prep, units, q):
    """subspace, complement, and their SUM as fractions of the exact-set effect on this prep."""
    b = g.direction_battery(backend, prep, units, q)
    s, c = b["subspace_fraction"], b["complement_fraction"]
    b["linearity_sum"] = None if s is None or c is None else s + c
    return b


def _direction(backend, module, units, q, preps, scale, with_pc=True):
    out = {"held": _battery(backend, preps["held"], units, q),
           "a2": _battery(backend, preps["a2"], units, q)}
    if with_pc:
        pc = g.pc_effects(backend, module, units, scale, q=q)
        out["p_effect"], out["c_effect"] = pc["P"], pc["C"]
    return out


def _brief(d):
    h, a = d["held"], d["a2"]
    r = lambda x: None if x is None else round(x, 3)
    return {"held": r(h["subspace_fraction"]), "comp": r(h["complement_fraction"]),
            "sum": r(h["linearity_sum"]), "a2": r(a["subspace_fraction"]),
            "P": r(d.get("p_effect")), "C": r(d.get("c_effect"))}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    instrument = g.verify_against_producer(backend, g.rows_of(m_pol, "A1"), layer=7,
                                           heads=(8,), mlp_layer=4)
    if not instrument["passed"]:
        raise SystemExit("new forward does not reproduce the producer")

    report = {}
    for label, (module, kind, units) in SETS.items():
        ts = time.perf_counter()
        a1 = g.rows_of(module, "A1")
        preps = {"fit": g.prepare(backend, a1[0::2]), "held": g.prepare(backend, a1[1::2]),
                 "a2": g.prepare(backend, g.rows_of(module, "A2"))}
        exact = {k: g.recovery(p, g.patched_axis(backend, p, units)) for k, p in preps.items()}
        scale = g.target_scale(preps["fit"])
        n_blocks = len(g.blocks_of(units))
        dim_total = sum(g.unit_dim(u) for u in units)

        # 1. semantics: cached-joint full rank vs block-live full rank vs exact, on held-out
        held = preps["held"]
        eye = torch.eye(dim_total, device=backend.device)
        cached_full = g.recovery(held, g.patched_axis(backend, held, units, q=eye))
        block_full = g.recovery(held, g.patched_axis(backend, held, units, q=g.block_identity(backend, units)))
        semantics = {"exact": exact["held"], "cached_joint_full_rank": cached_full,
                     "block_live_full_rank": block_full,
                     "cached_bias_fraction": _frac(exact["held"], cached_full - exact["held"]),
                     "block_error": abs(block_full - exact["held"]),
                     "layers": sorted({g.block_key(u)[0] for u in units}), "blocks": n_blocks}

        # 2./3. block-live directions
        q_dim = g.block_diff_in_means(backend, preps["fit"], units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        das = {}
        for seed in SEEDS:
            q, hist = g.fit_block_subspace(backend, preps["fit"], units, rank=1, steps=STEPS,
                                           lr=LR, seed=seed)
            das[seed] = (q, hist)
        q_cdas, h_cdas = g.fit_block_subspace(backend, preps["fit"], units, rank=1, steps=STEPS,
                                              lr=LR, seed=0, complement_weight=LAMBDA)
        directions = {"dim": _direction(backend, module, units, q_dim, preps, scale),
                      "rand": _direction(backend, module, units, q_rand, preps, scale, with_pc=False),
                      "cdas": _direction(backend, module, units, q_cdas, preps, scale)}
        directions["cdas"]["loss_history"] = h_cdas
        for seed, (q, hist) in das.items():
            d = _direction(backend, module, units, q, preps, scale, with_pc=(seed == 0))
            d["loss_history"] = hist
            d["cosine_to_dim"] = g.block_cosines(q, q_dim)
            directions[f"das_seed{seed}"] = d
        pair_cos = {}
        for i in SEEDS:
            for j in SEEDS:
                if i < j:
                    pair_cos[f"{i}-{j}"] = g.block_cosines(das[i][0], das[j][0])
        mean_pair_cos = sum(v for c in pair_cos.values() for v in c.values()) / max(
            1, sum(len(c) for c in pair_cos.values()))
        directions["cdas"]["cosine_to_dim"] = g.block_cosines(q_cdas, q_dim)

        # old semantics, side by side: cached-joint rank-1 dim and das with the sum diagnostic
        qj_dim = g.diff_in_means_direction(backend, preps["fit"], units)
        qj_das, hj = g.fit_joint_subspace(backend, preps["fit"], units, rank=1, steps=STEPS, lr=LR)
        cached_joint = {"dim": _direction(backend, module, units, qj_dim, preps, scale, with_pc=False),
                        "das": _direction(backend, module, units, qj_das, preps, scale, with_pc=False),
                        "dim_norm_shares": g.norm_shares(qj_dim, units),
                        "das_norm_shares": g.norm_shares(qj_das, units)}

        # 4. cdas rank sweep on MLP sets
        rank_sweep = {}
        if kind == "mlp":
            rank_sweep["1"] = _brief(directions["cdas"])
            for rank in CDAS_RANKS[1:]:
                qr, hr = g.fit_block_subspace(backend, preps["fit"], units, rank=rank, steps=STEPS,
                                              lr=LR, seed=0, complement_weight=LAMBDA)
                d = _direction(backend, module, units, qr, preps, scale)
                d["loss_history"] = hr
                d["rank"] = rank
                directions[f"cdas_rank{rank}"] = d
                rank_sweep[str(rank)] = _brief(d)

        # direction purity control: mixed halves of A1
        half = len(a1) // 2
        mixed_fit, mixed_eval = g.prepare(backend, a1[:half]), g.prepare(backend, a1[half:])
        dirs_fit = sorted({r.get("direction_id") for r in a1[:half]})
        exact_mixed = g.recovery(mixed_eval, g.patched_axis(backend, mixed_eval, units))
        q_signed = g.block_diff_in_means(backend, mixed_fit, units)
        # unsigned: plain mean over the mixed rows
        # (the unsigned mean can cancel EXACTLY when the rows pair up as mirrors; then the old
        # code normalised float residue -- record the raw norm and refuse to normalise it)
        q_unsigned, unsigned_norm, flipped = {}, {}, {}
        for key, us in g.blocks_of(units).items():
            delta = g._cached_delta(backend, mixed_fit, us)
            flipped[f"{key[0]:02d}:{key[1]}"] = int((g._orientation(delta) < 0).sum())
            d = delta.mean(0)
            unsigned_norm[f"{key[0]:02d}:{key[1]}"] = float(d.norm())
            q_unsigned[key] = (d / d.norm()).unsqueeze(1) if float(d.norm()) > 1e-3 else None
        degenerate = any(v is None for v in q_unsigned.values())
        purity = {"directions_in_fit_half": dirs_fit, "exact_mixed_eval": exact_mixed,
                  "signed_fraction": _frac(exact_mixed, g.recovery(
                      mixed_eval, g.patched_axis(backend, mixed_eval, units, q=q_signed))),
                  "unsigned_mean_norm": unsigned_norm, "unsigned_degenerate": degenerate,
                  "rows_flipped_by_sign_alignment": flipped, "mixed_rows": len(mixed_fit.rows),
                  "unsigned_fraction": None if degenerate else _frac(exact_mixed, g.recovery(
                      mixed_eval, g.patched_axis(backend, mixed_eval, units, q=q_unsigned))),
                  "even_rows_directions": sorted({r.get("direction_id") for r in a1[0::2]}),
                  "odd_rows_directions": sorted({r.get("direction_id") for r in a1[1::2]})}

        report[label] = {"units": units, "kind": kind, "exact_set": exact, "semantics": semantics,
                         "block_live": directions, "pairwise_seed_cosines": pair_cos,
                         "mean_pairwise_seed_cosine": mean_pair_cos,
                         "cached_joint": cached_joint, "cdas_rank_sweep": rank_sweep,
                         "direction_purity": purity,
                         "fit_rows": len(preps["fit"].rows), "parameters_per_rank": dim_total,
                         "seconds": time.perf_counter() - ts}
        print(label, json.dumps({
            "semantics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in semantics.items()},
            "block": {n: _brief(d) for n, d in directions.items()},
            "cached_joint": {n: _brief(cached_joint[n]) for n in ("dim", "das")},
            "seed_cos": round(mean_pair_cos, 3),
            "purity": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in purity.items()},
            "s": round(report[label]["seconds"], 1)}))

    # registered predictions
    f = lambda x: 0.0 if x is None else x
    B = lambda k, n: report[k]["block_live"][n]
    in_band = lambda d: LO <= f(d["held"]["subspace_fraction"]) <= HI and LO <= f(d["a2"]["subspace_fraction"]) <= HI
    multi = [k for k, r in report.items() if len(r["semantics"]["layers"]) > 1]
    mlps = [k for k, v in SETS.items() if v[1] == "mlp"]
    seed_in_band = {k: all(in_band(B(k, f"das_seed{s}")) for s in SEEDS) for k in report}
    predictions = {
        'pred_a_block_full_rank_equals_exact': all(r["semantics"]["block_error"] <= 1e-3 for r in report.values()),
        'pred_b_cached_cross_layer_bias_bounded': all(
            abs(f(report[k]["semantics"]["cached_bias_fraction"])) <= BIAS_BAR for k in multi),
        'pred_c_dim_is_linear': all(LIN_LO <= f(B(k, "dim")["held"]["linearity_sum"]) <= LIN_HI for k in report),
        'pred_d_fitted_directions_exploit_nonlin': any(
            f(B(k, "das_seed0")["held"]["linearity_sum"]) > LIN_HI for k in report),
        'pred_e_das_directions_not_unique': (
            sum(r["mean_pairwise_seed_cosine"] for r in report.values()) / len(report) < COS_BAR
            and sum(seed_in_band.values()) >= 5),
        'pred_f_cdas_rank_repairs_mlp_sets': all(any(
            in_band(B(k, n)) and abs(f(B(k, n)["held"]["complement_fraction"])) <= COMP_BAR
            for n in ("cdas", "cdas_rank2", "cdas_rank4")) for k in mlps),
        'pred_g_signed_dim_survives_mixed_rows': all(
            LO <= f(r["direction_purity"]["signed_fraction"]) <= HI
            and abs(f(r["direction_purity"]["unsigned_fraction"])) < 0.5 * abs(f(r["direction_purity"]["signed_fraction"]))
            for r in report.values()),
    }
    predictions = {k: bool(v) for k, v in predictions.items()}
    result = {"schema": "circuit_unit_subspace_redteam_result_v7",
              "candidate_id": "corpus.unit_subspace_redteam_v7", "instrument": instrument,
              "registered": {"steps": STEPS, "lr": LR, "lambda": LAMBDA, "seeds": SEEDS,
                             "cdas_ranks": CDAS_RANKS, "band": [LO, HI], "complement_bar": COMP_BAR,
                             "linear_band": [LIN_LO, LIN_HI], "bias_bar": BIAS_BAR, "cos_bar": COS_BAR,
                             "block_mode": "one subspace per (layer, kind) block applied to the LIVE value",
                             "das_objective": "match_exact_set_patch"},
              "predictions": predictions, "seed_in_band": seed_in_band, "sets": report,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
