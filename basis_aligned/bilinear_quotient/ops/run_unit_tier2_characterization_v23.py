#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 greedy), families (A1/A2/P/C), bars fixed before the run.
"""v23: Tier 2 behavioural characterization of four localized head sets (TIER_RUBRIC.md).

Tier 2 = Tier 1 plus "an exact definition of the affected tokens, positions, best competitor, effect
direction, and magnitude; the effect separates target from matched-negative and off-target positions".
The v9 head sets for quantifier_number, dative, voice_frame and polarity_licensing have Tier 1 (causal
localization, block-live exact set, held-out A2) but no record states those fields. One runner, one
intervention per family: EXACT block-live replacement of the set (no fitted axis) at the semantic
position (= final input position for every row, so "off-target positions" reduces to off-target TOKENS
at that position and to the same-answer families P and C).

Per behaviour x family (A1, A2 answer-changing; P positional, C related, same-answer) from the full
final-position logits (forward_units(return_logits=True), model unchanged):
    tokens       donor-answer / base-answer token strings, their log-prob shift under the patch (nats)
    competitor   argmax over the vocabulary excluding the donor answer, AFTER the patch; is it the foil?
    direction    fraction of rows whose margin moves toward the donor (sign consistency)
    magnitude    median margin shift (nats) and median signed recovery (fraction of donor - base)
    flip         fraction of rows whose full-vocabulary argmax becomes the donor answer
    off-target   KL(base || patched) at the final position over the vocabulary with the answer and foil
                 tokens removed and renormalised (nats): what the patch does to everything else
    negatives    P/C: median |margin movement| / target scale, and the full-vocab KL(base || patched)
Model-load, 4 behaviours x 4 families x (2 native + 1 patched) forwards.

REGISTERED BEFORE THE RUN
    pred_a_direction_consistent   >= 0.90 of A1 rows move toward the donor, every behaviour.
                                  Worked example: 31/32 = 0.97 -> True; 27/32 = 0.84 -> False.
    pred_b_competitor_is_foil     after the patch the best non-donor-answer token is the base answer
                                  (the foil) on >= 0.80 of A1 rows, every behaviour: a two-way decision.
                                  Worked example: 30/32 -> True.
    pred_c_flips_majority         full-vocabulary argmax becomes the donor answer on >= 0.50 of A1 rows
                                  for every behaviour (the sets recover 0.5-0.9 of the margin; whether
                                  that crosses the argmax is what this measures). Worked: 0.62 -> True.
    pred_d_negatives_separated    median |P| and |C| movement <= 0.10 of target scale AND median A1
                                  recovery >= 0.50, every behaviour (separation ratio >= 5).
                                  Worked example: A1 0.61, P 0.04, C 0.07 -> True; C 0.15 -> False.
    pred_e_off_target_small       median off-target KL on A1 <= 0.05 nat, every behaviour.
                                  Worked example: 0.02 -> True; 0.12 -> False.
    Reading rule. a,d True: the set's effect is a signed, target-specific push on the answer axis; the
    Tier 2 fields are the recorded values. e False on a behaviour: the set moves more than the answer
    pair -- record the top off-target tokens; the set is a broader feature writer, characterize before
    any Tier 3 claim. c False: the set is necessary-but-insufficient at the argmax; state it.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import circuit_fast_screen_candidate_voice_frame as m_voice
import circuit_fast_screen_candidate_sentence_terminal_context_control as spec_builder

ENC = spec_builder.ENCODING

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier2_characterization_v23_result.json"
SETS = {
    "quantifier_number": (v15.SETS["quantifier_number"][0], v15.SETS["quantifier_number"][1]),
    "dative": (v15.SETS["dative"][0], v15.SETS["dative"][1]),
    "polarity_licensing": (v15.SETS["polarity_licensing"][0], v15.SETS["polarity_licensing"][1]),
    "voice_frame": (m_voice, ["attn:07:head:08", "attn:01:head:05", "attn:00:head:03", "attn:04:head:01"]),
}
FAMILIES = ("A1", "A2", "P", "C")
SIGN_BAR, FOIL_BAR, FLIP_BAR, NEG_BAR, REC_BAR, KL_BAR, TOPK = 0.90, 0.80, 0.50, 0.10, 0.50, 0.05, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 2000


def _plan():
    return {"candidate_id": "corpus.unit_tier2_characterization_v23", "sets": {k: v[1] for k, v in SETS.items()},
            "families": list(FAMILIES), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def _median(xs):
    return float(statistics.median(xs)) if xs else None


def _characterize(backend, module, units, family, scale):
    torch = backend.torch
    rows = g.rows_of(module, family)
    prep = g.prepare(backend, rows, valid_only=(family in ("A1", "A2")))
    rows = prep.rows
    _, base_logits = g.forward_units(backend, prep.base_batch, return_logits=True)
    af, pat_logits = g.forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                                     base_cache=prep.base_cache, return_logits=True)
    base_lp, pat_lp = torch.log_softmax(base_logits, -1), torch.log_softmax(pat_logits, -1)
    n = len(rows)
    idx = torch.arange(n, device=base_logits.device)
    d_ans = torch.tensor([r["donor_answer_id"] for r in rows], device=idx.device)
    b_ans = torch.tensor([r["base_answer_id"] for r in rows], device=idx.device)
    patched_axis = [-(float(a) - float(f)) for a, f in af.tolist()]
    # competitor: best token other than the donor answer, after the patch
    masked = pat_logits.clone(); masked[idx, d_ans] = -1e9
    comp = masked.argmax(-1)
    comp_is_foil = (comp == b_ans).float().mean().item()
    flips = (pat_logits.argmax(-1) == d_ans).float().mean().item()
    base_top_is_base_answer = (base_logits.argmax(-1) == b_ans).float().mean().item()
    # off-target KL: drop answer + foil columns, renormalise
    keep = torch.ones_like(base_logits, dtype=torch.bool); keep[idx, d_ans] = False; keep[idx, b_ans] = False
    bl = base_logits.masked_fill(~keep, -1e9); pl = pat_logits.masked_fill(~keep, -1e9)
    blp, plp = torch.log_softmax(bl, -1), torch.log_softmax(pl, -1)
    off_kl = (blp.exp() * (blp - plp)).masked_fill(~keep, 0.0).sum(-1)
    full_kl = (base_lp.exp() * (base_lp - pat_lp)).sum(-1)
    shift = [p - b for p, b in zip(patched_axis, prep.base_axis)]
    out = {"rows": n, "dropped": prep.dropped, "units": list(units),
           "donor_answer_tokens": sorted({r["donor_answer"] for r in rows}),
           "base_answer_tokens": sorted({r["base_answer"] for r in rows}),
           "median_margin_shift_nats": _median(shift),
           "median_donor_answer_logprob_shift": (pat_lp[idx, d_ans] - base_lp[idx, d_ans]).median().item(),
           "median_base_answer_logprob_shift": (pat_lp[idx, b_ans] - base_lp[idx, b_ans]).median().item(),
           "competitor_is_foil_fraction": comp_is_foil,
           "competitor_tokens_top": [t for t, _ in __import__("collections").Counter(ENC.decode([int(t)]) for t in comp.tolist()).most_common(TOPK)],
           "donor_answer_is_argmax_after": flips,
           "base_answer_is_argmax_before": base_top_is_base_answer,
           "median_off_target_kl_nats": off_kl.median().item(),
           "median_full_kl_nats": full_kl.median().item(),
           "max_off_target_kl_nats": off_kl.max().item()}
    if family in ("A1", "A2"):
        rec = [kernel.signed_pairwise_donor_recovery(b, d, p) for b, d, p in zip(prep.base_axis, prep.donor_axis, patched_axis)]
        toward = [ (d - b) * (p - b) > 0 for b, d, p in zip(prep.base_axis, prep.donor_axis, patched_axis)]
        out.update({"median_recovery": _median(rec), "mean_recovery": sum(rec) / n,
                    "toward_donor_fraction": sum(toward) / n, "target_scale_nats": g.target_scale(prep)})
    else:
        mv = [abs(p - b) / scale for b, p in zip(prep.base_axis, patched_axis)]
        out.update({"median_abs_movement_over_scale": _median(mv), "max_abs_movement_over_scale": max(mv)})
    # top off-target movers on the worst row, for the record
    worst = int(off_kl.argmax())
    dlp = (pat_lp[worst] - base_lp[worst]).masked_fill(~keep[worst], 0.0)
    top = dlp.topk(TOPK).indices.tolist()
    out["worst_row_top_gainers"] = [ENC.decode([t]) for t in top]
    return out


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        a1 = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        scale = g.target_scale(a1)
        report[name] = {fam: _characterize(backend, module, units, fam, scale) for fam in FAMILIES}
        r = report[name]
        print(name, "A1 toward %.2f rec %.2f flip %.2f compfoil %.2f offKL %.3f | P %.2f C %.2f" % (
            r["A1"]["toward_donor_fraction"], r["A1"]["median_recovery"], r["A1"]["donor_answer_is_argmax_after"],
            r["A1"]["competitor_is_foil_fraction"], r["A1"]["median_off_target_kl_nats"],
            r["P"]["median_abs_movement_over_scale"], r["C"]["median_abs_movement_over_scale"]), flush=True)

    a1 = {k: v["A1"] for k, v in report.items()}
    predictions = {
        'pred_a_direction_consistent': all(v["toward_donor_fraction"] >= SIGN_BAR for v in a1.values()),
        'pred_b_competitor_is_foil': all(v["competitor_is_foil_fraction"] >= FOIL_BAR for v in a1.values()),
        'pred_c_flips_majority': all(v["donor_answer_is_argmax_after"] >= FLIP_BAR for v in a1.values()),
        'pred_d_negatives_separated': all(v["A1"]["median_recovery"] >= REC_BAR and v["P"]["median_abs_movement_over_scale"] <= NEG_BAR
                                          and v["C"]["median_abs_movement_over_scale"] <= NEG_BAR for v in report.values()),
        'pred_e_off_target_small': all(v["median_off_target_kl_nats"] <= KL_BAR for v in a1.values()),
    }
    tier2 = {k: bool(v["A1"]["toward_donor_fraction"] >= SIGN_BAR and v["A1"]["median_recovery"] >= REC_BAR
                     and v["P"]["median_abs_movement_over_scale"] <= NEG_BAR and v["C"]["median_abs_movement_over_scale"] <= NEG_BAR
                     and v["A1"]["median_off_target_kl_nats"] <= KL_BAR) for k, v in report.items()}
    result = {"predictions": predictions, "tier2_supported": tier2, "schema": "circuit_unit_tier2_characterization_result_v1",
              "candidate_id": "corpus.unit_tier2_characterization_v23", "semantics": "block_live_exact_set",
              "bars": {"sign": SIGN_BAR, "foil": FOIL_BAR, "flip": FLIP_BAR, "negative": NEG_BAR, "recovery": REC_BAR, "off_target_kl": KL_BAR},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "tier2_supported": tier2, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
