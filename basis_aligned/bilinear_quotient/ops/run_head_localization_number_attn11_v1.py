#!/usr/bin/env python3
# BQGATE: three frozen science predictions; head set fixed before the run.
"""Which HEADS of attn:11 carry number -- and do they carry a non-number variable at the same slot?

This replaces a retracted line of work rather than extending it. `resid:18` selectivity is
arithmetic: the answer margin there is a linear functional of the residual, so copying the donor's
final residual copies its logits, and target_recovery is exactly 1.0000 for all 50 screened
behaviours. Rank-1 DAS at that site recovers approximately the readout direction `w_a - w_f`,
which is why my "shared number direction" cosine of 0.651 sits on an `lm_head` difference cosine
of 0.558 and my "specificity control" of 0.034 sits on 0.006. Those were facts about the
unembedding, not about the model's computation.

The component sweeps already on disk point somewhere real: `attn:11` is the strongest single
module for the number behaviours (0.567 - 0.613) and markedly weaker for the same-slot tense
behaviour (0.394). This asks the question at HEAD grain, which is where a localization claim is
not recoverable from the weights alone.

Method: the producer's own `patched_heads`, which replaces one head's pre-projection slice with
the donor's and runs the real forward. Nothing about the readout is being fitted.

    behaviours   lexical_number.pp_intervener      number COPIED from a plural token
                 coordination_agreement.and_vs_or  number COMPOSED from two singulars
                 perfect_number.have_vs_has        number, disjoint answer vocabulary
                 temporal_auxiliary.will_vs_had    TENSE at the same read slot -- the control

REGISTERED BEFORE THE RUN (head set is all 9 of layer 11; no head is chosen after seeing results):
  pred_a_single_head_carries_number -> some ONE head reaches >= 0.30 recovery on all three number
                                       behaviours. Below that, number is distributed within the
                                       layer and I report that instead.
  pred_b_same_head_across_routes    -> the best head is the SAME for copy, compose and perfect.
                                       This is the non-trivial cross-behaviour claim, and unlike
                                       the resid:18 version a head is not the readout.
  pred_c_head_is_number_specific    -> that head's recovery on temporal_auxiliary is at most HALF
                                       its mean recovery on the three number behaviours.

  Stated prior: attn:11 as a whole reaches only 0.57-0.61 on number, so I do NOT expect a single
  head to carry it cleanly; the honest likely outcome is a distributed answer with two or three
  contributing heads. Registering pred_a as a real bar rather than a formality.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer

import circuit_fast_screen_candidate_lexical_number_pp as copy_num
import circuit_fast_screen_candidate_coordination_agreement as comp_num
import circuit_fast_screen_candidate_perfect_number as perf_num
import circuit_fast_screen_candidate_temporal_auxiliary as tense

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/head_localization_number_attn11_v1_result.json"
LAYER, N_HEADS = 11, 9
CARRY, SPECIFIC_RATIO = 0.30, 0.50
BEHAVIOURS = (("copy_lexical", copy_num), ("compose_coordination", comp_num),
              ("perfect_disjoint_vocab", perf_num), ("control_tense", tense))
NUMBER_KEYS = ("copy_lexical", "compose_coordination", "perfect_disjoint_vocab")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 25600


def _plan():
    return {"candidate_id": "number.head_localization_attn11_v1", "layer": LAYER,
            "heads": list(range(N_HEADS)),
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _a1(m):
    return [r for r in m.build_rows() if r["family"] == "A1"]


def _margins(out):
    return [a - f for a, f in out.answer_foil]


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    report = {}
    for label, module in BEHAVIOURS:
        rows = _a1(module)
        base_batch = das._batch(backend, rows, side="base")
        donor_batch = das._batch(backend, rows, side="donor")
        base_native = backend.native(base_batch, capture=False)
        donor_native = backend.native(donor_batch, capture=True)

        # donor head activations, keyed exactly as the producer expects
        donor_cache = donor_native.captured

        b = [-m for m in _margins(base_native)]          # base value on the donor's answer axis
        d = _margins(donor_native)

        per_head = {}
        for head in range(N_HEADS):
            out = backend.patched_heads(
                base_batch, layer=LAYER, heads=(head,), donor_cache=donor_cache)
            p = [-m for m in _margins(out)]
            rec = [kernel.signed_pairwise_donor_recovery(bi, di, pi)
                   for bi, di, pi in zip(b, d, p)]
            per_head[f"head:{head:02d}"] = sum(rec) / len(rec)

        out_all = backend.patched_heads(
            base_batch, layer=LAYER, heads=tuple(range(N_HEADS)), donor_cache=donor_cache)
        p_all = [-m for m in _margins(out_all)]
        all_rec = [kernel.signed_pairwise_donor_recovery(bi, di, pi)
                   for bi, di, pi in zip(b, d, p_all)]
        joint = sum(all_rec) / len(all_rec)
        summed = sum(per_head.values())
        # mlp:11 at the same layer, for a module-level comparison
        mlp_site = kernel.SiteRef(site_id=f"mlp:{LAYER:02d}", evidence_kind="residual")
        mlp_donor = backend.native(donor_batch, capture=True).captured
        mlp_out = backend.patched(base_batch, site=mlp_site, donor_cache=mlp_donor)
        p_mlp = [-m for m in _margins(mlp_out)]
        mlp_rec = sum(kernel.signed_pairwise_donor_recovery(bi, di, pi)
                      for bi, di, pi in zip(b, d, p_mlp)) / len(b)
        report[label] = {"per_head": per_head,
                         "all_heads_layer_11_joint": joint,
                         "sum_of_individual_heads": summed,
                         "additivity_ratio_joint_over_sum": (joint / summed) if summed else None,
                         "mlp_11_whole_module": mlp_rec,
                         "rows": len(rows)}

    # the best head is chosen by the NUMBER behaviours only; the control never influences it
    head_names = [f"head:{h:02d}" for h in range(N_HEADS)]
    mean_number = {h: sum(report[k]["per_head"][h] for k in NUMBER_KEYS) / len(NUMBER_KEYS)
                   for h in head_names}
    best = max(mean_number, key=mean_number.get)
    best_per_behaviour = {k: report[k]["per_head"][best] for k in NUMBER_KEYS}
    control = report["control_tense"]["per_head"][best]

    carries = bool(min(best_per_behaviour.values()) >= CARRY)
    same_head = bool(len({max(report[k]["per_head"], key=report[k]["per_head"].get)
                          for k in NUMBER_KEYS}) == 1)
    number_mean = sum(best_per_behaviour.values()) / len(best_per_behaviour)
    specific = bool(control <= SPECIFIC_RATIO * number_mean)

    predictions = {
        "pred_a_single_head_carries_number": carries,
        "pred_b_same_head_across_routes": same_head,
        "pred_c_head_is_number_specific": specific,
    }
    reading = ("single_head_carries_number_and_is_specific" if (carries and specific)
               else "number_is_distributed_within_layer_11" if not carries
               else "head_carries_number_but_is_not_specific")

    result = {"predictions": predictions,
              "schema": "circuit_head_localization_result_v1",
              "candidate_id": "number.head_localization_attn11_v1",
              "layer": LAYER, "heads_tested": list(range(N_HEADS)),
              "registered_thresholds": {"carry": CARRY, "specific_ratio": SPECIFIC_RATIO},
              "best_head_by_number_mean": best,
              "best_head_recovery_per_number_behaviour": best_per_behaviour,
              "best_head_recovery_on_tense_control": control,
              "reading": reading, "families": report,
              "supersedes": ("resid:18 DAS localization for the number family, retracted: "
                             "resid:18 recovery is 1.0000 for all 50 screened behaviours by "
                             "construction, and rank-1 DAS there recovers the lm_head readout "
                             "direction rather than an internal feature"),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "best_head": best,
                      "number": best_per_behaviour, "tense_control": control,
                      "families": report, "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
