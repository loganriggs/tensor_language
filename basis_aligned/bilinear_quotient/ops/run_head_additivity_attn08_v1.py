#!/usr/bin/env python3
# BQGATE: three frozen science predictions; layer and head set fixed before the run.
"""Do head interchange effects ADD, when more than one head actually contributes?

`head_localization_number_attn11_v1` found joint patching of all nine heads equals the sum of the
nine individual effects at 0.989 - 1.005. That looked like clean additivity but it was a WEAK
test: eight of the nine terms were zero, so with a single non-zero term additivity is nearly
vacuous. I said so at the time; this is the test that is not.

`attn:08` is the strongest single module in the corpus -- 0.898 for
`numbered_list.control_choice_discriminator`, 0.700 for `numeric_sequence` -- and a module that
strong is the best available candidate for a layer where SEVERAL heads carry part of the signal.
If two or more heads contribute here, joint-versus-sum becomes a real measurement of whether the
model's head contributions compose linearly at the read position.

    behaviours   numbered_list.control_choice_discriminator   attn:08 whole module 0.898
                 numbered_list.p_family_discriminator          attn:08 whole module 0.898
                 numeric_sequence (shared)                     attn:08 whole module 0.700
                 lexical_number.pp_intervener                  reference: attn:08 is NOT its module

REGISTERED BEFORE THE RUN (all 9 heads of layer 8; no head chosen after seeing results):
  pred_a_multiple_heads_contribute -> at least TWO heads reach 0.05 individual recovery on the
                                      same behaviour. If this FAILS the additivity test is again
                                      vacuous and I will report it as untested, not as passed.
  pred_b_effects_are_additive      -> joint / sum lies within 0.90 - 1.10 on every behaviour where
                                      pred_a holds.
  pred_c_subadditive               -> joint / sum < 0.90 on some behaviour, i.e. heads carry
                                      overlapping information and patching both double-counts.

  Stated prior: genuinely unsure. Attention head outputs enter the residual stream by ADDITION, so
  linearity into the residual is guaranteed; what is NOT guaranteed is linearity into the answer
  MARGIN after ten further layers of processing and a tanh soft cap. Sub-additivity from the cap
  alone is plausible at large effect sizes, and 0.898 is large.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer

import circuit_fast_screen_candidate_control_choice as list_choice
import circuit_fast_screen_candidate_list_p_family as list_p
import circuit_fast_screen_candidate_numbered_list_sufficiency as numseq
import circuit_fast_screen_candidate_lexical_number_pp as number_ref

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/head_additivity_attn08_v1_result.json"
LAYER, N_HEADS = 8, 9
BEHAVIOURS = (("list_control_choice", list_choice), ("list_p_family", list_p),
              ("list_cached_value_sufficiency", numseq), ("number_reference", number_ref))
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 25600


def _plan():
    return {"candidate_id": "numbered_list.head_additivity_attn08_v1", "layer": LAYER,
            "heads": list(range(N_HEADS)),
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _a1(m):
    # spec-authored candidates key this as "family"; the older modules use "transform_id"
    rows = m.build_rows()
    key = "family" if "family" in rows[0] else "transform_id"
    return [r for r in rows if r[key] == "A1"]


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
        # the MLP at the same layer, for a module-level comparison
        mlp_site = kernel.SiteRef(site_id=f"mlp:{LAYER:02d}", evidence_kind="residual")
        mlp_donor = backend.native(donor_batch, capture=True).captured
        mlp_out = backend.patched(base_batch, site=mlp_site, donor_cache=mlp_donor)
        p_mlp = [-m for m in _margins(mlp_out)]
        mlp_rec = sum(kernel.signed_pairwise_donor_recovery(bi, di, pi)
                      for bi, di, pi in zip(b, d, p_mlp)) / len(b)
        report[label] = {"per_head": per_head,
                         f"all_heads_layer_{LAYER:02d}_joint": joint,
                         "sum_of_individual_heads": summed,
                         "additivity_ratio_joint_over_sum": (joint / summed) if summed else None,
                         f"mlp_{LAYER:02d}_whole_module": mlp_rec,
                         "rows": len(rows)}

    # Registered predictions (docstring). pred_a gates the other two: with fewer than two
    # contributing heads the additivity ratio is vacuous and is reported as UNTESTED.
    CONTRIBUTE = 0.05
    contributing = {k: [h for h, v in report[k]["per_head"].items() if v >= CONTRIBUTE]
                    for k, _ in BEHAVIOURS}
    tested = [k for k, heads in contributing.items() if len(heads) >= 2]
    ratios = {k: report[k]["additivity_ratio_joint_over_sum"] for k in tested}
    pred_a = bool(tested)
    pred_b = bool(tested) and all(r is not None and 0.90 <= r <= 1.10 for r in ratios.values())
    pred_c = any(r is not None and r < 0.90 for r in ratios.values())
    predictions = {
        "pred_a_multiple_heads_contribute": pred_a,
        "pred_b_effects_are_additive": pred_b,
        "pred_c_subadditive": pred_c,
    }
    reading = ("additivity_untested_single_head_per_behaviour" if not pred_a
               else "head_effects_additive" if pred_b
               else "head_effects_subadditive" if pred_c
               else "head_effects_superadditive")

    result = {"predictions": predictions,
              "schema": "circuit_head_additivity_result_v1",
              "candidate_id": "numbered_list.head_additivity_attn08_v1",
              "layer": LAYER, "heads_tested": list(range(N_HEADS)),
              "registered_thresholds": {"contribute": CONTRIBUTE, "additive_band": [0.90, 1.10]},
              "contributing_heads": contributing,
              "behaviours_where_additivity_is_testable": tested,
              "additivity_ratios": ratios,
              "reading": reading, "families": report,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reading": reading, "contributing": contributing, "ratios": ratios,
                      "families": report, "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
