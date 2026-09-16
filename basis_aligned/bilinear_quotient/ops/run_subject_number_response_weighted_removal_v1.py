#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_replay_and_instrument pred_b_candidate_removes_native_effect pred_c_candidate_beats_law_and_null pred_d_unrelated_reader_selectivity
"""Selective removal test for the fresh response-weighted L11H3 write."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_candidate_subject_number_response_weighted_fresh as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_subject_number_response_weighted_fresh_capability_v1 as capability
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parent.parent
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_REMOVAL_V1_PREREGISTRATION.md"
ARTIFACT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FROZEN_V1_ARTIFACT.json"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
FRESH = ROOT / "circuits/fast_screens/subject_number_response_weighted_fresh_causal_v1_result.json"
BINDING = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_REMOVAL_V1_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_response_weighted_removal_v1_result.json"
NULLS = 4
SEED = 2026091603
METHODS = ("base", "exact", "candidate_removed", "law_removed", *(f"null_removed_{i}" for i in range(NULLS)))
READOUTS = (("work_jobs", (670, 3946)), ("cat_dog", (3797, 3290)),
            ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
            ("apple_orange", (17180, 10912)))
PATCH_CHUNK_ROWS = 256
PRICE = {"physical_model_forwards": 1 + math.ceil(32 * 16 * len(METHODS) / PATCH_CHUNK_ROWS),
         "role_sequences": 96, "causal_installations": 32 * 16 * len(METHODS),
         "offline_head_function_row_evaluations": 32 * 16 * 3,
         "equal_norm_nulls": NULLS, "coefficient_fits": 0, "backwards": 0,
         "parameter_updates": 0, "maximum_patch_chunk_rows": PATCH_CHUNK_ROWS}
PREDICTION_REGISTRY = {"pred_a_replay_and_instrument": None,
                       "pred_b_candidate_removes_native_effect": None,
                       "pred_c_candidate_beats_law_and_null": None,
                       "pred_d_unrelated_reader_selectivity": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    an, pn = np.linalg.norm(actual), np.linalg.norm(predicted)
    return {"count": int(len(actual)), "cosine": float(actual @ predicted / max(an * pn, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(actual - predicted) / max(an, 1e-30)),
            "sign_agreement": float(np.mean((actual > 0) == (predicted > 0))),
            "rms_actual": float(np.sqrt(np.mean(actual ** 2))),
            "rms_predicted": float(np.sqrt(np.mean(predicted ** 2)))}


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "artifact": ARTIFACT, "law": LAW,
             "fresh_result": FRESH, "authority": Path(authority.__file__),
             "capability_result": capability.RESULT, "capability_license": capability.LICENSE}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["price"] != PRICE or binding["methods"] != list(METHODS) \
            or binding["readouts"] != [[name, list(pair)] for name, pair in READOUTS] \
            or binding["nulls"] != NULLS or binding["seed"] != SEED:
        raise ValueError("binding changed")
    artifact, law, fresh = (json.loads(path.read_text()) for path in (ARTIFACT, LAW, FRESH))
    if artifact["terminal"] != "response_weighted_prototypes_frozen_opened_only" \
            or law["terminal"] != "bilinear_scalar_law_frozen_weights_only" \
            or fresh["terminal"] != "response_weighted_fresh_causal_null" \
            or fresh["predictions"]["pred_a_license_and_instrument"] is not True \
            or fresh["predictions"]["pred_b_fresh_coefficient_and_null"] is not True \
            or fresh["predictions"]["pred_d_fresh_native_and_strata"] is not True:
        raise ValueError("parent status changed")
    licensing.validate_causal_preflight(capability.build_gate(), capability.RESULT, capability.LICENSE,
        expected_license_sha256=binding["files"]["capability_license"],
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return binding, artifact, law, fresh


def plan():
    binding, _, _, _ = load_bound(); rows = authority.build_rows()
    return {"schema": "subject_number_response_weighted_removal_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "role_sequences": 3 * len(rows),
            "background_subsets": list(factor_gate.BACKGROUND_SUBSETS),
            "methods": list(METHODS), "readouts": [name for name, _ in READOUTS],
            "price": PRICE, "authority_sha256": authority.validate_rows(rows),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def compile_patch(tokens, heads, rows, torch):
    indices, replacements, specs = [], [], []
    for i, _ in enumerate(rows):
        for subset in factor_gate.BACKGROUND_SUBSETS:
            for method_name in METHODS:
                indices.append(i); replacements.append(heads[(i, subset, method_name)]); specs.append((i, subset, method_name))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {"tokens": tokens[:len(rows)][index], "finals": torch.full_like(index, tangent.parent.SUBJECT_POSITION),
            "replacement_heads": torch.stack(replacements),
            "native_reinstall_mask": torch.zeros(len(specs), dtype=torch.bool, device=tokens.device), "specs": specs}


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, artifact, law, fresh = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    rows = authority.build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = tangent.parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    prior_alpha = {(x["row_id"], x["background"]): x["candidate_alpha"] for x in fresh["scalar_evidence"]}
    with torch.no_grad():
        _, captured, projection, role_closure, inputs = tangent.parent._decomposed_forward(model, tokens, finals, torch, F, facade)
        recipient = tangent._role_slice(captured, 0, n); opposite = tangent._role_slice(captured, n, 2 * n)
        ir = tangent._role_slice(inputs, 0, n); io = tangent._role_slice(inputs, n, 2 * n)
        function = tangent._head_function(model, recipient, opposite, model.transformer.h[tangent.parent.LAYER].attn, projection, torch, F)
        axis = torch.tensor(artifact["native_axis"], device=device, dtype=inputs["raw_state"].dtype)
        beta = torch.tensor(artifact["interaction_beta"], device=device, dtype=torch.float64)
        prototypes = {key: torch.tensor(value, device=device, dtype=axis.dtype) for key, value in artifact["prototypes"].items()}
        generator = torch.Generator(device=device); generator.manual_seed(SEED)
        random_directions = []
        for _ in range(NULLS):
            value = torch.randn(axis.shape, generator=generator, device=device, dtype=axis.dtype)
            value = value - (value @ axis) * axis
            random_directions.append(value / value.norm().clamp_min(1e-30))
        heads, coefficient_replay, norm_errors = {}, [], []
        evaluations = 0
        for subset in factor_gate.BACKGROUND_SUBSETS:
            xb = factor_gate._raw_for(ir, io, subset, F); xy = factor_gate._raw_for(ir, io, subset + "YZ", F)
            h0, h1 = function(xb), function(xy); evaluations += 2 * n
            p_batch = torch.stack([prototypes[row["direction_id"]] for row in rows])
            hp = function(xb + p_batch); evaluations += n
            z = (h0 @ axis).double(); s = ((hp - h0) @ axis).double()
            alpha = torch.stack([torch.ones_like(z), z, s, z * s], dim=1) @ beta
            for i, row in enumerate(rows):
                target = float(law["predicted_coefficients"][f"{row['direction_id']}.cardinality_{len(subset)}"])
                coefficient_replay.append(abs(float(alpha[i]) - prior_alpha[(row["row_id"], subset)]))
                candidate_write = alpha[i].to(axis.dtype) * axis
                heads[(i, subset, "base")] = h0[i]; heads[(i, subset, "exact")] = h1[i]
                heads[(i, subset, "candidate_removed")] = h1[i] - candidate_write
                heads[(i, subset, "law_removed")] = h1[i] - target * axis
                for k, direction in enumerate(random_directions):
                    null_write = alpha[i].to(axis.dtype) * direction
                    norm_errors.append(float(abs(null_write.norm() - candidate_write.norm()) / candidate_write.norm().clamp_min(1e-30)))
                    heads[(i, subset, f"null_removed_{k}")] = h1[i] - null_write
        patch = compile_patch(tokens, heads, rows, torch)
        target_margins, controls, downstream_closures = {}, {}, []
        for start in range(0, len(patch["specs"]), PATCH_CHUNK_ROWS):
            stop = min(start + PATCH_CHUNK_ROWS, len(patch["specs"]))
            logits, _, _, closure = tangent.parent.downstream._decomposed_forward(
                model, patch["tokens"][start:stop], patch["finals"][start:stop], torch, F, facade,
                replacement_heads=patch["replacement_heads"][start:stop], native_reinstall_mask=patch["native_reinstall_mask"][start:stop])
            downstream_closures.append(closure)
            for local, spec in enumerate(patch["specs"][start:stop]):
                i, subset, method_name = spec; endpoint = rows[i]["endpoints"]["opposite_same_lemma"]
                final_logits = logits[local, tangent.parent.SUBJECT_POSITION]
                target_margins[spec] = float(final_logits[endpoint["answer_id"]] - final_logits[endpoint["foil_id"]])
                controls[spec] = {name: float(final_logits[a] - final_logits[b]) for name, (a, b) in READOUTS}
    evidence = []
    for i, row in enumerate(rows):
        for subset in factor_gate.BACKGROUND_SUBSETS:
            base = target_margins[(i, subset, "base")]
            item = {"row_id": row["row_id"], "direction": row["direction_id"], "template": row["template_id"],
                    "background": subset, "cardinality": len(subset)}
            for name in METHODS[1:]:
                item[f"{name}_q"] = target_margins[(i, subset, name)] - base
            item["candidate_control_changes"] = {name: controls[(i, subset, "candidate_removed")][name] - controls[(i, subset, "exact")][name]
                                                   for name, _ in READOUTS}
            evidence.append(item)
    exact_effect = np.asarray([x["exact_q"] for x in evidence])
    candidate_residual = np.asarray([x["candidate_removed_q"] for x in evidence])
    law_residual = np.asarray([x["law_removed_q"] for x in evidence])
    candidate_removed_amount = exact_effect - candidate_residual
    removal_metrics = stats(exact_effect, candidate_removed_amount)
    exact_rms = float(np.sqrt(np.mean(exact_effect ** 2)))
    candidate_residual_ratio = float(np.sqrt(np.mean(candidate_residual ** 2)) / max(exact_rms, 1e-30))
    law_residual_ratio = float(np.sqrt(np.mean(law_residual ** 2)) / max(exact_rms, 1e-30))
    null_residual_ratios = [float(np.sqrt(np.mean(np.asarray([x[f"null_removed_{k}_q"] for x in evidence]) ** 2)) / max(exact_rms, 1e-30)) for k in range(NULLS)]
    null_median = float(np.median(null_residual_ratios))
    target_removal_rms = float(np.sqrt(np.mean(candidate_removed_amount ** 2)))
    unrelated = {name: float(np.sqrt(np.mean(np.asarray([x["candidate_control_changes"][name] for x in evidence]) ** 2)) / max(target_removal_rms, 1e-30))
                 for name, _ in READOUTS}
    exactness = {"role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
                 "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
                 "downstream_state_closure_max_absolute_error": max(x["state_sum_max_absolute_error"] for x in downstream_closures),
                 "downstream_normalized_closure_max_absolute_error": max(x["normalized_state_max_absolute_error"] for x in downstream_closures)}
    instrument = (len(evidence) == 512 and len(patch["specs"]) == PRICE["causal_installations"]
                  and evaluations == PRICE["offline_head_function_row_evaluations"] and exact_rms >= .02
                  and max(coefficient_replay) <= 1e-5 and max(norm_errors) <= 1e-6
                  and max(exactness.values()) <= 5e-5 and all(np.isfinite(x) for x in [candidate_residual_ratio, law_residual_ratio, null_median, *unrelated.values()]))
    removal_pass = candidate_residual_ratio <= .50 and removal_metrics["cosine"] >= .90 \
        and removal_metrics["relative_l2_error"] <= .50 and removal_metrics["sign_agreement"] >= .90
    comparison_pass = law_residual_ratio - candidate_residual_ratio >= .10 and null_median - candidate_residual_ratio >= .20
    selectivity_pass = max(unrelated.values()) <= .25
    predictions = {"pred_a_replay_and_instrument": bool(instrument),
                   "pred_b_candidate_removes_native_effect": bool(instrument and removal_pass),
                   "pred_c_candidate_beats_law_and_null": bool(instrument and comparison_pass),
                   "pred_d_unrelated_reader_selectivity": bool(instrument and selectivity_pass)}
    terminal = "response_weighted_selective_removal_held" if all(predictions.values()) else "invalid" if not instrument else "response_weighted_selective_removal_null"
    result = {"schema": "subject_number_response_weighted_removal_v1_result", "terminal": terminal,
              "predictions": predictions, "exact_effect_rms": exact_rms,
              "candidate_removal_metrics": removal_metrics, "candidate_residual_ratio": candidate_residual_ratio,
              "law_residual_ratio": law_residual_ratio, "null_residual_ratios": null_residual_ratios,
              "null_median_residual_ratio": null_median,
              "candidate_advantage_over_law": law_residual_ratio - candidate_residual_ratio,
              "candidate_advantage_over_null_median": null_median - candidate_residual_ratio,
              "target_removal_rms": target_removal_rms, "unrelated_reader_ratios": unrelated,
              "coefficient_replay_max_absolute_error": max(coefficient_replay),
              "equal_norm_max_relative_error": max(norm_errors), "exactness": exactness,
              "evidence": evidence,
              "instrument": {"examples": len(evidence), "causal_installations": len(patch["specs"]),
                             "offline_head_function_row_evaluations": evaluations, "coefficient_fits": 0,
                             "fresh_opposite_state_used_only_by_exact_ceiling": True},
              "price": PRICE, "authority_sha256": authority.validate_rows(rows),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Prospective-on-opened-fourth-corpus selective removal of the frozen response-weighted L11H3 write against symbolic-law, equal-norm same-site, and five unrelated-reader controls."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "exact_effect_rms",
                                                    "candidate_removal_metrics", "candidate_residual_ratio",
                                                    "law_residual_ratio", "null_residual_ratios",
                                                    "candidate_advantage_over_law", "candidate_advantage_over_null_median",
                                                    "target_removal_rms", "unrelated_reader_ratios",
                                                    "coefficient_replay_max_absolute_error", "equal_norm_max_relative_error",
                                                    "exactness", "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
