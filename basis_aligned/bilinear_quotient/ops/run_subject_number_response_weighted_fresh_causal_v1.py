#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_license_and_instrument pred_b_fresh_coefficient_and_null pred_c_fresh_law_causal_replay pred_d_fresh_native_and_strata
"""Prospective fourth-corpus causal test of frozen response-weighted prototypes."""
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
PREREG = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_FRESH_CAUSAL_V1_PREREGISTRATION.md"
ARTIFACT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FROZEN_V1_ARTIFACT.json"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
BINDING = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_FRESH_CAUSAL_V1_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_response_weighted_fresh_causal_v1_result.json"
NULLS = 4
SEED = 2026091602
METHODS = ("base", "exact", "law", "candidate", *(f"null_{i}" for i in range(NULLS)))
PATCH_CHUNK_ROWS = 256
PRICE = {"physical_model_forwards": 1 + math.ceil(32 * 16 * len(METHODS) / PATCH_CHUNK_ROWS),
         "role_sequences": 96, "causal_installations": 32 * 16 * len(METHODS),
         "offline_head_function_row_evaluations": 32 * 16 * (3 + NULLS),
         "equal_norm_nulls_per_direction": NULLS, "coefficient_fits": 0,
         "backwards": 0, "parameter_updates": 0, "maximum_patch_chunk_rows": PATCH_CHUNK_ROWS}
PREDICTION_REGISTRY = {"pred_a_license_and_instrument": None,
                       "pred_b_fresh_coefficient_and_null": None,
                       "pred_c_fresh_law_causal_replay": None,
                       "pred_d_fresh_native_and_strata": None}


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


def passes(report, cosine, error, sign):
    return report["cosine"] >= cosine and report["relative_l2_error"] <= error and report["sign_agreement"] >= sign


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "artifact": ARTIFACT, "law": LAW,
             "authority": Path(authority.__file__), "capability_result": capability.RESULT,
             "capability_license": capability.LICENSE, "capability_runner": Path(capability.__file__)}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["price"] != PRICE or binding["methods"] != list(METHODS) \
            or binding["nulls"] != NULLS or binding["seed"] != SEED:
        raise ValueError("binding changed")
    artifact, law = json.loads(ARTIFACT.read_text()), json.loads(LAW.read_text())
    if artifact["terminal"] != "response_weighted_prototypes_frozen_opened_only" \
            or not all(artifact["predictions"].values()) \
            or law["terminal"] != "bilinear_scalar_law_frozen_weights_only":
        raise ValueError("parent status changed")
    licensing.validate_causal_preflight(capability.build_gate(), capability.RESULT, capability.LICENSE,
        expected_license_sha256=binding["files"]["capability_license"],
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return binding, artifact, law


def plan():
    binding, artifact, _ = load_bound(); rows = authority.build_rows()
    return {"schema": "subject_number_response_weighted_fresh_causal_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "role_sequences": 3 * len(rows),
            "background_subsets": list(factor_gate.BACKGROUND_SUBSETS),
            "methods": list(METHODS), "ports": ["native_mlp8_state", "frozen_head_weights", "counterfactual_direction"],
            "prototype_hashes": {key: value["float32_sha256"] for key, value in artifact["prototype_audit"].items()},
            "price": PRICE, "authority_sha256": authority.validate_rows(rows),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def compile_patch(tokens, heads, rows, torch):
    indices, replacements, specs = [], [], []
    for i, _ in enumerate(rows):
        for subset in factor_gate.BACKGROUND_SUBSETS:
            for method_name in METHODS:
                indices.append(i); replacements.append(heads[(i, subset, method_name)])
                specs.append((i, subset, method_name))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {"tokens": tokens[:len(rows)][index],
            "finals": torch.full_like(index, tangent.parent.SUBJECT_POSITION),
            "replacement_heads": torch.stack(replacements),
            "native_reinstall_mask": torch.zeros(len(specs), dtype=torch.bool, device=tokens.device),
            "specs": specs}


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, artifact, law = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    rows = authority.build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = tangent.parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    with torch.no_grad():
        _, captured, projection, role_closure, inputs = tangent.parent._decomposed_forward(
            model, tokens, finals, torch, F, facade)
        recipient = tangent._role_slice(captured, 0, n)
        opposite = tangent._role_slice(captured, n, 2 * n)
        ir = tangent._role_slice(inputs, 0, n)
        io = tangent._role_slice(inputs, n, 2 * n)
        function = tangent._head_function(model, recipient, opposite,
            model.transformer.h[tangent.parent.LAYER].attn, projection, torch, F)
        axis = torch.tensor(artifact["native_axis"], device=device, dtype=inputs["raw_state"].dtype)
        beta = torch.tensor(artifact["interaction_beta"], device=device, dtype=torch.float64)
        prototypes = {key: torch.tensor(value, device=device, dtype=axis.dtype)
                      for key, value in artifact["prototypes"].items()}
        generator = torch.Generator(device=device); generator.manual_seed(SEED)
        nulls = {}
        for direction, prototype in prototypes.items():
            values = []
            for _ in range(NULLS):
                value = torch.randn(prototype.shape, generator=generator, device=device, dtype=prototype.dtype)
                values.append(value * (prototype.norm() / value.norm().clamp_min(1e-30)))
            nulls[direction] = values
        heads, scalar_evidence = {}, []
        evaluations = 0
        for subset in factor_gate.BACKGROUND_SUBSETS:
            xb = factor_gate._raw_for(ir, io, subset, F)
            xy = factor_gate._raw_for(ir, io, subset + "YZ", F)
            h0, h1 = function(xb), function(xy); evaluations += 2 * n
            p_batch = torch.stack([prototypes[row["direction_id"]] for row in rows])
            hp = function(xb + p_batch); evaluations += n
            z = (h0 @ axis).double(); s = ((hp - h0) @ axis).double()
            candidate_alpha = torch.stack([torch.ones_like(z), z, s, z * s], dim=1) @ beta
            null_alphas = []
            for k in range(NULLS):
                np_batch = torch.stack([nulls[row["direction_id"]][k] for row in rows])
                hn = function(xb + np_batch); evaluations += n
                ns = ((hn - h0) @ axis).double()
                null_alphas.append(torch.stack([torch.ones_like(z), z, ns, z * ns], dim=1) @ beta)
            for i, row in enumerate(rows):
                target = float(law["predicted_coefficients"][f"{row['direction_id']}.cardinality_{len(subset)}"])
                heads[(i, subset, "base")] = h0[i]
                heads[(i, subset, "exact")] = h1[i]
                heads[(i, subset, "law")] = h0[i] + target * axis
                heads[(i, subset, "candidate")] = h0[i] + candidate_alpha[i].to(axis.dtype) * axis
                for k in range(NULLS):
                    heads[(i, subset, f"null_{k}")] = h0[i] + null_alphas[k][i].to(axis.dtype) * axis
                scalar_evidence.append({"row_id": row["row_id"], "direction": row["direction_id"],
                                        "template": row["template_id"], "background": subset,
                                        "cardinality": len(subset), "z": float(z[i]), "s": float(s[i]),
                                        "target_alpha": target, "candidate_alpha": float(candidate_alpha[i]),
                                        "null_alphas": [float(value[i]) for value in null_alphas]})
        patch = compile_patch(tokens, heads, rows, torch)
        margins, downstream_closures = {}, []
        for start in range(0, len(patch["specs"]), PATCH_CHUNK_ROWS):
            stop = min(start + PATCH_CHUNK_ROWS, len(patch["specs"]))
            logits, _, _, closure = tangent.parent.downstream._decomposed_forward(
                model, patch["tokens"][start:stop], patch["finals"][start:stop], torch, F, facade,
                replacement_heads=patch["replacement_heads"][start:stop],
                native_reinstall_mask=patch["native_reinstall_mask"][start:stop])
            downstream_closures.append(closure)
            for local, spec in enumerate(patch["specs"][start:stop]):
                i, subset, method_name = spec; endpoint = rows[i]["endpoints"]["opposite_same_lemma"]
                margins[(i, subset, method_name)] = float(logits[local, tangent.parent.SUBJECT_POSITION, endpoint["answer_id"]]
                                                          - logits[local, tangent.parent.SUBJECT_POSITION, endpoint["foil_id"]])
    evidence = []
    for i, row in enumerate(rows):
        for subset in factor_gate.BACKGROUND_SUBSETS:
            base = margins[(i, subset, "base")]
            evidence.append({"row_id": row["row_id"], "direction": row["direction_id"],
                             "template": row["template_id"], "background": subset,
                             "cardinality": len(subset),
                             **{f"{name}_q": margins[(i, subset, name)] - base for name in METHODS[1:]}})
    targets = [x["target_alpha"] for x in scalar_evidence]
    candidate_coefficients = [x["candidate_alpha"] for x in scalar_evidence]
    coefficient_metrics = stats(targets, candidate_coefficients)
    null_coefficient_metrics = [stats(targets, [x["null_alphas"][k] for x in scalar_evidence]) for k in range(NULLS)]
    native = stats([x["exact_q"] for x in evidence], [x["candidate_q"] for x in evidence])
    law_effect = stats([x["law_q"] for x in evidence], [x["candidate_q"] for x in evidence])
    null_law_metrics = [stats([x["law_q"] for x in evidence], [x[f"null_{k}_q"] for x in evidence]) for k in range(NULLS)]
    null_coefficient_median = float(np.median([x["relative_l2_error"] for x in null_coefficient_metrics]))
    null_law_median = float(np.median([x["relative_l2_error"] for x in null_law_metrics]))
    templates = {name: stats([x["exact_q"] for x in evidence if x["template"] == name],
                             [x["candidate_q"] for x in evidence if x["template"] == name])
                 for name, _ in authority.TEMPLATES}
    intermediate_rows = [x for x in evidence if x["background"] not in {"", "EAUW"}]
    intermediate = stats([x["exact_q"] for x in intermediate_rows], [x["candidate_q"] for x in intermediate_rows])
    exactness = {"role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
                 "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
                 "downstream_state_closure_max_absolute_error": max(x["state_sum_max_absolute_error"] for x in downstream_closures),
                 "downstream_normalized_closure_max_absolute_error": max(x["normalized_state_max_absolute_error"] for x in downstream_closures)}
    finite = all(np.isfinite(value) for report in [coefficient_metrics, native, law_effect, intermediate, *templates.values(), *null_coefficient_metrics, *null_law_metrics]
                 for value in report.values() if isinstance(value, float))
    instrument = (len(evidence) == 512 and len(scalar_evidence) == 512 and len(patch["specs"]) == PRICE["causal_installations"]
                  and evaluations == PRICE["offline_head_function_row_evaluations"] and finite
                  and max(exactness.values()) <= 5e-5)
    coefficient_pass = passes(coefficient_metrics, .80, .55, .90) and null_coefficient_median - coefficient_metrics["relative_l2_error"] >= .05
    law_pass = passes(law_effect, .85, .50, .90) and null_law_median - law_effect["relative_l2_error"] >= .05
    native_pass = passes(native, .70, .80, .75) and passes(intermediate, .60, .90, .65) \
        and all(passes(value, .60, .90, .65) for value in templates.values())
    predictions = {"pred_a_license_and_instrument": bool(instrument),
                   "pred_b_fresh_coefficient_and_null": bool(instrument and coefficient_pass),
                   "pred_c_fresh_law_causal_replay": bool(instrument and law_pass),
                   "pred_d_fresh_native_and_strata": bool(instrument and native_pass)}
    terminal = "response_weighted_fresh_causal_held" if all(predictions.values()) else "invalid" if not instrument else "response_weighted_fresh_causal_null"
    result = {"schema": "subject_number_response_weighted_fresh_causal_v1_result",
              "terminal": terminal, "predictions": predictions,
              "coefficient_metrics": coefficient_metrics, "law_effect_metrics": law_effect,
              "native_effect_metrics": native, "intermediate_native_effect_metrics": intermediate,
              "template_native_effect_metrics": templates,
              "null_coefficient_metrics": null_coefficient_metrics, "null_law_effect_metrics": null_law_metrics,
              "null_coefficient_median_error": null_coefficient_median,
              "null_law_effect_median_error": null_law_median,
              "coefficient_null_advantage": null_coefficient_median - coefficient_metrics["relative_l2_error"],
              "law_effect_null_advantage": null_law_median - law_effect["relative_l2_error"],
              "exactness": exactness, "scalar_evidence": scalar_evidence, "causal_evidence": evidence,
              "instrument": {"examples": len(evidence), "scalar_examples": len(scalar_evidence),
                             "causal_installations": len(patch["specs"]), "offline_head_function_row_evaluations": evaluations,
                             "finite": finite, "fresh_opposite_state_used_by_candidate": False,
                             "declared_ports": ["native_mlp8_state", "frozen_head_weights", "counterfactual_direction"]},
              "price": PRICE, "authority_sha256": authority.validate_rows(rows),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Prospective fourth-corpus coefficient and causal-effect prediction by a frozen donor-free response-weighted generator; direction remains a categorical input port."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "coefficient_metrics",
                                                    "law_effect_metrics", "native_effect_metrics",
                                                    "intermediate_native_effect_metrics", "template_native_effect_metrics",
                                                    "null_coefficient_median_error", "null_law_effect_median_error",
                                                    "coefficient_null_advantage", "law_effect_null_advantage",
                                                    "exactness", "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
