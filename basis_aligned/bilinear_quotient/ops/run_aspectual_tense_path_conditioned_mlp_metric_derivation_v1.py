#!/usr/bin/env python3
"""Develop a suffix-Jacobian-conditioned MLP response score."""

# BQGATE: EXPERIMENT pred_a_exact_gradient_response_pairing_and_finiteness pred_b_path_conditioning_improves_both_development_correlations pred_c_path_conditioning_is_strong_enough_for_fresh_test pred_d_zero_fit_and_declared_development_scope
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v3 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import path_conditioned_component_score as path_score
import positioned_component_program_eval as positioned
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_path_conditioned_mlp_metric_derivation_v1.json"
FRESH_RESULT = ROOT / "circuits/followups/aspectual_tense_activation_conditioned_mlp_fresh_validation_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v3.py"
SCORE_LIBRARY = ROOT / "ops/path_conditioned_component_score.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/aspectual_tense_path_conditioned_mlp_metric_derivation_v1_result.json"
CANDIDATE_ID = "aspectual_tense.path_conditioned_mlp_metric_derivation_v1"
EXPECTED = {
    "prior": "637389700a6888c1317cef11ce1e344380b25e89a2eb44abc176729cddc2191c",
    "fresh_result": "7f0189ed40879285bb9d17f167f025e0fce345ae0402959d851ce189373e4d78",
    "builder": "9ba3fb077e1019a77b51415a64f5f0cda1e2ff93d82a88d47968dcdf5dac66ee",
    "score_library": "8614522e13fa7aef77429f40e89971077a848e6d433ccadf18ec4ef282463d9b",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
MLP_COMPONENTS = tuple(positioned.Component("mlp", layer) for layer in range(9))
BASELINES = {"has": 0.5, "is": 0.38333333333333336}
FORWARDS, EVALUATIONS, BACKWARDS = 12, 184, 2
FINITE_DIFFERENCE_EPSILON = 1e-2
FINITE_DIFFERENCE_RELATIVE_TOLERANCE = 0.03


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def spearman(left, right):
    return statistics.correlation(ranks(left), ranks(right))


def margins(output):
    return [float(answer) - float(foil) for answer, foil in output.answer_foil]


def gradient_forward(backend, batch):
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    components = {}
    try:
        with torch.enable_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x = x.detach().requires_grad_(True)
            x0, v1 = x, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                mlp = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if layer < 9:
                    mlp.retain_grad()
                    components[layer] = mlp
                x = x + mlp
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)
            row_margins = torch.stack(tuple(
                logits[index, length - 1, batch.answer_ids[index]]
                - logits[index, length - 1, batch.foil_ids[index]]
                for index, length in enumerate(lengths)))
            row_margins.sum().backward()
        values = tuple((float(logits[index, length - 1, batch.answer_ids[index]].detach()),
                        float(logits[index, length - 1, batch.foil_ids[index]].detach()))
                       for index, length in enumerate(lengths))
        base = {layer: components[layer].detach().clone() for layer in range(9)}
        gradients = {layer: components[layer].grad.detach().clone() for layer in range(9)}
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.zero_grad(set_to_none=True)
    return producer.BatchOutput(values, {}), base, gradients


def scaled_patch(backend, batch, layer, base, donor, banks, alpha):
    def hook(_module, _arguments, output):
        changed = output.clone()
        for row, bank in enumerate(banks):
            for position in bank:
                changed[row, position] = (changed[row, position].float() + alpha * (
                    donor[row, position].float() - base[row, position].float())).to(output.dtype)
        return changed
    handle = backend.model.transformer.h[layer].mlp.register_forward_hook(hook)
    try:
        return backend.native(batch, capture=False)
    finally:
        handle.remove()


def validate_static():
    paths = {"prior": PRIOR, "fresh_result": FRESH_RESULT, "builder": BUILDER,
             "score_library": SCORE_LIBRARY, "positioned": POSITIONED, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, development data, or implementation hash changed")
    prior, result = [json.loads(path.read_text()) for path in (PRIOR, FRESH_RESULT)]
    rows_by_bank = fresh.build_rows_by_bank()
    fresh.validate_rows_by_bank(rows_by_bank)
    if (prior.get("candidate_id") != CANDIDATE_ID or result.get("terminal") != "null"
            or result["correlations"] != BASELINES):
        raise ExperimentError("development causal result or baseline changed")
    rows = {"has": [row for row in rows_by_bank["has_had"] if row["transform_id"] == "A1"],
            "is": [row for row in rows_by_bank["is_was"] if row["transform_id"] == "A1"]}
    return result, rows


def main():
    causal, rows_by_task = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "model_forwards": FORWARDS,
              "example_evaluations": EVALUATIONS, "transformer_backwards": BACKWARDS,
              "task_layer_scores": 18, "finite_difference_layers": {"has": 4, "is": 4},
              "fitted_scalars": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    forwards = evaluations = backwards = 0
    scores, correlations, rankings, finite_difference = {}, {}, {}, {}
    all_finite, all_capable = True, True
    for task in ("has", "is"):
        rows = rows_by_task[task]
        # Capability-only selection reproduces the development panel without
        # consulting any component outcome.
        native = {}
        for side in ("base", "donor"):
            batch = das._batch(backend, rows, side=side)
            native[side] = backend.native(batch, capture=False)
            forwards += 1
            evaluations += len(rows)
        selected = [row for index, row in enumerate(rows)
                    if margins(native["base"])[index] > 0 and margins(native["donor"])[index] > 0]
        expected_count = causal["capability"][f"{task}_A1"]["jointly_capable"]
        all_capable = all_capable and len(selected) == expected_count
        base_batch, donor_batch = das._batch(backend, selected, side="base"), das._batch(backend, selected, side="donor")
        base_output, base_components, gradients = gradient_forward(backend, base_batch)
        donor_output, donor_components = positioned.capture_full_components(
            backend, donor_batch, MLP_COMPONENTS)
        forwards += 2
        evaluations += 2 * len(selected)
        backwards += 1
        all_capable = all_capable and all(value > 0 for value in margins(base_output) + margins(donor_output))
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        task_scores = []
        for layer in range(9):
            donor = donor_components[MLP_COMPONENTS[layer].site_id]
            effects = path_score.directional_effects(
                gradients[layer], base_components[layer], donor, banks)
            value = float(effects.abs().mean())
            task_scores.append({"label": f"MLP{layer}", "layer": layer,
                                "mean_absolute_directional_effect": value,
                                "mean_signed_directional_effect": float(effects.mean())})
            all_finite = all_finite and bool(backend.torch.isfinite(effects).all()) and math.isfinite(value)
        scores[task] = task_scores
        causal_summaries = causal["causal_summaries"][task]
        labels = sorted(causal_summaries)
        correlations[task] = {
            "activation_conditioned_baseline": BASELINES[task],
            "path_conditioned": spearman(
                [next(row["mean_absolute_directional_effect"] for row in task_scores
                      if row["label"] == label) for label in labels],
                [causal_summaries[label]["mean_absolute_recovery"] for label in labels])}
        rankings[task] = [row["label"] for row in sorted(task_scores,
            key=lambda row: (-row["mean_absolute_directional_effect"], row["label"]))]

        plus = scaled_patch(backend, base_batch, 4, base_components[4],
                            donor_components[MLP_COMPONENTS[4].site_id], banks,
                            FINITE_DIFFERENCE_EPSILON)
        minus = scaled_patch(backend, base_batch, 4, base_components[4],
                             donor_components[MLP_COMPONENTS[4].site_id], banks,
                             -FINITE_DIFFERENCE_EPSILON)
        forwards += 2
        evaluations += 2 * len(selected)
        empirical = backend.torch.tensor([(a - b) / (2 * FINITE_DIFFERENCE_EPSILON)
            for a, b in zip(margins(plus), margins(minus))], device=backend.device)
        exact = path_score.directional_effects(
            gradients[4], base_components[4], donor_components[MLP_COMPONENTS[4].site_id], banks)
        max_abs = float((empirical - exact).abs().max())
        relative = max_abs / max(float(exact.abs().max()), 1e-6)
        finite_difference[task] = {"layer": 4, "epsilon": FINITE_DIFFERENCE_EPSILON,
                                   "max_absolute_error": max_abs,
                                   "relative_max_error": relative,
                                   "tolerance": FINITE_DIFFERENCE_RELATIVE_TOLERANCE,
                                   "passed": relative <= FINITE_DIFFERENCE_RELATIVE_TOLERANCE}

    pred_a = bool(all_capable and all_finite and all(item["passed"] for item in finite_difference.values()))
    pred_b = all(correlations[task]["path_conditioned"] > BASELINES[task] for task in ("has", "is"))
    pred_c = all(correlations[task]["path_conditioned"] > 0.65 and "MLP4" in rankings[task][:3]
                 for task in ("has", "is"))
    pred_d = forwards == FORWARDS and evaluations == EVALUATIONS and backwards == BACKWARDS
    predictions = {
        "pred_a_exact_gradient_response_pairing_and_finiteness": pred_a,
        "pred_b_path_conditioning_improves_both_development_correlations": pred_b,
        "pred_c_path_conditioning_is_strong_enough_for_fresh_test": pred_c,
        "pred_d_zero_fit_and_declared_development_scope": pred_d,
    }
    terminal = "invalid" if not pred_a or not pred_d else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_path_conditioned_mlp_metric_derivation_result_v1",
        "candidate_id": CANDIDATE_ID, "scope": "development_metric_derivation_only",
        "execution_policy": "managed_queue_only", "started_utc": started_utc,
        "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun, "scores": scores,
        "rankings": rankings, "correlations": correlations,
        "finite_difference_validation": finite_difference, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "transformer_backwards": backwards, "fitted_scalars": 0,
                  "model_updates": 0}, "terminal": terminal,
        "reason": "path_conditioned_metric_graduates_to_second_fresh_test" if terminal == "screen"
                  else "first_order_path_conditioning_does_not_resolve_mlp_ranking" if terminal == "null"
                  else "gradient_response_pairing_finiteness_or_accounting_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "rankings", "correlations",
        "finite_difference_validation", "predictions", "price", "terminal", "reason")},
        sort_keys=True))


if __name__ == "__main__":
    main()
