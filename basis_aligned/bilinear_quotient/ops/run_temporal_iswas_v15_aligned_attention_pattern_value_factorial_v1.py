#!/usr/bin/env python3
"""Exact P/V/interaction factorial for the aligned v15 attention core."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_reconstruction_closure_finiteness_and_exact_price pred_b_selective_attention_operation_subset_exists pred_c_selected_operation_subset_confirms_on_a2 pred_d_selected_operation_subset_is_proper pred_e_factor_roles_are_nonexchangeable pred_f_selected_operation_tracks_final_residual
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import attention_source_destination_eval as attention_eval
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1 as greedy


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1.json"
LINEAR_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_crossfit_head_response_task_p_complement_v1_result.json"
LINEAR_RUNNER = ROOT / "ops/run_temporal_iswas_v15_crossfit_head_response_task_p_complement_v1.py"
LATTICE_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_aligned_control_attention_lattice_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_aligned_attention_pattern_value_factorial_v1"
ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
FACTORS = attention_eval.RESPONSE_FACTORS
LAYERS = {8: (1,), 9: (1, 4), 11: (3,)}
COMPLETE_LAYER, COMPLETE_HEADS = 15, tuple(range(9))
TARGET_PANELS, CONTROL_PANELS = ("A1", "A2"), ("P", "C")
EXACT_FORWARDS = 17
EXPECTED = {
    "prior": "1e04936a82702b00aac788f0a2e4cbdd347b8244305eb3b97cd49237d99bba84",
    "linear_result": "053eca39027e5753c2e26812044c85f5a6116e4ee7e36d0d03c9c73a4a512f28",
    "linear_runner": "449d3de7bebda460445997c60438095d6555222747ef33b27b70f4559bdfe9d5",
    "lattice_result": "8eee3c031f9e7bae8bac578bd6fdd368f19fc5f027e1f184d58c7f325c4fee43",
    "builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def factor_subsets():
    return tuple(
        subset for size in range(len(FACTORS) + 1) for subset in itertools.combinations(FACTORS, size)
    )


def arm_name(subset):
    return "none" if not subset else "+".join(subset)


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def numeric_max_abs_difference(left, right) -> float:
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return max((numeric_max_abs_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def main() -> None:
    paths = {
        "prior": PRIOR, "linear_result": LINEAR_RESULT, "linear_runner": LINEAR_RUNNER,
        "lattice_result": LATTICE_RESULT, "builder": BUILDER, "attention_library": ATTENTION_LIBRARY,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    linear = json.loads(LINEAR_RESULT.read_text())
    lattice = json.loads(LATTICE_RESULT.read_text())
    rows = fresh.build_rows()
    alignment = derive_full_sequence_alignment_contract(
        rows, required_panels=TARGET_PANELS + CONTROL_PANELS
    )
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and fresh.validate_rows(rows) == ROWS_SHA256
        and alignment["panel_counts"] == {panel: 16 for panel in TARGET_PANELS + CONTROL_PANELS}
        and linear.get("terminal") == "linear_response_split_insufficient"
        and lattice.get("terminal") == "aligned_no_selective_attention_subset"
        and len(factor_subsets()) == 8 and len(rows) == len({row["row_id"] for row in rows}) == 64
    )
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "factorized_layers": {str(layer): list(heads) for layer, heads in LAYERS.items()},
        "complete_branch": "attn:15", "factor_subsets": 8,
        "model_forwards_exact": EXACT_FORWARDS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    native, forwards = backend.native, 0

    def counted(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        return native(*args, **kwargs)

    backend.native = counted
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    captures = {"base": {}, "donor": {}}
    outputs = {"base": {}, "donor": {}}
    identity_error, reconstruction_error = 0.0, 0.0
    for side, batch in (("base", base_batch), ("donor", donor_batch)):
        for layer in tuple(LAYERS) + (COMPLETE_LAYER,):
            output, capture = attention_eval.capture_layer_attention(backend, batch, layer)
            outputs[side][layer], captures[side][layer] = output, capture
            reconstruction_error = max(reconstruction_error, float(capture["reconstruction_max_abs"]))
        reference = outputs[side][COMPLETE_LAYER].answer_foil
        for layer in LAYERS:
            identity_error = max(identity_error, max(
                abs(float(left) - float(right))
                for pair_left, pair_right in zip(reference, outputs[side][layer].answer_foil)
                for left, right in zip(pair_left, pair_right)
            ))

    factors, factor_closure_error = {}, 0.0
    for layer, heads in LAYERS.items():
        base, donor = captures["base"][layer], captures["donor"][layer]
        pattern_delta = donor["pattern"].float() - base["pattern"].float()
        value_delta = donor["value"].float() - base["value"].float()
        factors[layer] = {
            "pattern_on_base_value": torch.einsum("bhqk,bkhd->bqhd", pattern_delta, base["value"].float()),
            "base_pattern_on_value_change": torch.einsum("bhqk,bkhd->bqhd", base["pattern"].float(), value_delta),
            "pattern_value_interaction": torch.einsum("bhqk,bkhd->bqhd", pattern_delta, value_delta),
        }
        reconstructed_delta = sum(
            (factors[layer][factor] for factor in FACTORS),
            torch.zeros_like(base["head_output"], dtype=torch.float32),
        )
        actual_delta = donor["head_output"].float() - base["head_output"].float()
        for index, query in enumerate(base_batch.semantic_positions):
            stop = int(query) + 1
            for head in heads:
                factor_closure_error = max(
                    factor_closure_error,
                    float((reconstructed_delta[index, :stop, head] - actual_delta[index, :stop, head]).abs().max()),
                )

    def specs(subset, source="donor"):
        result = []
        for layer, heads in LAYERS.items():
            base = captures["base"][layer]
            delta = sum(
                (factors[layer][factor] for factor in subset),
                torch.zeros_like(base["head_output"], dtype=torch.float32),
            )
            changed = base["head_output"].float() + (delta if source == "donor" else 0.0)
            result.append({"layer": layer, "base_capture": base,
                           "changed_capture": {"head_output": changed}, "selected_heads": heads})
        base15 = captures["base"][COMPLETE_LAYER]
        changed15 = captures[source][COMPLETE_LAYER]["head_output"]
        result.append({"layer": COMPLETE_LAYER, "base_capture": base15,
                       "changed_capture": {"head_output": changed15}, "selected_heads": COMPLETE_HEADS})
        return result

    base_output, donor_output = outputs["base"][COMPLETE_LAYER], outputs["donor"][COMPLETE_LAYER]
    base_state = greedy.module_impl.states(torch, backend, base_output, rows)
    donor_state = greedy.module_impl.states(torch, backend, donor_output, rows)
    base_logits = das.head_logits(backend, base_state).float()
    donor_logits = das.head_logits(backend, donor_state).float()
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    all_index = torch.arange(len(rows), device=backend.device)
    panel_indices = {panel: torch.as_tensor(
        [index for index, row in enumerate(rows) if row["transform_id"] == panel], device=backend.device
    ) for panel in TARGET_PANELS + CONTROL_PANELS}

    def margins(logits):
        return logits[all_index, answer] - logits[all_index, foil]

    base_margin, donor_margin = margins(base_logits), margins(donor_logits)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def control_report(logits, panel):
        index = panel_indices[panel]
        log_base, log_patch = F.log_softmax(base_logits[index], -1), F.log_softmax(logits[index], -1)
        kl = (log_base.exp() * (log_base - log_patch)).sum(-1)
        flips = base_logits[index].argmax(-1) != logits[index].argmax(-1)
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        return {"median_kl": float(kl.median()), "max_kl": float(kl.max()),
                "top1_flip_count": int(flips.sum()), "top1_flip_fraction": float(flips.float().mean()),
                "flipped_row_ids": [panel_rows[i]["row_id"] for i in range(len(panel_rows)) if bool(flips[i])]}

    def report(output):
        state = greedy.module_impl.states(torch, backend, output, rows)
        logits = das.head_logits(backend, state).float()
        delta_margin, delta_state = margins(logits) - base_margin, state - base_state
        targets = {}
        for panel in TARGET_PANELS:
            index = panel_indices[panel]
            behavior = greedy.module_impl.vector_metrics(torch, delta_margin[index], target_margin[index])
            behavior["direction_fraction"] = float(((delta_margin[index] * target_margin[index]) > 0).float().mean())
            targets[panel] = {"behavior": behavior,
                "final_residual": greedy.module_impl.vector_metrics(torch, delta_state[index], target_state[index])}
        return {"targets": targets, "controls": {panel: control_report(logits, panel) for panel in CONTROL_PANELS}}

    def eligible(value):
        return bool(value["targets"]["A1"]["behavior"]["signed_projection"] >= .75
                    and value["targets"]["A1"]["behavior"]["direction_fraction"] >= .875
                    and all(value["controls"][panel]["top1_flip_count"] == 0 for panel in CONTROL_PANELS)
                    and max(value["controls"][panel]["median_kl"] for panel in CONTROL_PANELS) <= .02)

    self_output = attention_eval.intervene_ordered_head_output_deltas(backend, base_batch, specs((), source="base"))
    self_state = greedy.module_impl.states(torch, backend, self_output, rows)
    self_logits = das.head_logits(backend, self_state).float()
    self_error = max(float((self_state - base_state).abs().max()), float((self_logits - base_logits).abs().max()))
    reports = {}
    for subset in factor_subsets():
        value = report(attention_eval.intervene_ordered_head_output_deltas(
            backend, base_batch, specs(subset, source="donor")
        ))
        reports[arm_name(subset)] = {"factors": list(subset), "eligible": eligible(value), "report": value}

    full_name = arm_name(FACTORS)
    full_parent = lattice["reports"]["31"]["report"]
    full_parent_replay_error = numeric_max_abs_difference(reports[full_name]["report"], full_parent)
    eligible_names = [name for name, value in reports.items() if value["eligible"]]
    selected_name = min(eligible_names, key=lambda name: (
        len(reports[name]["factors"]),
        -reports[name]["report"]["targets"]["A1"]["behavior"]["signed_projection"],
        tuple(FACTORS.index(factor) for factor in reports[name]["factors"]),
    )) if eligible_names else None
    selected = reports[selected_name] if selected_name is not None else None
    pred_a = bool(authority_ok and identity_error <= 1e-4 and reconstruction_error <= 5e-4
                  and factor_closure_error <= 1e-4 and self_error <= 1e-4
                  and full_parent_replay_error <= 1e-4 and finite(reports) and forwards == EXACT_FORWARDS)
    pred_b = selected is not None
    pred_c = bool(selected and selected["report"]["targets"]["A2"]["behavior"]["signed_projection"] >= .75
                  and selected["report"]["targets"]["A2"]["behavior"]["direction_fraction"] >= .875)
    pred_d = bool(selected and len(selected["factors"]) <= 2)
    pred_e = False
    if selected:
        size = len(selected["factors"])
        selected_flips = sum(selected["report"]["controls"][panel]["top1_flip_count"] for panel in CONTROL_PANELS)
        for name, alternative in reports.items():
            if name == selected_name or len(alternative["factors"]) != size:
                continue
            target_advantage = all(
                selected["report"]["targets"][panel]["behavior"]["signed_projection"]
                - alternative["report"]["targets"][panel]["behavior"]["signed_projection"] >= .10
                for panel in TARGET_PANELS
            )
            alternative_flips = sum(alternative["report"]["controls"][panel]["top1_flip_count"] for panel in CONTROL_PANELS)
            if target_advantage or alternative_flips - selected_flips >= 2:
                pred_e = True
                break
    pred_f = bool(selected and all(
        selected["report"]["targets"][panel]["final_residual"]["signed_projection"] >= .50
        for panel in TARGET_PANELS
    ))
    predictions = {
        "pred_a_authority_alignment_reconstruction_closure_finiteness_and_exact_price": pred_a,
        "pred_b_selective_attention_operation_subset_exists": pred_b,
        "pred_c_selected_operation_subset_confirms_on_a2": pred_c,
        "pred_d_selected_operation_subset_is_proper": pred_d,
        "pred_e_factor_roles_are_nonexchangeable": pred_e,
        "pred_f_selected_operation_tracks_final_residual": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "selective_pattern_value_operation_circuit"
    elif all(predictions[key] for key in list(predictions)[:5]) and not pred_f:
        terminal = "selective_pattern_value_behavior_only"
    elif pred_b and not pred_c:
        terminal = "pattern_value_construction_failure"
    elif pred_b and pred_c and not pred_d:
        terminal = "all_attention_factors_required"
    elif pred_b and pred_c and pred_d and not pred_e:
        terminal = "factor_roles_exchangeable"
    elif not pred_b:
        terminal = "no_selective_attention_operation_subset"
    else:
        terminal = "partial"
    result = {
        "schema": "temporal_iswas_v15_aligned_attention_pattern_value_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "rows_sha256": ROWS_SHA256, "alignment_contract": alignment,
        "factors": list(FACTORS), "factorized_layers": {str(layer): list(heads) for layer, heads in LAYERS.items()},
        "complete_branch": "attn:15", "instrument": {
            "native_capture_identity_max_abs_error": identity_error,
            "attention_reconstruction_max_abs_error": reconstruction_error,
            "factor_closure_max_abs_error": factor_closure_error,
            "base_self_max_abs_error": self_error,
            "full_parent_replay_max_abs_error": full_parent_replay_error,
        },
        "reports": reports, "eligible_arms": eligible_names, "selected_arm": selected_name,
        "selected": selected, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_exact": EXACT_FORWARDS, "model_forwards_observed": forwards,
                  "example_evaluations": forwards * len(rows), "fit_updates": 0,
                  "model_updates": 0, "transformer_backwards": 0}, "dryrun": dryrun,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "instrument", "eligible_arms", "selected_arm", "selected", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
