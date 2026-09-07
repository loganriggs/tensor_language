#!/usr/bin/env python3
"""Exact head-specific value-factor lattice for the aligned v15 attention core."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_absolute_clamp_closure_replay_finiteness_and_exact_price pred_b_selective_head_specific_value_program_exists pred_c_selected_program_confirms_on_a2 pred_d_value_allocation_is_proper_and_nonuniform pred_e_selected_program_tracks_parent_residual pred_f_selected_program_is_necessary
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from absolute_head_response_clamp_contract import build_absolute_head_response_plan
from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_head_specific_value_factor_lattice_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1_retry1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
CLAMP_CONTRACT = ROOT / "ops/absolute_head_response_clamp_contract.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_head_specific_value_factor_lattice_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_head_specific_value_factor_lattice_v1"
EXPECTED = {
    "prior": "ed1000535341ba595bdf98eac0a9cf1d73ba7bafb7a952687e21a76d22e8e87d",
    "parent_result": "6ac7bfe60cfbbeada55a509823f8a10b85f3624975860bf6b7d73e552d5160de",
    "parent_runner": "3424f394930cd9a0c25f05b5620f17fb37cfb852ce4e07154c8f4a8f979d64ab",
    "builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "clamp_contract": "b16b86b23d50b1284eecaa01855301d50ced82e2112bf44167b9596c0b7fd0f3",
}
VALUE_FACTOR = "base_pattern_on_value_change"
PATTERN_FACTOR = "pattern_on_base_value"
INTERACTION_FACTOR = "pattern_value_interaction"
VALUE_BITS = ((8, 1), (9, 1), (9, 4), (11, 3))
BITS = tuple(f"value_L{layer}H{head}" for layer, head in VALUE_BITS) + (
    "pattern_all_four_heads", "interaction_all_four_heads",
)
EXACT_FORWARDS = 73


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_name(mask: int) -> str:
    names = [name for bit, name in enumerate(BITS) if mask & (1 << bit)]
    return "none" if not names else "+".join(names)


def main() -> None:
    paths = {
        "prior": PRIOR, "parent_result": PARENT_RESULT, "parent_runner": PARENT_RUNNER,
        "builder": BUILDER, "attention_library": ATTENTION_LIBRARY,
        "clamp_contract": CLAMP_CONTRACT,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    parent_result = json.loads(PARENT_RESULT.read_text())
    rows = parent.fresh.build_rows()
    alignment = derive_full_sequence_alignment_contract(
        rows, required_panels=parent.TARGET_PANELS + parent.CONTROL_PANELS
    )
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and parent.fresh.validate_rows(rows) == parent.ROWS_SHA256
        and alignment["panel_counts"] == {panel: 16 for panel in parent.TARGET_PANELS + parent.CONTROL_PANELS}
        and parent_result.get("terminal") == "no_selective_attention_operation_subset"
        and parent_result["predictions"]["pred_a_authority_alignment_reconstruction_closure_finiteness_and_exact_price"]
        and len(BITS) == 6 and len(rows) == len({row["row_id"] for row in rows}) == 64
    )
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "bits": list(BITS), "subset_arms": 64, "complete_branch": "attn:15",
        "model_forwards_exact": EXACT_FORWARDS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
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
    captures, outputs = {"base": {}, "donor": {}}, {"base": {}, "donor": {}}
    identity_error, reconstruction_error = 0.0, 0.0
    for side, batch in (("base", base_batch), ("donor", donor_batch)):
        for layer in tuple(parent.LAYERS) + (parent.COMPLETE_LAYER,):
            output, capture = parent.attention_eval.capture_layer_attention(
                backend, batch, layer,
                call=(lambda batch=batch: backend.native(batch, capture=True))
                if layer == parent.COMPLETE_LAYER else None,
            )
            outputs[side][layer], captures[side][layer] = output, capture
            reconstruction_error = max(reconstruction_error, float(capture["reconstruction_max_abs"]))
        reference = outputs[side][parent.COMPLETE_LAYER].answer_foil
        for layer in parent.LAYERS:
            identity_error = max(identity_error, max(
                abs(float(left) - float(right))
                for left_pair, right_pair in zip(reference, outputs[side][layer].answer_foil)
                for left, right in zip(left_pair, right_pair)
            ))

    factors, factor_closure_error = {}, 0.0
    for layer, heads in parent.LAYERS.items():
        base, donor = captures["base"][layer], captures["donor"][layer]
        dp = donor["pattern"].float() - base["pattern"].float()
        dv = donor["value"].float() - base["value"].float()
        factors[layer] = {
            PATTERN_FACTOR: torch.einsum("bhqk,bkhd->bqhd", dp, base["value"].float()),
            VALUE_FACTOR: torch.einsum("bhqk,bkhd->bqhd", base["pattern"].float(), dv),
            INTERACTION_FACTOR: torch.einsum("bhqk,bkhd->bqhd", dp, dv),
        }
        closure = sum((factors[layer][factor] for factor in parent.FACTORS),
                      torch.zeros_like(base["head_output"], dtype=torch.float32))
        actual = donor["head_output"].float() - base["head_output"].float()
        for index, query in enumerate(base_batch.semantic_positions):
            for head in heads:
                factor_closure_error = max(factor_closure_error, float(
                    (closure[index, :int(query) + 1, head]
                     - actual[index, :int(query) + 1, head]).abs().max()
                ))

    def specs(mask: int, source="donor"):
        result = []
        pattern_on = bool(mask & (1 << 4))
        interaction_on = bool(mask & (1 << 5))
        for layer, heads in parent.LAYERS.items():
            base = captures["base"][layer]
            delta = torch.zeros_like(base["head_output"], dtype=torch.float32)
            if pattern_on:
                delta[:, :, list(heads)] += factors[layer][PATTERN_FACTOR][:, :, list(heads)]
            if interaction_on:
                delta[:, :, list(heads)] += factors[layer][INTERACTION_FACTOR][:, :, list(heads)]
            for bit, (value_layer, head) in enumerate(VALUE_BITS):
                if layer == value_layer and mask & (1 << bit):
                    delta[:, :, head] += factors[layer][VALUE_FACTOR][:, :, head]
            changed = base["head_output"].float() + (delta if source == "donor" else 0.0)
            result.append({"layer": layer, "base_capture": base,
                           "changed_capture": {"head_output": changed}, "selected_heads": heads})
        base15 = captures["base"][parent.COMPLETE_LAYER]
        result.append({
            "layer": parent.COMPLETE_LAYER, "base_capture": base15,
            "changed_capture": {"head_output": captures[source][parent.COMPLETE_LAYER]["head_output"]},
            "selected_heads": parent.COMPLETE_HEADS,
        })
        return result

    def intervene(mask: int, source="donor"):
        cache, support = build_absolute_head_response_plan(
            specs(mask, source), complete_head_sites={parent.COMPLETE_LAYER: parent.COMPLETE_HEADS}
        )
        return parent.greedy.run_patch(backend, base_batch, cache, support)

    base_output, donor_output = outputs["base"][parent.COMPLETE_LAYER], outputs["donor"][parent.COMPLETE_LAYER]
    base_state = parent.greedy.module_impl.states(torch, backend, base_output, rows)
    donor_state = parent.greedy.module_impl.states(torch, backend, donor_output, rows)
    base_logits = das.head_logits(backend, base_state).float()
    donor_logits = das.head_logits(backend, donor_state).float()
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    all_index = torch.arange(len(rows), device=backend.device)
    panel_indices = {panel: torch.as_tensor(
        [index for index, row in enumerate(rows) if row["transform_id"] == panel], device=backend.device
    ) for panel in parent.TARGET_PANELS + parent.CONTROL_PANELS}

    def margins(logits):
        return logits[all_index, answer] - logits[all_index, foil]

    base_margin, donor_margin = margins(base_logits), margins(donor_logits)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def report(output):
        state = parent.greedy.module_impl.states(torch, backend, output, rows)
        logits = das.head_logits(backend, state).float()
        delta_margin, delta_state = margins(logits) - base_margin, state - base_state
        targets = {}
        for panel in parent.TARGET_PANELS:
            index = panel_indices[panel]
            behavior = parent.greedy.module_impl.vector_metrics(torch, delta_margin[index], target_margin[index])
            behavior["direction_fraction"] = float(((delta_margin[index] * target_margin[index]) > 0).float().mean())
            targets[panel] = {
                "behavior": behavior,
                "final_residual": parent.greedy.module_impl.vector_metrics(
                    torch, delta_state[index], target_state[index]
                ),
            }
        controls = {}
        for panel in parent.CONTROL_PANELS:
            index = panel_indices[panel]
            log_base, log_patch = F.log_softmax(base_logits[index], -1), F.log_softmax(logits[index], -1)
            kl = (log_base.exp() * (log_base - log_patch)).sum(-1)
            flips = base_logits[index].argmax(-1) != logits[index].argmax(-1)
            panel_rows = [row for row in rows if row["transform_id"] == panel]
            controls[panel] = {
                "median_kl": float(kl.median()), "max_kl": float(kl.max()),
                "top1_flip_count": int(flips.sum()), "top1_flip_fraction": float(flips.float().mean()),
                "flipped_row_ids": [panel_rows[i]["row_id"] for i in range(len(panel_rows)) if bool(flips[i])],
            }
        return {"targets": targets, "controls": controls}

    def eligible(value):
        a1 = value["targets"]["A1"]["behavior"]
        return bool(a1["signed_projection"] >= .75 and a1["direction_fraction"] >= .875
                    and all(value["controls"][panel]["top1_flip_count"] == 0 for panel in parent.CONTROL_PANELS)
                    and max(value["controls"][panel]["median_kl"] for panel in parent.CONTROL_PANELS) <= .02)

    self_output = intervene(0, source="base")
    self_state = parent.greedy.module_impl.states(torch, backend, self_output, rows)
    self_logits = das.head_logits(backend, self_state).float()
    self_error = max(float((self_state - base_state).abs().max()), float((self_logits - base_logits).abs().max()))
    reports = {}
    for mask in range(64):
        value = report(intervene(mask))
        reports[str(mask)] = {"mask": mask, "bits": [BITS[bit] for bit in range(6) if mask & (1 << bit)],
                              "eligible": eligible(value), "report": value}

    full_parent = parent_result["reports"][parent.arm_name(parent.FACTORS)]["report"]
    full_replay = parent.replay_comparison(reports["63"]["report"], full_parent)
    eligible_masks = [mask for mask in range(64) if reports[str(mask)]["eligible"]]
    selected_mask = min(eligible_masks, key=lambda mask: (
        mask.bit_count(), -reports[str(mask)]["report"]["targets"]["A1"]["behavior"]["signed_projection"], mask
    )) if eligible_masks else None
    selected = reports[str(selected_mask)] if selected_mask is not None else None
    value_bits_selected = [bit for bit in range(4) if selected_mask is not None and selected_mask & (1 << bit)]
    necessity = {}
    if selected is not None:
        selected_a1 = selected["report"]["targets"]["A1"]["behavior"]["signed_projection"]
        for bit in value_bits_selected:
            removed_mask = selected_mask & ~(1 << bit)
            removed = reports[str(removed_mask)]
            drop = selected_a1 - removed["report"]["targets"]["A1"]["behavior"]["signed_projection"]
            necessity[BITS[bit]] = {"removed_mask": removed_mask, "a1_projection_drop": drop,
                                    "removed_eligible": removed["eligible"],
                                    "necessary": bool(drop >= .05 or not removed["eligible"])}

    pred_a = bool(authority_ok and identity_error <= 1e-4 and reconstruction_error <= 5e-4
                  and factor_closure_error <= 1e-4 and self_error <= 1e-4
                  and full_replay["numeric_schema_match"] and full_replay["categorical_match"]
                  and full_replay["numeric_max_abs_error"] <= 1e-4
                  and parent.finite(reports) and forwards == EXACT_FORWARDS)
    pred_b = selected is not None
    pred_c = bool(selected and selected["report"]["targets"]["A2"]["behavior"]["signed_projection"] >= .75
                  and selected["report"]["targets"]["A2"]["behavior"]["direction_fraction"] >= .875)
    pred_d = bool(0 < len(value_bits_selected) < 4)
    pred_e = bool(selected and selected["report"]["targets"]["A1"]["final_residual"]["signed_projection"] >= .25
                  and selected["report"]["targets"]["A2"]["final_residual"]["signed_projection"] >= .32)
    pred_f = bool(value_bits_selected and all(value["necessary"] for value in necessity.values()))
    predictions = {
        "pred_a_authority_alignment_absolute_clamp_closure_replay_finiteness_and_exact_price": pred_a,
        "pred_b_selective_head_specific_value_program_exists": pred_b,
        "pred_c_selected_program_confirms_on_a2": pred_c,
        "pred_d_value_allocation_is_proper_and_nonuniform": pred_d,
        "pred_e_selected_program_tracks_parent_residual": pred_e,
        "pred_f_selected_program_is_necessary": pred_f,
    }
    if not pred_a: terminal = "invalid"
    elif all(predictions.values()): terminal = "selective_head_specific_value_operation_circuit"
    elif pred_b and pred_c and pred_d and not pred_e and pred_f: terminal = "selective_value_behavior_only"
    elif pred_b and not pred_c: terminal = "head_specific_value_construction_failure"
    elif pred_b and pred_c and not pred_d: terminal = "uniform_value_allocation_required"
    elif pred_b and pred_c and pred_d and pred_e and not pred_f: terminal = "value_program_not_necessary"
    elif not pred_b: terminal = "no_selective_head_specific_value_program"
    else: terminal = "partial"
    result = {
        "schema": "temporal_iswas_v15_head_specific_value_factor_lattice_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "rows_sha256": parent.ROWS_SHA256, "alignment_contract": alignment, "bits": list(BITS),
        "instrument": {"native_capture_identity_max_abs_error": identity_error,
                       "attention_reconstruction_max_abs_error": reconstruction_error,
                       "factor_closure_max_abs_error": factor_closure_error,
                       "base_self_max_abs_error": self_error,
                       "full_parent_replay": full_replay},
        "reports": reports, "eligible_masks": eligible_masks, "selected_mask": selected_mask,
        "selected": selected, "necessity": necessity, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_exact": EXACT_FORWARDS, "model_forwards_observed": forwards,
                  "example_evaluations": forwards * len(rows), "fit_updates": 0,
                  "model_updates": 0, "transformer_backwards": 0}, "dryrun": dryrun,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "instrument", "eligible_masks", "selected_mask", "selected", "necessity",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
