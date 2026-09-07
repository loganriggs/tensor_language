#!/usr/bin/env python3
"""Dual greedy search over twelve exact head-by-attention-operation factors."""

# BQGATE: EXPERIMENT pred_a_authority_absolute_clamp_factor_closure_self_full_replay_finiteness_and_price pred_b_dual_greedy_finds_selective_program pred_c_selected_program_confirms_on_a2 pred_d_selected_allocation_is_head_heterogeneous pred_e_selected_program_tracks_parent_residual pred_f_selected_components_are_necessary
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
import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as factor_parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_head_factor_dual_greedy_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_head_specific_value_factor_lattice_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_v15_head_specific_value_factor_lattice_v1.py"
CLAMP_CONTRACT = ROOT / "ops/absolute_head_response_clamp_contract.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_head_factor_dual_greedy_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_head_factor_dual_greedy_v1"
EXPECTED = {
    "prior": "99425c2bc5ddbe266bd966ee186bd603b98e417e4b9d259f0ab2b08770df9af6",
    "parent_result": "36f4e3a94f8eac9adfba233c14d78e0fe8a5efe87fa739ed0ae2205fd397c853",
    "parent_runner": "a09ad33fc4b3ce156533024c0ab5227655e462d663cf35b12bb50becc3ee11d6",
    "clamp_contract": "b16b86b23d50b1284eecaa01855301d50ced82e2112bf44167b9596c0b7fd0f3",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
}
HEADS = ((8, 1), (9, 1), (9, 4), (11, 3))
COMPONENTS = tuple(
    (layer, head, factor)
    for layer, head in HEADS
    for factor in factor_parent.FACTORS
)
COMPONENT_NAMES = tuple(f"L{layer}H{head}:{factor}" for layer, head, factor in COMPONENTS)
MAX_FORWARDS = 178


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligibility_distance(value) -> float:
    a1 = value["targets"]["A1"]["behavior"]
    controls = value["controls"]
    flips = sum(controls[panel]["top1_flip_count"] for panel in factor_parent.CONTROL_PANELS)
    max_kl = max(controls[panel]["median_kl"] for panel in factor_parent.CONTROL_PANELS)
    return (max(0.0, .75 - a1["signed_projection"]) / .75
            + max(0.0, .875 - a1["direction_fraction"]) / .875
            + flips + 10.0 * max(0.0, max_kl - .02))


def main() -> None:
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "parent_runner": PARENT_RUNNER,
             "clamp_contract": CLAMP_CONTRACT, "attention_library": ATTENTION_LIBRARY}
    observed = {name: sha256(path) for name, path in paths.items()}
    prior, parent_result = json.loads(PRIOR.read_text()), json.loads(PARENT_RESULT.read_text())
    rows = factor_parent.fresh.build_rows()
    alignment = derive_full_sequence_alignment_contract(
        rows, required_panels=factor_parent.TARGET_PANELS + factor_parent.CONTROL_PANELS
    )
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and factor_parent.fresh.validate_rows(rows) == factor_parent.ROWS_SHA256
        and alignment["panel_counts"] == {panel: 16 for panel in factor_parent.TARGET_PANELS + factor_parent.CONTROL_PANELS}
        and parent_result.get("terminal") == "no_selective_head_specific_value_program"
        and parent_result["predictions"]["pred_a_authority_alignment_absolute_clamp_closure_replay_finiteness_and_exact_price"]
        and len(COMPONENTS) == len(set(COMPONENTS)) == 12
    )
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "rows": len(rows),
              "components": list(COMPONENT_NAMES), "paths": ["target_first", "constrained"],
              "model_forwards_max": MAX_FORWARDS, "fit_updates": 0,
              "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
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
        for layer in tuple(factor_parent.LAYERS) + (factor_parent.COMPLETE_LAYER,):
            output, capture = factor_parent.attention_eval.capture_layer_attention(
                backend, batch, layer,
                call=(lambda batch=batch: backend.native(batch, capture=True))
                if layer == factor_parent.COMPLETE_LAYER else None)
            outputs[side][layer], captures[side][layer] = output, capture
            reconstruction_error = max(reconstruction_error, float(capture["reconstruction_max_abs"]))
        reference = outputs[side][factor_parent.COMPLETE_LAYER].answer_foil
        for layer in factor_parent.LAYERS:
            identity_error = max(identity_error, max(
                abs(float(a) - float(b))
                for pair_a, pair_b in zip(reference, outputs[side][layer].answer_foil)
                for a, b in zip(pair_a, pair_b)))

    factors, closure_error = {}, 0.0
    for layer, selected_heads in factor_parent.LAYERS.items():
        base, donor = captures["base"][layer], captures["donor"][layer]
        dp = donor["pattern"].float() - base["pattern"].float()
        dv = donor["value"].float() - base["value"].float()
        factors[layer] = {
            "pattern_on_base_value": torch.einsum("bhqk,bkhd->bqhd", dp, base["value"].float()),
            "base_pattern_on_value_change": torch.einsum("bhqk,bkhd->bqhd", base["pattern"].float(), dv),
            "pattern_value_interaction": torch.einsum("bhqk,bkhd->bqhd", dp, dv),
        }
        closure = sum((factors[layer][factor] for factor in factor_parent.FACTORS),
                      torch.zeros_like(base["head_output"], dtype=torch.float32))
        actual = donor["head_output"].float() - base["head_output"].float()
        for index, query in enumerate(base_batch.semantic_positions):
            for head in selected_heads:
                closure_error = max(closure_error, float(
                    (closure[index, :int(query)+1, head] - actual[index, :int(query)+1, head]).abs().max()))

    def specs(mask: int, source="donor"):
        result = []
        for layer, selected_heads in factor_parent.LAYERS.items():
            base = captures["base"][layer]
            delta = torch.zeros_like(base["head_output"], dtype=torch.float32)
            if source == "donor":
                for bit, (component_layer, head, factor) in enumerate(COMPONENTS):
                    if layer == component_layer and mask & (1 << bit):
                        delta[:, :, head] += factors[layer][factor][:, :, head]
            result.append({"layer": layer, "base_capture": base,
                           "changed_capture": {"head_output": base["head_output"].float() + delta},
                           "selected_heads": selected_heads})
        base15 = captures["base"][factor_parent.COMPLETE_LAYER]
        result.append({"layer": factor_parent.COMPLETE_LAYER, "base_capture": base15,
                       "changed_capture": {"head_output": captures[source][factor_parent.COMPLETE_LAYER]["head_output"]},
                       "selected_heads": factor_parent.COMPLETE_HEADS})
        return result

    def intervene(mask: int, source="donor"):
        cache, support = build_absolute_head_response_plan(
            specs(mask, source), complete_head_sites={factor_parent.COMPLETE_LAYER: factor_parent.COMPLETE_HEADS})
        return factor_parent.greedy.run_patch(backend, base_batch, cache, support)

    base_output, donor_output = outputs["base"][factor_parent.COMPLETE_LAYER], outputs["donor"][factor_parent.COMPLETE_LAYER]
    base_state = factor_parent.greedy.module_impl.states(torch, backend, base_output, rows)
    donor_state = factor_parent.greedy.module_impl.states(torch, backend, donor_output, rows)
    base_logits, donor_logits = das.head_logits(backend, base_state).float(), das.head_logits(backend, donor_state).float()
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    all_index = torch.arange(len(rows), device=backend.device)
    panel_indices = {panel: torch.as_tensor([i for i, row in enumerate(rows) if row["transform_id"] == panel], device=backend.device)
                     for panel in factor_parent.TARGET_PANELS + factor_parent.CONTROL_PANELS}
    margins = lambda logits: logits[all_index, answer] - logits[all_index, foil]
    base_margin, donor_margin = margins(base_logits), margins(donor_logits)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def report(output):
        state = factor_parent.greedy.module_impl.states(torch, backend, output, rows)
        logits = das.head_logits(backend, state).float()
        delta_margin, delta_state = margins(logits) - base_margin, state - base_state
        targets, controls = {}, {}
        for panel in factor_parent.TARGET_PANELS:
            index = panel_indices[panel]
            behavior = factor_parent.greedy.module_impl.vector_metrics(torch, delta_margin[index], target_margin[index])
            behavior["direction_fraction"] = float(((delta_margin[index] * target_margin[index]) > 0).float().mean())
            targets[panel] = {"behavior": behavior, "final_residual": factor_parent.greedy.module_impl.vector_metrics(
                torch, delta_state[index], target_state[index])}
        for panel in factor_parent.CONTROL_PANELS:
            index = panel_indices[panel]
            log_base, log_patch = F.log_softmax(base_logits[index], -1), F.log_softmax(logits[index], -1)
            kl = (log_base.exp() * (log_base - log_patch)).sum(-1)
            flips = base_logits[index].argmax(-1) != logits[index].argmax(-1)
            panel_rows = [row for row in rows if row["transform_id"] == panel]
            controls[panel] = {"median_kl": float(kl.median()), "max_kl": float(kl.max()),
                               "top1_flip_count": int(flips.sum()), "top1_flip_fraction": float(flips.float().mean()),
                               "flipped_row_ids": [panel_rows[i]["row_id"] for i in range(len(panel_rows)) if bool(flips[i])]}
        return {"targets": targets, "controls": controls}

    def eligible(value):
        a1, controls = value["targets"]["A1"]["behavior"], value["controls"]
        return bool(a1["signed_projection"] >= .75 and a1["direction_fraction"] >= .875
                    and all(controls[panel]["top1_flip_count"] == 0 for panel in factor_parent.CONTROL_PANELS)
                    and max(controls[panel]["median_kl"] for panel in factor_parent.CONTROL_PANELS) <= .02)

    self_output = intervene(0, source="base")
    self_state = factor_parent.greedy.module_impl.states(torch, backend, self_output, rows)
    self_logits = das.head_logits(backend, self_state).float()
    self_error = max(float((self_state-base_state).abs().max()), float((self_logits-base_logits).abs().max()))
    reports = {}

    def get(mask: int):
        key = str(mask)
        if key not in reports:
            value = report(intervene(mask))
            reports[key] = {"mask": mask, "components": [COMPONENT_NAMES[b] for b in range(12) if mask & (1 << b)],
                            "eligible": eligible(value), "eligibility_distance": eligibility_distance(value), "report": value}
        return reports[key]

    get(0)
    paths_taken = {}
    for path_name in ("target_first", "constrained"):
        current, path_masks = 0, [0]
        for _depth in range(1, 13):
            candidates = [current | (1 << bit) for bit in range(12) if not current & (1 << bit)]
            for mask in candidates: get(mask)
            if path_name == "target_first":
                current = min(candidates, key=lambda mask: (
                    -reports[str(mask)]["report"]["targets"]["A1"]["behavior"]["signed_projection"],
                    -reports[str(mask)]["report"]["targets"]["A1"]["behavior"]["direction_fraction"],
                    sum(reports[str(mask)]["report"]["controls"][p]["top1_flip_count"] for p in factor_parent.CONTROL_PANELS),
                    max(reports[str(mask)]["report"]["controls"][p]["median_kl"] for p in factor_parent.CONTROL_PANELS), mask))
            else:
                current = min(candidates, key=lambda mask: (
                    reports[str(mask)]["eligibility_distance"],
                    -reports[str(mask)]["report"]["targets"]["A1"]["behavior"]["signed_projection"], mask))
            path_masks.append(current)
        paths_taken[path_name] = path_masks

    eligible_masks = [int(mask) for mask, value in reports.items() if value["eligible"]]
    selected_mask = min(eligible_masks, key=lambda mask: (
        mask.bit_count(), -reports[str(mask)]["report"]["targets"]["A1"]["behavior"]["signed_projection"], mask
    )) if eligible_masks else None
    selected = reports[str(selected_mask)] if selected_mask is not None else None
    necessity = {}
    if selected is not None:
        selected_a1 = selected["report"]["targets"]["A1"]["behavior"]["signed_projection"]
        for bit in range(12):
            if not selected_mask & (1 << bit): continue
            removed_mask = selected_mask & ~(1 << bit)
            removed = get(removed_mask)
            drop = selected_a1 - removed["report"]["targets"]["A1"]["behavior"]["signed_projection"]
            necessity[COMPONENT_NAMES[bit]] = {"removed_mask": removed_mask, "a1_projection_drop": drop,
                                               "removed_eligible": removed["eligible"],
                                               "necessary": bool(drop >= .05 or not removed["eligible"])}

    full_mask = (1 << 12) - 1
    full_parent_report = parent_result["reports"]["63"]["report"]
    full_replay = factor_parent.replay_comparison(get(full_mask)["report"], full_parent_report)
    pred_a = bool(authority_ok and identity_error <= 1e-4 and reconstruction_error <= 5e-4
                  and closure_error <= 1e-4 and self_error <= 1e-4
                  and full_replay["numeric_schema_match"] and full_replay["categorical_match"]
                  and full_replay["numeric_max_abs_error"] <= 1e-4
                  and factor_parent.finite(reports) and forwards <= MAX_FORWARDS)
    pred_b = selected is not None
    pred_c = bool(selected and selected["report"]["targets"]["A2"]["behavior"]["signed_projection"] >= .75
                  and selected["report"]["targets"]["A2"]["behavior"]["direction_fraction"] >= .875)
    per_head = [] if selected_mask is None else [tuple(factor for bit, (l, h, factor) in enumerate(COMPONENTS)
                                                       if (l, h) == head and selected_mask & (1 << bit)) for head in HEADS]
    pred_d = bool(selected and len(set(per_head)) > 1)
    pred_e = bool(selected and selected["report"]["targets"]["A1"]["final_residual"]["signed_projection"] >= .25
                  and selected["report"]["targets"]["A2"]["final_residual"]["signed_projection"] >= .32)
    pred_f = bool(selected and necessity and all(value["necessary"] for value in necessity.values()))
    predictions = {
        "pred_a_authority_absolute_clamp_factor_closure_self_full_replay_finiteness_and_price": pred_a,
        "pred_b_dual_greedy_finds_selective_program": pred_b,
        "pred_c_selected_program_confirms_on_a2": pred_c,
        "pred_d_selected_allocation_is_head_heterogeneous": pred_d,
        "pred_e_selected_program_tracks_parent_residual": pred_e,
        "pred_f_selected_components_are_necessary": pred_f,
    }
    if not pred_a: terminal = "invalid"
    elif all(predictions.values()): terminal = "selective_head_factor_greedy_circuit"
    elif pred_b and pred_c and pred_d and not pred_e and pred_f: terminal = "selective_head_factor_behavior_only"
    elif pred_b and not pred_c: terminal = "head_factor_construction_failure"
    elif pred_b and pred_c and (not pred_d or not pred_f): terminal = "head_factor_uniform_or_nonnecessary"
    elif not pred_b: terminal = "no_selective_dual_greedy_program"
    else: terminal = "partial"
    result = {"schema": "temporal_iswas_v15_head_factor_dual_greedy_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
              "rows_sha256": factor_parent.ROWS_SHA256, "alignment_contract": alignment,
              "components": list(COMPONENT_NAMES), "paths": paths_taken,
              "instrument": {"native_capture_identity_max_abs_error": identity_error,
                             "attention_reconstruction_max_abs_error": reconstruction_error,
                             "factor_closure_max_abs_error": closure_error, "base_self_max_abs_error": self_error,
                             "full_parent_replay": full_replay},
              "reports": reports, "eligible_masks": eligible_masks, "selected_mask": selected_mask,
              "selected": selected, "selected_per_head_factors": per_head, "necessity": necessity,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards_observed": forwards, "model_forwards_max": MAX_FORWARDS,
                        "example_evaluations": forwards*len(rows), "unique_scientific_arms": len(reports),
                        "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}, "dryrun": dryrun}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("instrument", "paths", "eligible_masks", "selected_mask",
        "selected", "selected_per_head_factors", "necessity", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
