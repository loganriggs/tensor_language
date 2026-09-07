#!/usr/bin/env python3
"""Replay the frozen five-piece attention lattice on aligned v15 controls."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_capability_lattice_closure_finiteness_and_exact_price pred_b_target_sufficient_separately_selective_subset_exists pred_c_selected_subset_confirms_without_reselection pred_d_selected_subset_is_proper pred_e_every_selected_piece_is_necessary pred_f_final_residual_tracks_selected_behavior
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as original
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as aligned_builder
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1 as greedy


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_aligned_control_attention_lattice_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
ALIGNMENT_CONTRACT = ROOT / "ops/aligned_full_sequence_patch_contract.py"
CAPABILITY_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_v15_aligned_controls_capability_v1_result.json"
CAPABILITY_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_v15_aligned_controls_capability_v1.py"
OLD_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_low_collateral_attention_lattice_v1_result.json"
OLD_RUNNER = ROOT / "ops/run_temporal_iswas_v15_low_collateral_attention_lattice_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_aligned_control_attention_lattice_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_aligned_control_attention_lattice_v1"
ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
COMPONENTS = ("L8H1", "L9H1", "L9H4", "L11H3", "attn:15")
TARGET_PANELS = ("A1", "A2")
CONTROL_PANELS = ("P", "C")
EXACT_FORWARDS = 35
EXPECTED = {
    "prior": "3ef37101e9a5efba620839b6750e91b81635bb3dfb1855ac9b2182b507df9e54",
    "builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "alignment_contract": "7b4b04cec6f28b47b7d22b8e4be32421bbfae6a04f1adc7c0b890939c9e32760",
    "capability_result": "cf4a6b5a63e632f17a40639d5419fa84e5a0415d4bb100a219282fc55a672284",
    "capability_runner": "8f22a06148ca70e45281eddb8ed29ea951977364fd78361220294141221ca4ef",
    "old_result": "9777d6b43619a04f89eb40d3df3a8f8a60aca62d1430d40066e265236322dd9b",
    "old_runner": "6747d2d5cc85772f1f20dceb3ab9ac6cf665bfdccad590a1ba028e7cac19f192",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


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
    if left == right:
        return 0.0
    return math.inf


def main() -> None:
    paths = {
        "prior": PRIOR,
        "builder": BUILDER,
        "alignment_contract": ALIGNMENT_CONTRACT,
        "capability_result": CAPABILITY_RESULT,
        "capability_runner": CAPABILITY_RUNNER,
        "old_result": OLD_RESULT,
        "old_runner": OLD_RUNNER,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    capability = json.loads(CAPABILITY_RESULT.read_text())
    old_result = json.loads(OLD_RESULT.read_text())
    rows = aligned_builder.build_rows()
    original_rows = original.build_rows()
    old_targets = [row for row in original_rows if row["transform_id"] in TARGET_PANELS]
    new_targets = [row for row in rows if row["transform_id"] in TARGET_PANELS]
    alignment = derive_full_sequence_alignment_contract(
        rows, required_panels=TARGET_PANELS + CONTROL_PANELS
    )
    panel_counts = {
        panel: sum(row["transform_id"] == panel for row in rows)
        for panel in TARGET_PANELS + CONTROL_PANELS
    }
    capability_ok = bool(
        capability.get("terminal") == "manifest"
        and all(capability.get("predictions", {}).values())
        and capability.get("a1_a2_outcomes_opened") is False
        and {
            panel: len(capability["jointly_capable_row_ids"][panel])
            for panel in CONTROL_PANELS
        } == {"P": 16, "C": 16}
    )
    authority_ok = bool(
        observed == EXPECTED
        and prior.get("candidate_id") == CANDIDATE_ID
        and aligned_builder.validate_rows(rows) == ROWS_SHA256
        and new_targets == old_targets
        and panel_counts == {panel: 16 for panel in TARGET_PANELS + CONTROL_PANELS}
        and alignment["panel_counts"] == panel_counts
        and len(rows) == len({row["row_id"] for row in rows}) == 64
        and old_result.get("components") == list(COMPONENTS)
        and capability_ok
    )
    dryrun = {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "components": list(COMPONENTS),
        "rows": len(rows),
        "subset_arms": 32,
        "model_forwards_exact": EXACT_FORWARDS,
        "fit_updates": 0,
        "model_updates": 0,
        "transformer_backwards": 0,
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
    base_output, base_cache = greedy.capture(backend, base_batch)
    donor_output, donor_cache = greedy.capture(backend, donor_batch)
    cache_shapes_match = set(base_cache) == set(donor_cache) and all(
        base_cache[key].shape == donor_cache[key].shape for key in base_cache
    )
    if not cache_shapes_match:
        raise RuntimeError("base/donor response shapes changed")
    base_state = greedy.module_impl.states(torch, backend, base_output, rows)
    donor_state = greedy.module_impl.states(torch, backend, donor_output, rows)
    base_logits = das.head_logits(backend, base_state).float()
    donor_logits = das.head_logits(backend, donor_state).float()
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    all_index = torch.arange(len(rows), device=backend.device)
    panel_indices = {
        panel: torch.as_tensor(
            [index for index, row in enumerate(rows) if row["transform_id"] == panel],
            device=backend.device,
        )
        for panel in TARGET_PANELS + CONTROL_PANELS
    }

    def margins(logits):
        return logits[all_index, answer] - logits[all_index, foil]

    base_margin, donor_margin = margins(base_logits), margins(donor_logits)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def control_report(base_panel_logits, patched_panel_logits, panel_rows):
        log_base, log_patched = F.log_softmax(base_panel_logits, -1), F.log_softmax(patched_panel_logits, -1)
        kl = (log_base.exp() * (log_base - log_patched)).sum(-1)
        flips = base_panel_logits.argmax(-1) != patched_panel_logits.argmax(-1)
        return {
            "median_kl": float(kl.median()),
            "max_kl": float(kl.max()),
            "top1_flip_fraction": float(flips.float().mean()),
            "top1_flip_count": int(flips.sum()),
            "flipped_row_ids": [
                panel_rows[index]["row_id"] for index in range(len(panel_rows)) if bool(flips[index])
            ],
        }

    def report(output):
        state = greedy.module_impl.states(torch, backend, output, rows)
        logits = das.head_logits(backend, state).float()
        delta_margin, delta_state = margins(logits) - base_margin, state - base_state
        targets = {}
        for panel in TARGET_PANELS:
            index = panel_indices[panel]
            behavior = greedy.module_impl.vector_metrics(
                torch, delta_margin[index], target_margin[index]
            )
            behavior["direction_fraction"] = float(
                ((delta_margin[index] * target_margin[index]) > 0).float().mean()
            )
            targets[panel] = {
                "behavior": behavior,
                "final_residual": greedy.module_impl.vector_metrics(
                    torch, delta_state[index], target_state[index]
                ),
            }
        controls = {}
        for panel in CONTROL_PANELS:
            index = panel_indices[panel]
            panel_rows = [row for row in rows if row["transform_id"] == panel]
            controls[panel] = control_report(base_logits[index], logits[index], panel_rows)
        control_index = torch.cat([panel_indices[panel] for panel in CONTROL_PANELS])
        pooled_rows = [row for row in rows if row["transform_id"] in CONTROL_PANELS]
        controls["pooled"] = control_report(
            base_logits[control_index], logits[control_index], pooled_rows
        )
        return {"targets": targets, "controls": controls}

    def eligible(value):
        a1 = value["targets"]["A1"]["behavior"]
        controls = value["controls"]
        return bool(
            a1["signed_projection"] >= 0.75
            and a1["direction_fraction"] >= 0.875
            and controls["P"]["top1_flip_count"] == 0
            and controls["C"]["top1_flip_count"] == 0
            and max(controls["P"]["median_kl"], controls["C"]["median_kl"]) <= 0.02
        )

    self_output = greedy.run_patch(backend, base_batch, base_cache, COMPONENTS)
    self_state = greedy.module_impl.states(torch, backend, self_output, rows)
    self_logits = das.head_logits(backend, self_state).float()
    self_error = max(
        float((self_state - base_state).abs().max()),
        float((self_logits - base_logits).abs().max()),
    )
    reports = {}
    for mask in range(32):
        support = tuple(site for bit, site in enumerate(COMPONENTS) if mask & (1 << bit))
        value = report(greedy.run_patch(backend, base_batch, donor_cache, support))
        reports[str(mask)] = {"support": list(support), "eligible": eligible(value), "report": value}

    target_replay_max_abs_error = max(
        numeric_max_abs_difference(
            reports[str(mask)]["report"]["targets"],
            old_result["reports"][str(mask)]["report"]["targets"],
        )
        for mask in range(32)
    )
    eligible_masks = [mask for mask in range(32) if reports[str(mask)]["eligible"]]
    selected_mask = min(
        eligible_masks,
        key=lambda mask: (
            mask.bit_count(),
            -reports[str(mask)]["report"]["targets"]["A1"]["behavior"]["signed_projection"],
            mask,
        ),
    ) if eligible_masks else None
    selected = reports[str(selected_mask)] if selected_mask is not None else None
    necessity = {}
    if selected_mask is not None:
        selected_a1 = selected["report"]["targets"]["A1"]["behavior"]["signed_projection"]
        for bit, site in enumerate(COMPONENTS):
            if not selected_mask & (1 << bit):
                continue
            removed_mask = selected_mask & ~(1 << bit)
            removed = reports[str(removed_mask)]
            drop = selected_a1 - removed["report"]["targets"]["A1"]["behavior"]["signed_projection"]
            necessity[site] = {
                "removed_mask": removed_mask,
                "a1_projection_drop": drop,
                "removed_eligible": removed["eligible"],
                "necessary": bool(drop >= 0.05 or not removed["eligible"]),
            }

    pred_a = bool(
        authority_ok
        and cache_shapes_match
        and self_error <= 1e-4
        and target_replay_max_abs_error <= 1e-4
        and len(reports) == 32
        and finite(reports)
        and forwards == EXACT_FORWARDS
    )
    pred_b = selected is not None
    pred_c = bool(
        selected
        and selected["report"]["targets"]["A2"]["behavior"]["signed_projection"] >= 0.75
        and selected["report"]["targets"]["A2"]["behavior"]["direction_fraction"] >= 0.875
    )
    pred_d = bool(selected_mask is not None and selected_mask.bit_count() <= 4)
    pred_e = bool(necessity and all(value["necessary"] for value in necessity.values()))
    pred_f = bool(
        selected
        and all(
            selected["report"]["targets"][panel]["final_residual"]["signed_projection"] >= 0.50
            for panel in TARGET_PANELS
        )
    )
    predictions = {
        "pred_a_authority_alignment_capability_lattice_closure_finiteness_and_exact_price": pred_a,
        "pred_b_target_sufficient_separately_selective_subset_exists": pred_b,
        "pred_c_selected_subset_confirms_without_reselection": pred_c,
        "pred_d_selected_subset_is_proper": pred_d,
        "pred_e_every_selected_piece_is_necessary": pred_e,
        "pred_f_final_residual_tracks_selected_behavior": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "aligned_selective_attention_circuit"
    elif all(predictions[key] for key in list(predictions)[:5]) and not pred_f:
        terminal = "aligned_selective_attention_behavior_only"
    elif pred_b and not pred_c:
        terminal = "aligned_attention_construction_failure"
    elif pred_b and pred_c and not pred_d:
        terminal = "aligned_attention_requires_all_five"
    elif pred_b and pred_c and pred_d and not pred_e:
        terminal = "aligned_attention_subset_redundant"
    elif not pred_b:
        terminal = "aligned_no_selective_attention_subset"
    else:
        terminal = "partial"
    result = {
        "schema": "temporal_iswas_v15_aligned_control_attention_lattice_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "population": {
            "panel_counts": panel_counts,
            "row_ids": [row["row_id"] for row in rows],
            "alignment_contract": alignment,
            "capability_result_terminal": capability["terminal"],
        },
        "components": list(COMPONENTS),
        "instrument": {
            "cache_shapes_match": cache_shapes_match,
            "base_self_max_abs_error": self_error,
            "target_replay_max_abs_error": target_replay_max_abs_error,
        },
        "reports": reports,
        "eligible_masks": eligible_masks,
        "selected_mask": selected_mask,
        "selected": selected,
        "necessity": necessity,
        "predictions": predictions,
        "terminal": terminal,
        "price": {
            "model_forwards_exact": EXACT_FORWARDS,
            "model_forwards_observed": forwards,
            "example_evaluations": forwards * len(rows),
            "fit_updates": 0,
            "model_updates": 0,
            "transformer_backwards": 0,
        },
        "dryrun": dryrun,
    }
    atomic_create_json(OUT, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "instrument", "eligible_masks", "selected_mask", "selected", "necessity",
                    "predictions", "terminal", "price",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
