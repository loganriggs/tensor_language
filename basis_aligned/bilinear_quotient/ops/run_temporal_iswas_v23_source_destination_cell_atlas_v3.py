#!/usr/bin/env python3
"""Reciprocal source-role by destination-role behavioral atlas for v23."""

# BQGATE: EXPERIMENT pred_a_authority_dependency_partitions_finiteness_enumeration_exact_price_and_no_v24 pred_b_exact_cell_closure_and_every_full_head_replay pred_c_changed_source_to_matched_suffix_destination_cell_is_reciprocal pred_d_at_least_three_heads_have_a_reciprocal_cell pred_e_a_reciprocal_cell_label_is_shared_by_at_least_two_heads
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from circuit_fast_screen_managed_runner import atomic_create_json
import attention_source_destination_cell_primitive as celllib
import run_temporal_iswas_v23_per_head_input_factor_source_atlas_v2 as parent_run


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT.parent / "polynomial_causal/TEMPORAL_ISWAS_V23_SOURCE_DESTINATION_CELL_ATLAS_V3_PREREGISTRATION.md"
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_source_destination_cell_atlas_v3.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_per_head_input_factor_source_atlas_v2_result.json"
CELL_LIBRARY = Path(celllib.__file__).resolve()
PARENT_SCRIPT = Path(parent_run.__file__).resolve()
OUT = ROOT / "circuits/followups/temporal_iswas_v23_source_destination_cell_atlas_v3_result.json"
EXPECTED = {
    "authority": "75773926c4fbc4d4f3246dfab331a5fc35c218d9636767502aa461962370366b",
    "prior": "db19c68a06c9f1e9e65e3c2d6887624d8b2b1c48fc83025274511077b94cdf4a",
    "parent_result": "f116c0bb0ee8bbc346c1a679869c3943ce1206a3dbe2690ddba1f1108a59dca0",
    "cell_library": "dc9920759f842a3383815666ebdd1e5475241f87477f4b454d95356374518f8a",
    "parent_script": "06ae1e9a731e6702208296ca9c7673c365999201beb546c920a974f080471d08",
}
union = parent_run.union
ROUTES = parent_run.ROUTES
PANELS = parent_run.PANELS
ROLE_NAMES = tuple(union.source.SOURCE_GROUPS)
BARS = {
    "closure": 1e-4, "parent_replay": 2e-3,
    "reciprocal_projection": .15, "reciprocal_cosine": .60,
    "reciprocal_direction": .60, "control": .25,
    "reciprocal_head_count": 3, "shared_head_count": 2,
}
PRICE = {
    "checkpoint_loads": 1, "model_forwards_exact": 84,
    "sequence_evaluations_exact": 5376, "transformer_backwards": 0,
    "model_updates": 0, "fit_parameters": 0,
}
PREDICTION_KEYS = (
    "pred_a_authority_dependency_partitions_finiteness_enumeration_exact_price_and_no_v24",
    "pred_b_exact_cell_closure_and_every_full_head_replay",
    "pred_c_changed_source_to_matched_suffix_destination_cell_is_reciprocal",
    "pred_d_at_least_three_heads_have_a_reciprocal_cell",
    "pred_e_a_reciprocal_cell_label_is_shared_by_at_least_two_heads",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cell_label(destination_index, source_index):
    return f"{ROLE_NAMES[destination_index]}<-{ROLE_NAMES[source_index]}"


def dependency_status():
    if not PARENT_RESULT.exists():
        return {"status": "pending"}
    result = json.loads(PARENT_RESULT.read_text())
    predictions = result.get("predictions", {})
    valid = bool(
        sha256(PARENT_RESULT) == EXPECTED["parent_result"]
        and result.get("candidate_id") ==
            "cross_task.temporal_iswas.v23_per_head_input_factor_source_atlas_v2"
        and predictions.get(
            "pred_a_authority_dependency_captures_closures_finiteness_enumeration_and_exact_price") is True
        and predictions.get("pred_b_every_full_head_replays_immutable_parent_singleton") is True
        and result.get("terminal") != "invalid_instrument"
        and result.get("v24_accessed") is False
    )
    return {
        "status": "valid" if valid else "invalid", "sha256": sha256(PARENT_RESULT),
        "terminal": result.get("terminal"),
        "predictions": {key: predictions.get(key) for key in sorted(predictions)},
    }


def is_reciprocal(report):
    for direction in ("sufficiency", "removed_by_reset"):
        arm = report[direction]
        target = arm["target"]
        if not (
            target["signed_projection"] >= BARS["reciprocal_projection"]
            and target["cosine"] >= BARS["reciprocal_cosine"]
            and target["direction_fraction"] >= BARS["reciprocal_direction"]
            and all(arm["controls"][panel] <= BARS["control"] for panel in ("P", "C"))
            and all(arm["halves"][half]["signed_projection"] > 0
                    for half in ("first", "second"))
        ):
            return False
    return True


def main():
    paths = {
        "authority": AUTHORITY, "prior": PRIOR, "parent_result": PARENT_RESULT,
        "cell_library": CELL_LIBRARY, "parent_script": PARENT_SCRIPT,
    }
    observed = {name: sha256(path) for name, path in paths.items() if path.exists()}
    dependency = dependency_status()
    capability = json.loads(parent_run.CAPABILITY.read_text())
    rows = union.fresh.build_rows()
    counts = {panel: sum(row["family"] == panel for row in rows) for panel in PANELS}
    capable = set(capability["jointly_capable_row_ids"]["A1"]
                  + capability["jointly_capable_row_ids"]["A2"])
    target_rows = [index for index, row in enumerate(rows) if row["row_id"] in capable]
    static_authority = bool(
        observed == EXPECTED and union.fresh.authority_sha256() == parent_run.ROWS_SHA256
        and counts == {panel: 16 for panel in PANELS} and len(target_rows) == 30
        and ROLE_NAMES == ("changed", "unchanged_prefix", "matched_suffix")
        and ROUTES == ((8, 1), (9, 1), (9, 4), (11, 3))
    )
    dryrun = {
        "candidate_id": "cross_task.temporal_iswas.v23_source_destination_cell_atlas_v3",
        "dryrun": True, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "static_authority_ok": static_authority,
        "dependency": dependency, "counts": counts, "target_rows": len(target_rows),
        "routes": [f"L{layer}H{head}" for layer, head in ROUTES],
        "role_names": list(ROLE_NAMES), "cells_per_head": 9,
        "sufficiency_arms_per_head": 9, "reset_arms_per_head": 9,
        "full_replay_arms_per_head": 1, "capture_forwards": 6,
        "bars": BARS, "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if not static_authority or dependency.get("status") != "valid":
        raise RuntimeError(f"v23 cell-atlas authority/dependency invalid: {dependency}")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = union.producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    base_batch = union.das._batch(backend, rows, side="base")
    donor_batch = union.das._batch(backend, rows, side="donor")
    base_logits, base_lengths, _, _, _, _ = union.shared.capture_native(
        backend, base_batch, False)
    donor_logits, donor_lengths, _, _, _, _ = union.shared.capture_native(
        backend, donor_batch, False)
    forwards = 2
    captures = {"base": {}, "donor": {}}
    closure_error = 0.0
    with torch.no_grad():
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            for layer in sorted(union.ROUTES_BY_LAYER):
                _output, capture = union.attention_eval.capture_layer_attention(
                    backend, batch, layer, include_qk_factors=True)
                captures[side][layer] = capture
                closure_error = max(closure_error, float(capture["reconstruction_max_abs"]))
                forwards += 1
    tokens, _lengths = backend._tensor_batch(base_batch)
    groups = union.source.batch_token_role_partitions(
        rows, tokens.shape[1], torch, device=backend.device)

    def complete_factors(side, layer, head):
        capture = captures[side][layer]
        return {
            "q": capture["q"][:, :, head].float(),
            "k": capture["k"][:, :, head].float(),
            "q2": capture["q2"][:, :, head].float(),
            "k2": capture["k2"][:, :, head].float(),
            "u": capture["value"][:, :, head].float(),
        }

    decompositions = {}
    for layer, head in ROUTES:
        route = f"L{layer}H{head}"
        decomposition = celllib.source_destination_cell_deltas(
            complete_factors("base", layer, head),
            complete_factors("donor", layer, head), groups, groups, torch)
        decompositions[route] = decomposition
        expected_base = captures["base"][layer]["head_output"][:, :, head].float()
        expected_donor = captures["donor"][layer]["head_output"][:, :, head].float()
        reconstructed = decomposition["cells"].sum((2, 3))
        for index, endpoint in enumerate(base_batch.semantic_positions):
            stop = int(endpoint) + 1
            closure_error = max(
                closure_error,
                float((decomposition["base"][index, :stop] - expected_base[index, :stop]).abs().max()),
                float((decomposition["donor"][index, :stop] - expected_donor[index, :stop]).abs().max()),
                float((reconstructed[index, :stop]
                       - (decomposition["donor"] - decomposition["base"])[index, :stop]).abs().max()),
            )
        full_mask = torch.ones(3, 3, dtype=torch.bool, device=backend.device)
        reset_to_base = celllib.compose_cell_intervention(
            decomposition, full_mask, "reset", torch)
        closure_error = max(closure_error, float((reset_to_base - decomposition["base"]).abs().max()))

    def cache_for(layer, head, replacement):
        base_capture = captures["base"][layer]
        changed = union.assemble_layer_head_outputs(
            base_capture["head_output"], {head: replacement})
        return {f"attn:{layer}": changed.reshape(changed.shape[0], changed.shape[1], -1)}

    def run_replacement(layer, head, replacement):
        nonlocal forwards
        route = f"L{layer}H{head}"
        logits, lengths, _final, _x = union.shared.run_patch(
            backend, base_batch, cache_for(layer, head, replacement), routes=(route,))
        forwards += 1
        return union.shared.margin(torch, logits, lengths, rows, backend.device)

    base_margin = union.shared.margin(torch, base_logits, base_lengths, rows, backend.device)
    donor_margin = union.shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
    native_delta = donor_margin - base_margin
    target_index = torch.as_tensor(target_rows, device=backend.device)
    control_index = {panel: torch.as_tensor(
        [index for index, row in enumerate(rows) if row["family"] == panel],
        device=backend.device) for panel in ("P", "C")}
    half_index = {name: torch.as_tensor(
        [index for index in target_rows if int(rows[index]["group_number"]) % 4 in residues],
        device=backend.device)
        for name, residues in (("first", (0, 1)), ("second", (2, 3)))}

    def report(delta, reference, control_scale):
        return {
            "target": union.shared.metrics(torch, delta[target_index], reference[target_index]),
            "controls": {panel: union.shared.rms_ratio(delta[index], control_scale)
                         for panel, index in control_index.items()},
            "halves": {name: union.shared.metrics(torch, delta[index], reference[index])
                       for name, index in half_index.items()},
        }

    reports, replay_errors = {}, {}
    native_scale = native_delta[target_index]
    with torch.no_grad():
        for layer, head in ROUTES:
            route = f"L{layer}H{head}"
            decomposition = decompositions[route]
            full_mask = torch.ones(3, 3, dtype=torch.bool, device=backend.device)
            full_margin = run_replacement(
                layer, head, celllib.compose_cell_intervention(
                    decomposition, full_mask, "sufficiency", torch))
            full_delta = full_margin - base_margin
            full_report = report(full_delta, native_delta, native_scale)
            cells = {}
            for destination_index in range(3):
                for source_index in range(3):
                    selected = torch.zeros(3, 3, dtype=torch.bool, device=backend.device)
                    selected[destination_index, source_index] = True
                    sufficient_margin = run_replacement(
                        layer, head, celllib.compose_cell_intervention(
                            decomposition, selected, "sufficiency", torch))
                    reset_margin = run_replacement(
                        layer, head, celllib.compose_cell_intervention(
                            decomposition, selected, "reset", torch))
                    sufficient_delta = sufficient_margin - base_margin
                    reset_delta = reset_margin - base_margin
                    removed_delta = full_delta - reset_delta
                    label = cell_label(destination_index, source_index)
                    cells[label] = {
                        "destination_role": ROLE_NAMES[destination_index],
                        "source_role": ROLE_NAMES[source_index],
                        "sufficiency": report(sufficient_delta, full_delta, full_delta[target_index]),
                        "removed_by_reset": report(removed_delta, full_delta, full_delta[target_index]),
                    }
                    cells[label]["reciprocal"] = is_reciprocal(cells[label])
            reports[route] = {"full_vs_native": full_report, "cells": cells}
            expected = json.loads(PARENT_RESULT.read_text())["reports"][route]["full_vs_native"]
            differences = [abs(full_report["target"][key] - expected["target"][key])
                           for key in ("signed_projection", "cosine", "direction_fraction",
                                       "relative_residual", "norm_ratio")]
            differences += [abs(full_report["controls"][panel] - expected["controls"][panel])
                            for panel in ("P", "C")]
            replay_errors[route] = max(differences)

    reciprocal_cells = {
        route: sorted(label for label, values in report_values["cells"].items()
                      if values["reciprocal"])
        for route, report_values in reports.items()
    }
    label_heads = {cell_label(d, s): sorted(
        route for route in reciprocal_cells if cell_label(d, s) in reciprocal_cells[route])
        for d in range(3) for s in range(3)}
    shared_labels = sorted(label for label, heads in label_heads.items()
                           if len(heads) >= BARS["shared_head_count"])
    finite_payload = [reports, replay_errors]
    A = bool(
        static_authority and dependency.get("status") == "valid"
        and forwards == PRICE["model_forwards_exact"]
        and forwards * len(rows) == PRICE["sequence_evaluations_exact"]
        and union.finite(finite_payload)
    )
    B = bool(closure_error <= BARS["closure"]
             and max(replay_errors.values()) <= BARS["parent_replay"])
    direct_label = "matched_suffix<-changed"
    C = any(direct_label in labels for labels in reciprocal_cells.values())
    D = sum(bool(labels) for labels in reciprocal_cells.values()) >= BARS["reciprocal_head_count"]
    E = bool(shared_labels)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "invalid_instrument" if not (A and B) else (
        "shared_directed_cell_program_screen" if C and D and E else "resolved_cell_atlas")
    result = {
        "schema": "temporal_iswas_v23_source_destination_cell_atlas_result_v3",
        "candidate_id": dryrun["candidate_id"], "started_utc": started_utc,
        "finished_utc": now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": observed, "parent_dependency": dependency,
        "population": {"counts": counts, "target_rows": len(target_rows)},
        "routes": dryrun["routes"], "role_names": list(ROLE_NAMES),
        "closure_max_abs": closure_error, "parent_replay_max_abs": replay_errors,
        "reports": reports, "reciprocal_cells": reciprocal_cells,
        "reciprocal_label_heads": label_heads, "shared_reciprocal_labels": shared_labels,
        "predictions": predictions, "terminal": terminal, "bars": BARS,
        "price": PRICE, "model_forwards": forwards, "v24_accessed": False,
        "fit_parameters": 0,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "closure_max_abs", "parent_replay_max_abs", "reciprocal_cells",
        "shared_reciprocal_labels", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
