#!/usr/bin/env python3
"""Shared-versus-typed L9H1/H4 value subspace for aspectual and tense tasks."""

# BQGATE: EXPERIMENT pred_a_authority_capability_capture_and_split pred_b_each_task_basis_is_sufficient pred_c_shared_geometric_mode_exists pred_d_shared_mode_is_causally_material pred_e_task_specific_complements_are_secondary pred_f_joint_basis_is_compact_and_sufficient
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import attention_source_group_eval as source_groups
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import cross_task_subspace_eval as subspace
import run_aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1 as has_factor
import run_tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1 as is_factor


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_l9h1h4_shared_value_weight_subspace_v1.json"
TYPED = ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_result.json"
HAS_RESULT = ROOT / "circuits/followups/aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1_result.json"
IS_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1_result.json"
HAS_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1.py"
IS_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1.py"
SUBSPACE = ROOT / "ops/cross_task_subspace_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v1_result.json"
CANDIDATE_ID = "aspectual_tense.l9h1h4_shared_value_weight_subspace_v1"
EXPECTED = {
    "prior": "06620c3b87209fc8431278c19530262df891ee56d83a5c2b3fd266394095e3d1",
    "typed": "afb17159330dc6abcf018d36313a7df2c78c6708b67feb8d2f2d9d2eee50faf0",
    "has_result": "0f15a432f15b9f4a0a5f4b7470eb097793135c7d01118266fa6d45db2e8fd2c4",
    "is_result": "9a00cefe8986b1459c334c445a28208ae54911b81e5a42d78e1bc878777f07e4",
    "has_runner": "33e208e2b256fa6916bb61f408ffddc376c4c4871fe419f555c8a76422006374",
    "is_runner": "6826d33fadd2af133000cb3c826b4d89c535f576c3f057e667e68dece98e7d39",
    "subspace": "5207690210fd8aa56e1e8a16fb34cb63e00db13a9995177b783e4dcaa4d5b11d",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
}
HEADS = (1, 4)
ARMS = ("exact_value", "own_basis", "cross_basis", "joint_basis",
        "shared_basis", "own_complement")
FORWARDS, EVALUATIONS, RECORDS = 48, 816, 432


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stratified(rows):
    counts, fit, heldout = {}, [], []
    for row in [row for row in rows if row["transform_id"] == "A1"]:
        direction = row["direction_id"]
        occurrence = counts.get(direction, 0)
        counts[direction] = occurrence + 1
        (fit if occurrence % 2 == 0 else heldout).append(row)
    return fit, heldout


def validate_static():
    paths = {"prior": PRIOR, "typed": TYPED, "has_result": HAS_RESULT,
             "is_result": IS_RESULT, "has_runner": HAS_RUNNER, "is_runner": IS_RUNNER,
             "subspace": SUBSPACE, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or implementation hash changed")
    if any(json.loads(path.read_text()).get("terminal") != "screen"
           for path in (TYPED, HAS_RESULT, IS_RESULT)):
        raise ExperimentError("shared-path authority is no longer a screen")
    has_rows, _spec = has_factor.validate_static()
    is_rows = is_factor.validate_static()
    has_fit, has_held = stratified(has_rows)
    is_fit, is_held = stratified(is_rows)
    splits = {
        "has_fit": has_fit,
        "has_heldout": has_held,
        "has_a2": [row for row in has_rows if row["transform_id"] == "A2"],
        "is_fit": is_fit,
        "is_heldout": is_held,
        "is_a2": [row for row in is_rows if row["transform_id"] == "A2"],
    }
    expected_counts = {"has_fit": 16, "has_heldout": 16, "has_a2": 32,
                       "is_fit": 8, "is_heldout": 8, "is_a2": 16}
    if {name: len(rows) for name, rows in splits.items()} != expected_counts:
        raise ExperimentError("fit/evaluation split changed")
    for name in ("has_fit", "has_heldout", "is_fit", "is_heldout"):
        counts = sorted(sum(row["direction_id"] == direction for row in splits[name])
                        for direction in {row["direction_id"] for row in splits[name]})
        if counts != [len(splits[name]) // 2] * 2:
            raise ExperimentError("direction-stratified split changed")
    return splits


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False,
            "split_counts": {"has_fit": 16, "has_heldout": 16, "has_a2": 32,
                             "is_fit": 8, "is_heldout": 8, "is_a2": 16},
            "arms": list(ARMS), "model_forwards": FORWARDS,
            "example_evaluations": EVALUATIONS, "records": RECORDS,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(left, right))


def capture_item(backend, rows, task):
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_capture = backend.capture_bilinear(base_batch)
    donor_output, donor_capture = backend.capture_bilinear(donor_batch)
    empty_output, base_attention, base_raw, error_a = backend.capture_writer_raw(
        base_batch, donor_batch, base_capture, donor_capture, ())
    factors = ("left_change", "right_change")
    writer_output, hybrid_attention, hybrid_raw, error_b = backend.capture_writer_raw(
        base_batch, donor_batch, base_capture, donor_capture, factors)
    positions = (has_factor.block4.source_positions(base_batch, donor_batch) if task == "has"
                 else is_factor.writer.source_positions(base_batch, donor_batch))
    return {"rows": rows, "task": task, "base_batch": base_batch, "donor_batch": donor_batch,
            "base_output": base_output, "donor_output": donor_output,
            "empty_output": empty_output, "writer_output": writer_output,
            "base_attention": base_attention, "hybrid_attention": hybrid_attention,
            "base_raw": base_raw, "hybrid_raw": hybrid_raw, "positions": positions,
            "tensor_error": max(error_a, error_b)}


def oriented_matrix(item):
    module = __import__("torch")
    per_row = []
    for index, positions in enumerate(item["positions"]):
        changes = []
        for position in positions:
            changes.append(module.cat([
                item["hybrid_raw"]["v9"][index, position, head].float()
                - item["base_raw"]["v9"][index, position, head].float() for head in HEADS]))
        per_row.append(changes)
    means = module.stack([module.stack(changes).mean(0) for changes in per_row])
    signs = (means @ means[0]).sign()
    signs[signs == 0] = 1
    return module.stack([change * signs[index]
                         for index, changes in enumerate(per_row) for change in changes])


def install_value(backend, item, basis=None, *, complement=False):
    torch = backend.torch
    head_dim = backend.model.config.n_embd // backend.model.config.n_head
    lamb = backend.model.transformer.h[9].attn.lamb.detach().float()

    def patch(_module, arguments):
        flattened = arguments[0]
        output = flattened.view(len(item["rows"]), flattened.shape[1],
                                backend.model.config.n_head, head_dim).clone()
        for index, (query, positions) in enumerate(zip(
                item["base_batch"].semantic_positions, item["positions"])):
            for position in positions:
                raw = torch.cat([
                    item["hybrid_raw"]["v9"][index, position, head].float()
                    - item["base_raw"]["v9"][index, position, head].float() for head in HEADS])
                if basis is None:
                    projected = raw
                elif basis.shape[1] == 0:
                    projected = torch.zeros_like(raw)
                else:
                    in_basis = basis @ (basis.T @ raw)
                    projected = raw - in_basis if complement else in_basis
                for offset, head in enumerate(HEADS):
                    segment = projected[offset * head_dim:(offset + 1) * head_dim]
                    pattern = item["base_attention"]["pattern"][index, head, query, position].float()
                    output[index, query, head] += (pattern * (1.0 - lamb) * segment).to(output.dtype)
        return (output.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch)
    try:
        result, _capture = backend.manual_forward(item["base_batch"])
        return result
    finally:
        handle.remove()


def basis_record(basis):
    values = basis.detach().cpu().float().contiguous()
    return {"shape": list(values.shape), "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
            "values_column_major": values.T.reshape(-1).tolist()}


def main():
    splits = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    has_backend = has_factor.Backend.load("cuda")
    is_backend = is_factor.Backend(has_backend.torch, has_backend.model, has_backend.device)
    items, capture_error, identity_error = {}, 0.0, 0.0
    all_capable = True
    forwards = evaluations = 0
    for name, rows in splits.items():
        task = "has" if name.startswith("has_") else "is"
        backend = has_backend if task == "has" else is_backend
        item = capture_item(backend, rows, task)
        items[name] = item
        forwards += 4
        evaluations += 4 * len(rows)
        capture_error = max(capture_error, float(item["tensor_error"]),
                            float(item["base_attention"]["reconstruction_max_abs"]),
                            float(item["hybrid_attention"]["reconstruction_max_abs"]))
        identity_error = max(identity_error, pair_error(item["base_output"], item["empty_output"]))
        all_capable = all_capable and capable(item["base_output"]) and capable(item["donor_output"])

    matrices = {name: oriented_matrix(item) for name, item in items.items()}
    has_basis, has_singular, has_explained = subspace.energy_basis(matrices["has_fit"], retained=0.95)
    is_basis, is_singular, is_explained = subspace.energy_basis(matrices["is_fit"], retained=0.95)
    balanced = has_backend.torch.cat((matrices["has_fit"] / matrices["has_fit"].norm(),
                                      matrices["is_fit"] / matrices["is_fit"].norm()))
    joint_basis, joint_singular, joint_explained = subspace.energy_basis(balanced, retained=0.95)
    shared_basis, principal = subspace.shared_midpoint_basis(
        has_basis, is_basis, cosine_threshold=0.80)
    projection = {
        task: {basis_name: subspace.projection_energy(matrices[f"{task}_heldout"], basis)
               for basis_name, basis in (("has", has_basis), ("is", is_basis),
                                         ("joint", joint_basis), ("shared", shared_basis))
               if basis.shape[1] > 0}
        for task in ("has", "is")
    }
    for task in ("has", "is"):
        if shared_basis.shape[1] == 0:
            projection[task]["shared"] = 0.0

    bases = {"has": has_basis, "is": is_basis, "joint": joint_basis, "shared": shared_basis}
    records = []
    summaries = {}
    for panel in ("has_heldout", "has_a2", "is_heldout", "is_a2"):
        item = items[panel]
        task = item["task"]
        backend = has_backend if task == "has" else is_backend
        other = "is" if task == "has" else "has"
        arm_bases = {
            "exact_value": (None, False), "own_basis": (bases[task], False),
            "cross_basis": (bases[other], False), "joint_basis": (joint_basis, False),
            "shared_basis": (shared_basis, False), "own_complement": (bases[task], True),
        }
        for arm in ARMS:
            basis, complement = arm_bases[arm]
            output = install_value(backend, item, basis, complement=complement)
            forwards += 1
            evaluations += len(item["rows"])
            arm_records = source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm)
            records.extend(dict(record, panel=panel, task=task) for record in arm_records)
        summaries[panel] = {arm: source_groups.summarize(
            [record for record in records if record["panel"] == panel and record["arm"] == arm])
            for arm in ARMS}
        exact = summaries[panel]["exact_value"]["mean_recovery"]
        for arm in ARMS[1:]:
            summaries[panel][arm]["fraction_of_exact_value"] = (
                summaries[panel][arm]["mean_recovery"] / exact if abs(exact) > 1e-8 else None)

    pred_a = bool(all_capable and capture_error <= 2e-3 and identity_error <= 1e-4
                  and forwards == FORWARDS and evaluations == EVALUATIONS and len(records) == RECORDS)
    pred_b = all(0.80 <= summaries[panel]["own_basis"]["fraction_of_exact_value"] <= 1.20
                 and summaries[panel]["own_basis"]["direction_fraction"] >= 0.75
                 for panel in summaries)
    pred_c = bool(shared_basis.shape[1] >= 1 and projection["has"]["is"] >= 0.30
                  and projection["is"]["has"] >= 0.30)
    pred_d = all(summaries[panel]["shared_basis"]["fraction_of_exact_value"] >= 0.40
                 for panel in summaries)
    pred_e = all(abs(summaries[panel]["own_complement"]["fraction_of_exact_value"]) <= 0.40
                 for panel in summaries)
    pred_f = bool(joint_basis.shape[1] < has_basis.shape[1] + is_basis.shape[1]
                  and all(0.80 <= summaries[panel]["joint_basis"]["fraction_of_exact_value"] <= 1.20
                          for panel in summaries))
    predictions = {
        "pred_a_authority_capability_capture_and_split": pred_a,
        "pred_b_each_task_basis_is_sufficient": pred_b,
        "pred_c_shared_geometric_mode_exists": pred_c,
        "pred_d_shared_mode_is_causally_material": pred_d,
        "pred_e_task_specific_complements_are_secondary": pred_e,
        "pred_f_joint_basis_is_compact_and_sufficient": pred_f,
    }
    terminal = "invalid" if not pred_a else ("screen" if all(predictions.values()) else "null")
    result = {
        "schema": "aspectual_tense_l9h1h4_shared_value_weight_subspace_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"all_native_capable": all_capable,
            "factor_capture_max_abs_error": capture_error,
            "empty_writer_identity_max_abs_logit_error": identity_error},
        "subspaces": {
            "has": {"rank": has_basis.shape[1], "explained_fit_energy": has_explained,
                    "singular_values": [float(x) for x in has_singular.detach().cpu()],
                    "basis": basis_record(has_basis)},
            "is": {"rank": is_basis.shape[1], "explained_fit_energy": is_explained,
                   "singular_values": [float(x) for x in is_singular.detach().cpu()],
                   "basis": basis_record(is_basis)},
            "joint": {"rank": joint_basis.shape[1], "explained_balanced_fit_energy": joint_explained,
                      "singular_values": [float(x) for x in joint_singular.detach().cpu()],
                      "basis": basis_record(joint_basis)},
            "shared": {"rank": shared_basis.shape[1], "basis": basis_record(shared_basis)},
        },
        "principal_cosines": [float(x) for x in principal.detach().cpu()],
        "heldout_projection_energy": projection, "summaries": summaries,
        "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "shared_weight_realized_value_subspace" if terminal == "screen" else (
            "shared_reader_weights_with_task_typed_value_axes" if terminal == "null"
            else "authority_capability_capture_split_closure_coverage_or_price_invalid"),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "principal_cosines",
          "heldout_projection_energy", "summaries", "predictions", "price", "terminal", "reason")},
          sort_keys=True))


if __name__ == "__main__":
    main()
