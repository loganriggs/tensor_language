#!/usr/bin/env python3
"""Localize the material will/had block-8 attention write by native head."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_head_instrument pred_b_material_attention_branch_recurrence pred_c_shared_single_head_concentration pred_d_complete_head_ranking_reported pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block8_subject_attention_heads_v1.json"
CUBE = ROOT / "circuits/followups/temporal_auxiliary_will_had_subject_onset_block8_component_cube_v2_result.json"
DEPTH = ROOT / "circuits/followups/temporal_auxiliary_will_had_subject_onset_state_depth_v1_result.json"
FAST_SCREEN = ROOT / "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
ONSET_EVAL = ROOT / "ops/residual_source_onset_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8_subject_attention_heads_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block8_subject_attention_heads_v1"
EXPECTED = {
    "prior": "bfe08e34c4ed35ae42975075e3ff90c4d7eaeabc88a862f9ac740cdabe49cb42",
    "cube": "973c0e490ed04c07fbc410e4b2960aa27db9a7ffc7f219948a68e5dd947a431b",
    "depth": "b1f5ca50c669019684b5308b10dcd083de67897327bc6eecbcd92c978fc9cdae",
    "fast_screen": "71a833c52c39b98be6f576d11749522e2725c540a29d61a2352e315978684ec9",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "onset_eval": "8f0235743c0450797ade21ac663d4bf735f1784931c2c60ef8f69d4f7cd113a7",
}
EXPECTED_ROWS_SHA256 = "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
HEADS = tuple(range(12))
ARMS = tuple(f"head:{head:02d}" for head in HEADS) + ("full_heads", "direct_output")
MODEL_FORWARDS = 32
EXAMPLE_EVALUATIONS = 1024
INTERVENTION_RECORDS = 896


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(
        abs(float(a) - float(b))
        for pair_a, pair_b in zip(first.answer_foil, second.answer_foil)
        for a, b in zip(pair_a, pair_b)
    )


def validate_static():
    paths = {
        "prior": PRIOR,
        "cube": CUBE,
        "depth": DEPTH,
        "fast_screen": FAST_SCREEN,
        "builder": BUILDER,
        "onset_eval": ONSET_EVAL,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    cube = json.loads(CUBE.read_text())
    depth = json.loads(DEPTH.read_text())
    fast_screen = json.loads(FAST_SCREEN.read_text())
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    capability = [
        cell for cell in fast_screen["run"]["capability_cells"]
        if cell["family"] in {"A1", "A2"}
    ]
    if (
        cube.get("terminal") != "null"
        or not cube["predictions"].get("pred_a_authority_capability_exact_cube")
        or not cube["predictions"].get("pred_d_attention8_is_material")
        or depth["score"].get("subject_onset_boundary") != 9
        or fast_screen.get("terminal") != "screen"
        or len(rows) != 64
        or len(capability) != 4
        or not all(cell["passed"] for cell in capability)
        or len(ARMS) != 14
    ):
        raise ExperimentError("population or parent causal evidence changed")
    return rows, fast_screen, capability


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "rows": 64,
        "heads": list(HEADS),
        "arms": list(ARMS),
        "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def capture_attention(backend, batch):
    captured = {}
    block = backend.model.transformer.h[8]
    heads = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // heads

    def capture_heads(_module, arguments):
        value = arguments[0]
        captured["heads"] = value.detach().clone().view(
            len(batch.row_ids), value.shape[1], heads, head_dim
        )

    def capture_output(_module, _arguments, output):
        captured["output"] = output.detach().clone()

    handles = [
        block.attn.c_proj.register_forward_pre_hook(capture_heads),
        block.attn.c_proj.register_forward_hook(capture_output),
    ]
    try:
        output = backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"heads", "output"}:
        raise ExperimentError("block8 attention capture incomplete")
    return output, captured


def intervene(backend, base_batch, donor_batch, donor_capture, arm):
    block = backend.model.transformer.h[8]
    destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
    heads = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // heads
    handles = []
    if arm == "direct_output":
        def patch_output(_module, _arguments, output):
            changed = output.clone()
            for index, positions in enumerate(destinations):
                for position in positions:
                    changed[index, position] = donor_capture["output"][index, position].to(
                        device=changed.device, dtype=changed.dtype
                    )
            return changed
        handles.append(block.attn.c_proj.register_forward_hook(patch_output))
    else:
        selected = HEADS if arm == "full_heads" else (int(arm.split(":")[1]),)
        def patch_heads(_module, arguments):
            flattened = arguments[0]
            changed = flattened.clone().view(
                len(base_batch.row_ids), flattened.shape[1], heads, head_dim
            )
            for index, positions in enumerate(destinations):
                for position in positions:
                    for head in selected:
                        changed[index, position, head] = donor_capture["heads"][
                            index, position, head
                        ].to(device=changed.device, dtype=changed.dtype)
            return (changed.reshape_as(flattened),) + tuple(arguments[1:])
        handles.append(block.attn.c_proj.register_forward_pre_hook(patch_heads))
    try:
        return backend.native(base_batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()


def main():
    rows, fast_screen, capability_cells = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    frozen_native = {
        (str(item["row_id"]), str(item["side"])): (
            float(item["answer_logit"]), float(item["foil_logit"])
        )
        for item in fast_screen["run"]["native_logits"]
        if item["family"] in {"A1", "A2"}
    }
    batches = []
    manual_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        outputs, captures = {}, {}
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            outputs[side], captures[side] = capture_attention(backend, batch)
            frozen = producer.BatchOutput(
                tuple(frozen_native[(row_id, side)] for row_id in batch.row_ids), {}
            )
            manual_error = max(manual_error, pair_error(outputs[side], frozen))
            forward_calls += 1
            evaluations += len(family_rows)
        batches.append((family_rows, base_batch, donor_batch, outputs, captures))

    records = []
    arm_outputs = {}
    for arm in ARMS:
        for family_rows, base_batch, donor_batch, outputs, captures in batches:
            family = str(family_rows[0]["transform_id"])
            patched = intervene(backend, base_batch, donor_batch, captures["donor"], arm)
            arm_outputs[(arm, family)] = patched
            forward_calls += 1
            evaluations += len(family_rows)
            records.extend(onset.recovery_records(
                family_rows,
                outputs["base"],
                outputs["donor"],
                patched,
                group=arm,
                boundary=8,
            ))

    closure_error = max(
        pair_error(arm_outputs[("full_heads", family)], arm_outputs[("direct_output", family)])
        for family in ("A1", "A2")
    )
    summaries = {
        arm: {
            family: onset.summarize([
                record for record in records
                if record["group"] == arm and record["family"] == family
            ])
            for family in ("A1", "A2")
        }
        for arm in ARMS
    }
    full = summaries["full_heads"]
    shared_passing_heads = [
        head for head in HEADS
        if all(
            summaries[f"head:{head:02d}"][family]["mean_recovery"]
            >= 0.50 * full[family]["mean_recovery"]
            and summaries[f"head:{head:02d}"][family]["direction_fraction"] >= 0.75
            for family in ("A1", "A2")
        )
    ]
    ranking = {
        family: sorted(
            (
                {
                    "head": head,
                    "mean_recovery": summaries[f"head:{head:02d}"][family]["mean_recovery"],
                    "fraction_of_full": summaries[f"head:{head:02d}"][family]["mean_recovery"]
                    / full[family]["mean_recovery"],
                    "direction_fraction": summaries[f"head:{head:02d}"][family]["direction_fraction"],
                }
                for head in HEADS
            ),
            key=lambda item: (-item["mean_recovery"], item["head"]),
        )
        for family in ("A1", "A2")
    }
    pred_a = bool(
        all(cell["passed"] for cell in capability_cells)
        and manual_error <= 1.0e-4
        and closure_error <= 1.0e-4
        and len(records) == INTERVENTION_RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in records)
    )
    pred_b = bool(
        full["A1"]["mean_recovery"] >= 0.10
        and full["A2"]["mean_recovery"] >= 0.06
        and all(full[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    )
    pred_c = bool(shared_passing_heads)
    pred_d = all(len(ranking[family]) == 12 for family in ("A1", "A2"))
    pred_e = bool(
        forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == INTERVENTION_RECORDS
        and len({(record["group"], record["row_id"]) for record in records})
        == INTERVENTION_RECORDS
    )
    predictions = {
        "pred_a_authority_capability_exact_head_instrument": pred_a,
        "pred_b_material_attention_branch_recurrence": pred_b,
        "pred_c_shared_single_head_concentration": pred_c,
        "pred_d_complete_head_ranking_reported": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid"
    )
    reason = {
        "screen": "shared_single_block8_attention_head_localized",
        "null": "material_block8_attention_is_distributed_at_singleton_resolution",
        "invalid": "authority_capability_closure_materiality_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_block8_subject_attention_heads_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "instrument": {
            "manual_scored_logit_max_abs": manual_error,
            "full_heads_direct_output_scored_logit_max_abs": closure_error,
        },
        "summaries": summaries,
        "ranking": ranking,
        "shared_passing_heads": shared_passing_heads,
        "predictions": predictions,
        "price": {
            "model_forwards": forward_calls,
            "example_evaluations": evaluations,
            "intervention_records": len(records),
            "fitted_scalars": 0,
            "grid_evaluations": 0,
            "root_evaluations": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        },
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "partition the shared passing head by exact source-token groups"
            if terminal == "screen"
            else "test preregistered small head groups only if the complete ranking supplies a stable cross-construction grouping hypothesis"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "instrument": result["instrument"],
        "full_heads": full,
        "shared_passing_heads": shared_passing_heads,
        "top_heads": {family: ranking[family][:4] for family in ("A1", "A2")},
        "predictions": predictions,
        "price": result["price"],
        "terminal": terminal,
        "reason": reason,
        "next_action": result["next_action"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
