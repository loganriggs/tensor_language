#!/usr/bin/env python3
# BQGATE: frozen A-E split head-compression predictions; CUDA is managed-queue only.
"""Select and disjointly validate compact attention11/15 head sets."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_battery_integration_contract as battery
import circuit_candidate_aspectual_lexical_holdout_v5 as holdout
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention11_15_head_compression_split_v1.json"
BLOCK11 = ROOT / "circuits/followups/aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1_result.json"
BLOCK15 = ROOT / "circuits/followups/aspectual_anchor_block15_crossing_confirmation_v1_result.json"
SUFFIX_RUNNER = ROOT / "ops/run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_attention11_15_head_compression_split_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.attention11_15_head_compression_split_v1"
EXPECTED_PRIOR_SHA256 = "71eee7cb5bbc920742ba395cf0cee2299eddcaeb525f65ec2c365517b5b367e1"
EXPECTED_BLOCK11_SHA256 = "f534448e1b6e27195928d0e748147f43703225666fa102ece9d9a59d2f70c7ab"
EXPECTED_BLOCK15_SHA256 = "0c53f422e8ae3a176737cecac0025f97f41cd5e5eda545eaea0f0766b5525252"
EXPECTED_SUFFIX_RUNNER_SHA256 = "38c6ed4d8e2f7c66d7c3a48bbcafb1d0848927a6c502caa4f322b2e1b0867c4d"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_SELECTION_SHA256 = "d150ff72d1423058a01aa2140563315c041b1be98a59066e8dc4a98688775fe8"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
BOUNDARIES = (11, 15)
HEADS = tuple(range(9))
SELECTED_WIDTH = 4
WRITER_FACTORS = ("left_change", "right_change")
MODEL_FORWARDS_MAX = 108
EXAMPLE_EVALUATIONS_MAX = 896


class ExperimentError(RuntimeError):
    pass


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def selection_arms():
    return (
        "no_heads", "all_heads",
        *(f"h{head}" for head in HEADS),
        *(f"all_except_h{head}" for head in HEADS),
    )


def validate_static():
    for file_path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        BLOCK11: EXPECTED_BLOCK11_SHA256,
        BLOCK15: EXPECTED_BLOCK15_SHA256,
        SUFFIX_RUNNER: EXPECTED_SUFFIX_RUNNER_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    block11_result = json.loads(BLOCK11.read_text())
    block15_result = json.loads(BLOCK15.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if block11_result.get("terminal") != "screen" or block15_result.get("terminal") != "screen":
        raise ExperimentError("parent attention crossing changed")
    rows_all = holdout.build_rows()
    if holdout.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    target = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    selection, confirmation = tuple(target[:16]), tuple(target[16:])
    if suffix.ids_sha256(selection) != EXPECTED_SELECTION_SHA256:
        raise ExperimentError("selection split changed")
    if suffix.ids_sha256(confirmation) != EXPECTED_CONFIRMATION_SHA256:
        raise ExperimentError("confirmation split changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-attention11-15-head-compression-split-v1",
        authority_sha256=EXPECTED_ROWS_SHA256,
        expected_fit_rows=len(rows_all),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=131072,
        ),
    )
    enriched_all = screen.validate_fit_authority(spec, rows_all)
    selection = tuple(enriched_all[str(row["row_id"])] for row in selection)
    confirmation = tuple(enriched_all[str(row["row_id"])] for row in confirmation)
    if len(selection) != 16 or len(confirmation) != 16 or len(selection_arms()) != 20:
        raise ExperimentError("population or head-arm inventory changed")
    return selection, confirmation, spec


class HeadCompressionBackend(suffix.SuffixBackend):
    def capture_suffix_heads(self, batch: producer.ModelBatch):
        captured = {}
        handles = []
        head_dim = self.model.config.n_embd // self.model.config.n_head
        for boundary in BOUNDARIES:
            def capture_heads(_module, arguments, boundary=boundary):
                flattened = arguments[0]
                captured[f"head_output{boundary}"] = flattened.view(
                    len(batch.row_ids), flattened.shape[1],
                    self.model.config.n_head, head_dim,
                ).detach().clone()
            handles.append(
                self.model.transformer.h[boundary].attn.c_proj.register_forward_pre_hook(
                    capture_heads
                )
            )
        try:
            output, state_capture = self.capture_suffix(batch)
        finally:
            for handle in handles:
                handle.remove()
        for boundary in BOUNDARIES:
            if f"head_output{boundary}" not in captured:
                raise ExperimentError("suffix head capture missing")
        state_capture.update(captured)
        return output, state_capture

    def capture_writer_suffix_heads(
        self, base_batch, donor_batch, base_capture, donor_capture
    ):
        projected, tensor_error = self.projected_terms(base_capture, donor_capture)
        positions = block4.source_positions(base_batch, donor_batch)

        def patch_mlp4(_module, _arguments, output):
            changed = output.clone()
            for i, bank in enumerate(positions):
                for position in bank:
                    delta = sum(
                        (projected[factor][i, position] for factor in WRITER_FACTORS),
                        self.torch.zeros_like(changed[i, position], dtype=self.torch.float32),
                    )
                    changed[i, position] = (
                        changed[i, position].float() + delta
                    ).to(changed.dtype)
            return changed

        handle = self.model.transformer.h[4].mlp.register_forward_hook(patch_mlp4)
        try:
            output, capture = self.capture_suffix_heads(base_batch)
        finally:
            handle.remove()
        return output, capture, tensor_error

    def projected_head_delta(self, base_capture, hybrid_capture, boundary, selected_heads):
        if len(selected_heads) != len(set(selected_heads)) or any(
            head not in HEADS for head in selected_heads
        ):
            raise ExperimentError("head subset changed")
        base = base_capture[f"head_output{boundary}"]
        hybrid = hybrid_capture[f"head_output{boundary}"]
        delta = self.torch.zeros_like(base, dtype=self.torch.float32)
        for head in selected_heads:
            delta[:, :, head] = hybrid[:, :, head].float() - base[:, :, head].float()
        flattened = delta.reshape(delta.shape[0], delta.shape[1], -1)
        weight = self.model.transformer.h[boundary].attn.c_proj.weight.float()
        return self.F.linear(flattened, weight, None)

    def head_crossing(
        self, batch, base_capture, hybrid_capture, boundary, selected_heads
    ):
        state = base_capture[f"resid{boundary + 1}"].clone()
        lambda0 = self.model.transformer.h[boundary].lambdas[0]
        projected_attention = self.projected_head_delta(
            base_capture, hybrid_capture, boundary, selected_heads
        )
        for i, query in enumerate(batch.semantic_positions):
            delta = (
                lambda0.float() * (
                    hybrid_capture[f"resid{boundary}"][i, query].float()
                    - base_capture[f"resid{boundary}"][i, query].float()
                )
                + projected_attention[i, query]
                + hybrid_capture[f"mlp{boundary}"][i, query].float()
                - base_capture[f"mlp{boundary}"][i, query].float()
            )
            state[i, query] = (state[i, query].float() + delta).to(state.dtype)
        return self.suffix_from_resid(
            batch, state, base_capture["x0"],
            base_capture[f"v1_after{boundary}"], boundary + 1,
        )


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery is missing or nonfinite")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    selection, confirmation, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_attention11_15_head_compression_split_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "selection_rows": len(selection),
        "confirmation_rows": len(confirmation),
        "boundaries": list(BOUNDARIES),
        "heads": list(HEADS),
        "selection_arms_per_boundary": len(selection_arms()),
        "selected_width": SELECTED_WIDTH,
        "confirmation_arms_per_boundary": 3,
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0,
        "model_updates": 0,
        "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = HeadCompressionBackend.load("cuda")
    native = {}
    captures = {"selection": [], "confirmation": []}
    writer_values = {
        phase: {"A1": [], "A2": []} for phase in captures
    }
    selection_values = {
        boundary: {
            arm: {"A1": [], "A2": []} for arm in selection_arms()
        } for boundary in BOUNDARIES
    }
    confirmation_values = {
        boundary: {
            arm: {"A1": [], "A2": []}
            for arm in ("no_heads", "selected_four", "all_heads")
        } for boundary in BOUNDARIES
    }
    raw_records = []
    manual_base_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    attention_projection_error_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    crossing_tensor_error_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    forward_calls = 0
    evaluations = 0

    for phase, phase_rows in (("selection", selection), ("confirmation", confirmation)):
        for family in ("A1", "A2"):
            family_rows = [row for row in phase_rows if row["transform_id"] == family]
            for chunk in producer._chunks(family_rows, spec.batch_size):
                base_batch = producer._batch(spec, chunk, "base")
                donor_batch = producer._batch(spec, chunk, "donor")
                base_native, base_bilinear = backend.capture_bilinear(base_batch)
                donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
                base_manual, base_capture = backend.capture_suffix_heads(base_batch)
                writer_output, hybrid_capture, writer_error = backend.capture_writer_suffix_heads(
                    base_batch, donor_batch, base_bilinear, donor_bilinear
                )
                forward_calls += 4
                evaluations += 4 * len(chunk)
                writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
                for reference, manual in zip(base_native.answer_foil, base_manual.answer_foil):
                    manual_base_max_abs = max(
                        manual_base_max_abs,
                        abs(reference[0] - manual[0]), abs(reference[1] - manual[1]),
                    )
                for side, output in (("base", base_native), ("donor", donor_native)):
                    for row, pair in zip(chunk, output.answer_foil):
                        answer, foil = producer._finite_pair(pair)
                        native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                            str(row["row_id"]), family, side, answer, foil
                        )
                for row, pair in zip(chunk, writer_output.answer_foil):
                    answer, foil, value = suffix.recovery(row, pair, native)
                    writer_values[phase][family].append(value)
                    raw_records.append({
                        "phase": phase, "boundary": "writer", "arm_id": "writer_two_term",
                        "family": family, "row_id": str(row["row_id"]),
                        "answer_logit": answer, "foil_logit": foil, "recovery": value,
                    })
                for boundary in BOUNDARIES:
                    projected = backend.projected_head_delta(
                        base_capture, hybrid_capture, boundary, HEADS
                    )
                    actual_attention = (
                        hybrid_capture[f"attention{boundary}"].float()
                        - base_capture[f"attention{boundary}"].float()
                    )
                    attention_projection_error_max_abs[boundary] = max(
                        attention_projection_error_max_abs[boundary],
                        float((projected - actual_attention).abs().max()),
                    )
                    lambda0 = backend.model.transformer.h[boundary].lambdas[0]
                    for i, query in enumerate(base_batch.semantic_positions):
                        reconstructed = (
                            lambda0.float() * (
                                hybrid_capture[f"resid{boundary}"][i, query].float()
                                - base_capture[f"resid{boundary}"][i, query].float()
                            )
                            + projected[i, query]
                            + hybrid_capture[f"mlp{boundary}"][i, query].float()
                            - base_capture[f"mlp{boundary}"][i, query].float()
                        )
                        direct = (
                            hybrid_capture[f"resid{boundary + 1}"][i, query].float()
                            - base_capture[f"resid{boundary + 1}"][i, query].float()
                        )
                        crossing_tensor_error_max_abs[boundary] = max(
                            crossing_tensor_error_max_abs[boundary],
                            float((reconstructed - direct).abs().max()),
                        )
                captures[phase].append(
                    (family, chunk, base_batch, base_capture, hybrid_capture)
                )

    for family, chunk, base_batch, base_capture, hybrid_capture in captures["selection"]:
        for boundary in BOUNDARIES:
            head_sets = {
                "no_heads": (), "all_heads": HEADS,
                **{f"h{head}": (head,) for head in HEADS},
                **{
                    f"all_except_h{head}": tuple(other for other in HEADS if other != head)
                    for head in HEADS
                },
            }
            for arm, selected_heads in head_sets.items():
                output = backend.head_crossing(
                    base_batch, base_capture, hybrid_capture, boundary, selected_heads
                )
                forward_calls += 1
                evaluations += len(chunk)
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil, value = suffix.recovery(row, pair, native)
                    selection_values[boundary][arm][family].append(value)
                    raw_records.append({
                        "phase": "selection", "boundary": boundary, "arm_id": arm,
                        "family": family, "row_id": str(row["row_id"]),
                        "answer_logit": answer, "foil_logit": foil, "recovery": value,
                    })

    selection_summary = {}
    selected_heads = {}
    for boundary in BOUNDARIES:
        summaries = {}
        targets = {}
        for arm in selection_arms():
            families = {
                family: summarize(selection_values[boundary][arm][family])
                for family in ("A1", "A2")
            }
            target = statistics.fmean(
                families[family]["mean_recovery"] for family in ("A1", "A2")
            )
            summaries[arm] = {"families": families, "mean_target_recovery": target}
            targets[arm] = target
        attributions = {}
        for head in HEADS:
            singleton = targets[f"h{head}"] - targets["no_heads"]
            necessity = targets["all_heads"] - targets[f"all_except_h{head}"]
            attributions[f"h{head}"] = {
                "singleton_increment": singleton,
                "full_minus_leave_one_out_increment": necessity,
                "selection_score": 0.5 * (singleton + necessity),
            }
        ranking = sorted(
            HEADS, key=lambda head: (-attributions[f"h{head}"]["selection_score"], head)
        )
        selected_heads[boundary] = tuple(ranking[:SELECTED_WIDTH])
        selection_summary[str(boundary)] = {
            "arms": summaries, "attributions": attributions,
            "ranking": [f"h{head}" for head in ranking],
            "selected_heads": list(selected_heads[boundary]),
            "all_minus_none_increment": targets["all_heads"] - targets["no_heads"],
        }

    for family, chunk, base_batch, base_capture, hybrid_capture in captures["confirmation"]:
        for boundary in BOUNDARIES:
            head_sets = {
                "no_heads": (),
                "selected_four": selected_heads[boundary],
                "all_heads": HEADS,
            }
            for arm, selected in head_sets.items():
                output = backend.head_crossing(
                    base_batch, base_capture, hybrid_capture, boundary, selected
                )
                forward_calls += 1
                evaluations += len(chunk)
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil, value = suffix.recovery(row, pair, native)
                    confirmation_values[boundary][arm][family].append(value)
                    raw_records.append({
                        "phase": "confirmation", "boundary": boundary, "arm_id": arm,
                        "family": family, "row_id": str(row["row_id"]),
                        "answer_logit": answer, "foil_logit": foil, "recovery": value,
                    })

    confirmation_summary = {}
    compression_pass = []
    for boundary in BOUNDARIES:
        summaries = {}
        targets = {}
        for arm in ("no_heads", "selected_four", "all_heads"):
            families = {
                family: summarize(confirmation_values[boundary][arm][family])
                for family in ("A1", "A2")
            }
            target = statistics.fmean(
                families[family]["mean_recovery"] for family in ("A1", "A2")
            )
            summaries[arm] = {"families": families, "mean_target_recovery": target}
            targets[arm] = target
        denominator = targets["all_heads"] - targets["no_heads"]
        numerator = targets["selected_four"] - targets["no_heads"]
        retained = numerator / denominator
        family_increments = {
            family: (
                summaries["selected_four"]["families"][family]["mean_recovery"]
                - summaries["no_heads"]["families"][family]["mean_recovery"]
            ) for family in ("A1", "A2")
        }
        compression_pass.append(
            retained >= 0.65 and all(value > 0.0 for value in family_increments.values())
        )
        confirmation_summary[str(boundary)] = {
            "arms": summaries,
            "selected_heads": list(selected_heads[boundary]),
            "selected_attention_increment": numerator,
            "all_attention_increment": denominator,
            "selected_to_all_attention_fraction": retained,
            "selected_family_increments": family_increments,
        }

    writer_summary = {
        phase: {
            family: summarize(writer_values[phase][family]) for family in ("A1", "A2")
        } for phase in ("selection", "confirmation")
    }
    pooled_writer = statistics.fmean(
        value for phase in ("selection", "confirmation")
        for family in ("A1", "A2") for value in writer_values[phase][family]
    )
    current_capability = all(
        native[(str(row["row_id"]), side)].margin > 0.0
        for row in selection + confirmation for side in ("base", "donor")
    )
    pred_a = (
        current_capability and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and all(value <= 0.04 for value in attention_projection_error_max_abs.values())
        and all(value <= 0.04 for value in crossing_tensor_error_max_abs.values())
    )
    pred_b = (
        abs(pooled_writer - 0.2835613798233539) <= 0.01
        and all(
            writer_summary[phase][family]["mean_recovery"] > 0.0
            and writer_summary[phase][family]["direction_fraction"] >= 0.75
            for phase in ("selection", "confirmation") for family in ("A1", "A2")
        )
    )
    pred_c = all(
        selection_summary[str(boundary)]["all_minus_none_increment"] > 0.0
        and len(set(selected_heads[boundary])) == SELECTED_WIDTH
        for boundary in BOUNDARIES
    )
    pred_d = all(compression_pass)
    pred_e = (
        len(raw_records) == 768 and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = (
        "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e))
        else "null" if pred_a and pred_b and pred_c and pred_e
        else "invalid"
    )
    reason = {
        "screen": "attention11_15_four_head_compression_transfers_disjointly",
        "null": "one_or_both_four_head_compressions_failed",
        "invalid": "authority_split_capability_instrument_writer_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_attention11_15_head_compression_split_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "block11_result_sha256": EXPECTED_BLOCK11_SHA256,
        "block15_result_sha256": EXPECTED_BLOCK15_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_split_capability_and_exact_instrument": pred_a,
            "pred_b_writer_recurrence": pred_b,
            "pred_c_positive_selection_attention": pred_c,
            "pred_d_disjoint_four_head_compression": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "selection": selection_summary,
            "confirmation": confirmation_summary,
            "writer": writer_summary,
            "pooled_writer_mean_recovery": pooled_writer,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "attention_projection_error_max_abs": {
                str(key): value for key, value in attention_projection_error_max_abs.items()
            },
            "crossing_tensor_error_max_abs": {
                str(key): value for key, value in crossing_tensor_error_max_abs.items()
            },
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records,
        "terminal": terminal, "reason": reason,
        "next_action": (
            "factor source terms for the validated attention11 and attention15 head sets"
            if terminal == "screen"
            else "retain module-level suffix attention operations"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"],
        "selection": {
            boundary: selection_summary[str(boundary)]["selected_heads"]
            for boundary in BOUNDARIES
        },
        "confirmation_fraction": {
            boundary: confirmation_summary[str(boundary)]["selected_to_all_attention_fraction"]
            for boundary in BOUNDARIES
        },
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
