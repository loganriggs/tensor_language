#!/usr/bin/env python3
# BQGATE: frozen A-E suffix selection and disjoint-confirmation predictions; CUDA is managed-queue only.
"""Leakage-controlled suffix depth selection and disjoint crossing confirmation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
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
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as path
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
PROGRAM_RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v2_result.json"
RESID10 = ROOT / "circuits/followups/aspectual_anchor_block9_crossing_factorial_lexical_holdout_v1_result.json"
BACKEND_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.suffix_depth_adaptive_factorial_lexical_holdout_v1"
EXPECTED_PRIOR_SHA256 = "8c0770228f625a0288931daca30c53b5dfd31fe79cb86fbc482ad2ab6cac9f6f"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_PROGRAM_RELEASE_SHA256 = "ce0cffd7dec596768fb9181ec546ff0764207717f244d13809a085d62f4dd3c1"
EXPECTED_RESID10_SHA256 = "d7df483de097811d93c2b5b92b8beed1dd44dcf165fa3bfc9ee89541e0e92bbc"
EXPECTED_BACKEND_RUNNER_SHA256 = "5449d45505267f2ebc92c9285b7a984fce773f834c2071e1d7e6d1a9f1949372"
EXPECTED_SELECTION_SHA256 = "d150ff72d1423058a01aa2140563315c041b1be98a59066e8dc4a98688775fe8"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
RESIDUAL_BOUNDARIES = tuple(range(10, 19))
CROSSING_BOUNDARIES = tuple(range(10, 18))
WRITER_FACTORS = ("left_change", "right_change")
BASE_FACTORS = ("carried", "attention", "mlp")
MODEL_FORWARDS_MAX = 52
EXAMPLE_EVALUATIONS_MAX = 480


class ExperimentError(RuntimeError):
    pass


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def ids_sha256(rows) -> str:
    payload = json.dumps(
        [row["row_id"] for row in rows], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets():
    return tuple(
        subset for width in range(len(BASE_FACTORS) + 1)
        for subset in itertools.combinations(BASE_FACTORS, width)
    )


def arm_id(subset: tuple[str, ...]) -> str:
    return "empty" if not subset else "+".join(subset)


def validate_static():
    for file_path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
        PROGRAM_RELEASE: EXPECTED_PROGRAM_RELEASE_SHA256,
        RESID10: EXPECTED_RESID10_SHA256,
        BACKEND_RUNNER: EXPECTED_BACKEND_RUNNER_SHA256,
    }.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    release = json.loads(PROGRAM_RELEASE.read_text())
    resid10 = json.loads(RESID10.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if release.get("terminal") != "release" or not all(release["predictions"].values()):
        raise ExperimentError("program release changed")
    if resid10.get("terminal") != "screen":
        raise ExperimentError("resid10 authority changed")
    rows_all = holdout.build_rows()
    if holdout.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    selection, confirmation = tuple(selected[:16]), tuple(selected[16:])
    if ids_sha256(selection) != EXPECTED_SELECTION_SHA256:
        raise ExperimentError("selection split changed")
    if ids_sha256(confirmation) != EXPECTED_CONFIRMATION_SHA256:
        raise ExperimentError("confirmation split changed")
    for split in (selection, confirmation):
        cells = {
            (family, direction): sum(
                row["transform_id"] == family and row["direction_id"] == direction
                for row in split
            )
            for family in ("A1", "A2")
            for direction in ("past_to_present", "present_to_past")
        }
        if set(cells.values()) != {4}:
            raise ExperimentError("split balance changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-suffix-depth-adaptive-factorial-lexical-holdout-v1",
        authority_sha256=EXPECTED_ROWS_SHA256,
        expected_fit_rows=len(rows_all),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=65536,
        ),
    )
    enriched_all = screen.validate_fit_authority(spec, rows_all)
    selection = tuple(enriched_all[str(row["row_id"])] for row in selection)
    confirmation = tuple(enriched_all[str(row["row_id"])] for row in confirmation)
    if len(selection) != 16 or len(confirmation) != 16 or len(subsets()) != 8:
        raise ExperimentError("population or factorial changed")
    return selection, confirmation, spec, resid10


class SuffixBackend(path.PathBackend):
    def capture_suffix(self, batch: producer.ModelBatch):
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        capture = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0 = x
            capture["x0"] = x0.detach().clone()
            v1 = None
            for layer, block in enumerate(model.transformer.h):
                if layer in CROSSING_BOUNDARIES:
                    capture[f"resid{layer}"] = x.detach().clone()
                    capture[f"v1_before{layer}"] = v1.detach().clone()
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                mlp_output = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                x = x + mlp_output
                if layer in CROSSING_BOUNDARIES:
                    capture[f"attention{layer}"] = attention.detach().clone()
                    capture[f"mlp{layer}"] = mlp_output.detach().clone()
                    capture[f"resid{layer + 1}"] = x.detach().clone()
                    capture[f"v1_after{layer}"] = v1.detach().clone()
            capture["v1_before18"] = v1.detach().clone()
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0
            )
            values = tuple(
                (
                    float(logits[i, length - 1, batch.answer_ids[i]].float()),
                    float(logits[i, length - 1, batch.foil_ids[i]].float()),
                )
                for i, length in enumerate(lengths)
            )
        expected = {"x0", "v1_before18"}
        for boundary in CROSSING_BOUNDARIES:
            expected.update({
                f"resid{boundary}", f"v1_before{boundary}",
                f"attention{boundary}", f"mlp{boundary}",
                f"resid{boundary + 1}", f"v1_after{boundary}",
            })
        if set(capture) != expected:
            raise ExperimentError("suffix capture incomplete")
        return producer.BatchOutput(values, {}), capture

    def capture_writer_suffix(self, base_batch, donor_batch, base_capture, donor_capture):
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
            output, capture = self.capture_suffix(base_batch)
        finally:
            handle.remove()
        return output, capture, tensor_error

    def suffix_from_resid(self, batch, state, x0, v1, boundary: int):
        torch, F, model = self.torch, self.F, self.model
        lengths = tuple(len(row) for row in batch.token_rows)
        x = state
        with torch.no_grad():
            for layer in range(boundary, 18):
                block = model.transformer.h[layer]
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0
            )
            values = tuple(
                (
                    float(logits[i, length - 1, batch.answer_ids[i]].float()),
                    float(logits[i, length - 1, batch.foil_ids[i]].float()),
                )
                for i, length in enumerate(lengths)
            )
        return producer.BatchOutput(values, {})

    def direct_query(self, batch, base_capture, hybrid_capture, boundary: int):
        if boundary not in RESIDUAL_BOUNDARIES:
            raise ExperimentError("residual boundary changed")
        state = base_capture[f"resid{boundary}"].clone()
        for i, query in enumerate(batch.semantic_positions):
            state[i, query] = hybrid_capture[f"resid{boundary}"][i, query]
        return self.suffix_from_resid(
            batch, state, base_capture["x0"],
            base_capture[f"v1_before{boundary}"], boundary,
        )

    def crossing(self, batch, base_capture, hybrid_capture, boundary, subset):
        if boundary not in CROSSING_BOUNDARIES or subset not in subsets():
            raise ExperimentError("selected crossing changed")
        state = base_capture[f"resid{boundary + 1}"].clone()
        lambda0 = self.model.transformer.h[boundary].lambdas[0]
        for i, query in enumerate(batch.semantic_positions):
            terms = {
                "carried": lambda0.float() * (
                    hybrid_capture[f"resid{boundary}"][i, query].float()
                    - base_capture[f"resid{boundary}"][i, query].float()
                ),
                "attention": (
                    hybrid_capture[f"attention{boundary}"][i, query].float()
                    - base_capture[f"attention{boundary}"][i, query].float()
                ),
                "mlp": (
                    hybrid_capture[f"mlp{boundary}"][i, query].float()
                    - base_capture[f"mlp{boundary}"][i, query].float()
                ),
            }
            delta = sum(
                (terms[factor] for factor in subset),
                self.torch.zeros_like(state[i, query], dtype=self.torch.float32),
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


def recovery(row, pair, native) -> tuple[float, float, float]:
    answer, foil = producer._finite_pair(pair)
    row_id = str(row["row_id"])
    value = kernel.signed_pairwise_donor_recovery(
        -native[(row_id, "base")].margin,
        native[(row_id, "donor")].margin,
        -(answer - foil),
    )
    return answer, foil, value


def main() -> None:
    selection, confirmation, spec, resid10_authority = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_dryrun_v1",
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
        "selection_curve": list(RESIDUAL_BOUNDARIES),
        "confirmation_factorial_arms": len(subsets()),
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
    backend = SuffixBackend.load("cuda")
    native = {}
    selection_curve = {
        boundary: {"A1": [], "A2": []} for boundary in RESIDUAL_BOUNDARIES
    }
    writer_values = {
        phase: {"A1": [], "A2": []} for phase in ("selection", "confirmation")
    }
    selection_records = []
    confirmation_records = []
    selection_captures = []
    confirmation_captures = []
    manual_base_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    resid18_to_writer_max_abs = 0.0
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
                base_manual, base_suffix = backend.capture_suffix(base_batch)
                writer_output, hybrid_suffix, writer_error = backend.capture_writer_suffix(
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
                    answer, foil, value = recovery(row, pair, native)
                    writer_values[phase][family].append(value)
                    record = {
                        "phase": phase, "boundary": "writer", "arm_id": "writer_two_term",
                        "family": family, "row_id": str(row["row_id"]),
                        "answer_logit": answer, "foil_logit": foil, "recovery": value,
                    }
                    (selection_records if phase == "selection" else confirmation_records).append(record)
                captures = selection_captures if phase == "selection" else confirmation_captures
                captures.append((family, chunk, base_batch, donor_batch, base_suffix, hybrid_suffix))

    for family, chunk, base_batch, _donor_batch, base_suffix, hybrid_suffix in selection_captures:
        for boundary in RESIDUAL_BOUNDARIES:
            output = backend.direct_query(base_batch, base_suffix, hybrid_suffix, boundary)
            forward_calls += 1
            evaluations += len(chunk)
            if boundary == 18:
                writer_pairs = {
                    record["row_id"]: (record["answer_logit"], record["foil_logit"])
                    for record in selection_records if record["boundary"] == "writer"
                }
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil, value = recovery(row, pair, native)
                selection_curve[boundary][family].append(value)
                if boundary == 18:
                    reference = writer_pairs[str(row["row_id"])]
                    resid18_to_writer_max_abs = max(
                        resid18_to_writer_max_abs,
                        abs(answer - reference[0]), abs(foil - reference[1]),
                    )
                selection_records.append({
                    "phase": "selection", "boundary": boundary,
                    "arm_id": f"direct_resid{boundary}_query",
                    "family": family, "row_id": str(row["row_id"]),
                    "answer_logit": answer, "foil_logit": foil, "recovery": value,
                })

    curve_summary = {}
    curve_targets = {}
    for boundary in RESIDUAL_BOUNDARIES:
        families = {
            family: summarize(selection_curve[boundary][family])
            for family in ("A1", "A2")
        }
        target = statistics.fmean(
            families[family]["mean_recovery"] for family in ("A1", "A2")
        )
        curve_summary[f"resid{boundary}"] = {
            "families": families, "mean_target_recovery": target,
        }
        curve_targets[boundary] = target
    increments = {
        boundary: curve_targets[boundary + 1] - curve_targets[boundary]
        for boundary in CROSSING_BOUNDARIES
    }
    selected_boundary = max(CROSSING_BOUNDARIES, key=lambda value: (increments[value], -value))

    factorial_values = {
        subset: {"A1": [], "A2": []} for subset in subsets()
    }
    ceiling_values = {"A1": [], "A2": []}
    selected_tensor_error_max_abs = 0.0
    full_to_ceiling_max_abs = 0.0
    for family, chunk, base_batch, _donor_batch, base_suffix, hybrid_suffix in confirmation_captures:
        lambda0 = backend.model.transformer.h[selected_boundary].lambdas[0]
        for i, query in enumerate(base_batch.semantic_positions):
            reconstructed = (
                lambda0.float() * (
                    hybrid_suffix[f"resid{selected_boundary}"][i, query].float()
                    - base_suffix[f"resid{selected_boundary}"][i, query].float()
                )
                + hybrid_suffix[f"attention{selected_boundary}"][i, query].float()
                - base_suffix[f"attention{selected_boundary}"][i, query].float()
                + hybrid_suffix[f"mlp{selected_boundary}"][i, query].float()
                - base_suffix[f"mlp{selected_boundary}"][i, query].float()
            )
            direct = (
                hybrid_suffix[f"resid{selected_boundary + 1}"][i, query].float()
                - base_suffix[f"resid{selected_boundary + 1}"][i, query].float()
            )
            selected_tensor_error_max_abs = max(
                selected_tensor_error_max_abs, float((reconstructed - direct).abs().max())
            )
        outputs = {
            arm_id(subset): backend.crossing(
                base_batch, base_suffix, hybrid_suffix, selected_boundary, subset
            ) for subset in subsets()
        }
        outputs["direct_query_ceiling"] = backend.direct_query(
            base_batch, base_suffix, hybrid_suffix, selected_boundary + 1
        )
        forward_calls += len(subsets()) + 1
        evaluations += (len(subsets()) + 1) * len(chunk)
        full_name = arm_id(BASE_FACTORS)
        for full_pair, direct_pair in zip(
            outputs[full_name].answer_foil, outputs["direct_query_ceiling"].answer_foil
        ):
            full_to_ceiling_max_abs = max(
                full_to_ceiling_max_abs,
                abs(full_pair[0] - direct_pair[0]), abs(full_pair[1] - direct_pair[1]),
            )
        for name, output in outputs.items():
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil, value = recovery(row, pair, native)
                if name == "direct_query_ceiling":
                    ceiling_values[family].append(value)
                else:
                    subset = next(item for item in subsets() if arm_id(item) == name)
                    factorial_values[subset][family].append(value)
                confirmation_records.append({
                    "phase": "confirmation", "boundary": selected_boundary,
                    "arm_id": name, "family": family, "row_id": str(row["row_id"]),
                    "answer_logit": answer, "foil_logit": foil, "recovery": value,
                })

    factorial_summary = {}
    factorial_targets = {}
    for subset in subsets():
        families = {
            family: summarize(factorial_values[subset][family])
            for family in ("A1", "A2")
        }
        target = statistics.fmean(
            families[family]["mean_recovery"] for family in ("A1", "A2")
        )
        factorial_summary[arm_id(subset)] = {
            "factors": list(subset), "families": families,
            "mean_target_recovery": target,
        }
        factorial_targets[subset] = target
    shapley = {}
    for factor in BASE_FACTORS:
        total = 0.0
        for subset in subsets():
            if factor in subset:
                continue
            extended = tuple(item for item in BASE_FACTORS if item in set(subset) | {factor})
            weight = (
                math.factorial(len(subset))
                * math.factorial(len(BASE_FACTORS) - len(subset) - 1)
                / math.factorial(len(BASE_FACTORS))
            )
            total += weight * (factorial_targets[extended] - factorial_targets[subset])
        shapley[factor] = total
    full_name = arm_id(BASE_FACTORS)
    removal_damage = {
        factor: {
            family: factorial_summary[full_name]["families"][family]["mean_recovery"]
            - factorial_summary[arm_id(tuple(item for item in BASE_FACTORS if item != factor))]["families"][family]["mean_recovery"]
            for family in ("A1", "A2")
        }
        for factor in BASE_FACTORS
    }
    ceiling_summary = {
        family: summarize(ceiling_values[family]) for family in ("A1", "A2")
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

    current_capability = True
    for phase_rows in (selection, confirmation):
        for family in ("A1", "A2"):
            for direction in ("past_to_present", "present_to_past"):
                cell_rows = [
                    row for row in phase_rows
                    if row["transform_id"] == family and row["direction_id"] == direction
                ]
                for side in ("base", "donor"):
                    accuracy = sum(
                        native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows
                    ) / len(cell_rows)
                    current_capability = current_capability and accuracy >= 0.75

    pred_a = (
        current_capability and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and selected_tensor_error_max_abs <= 0.04
        and full_to_ceiling_max_abs <= 0.125
    )
    writer_family_ok = all(
        writer_summary[phase][family]["mean_recovery"] > 0.0
        and writer_summary[phase][family]["direction_fraction"] >= 0.75
        for phase in ("selection", "confirmation") for family in ("A1", "A2")
    )
    pred_b = (
        writer_family_ok and abs(pooled_writer - 0.2835613798233539) <= 0.01
        and resid18_to_writer_max_abs <= 1.0e-4
        and abs(
            curve_targets[18]
            - statistics.fmean(
                writer_summary["selection"][family]["mean_recovery"]
                for family in ("A1", "A2")
            )
        ) <= 1.0e-6
    )
    pred_c = (
        curve_targets[18] - curve_targets[10] >= 0.04
        and increments[selected_boundary] >= 0.007
        and all(
            curve_summary[f"resid{boundary}"]["families"][family]["direction_fraction"] >= 0.75
            for boundary in (10, 18) for family in ("A1", "A2")
        )
    )
    new_factors = ("attention", "mlp")
    pred_d = (
        shapley["attention"] + shapley["mlp"] > 0.0
        and factorial_targets[BASE_FACTORS] > factorial_targets[("carried",)]
        and any(
            all(removal_damage[factor][family] > 0.0 for family in ("A1", "A2"))
            for factor in new_factors
        )
    )
    pred_e = (
        len(selection_records) + len(confirmation_records) == 320
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = (
        "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e))
        else "null" if pred_a and pred_b and pred_e
        else "invalid"
    )
    reason = {
        "screen": "suffix_depth_selected_and_disjointly_factorized",
        "null": "suffix_gain_or_disjoint_component_confirmation_failed",
        "invalid": "authority_split_capability_instrument_endpoint_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "program_release_sha256": EXPECTED_PROGRAM_RELEASE_SHA256,
        "resid10_result_sha256": EXPECTED_RESID10_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_capability_and_exact_instrument": pred_a,
            "pred_b_writer_and_endpoint_recurrence": pred_b,
            "pred_c_suffix_gain_and_selection": pred_c,
            "pred_d_disjoint_factorial_confirmation": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "selection_curve": curve_summary,
            "selection_increments": {f"block{key}": value for key, value in increments.items()},
            "selected_boundary": selected_boundary,
            "selected_increment": increments[selected_boundary],
            "confirmation_factorial_arms": factorial_summary,
            "confirmation_shapley": shapley,
            "confirmation_full_removal_damage": removal_damage,
            "confirmation_direct_ceiling": {
                "families": ceiling_summary,
                "mean_target_recovery": statistics.fmean(
                    ceiling_summary[family]["mean_recovery"] for family in ("A1", "A2")
                ),
            },
            "writer": writer_summary,
            "pooled_writer_mean_recovery": pooled_writer,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "selected_boundary_tensor_reconstruction_max_abs": selected_tensor_error_max_abs,
            "full_to_direct_ceiling_scored_logit_max_abs": full_to_ceiling_max_abs,
            "resid18_to_writer_scored_logit_max_abs": resid18_to_writer_max_abs,
            "open_pooled_resid10_mean_recovery": resid10_authority["score"]["factorial_arms"]["carried9+attention9+mlp9"]["mean_target_recovery"],
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(selection_records) + len(confirmation_records),
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
        },
        "selection_records": selection_records,
        "confirmation_records": confirmation_records,
        "terminal": terminal, "reason": reason,
        "next_action": (
            "compile the disjointly confirmed suffix crossing into executable program v3"
            if terminal == "screen"
            else "retain native suffix dependency and do not promote a selected crossing"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"],
        "selection_curve": {key: value["mean_target_recovery"] for key, value in curve_summary.items()},
        "selected_boundary": selected_boundary,
        "selected_increment": increments[selected_boundary],
        "confirmation_shapley": shapley,
        "confirmation_full": factorial_targets[BASE_FACTORS],
        "confirmation_carried": factorial_targets[("carried",)],
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
