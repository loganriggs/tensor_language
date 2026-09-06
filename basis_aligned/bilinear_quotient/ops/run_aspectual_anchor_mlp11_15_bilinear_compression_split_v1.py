#!/usr/bin/env python3
# BQGATE: frozen A-E split suffix-MLP bilinear predictions; CUDA is managed-queue only.
"""Select and confirm exact bilinear response terms for MLP11 and MLP15."""

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
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_attention11h3_15h5_source_compression_split_v1 as source_parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp11_15_bilinear_compression_split_v1.json"
SOURCE_RELEASE = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_release_v1_result.json"
SOURCE_RESULT = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_split_v1_result.json"
SOURCE_RUNNER = ROOT / "ops/run_aspectual_anchor_attention11h3_15h5_source_compression_split_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_split_v1"
EXPECTED_PRIOR_SHA256 = "e0f6f263fce698c93378bbf9ecfda8274ac113119d5e9fe0efa4a0f8dc254622"
EXPECTED_SOURCE_RELEASE_SHA256 = "e80f06ef21344139d33d7bc0793a20f564bf360ad2f7ea2d76edd52ab3421df5"
EXPECTED_SOURCE_RESULT_SHA256 = "7b4ba19260f4d311ca959331e60f29c64136f416566582d510591c96d5243adb"
EXPECTED_SOURCE_RUNNER_SHA256 = "419c0be9a1c9cebf594225c492b9402af7240d7ffd95d7987320d3e2ecb18a30"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_SELECTION_SHA256 = "d150ff72d1423058a01aa2140563315c041b1be98a59066e8dc4a98688775fe8"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
BOUNDARIES = (11, 15)
FACTORS = ("left_change", "right_change", "bilinear_interaction")
SELECTED_WIDTH = 2
MODEL_FORWARDS_MAX = 60
EXAMPLE_EVALUATIONS_MAX = 480


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets() -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset for width in range(len(FACTORS) + 1)
        for subset in itertools.combinations(FACTORS, width)
    )


def arm_id(subset: tuple[str, ...]) -> str:
    return "empty" if not subset else "+".join(subset)


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        SOURCE_RELEASE: EXPECTED_SOURCE_RELEASE_SHA256,
        SOURCE_RESULT: EXPECTED_SOURCE_RESULT_SHA256,
        SOURCE_RUNNER: EXPECTED_SOURCE_RUNNER_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    source_release = json.loads(SOURCE_RELEASE.read_text())
    source_result = json.loads(SOURCE_RESULT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or source_release.get("terminal") != "release"
        or source_result.get("terminal") != "invalid"
    ):
        raise ExperimentError("prior or source authority changed")
    expected_banks = {
        "11": {"head": 3, "source_roles": ["determiner", "period", "self"]},
        "15": {"head": 5, "source_roles": ["period", "determiner", "self"]},
    }
    for boundary, expected in expected_banks.items():
        actual = source_release["released_banks"][boundary]
        if actual["head"] != expected["head"] or actual["source_roles"] != expected["source_roles"]:
            raise ExperimentError("released source bank changed")
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
        experiment_id="aspectual-anchor-mlp11-15-bilinear-compression-split-v1",
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
    if len(selection) != 16 or len(confirmation) != 16 or len(subsets()) != 8:
        raise ExperimentError("population or factorial changed")
    return selection, confirmation, spec, source_release, source_result


class SuffixMlpBackend(source_parent.SourceCompressionBackend):
    def mlp_states(self, capture, boundary):
        F = self.F
        block = self.model.transformer.h[boundary]
        live = block.lambdas[0] * capture[f"resid{boundary}"] + block.lambdas[1] * capture["x0"]
        normalized = F.rms_norm(live + capture[f"attention{boundary}"], (self.model.config.n_embd,))
        return block.mlp.Left(normalized).detach(), block.mlp.Right(normalized).detach()

    def projected_mlp_terms(self, base_capture, hybrid_capture, boundary):
        F = self.F
        left_base, right_base = self.mlp_states(base_capture, boundary)
        left_hybrid, right_hybrid = self.mlp_states(hybrid_capture, boundary)
        delta_left = left_hybrid.float() - left_base.float()
        delta_right = right_hybrid.float() - right_base.float()
        hidden = {
            "left_change": delta_left * right_base.float(),
            "right_change": left_base.float() * delta_right,
            "bilinear_interaction": delta_left * delta_right,
        }
        weight = self.model.transformer.h[boundary].mlp.Down.weight.float()
        projected = {name: F.linear(value, weight, None) for name, value in hidden.items()}
        direct = hybrid_capture[f"mlp{boundary}"].float() - base_capture[f"mlp{boundary}"].float()
        error = float((sum(projected.values()) - direct).abs().max())
        return projected, error

    def factor_crossing(
        self, batch, role_banks, base_capture, hybrid_capture, base_terms,
        hybrid_terms, boundary, selected_factors,
    ):
        if selected_factors not in subsets():
            raise ExperimentError("MLP factor subset changed")
        state = base_capture[f"resid{boundary + 1}"].clone()
        lambda0 = self.model.transformer.h[boundary].lambdas[0]
        projected_attention = self.projected_source_delta(
            batch, role_banks, base_terms, hybrid_terms, boundary,
            tuple(source_parent.ROLES[index] for index in range(len(source_parent.ROLES)))
        )
        projected_mlp, _error = self.projected_mlp_terms(base_capture, hybrid_capture, boundary)
        for i, query in enumerate(batch.semantic_positions):
            delta = (
                lambda0.float() * (
                    hybrid_capture[f"resid{boundary}"][i, query].float()
                    - base_capture[f"resid{boundary}"][i, query].float()
                )
                + projected_attention[i, query]
            )
            for factor in selected_factors:
                delta = delta + projected_mlp[factor][i, query]
            state[i, query] = (state[i, query].float() + delta).to(state.dtype)
        return self.suffix_from_resid(
            batch, state, base_capture["x0"],
            base_capture[f"v1_after{boundary}"], boundary + 1,
        )


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery is missing or nonfinite")
    return {
        "count": len(values), "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def shapley(targets: dict[tuple[str, ...], float]) -> dict[str, float]:
    values = {}
    factorial = math.factorial
    for factor in FACTORS:
        others = tuple(other for other in FACTORS if other != factor)
        contribution = 0.0
        for width in range(len(others) + 1):
            for subset in itertools.combinations(others, width):
                with_factor = tuple(item for item in FACTORS if item in set(subset) | {factor})
                ordered_subset = tuple(item for item in FACTORS if item in subset)
                weight = factorial(width) * factorial(len(FACTORS) - width - 1) / factorial(len(FACTORS))
                contribution += weight * (targets[with_factor] - targets[ordered_subset])
        values[factor] = contribution
    return values


def main() -> None:
    selection, confirmation, spec, source_release, source_result = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_mlp11_15_bilinear_compression_split_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "selection_rows": len(selection), "confirmation_rows": len(confirmation),
        "boundaries": list(BOUNDARIES), "factors": list(FACTORS),
        "factorial_arms_per_boundary": len(subsets()), "selected_width": SELECTED_WIDTH,
        "confirmation_arms_per_boundary": 3,
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = SuffixMlpBackend.load("cuda")
    native = {}
    captures = {"selection": [], "confirmation": []}
    writer_values = {phase: {family: [] for family in ("A1", "A2")} for phase in captures}
    selection_values = {
        boundary: {subset: {family: [] for family in ("A1", "A2")} for subset in subsets()}
        for boundary in BOUNDARIES
    }
    confirmation_values = {
        boundary: {arm: {family: [] for family in ("A1", "A2")} for arm in ("empty", "selected_two", "all_three")}
        for boundary in BOUNDARIES
    }
    parent_logits = {
        (record["phase"], int(record["boundary"]), record["row_id"]): (record["answer_logit"], record["foil_logit"])
        for record in source_result["intervention_logits"]
        if record["boundary"] in BOUNDARIES and record["arm_id"] == "all_sources"
    }
    raw_records = []
    manual_base_max_abs = writer_tensor_error_max_abs = 0.0
    term_reconstruction_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    source_query_projection_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    mlp_reconstruction_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    all_three_control_logit_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    forward_calls = evaluations = 0

    for phase, phase_rows in (("selection", selection), ("confirmation", confirmation)):
        for family in ("A1", "A2"):
            family_rows = [row for row in phase_rows if row["transform_id"] == family]
            for chunk in producer._chunks(family_rows, spec.batch_size):
                base_batch = producer._batch(spec, chunk, "base")
                donor_batch = producer._batch(spec, chunk, "donor")
                role_banks = backend.role_positions(base_batch, donor_batch)
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
                        manual_base_max_abs, abs(reference[0] - manual[0]), abs(reference[1] - manual[1])
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
                terms = {}
                for boundary in BOUNDARIES:
                    base_pattern, base_value, base_error = backend.attention_terms(base_batch, base_capture, boundary)
                    hybrid_pattern, hybrid_value, hybrid_error = backend.attention_terms(base_batch, hybrid_capture, boundary)
                    terms[boundary] = ((base_pattern, base_value), (hybrid_pattern, hybrid_value))
                    term_reconstruction_max_abs[boundary] = max(
                        term_reconstruction_max_abs[boundary], base_error, hybrid_error
                    )
                    projected_sources = backend.projected_source_delta(
                        base_batch, role_banks, *terms[boundary], boundary, source_parent.ROLES
                    )
                    projected_head = backend.projected_head_delta(
                        base_capture, hybrid_capture, boundary,
                        (source_parent.HEAD_BY_BOUNDARY[boundary],),
                    )
                    for i, query in enumerate(base_batch.semantic_positions):
                        source_query_projection_max_abs[boundary] = max(
                            source_query_projection_max_abs[boundary],
                            float((projected_sources[i, query] - projected_head[i, query]).abs().max()),
                        )
                    _projected_mlp, mlp_error = backend.projected_mlp_terms(
                        base_capture, hybrid_capture, boundary
                    )
                    mlp_reconstruction_max_abs[boundary] = max(
                        mlp_reconstruction_max_abs[boundary], mlp_error
                    )
                captures[phase].append((family, chunk, base_batch, role_banks, base_capture, hybrid_capture, terms))

    for family, chunk, base_batch, role_banks, base_capture, hybrid_capture, terms in captures["selection"]:
        for boundary in BOUNDARIES:
            for subset in subsets():
                output = backend.factor_crossing(
                    base_batch, role_banks, base_capture, hybrid_capture,
                    *terms[boundary], boundary, subset
                )
                forward_calls += 1
                evaluations += len(chunk)
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil, value = suffix.recovery(row, pair, native)
                    selection_values[boundary][subset][family].append(value)
                    raw_records.append({
                        "phase": "selection", "boundary": boundary, "arm_id": arm_id(subset),
                        "family": family, "row_id": str(row["row_id"]),
                        "answer_logit": answer, "foil_logit": foil, "recovery": value,
                    })
                    if subset == FACTORS:
                        control = parent_logits[("selection", boundary, str(row["row_id"]))]
                        all_three_control_logit_max_abs[boundary] = max(
                            all_three_control_logit_max_abs[boundary],
                            abs(answer - control[0]), abs(foil - control[1]),
                        )

    selection_summary, selected_factors = {}, {}
    for boundary in BOUNDARIES:
        arm_summaries, targets = {}, {}
        for subset in subsets():
            families = {family: summarize(selection_values[boundary][subset][family]) for family in ("A1", "A2")}
            target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
            arm_summaries[arm_id(subset)] = {"families": families, "mean_target_recovery": target}
            targets[subset] = target
        contributions = shapley(targets)
        ranking = sorted(FACTORS, key=lambda factor: (-contributions[factor], FACTORS.index(factor)))
        selected_factors[boundary] = tuple(factor for factor in FACTORS if factor in ranking[:SELECTED_WIDTH])
        selection_summary[str(boundary)] = {
            "arms": arm_summaries, "shapley": contributions, "ranking": ranking,
            "selected_factors": list(selected_factors[boundary]),
            "all_minus_empty_increment": targets[FACTORS] - targets[()],
        }

    for family, chunk, base_batch, role_banks, base_capture, hybrid_capture, terms in captures["confirmation"]:
        for boundary in BOUNDARIES:
            for arm, selected in {
                "empty": (), "selected_two": selected_factors[boundary], "all_three": FACTORS,
            }.items():
                output = backend.factor_crossing(
                    base_batch, role_banks, base_capture, hybrid_capture,
                    *terms[boundary], boundary, selected
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
                    if arm == "all_three":
                        control = parent_logits[("confirmation", boundary, str(row["row_id"]))]
                        all_three_control_logit_max_abs[boundary] = max(
                            all_three_control_logit_max_abs[boundary],
                            abs(answer - control[0]), abs(foil - control[1]),
                        )

    confirmation_summary, compression_pass = {}, []
    for boundary in BOUNDARIES:
        arm_summaries, targets = {}, {}
        for arm in ("empty", "selected_two", "all_three"):
            families = {family: summarize(confirmation_values[boundary][arm][family]) for family in ("A1", "A2")}
            target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
            arm_summaries[arm] = {"families": families, "mean_target_recovery": target}
            targets[arm] = target
        denominator = targets["all_three"] - targets["empty"]
        numerator = targets["selected_two"] - targets["empty"]
        retained = numerator / denominator
        family_increments = {
            family: arm_summaries["selected_two"]["families"][family]["mean_recovery"]
            - arm_summaries["empty"]["families"][family]["mean_recovery"]
            for family in ("A1", "A2")
        }
        compression_pass.append(retained >= 0.70 and all(value > 0.0 for value in family_increments.values()))
        confirmation_summary[str(boundary)] = {
            "arms": arm_summaries, "selected_factors": list(selected_factors[boundary]),
            "selected_mlp_increment": numerator, "all_mlp_increment": denominator,
            "selected_to_all_mlp_fraction": retained,
            "selected_family_increments": family_increments,
        }

    writer_summary = {
        phase: {family: summarize(writer_values[phase][family]) for family in ("A1", "A2")}
        for phase in ("selection", "confirmation")
    }
    pooled_writer = statistics.fmean(
        value for phase in ("selection", "confirmation") for family in ("A1", "A2")
        for value in writer_values[phase][family]
    )
    current_capability = all(
        native[(str(row["row_id"]), side)].margin > 0.0
        for row in selection + confirmation for side in ("base", "donor")
    )
    pred_a = (
        current_capability and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and all(value <= 1.0e-4 for value in term_reconstruction_max_abs.values())
        and all(value <= 0.04 for value in source_query_projection_max_abs.values())
        and all(value <= 0.04 for value in mlp_reconstruction_max_abs.values())
    )
    pred_b = (
        abs(pooled_writer - 0.2835613798233539) <= 0.01
        and all(
            writer_summary[phase][family]["mean_recovery"] > 0.0
            and writer_summary[phase][family]["direction_fraction"] >= 0.75
            for phase in ("selection", "confirmation") for family in ("A1", "A2")
        )
        and all(value <= 0.125 for value in all_three_control_logit_max_abs.values())
    )
    pred_c = all(
        selection_summary[str(boundary)]["all_minus_empty_increment"] > 0.0
        and len(set(selected_factors[boundary])) == SELECTED_WIDTH for boundary in BOUNDARIES
    )
    pred_d = all(compression_pass)
    pred_e = (
        len(raw_records) == 384
        and len({(record["phase"], str(record["boundary"]), record["arm_id"], record["row_id"]) for record in raw_records}) == 384
        and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else (
        "null" if pred_a and pred_b and pred_c and pred_e else "invalid"
    )
    reason = {
        "screen": "mlp11_15_two_term_bilinear_responses_transfer_disjointly",
        "null": "one_or_both_two_term_mlp_responses_failed_disjoint_compression",
        "invalid": "authority_split_capability_instrument_control_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_mlp11_15_bilinear_compression_split_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "source_release_sha256": EXPECTED_SOURCE_RELEASE_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_split_capability_and_exact_instrument": pred_a,
            "pred_b_writer_and_source_control_recurrence": pred_b,
            "pred_c_positive_selection_mlp_signal": pred_c,
            "pred_d_disjoint_two_term_compression": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "selection": selection_summary, "confirmation": confirmation_summary,
            "writer": writer_summary, "pooled_writer_mean_recovery": pooled_writer,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "attention_term_reconstruction_max_abs": {str(key): value for key, value in term_reconstruction_max_abs.items()},
            "source_query_projection_max_abs": {str(key): value for key, value in source_query_projection_max_abs.items()},
            "mlp_bilinear_reconstruction_max_abs": {str(key): value for key, value in mlp_reconstruction_max_abs.items()},
            "all_three_to_source_control_logit_max_abs": {str(key): value for key, value in all_three_control_logit_max_abs.items()},
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": (
            "compile the selected MLP11 and MLP15 bilinear terms into transparent program v4"
            if terminal == "screen" else "retain full native MLP11 and MLP15 deltas"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"],
        "selection": {boundary: list(selected_factors[boundary]) for boundary in BOUNDARIES},
        "confirmation_fraction": {
            boundary: confirmation_summary[str(boundary)]["selected_to_all_mlp_fraction"]
            for boundary in BOUNDARIES
        },
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
