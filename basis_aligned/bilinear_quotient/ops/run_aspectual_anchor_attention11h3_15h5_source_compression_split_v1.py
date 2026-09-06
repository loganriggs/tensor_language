#!/usr/bin/env python3
# BQGATE: frozen A-E split source-compression predictions; CUDA is managed-queue only.
"""Localize and confirm compact source-role banks for suffix heads 11H3 and 15H5."""

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
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_attention11_15_head_compression_split_v1 as head_parent
import run_aspectual_anchor_attention11_15_single_head_confirmation_v1 as single_parent
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source_math
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention11h3_15h5_source_compression_split_v1.json"
SINGLE = ROOT / "circuits/followups/aspectual_anchor_attention11_15_single_head_confirmation_v1_result.json"
SINGLE_RUNNER = ROOT / "ops/run_aspectual_anchor_attention11_15_single_head_confirmation_v1.py"
HEAD_RUNNER = ROOT / "ops/run_aspectual_anchor_attention11_15_head_compression_split_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_split_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.attention11h3_15h5_source_compression_split_v1"
EXPECTED_PRIOR_SHA256 = "6cfd9e357cc3c7797c35ef206316918fbb832c6f5792d2c52a62e9022a688a96"
EXPECTED_SINGLE_SHA256 = "be2c405c4b7023c57d4a11baf9be0bc999fb3835908f34fb88c1011ab146353f"
EXPECTED_SINGLE_RUNNER_SHA256 = "36a3fe7c5e5066a1637877333b361bc6d87ba57b3f1fe8591fa0786108b448ec"
EXPECTED_HEAD_RUNNER_SHA256 = "225c822c37007445353febd133fe308230b4f47b25b572bfe2c2614cc54950f2"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_SELECTION_SHA256 = "d150ff72d1423058a01aa2140563315c041b1be98a59066e8dc4a98688775fe8"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
BOUNDARIES = (11, 15)
HEAD_BY_BOUNDARY = {11: 3, 15: 5}
ROLES = ("cue", "last", "period", "determiner", "self", "other")
SELECTED_WIDTH = 3
WRITER_FACTORS = ("left_change", "right_change")
MODEL_FORWARDS_MAX = 84
EXAMPLE_EVALUATIONS_MAX = 704


class ExperimentError(RuntimeError):
    pass


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def selection_arms() -> tuple[str, ...]:
    return (
        "no_sources", "all_sources",
        *(f"only_{role}" for role in ROLES),
        *(f"all_except_{role}" for role in ROLES),
    )


def validate_static():
    for file_path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        SINGLE: EXPECTED_SINGLE_SHA256,
        SINGLE_RUNNER: EXPECTED_SINGLE_RUNNER_SHA256,
        HEAD_RUNNER: EXPECTED_HEAD_RUNNER_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    single = json.loads(SINGLE.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID or single.get("terminal") != "screen":
        raise ExperimentError("prior or parent terminal changed")
    if single_parent.HEAD_BY_BOUNDARY != HEAD_BY_BOUNDARY:
        raise ExperimentError("dominant-head authority changed")
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
        experiment_id="aspectual-anchor-attention11h3-15h5-source-compression-split-v1",
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
    if len(selection) != 16 or len(confirmation) != 16 or len(selection_arms()) != 14:
        raise ExperimentError("population or source-arm inventory changed")
    return selection, confirmation, spec, single


class SourceCompressionBackend(head_parent.HeadCompressionBackend):
    def attention_terms(self, batch, capture, boundary):
        if boundary not in BOUNDARIES:
            raise ExperimentError("attention boundary changed")
        torch, F, model = self.torch, self.F, self.model
        tokens, _lengths = self._tensor_batch(batch)
        width, heads = model.config.n_embd, model.config.n_head
        head_dim, maximum = width // heads, tokens.shape[1]
        mask = torch.tril(torch.ones(maximum, maximum, device=self.device, dtype=torch.bool))
        block = model.transformer.h[boundary]
        live = block.lambdas[0] * capture[f"resid{boundary}"] + block.lambdas[1] * capture["x0"]
        current = F.rms_norm(live, (width,))
        cosine, sine = source_math.rope_tables(
            torch, maximum, head_dim, self.device, current.dtype
        )
        cosine, sine = cosine[None, :, None, :], sine[None, :, None, :]

        def qk(linear):
            value = F.rms_norm(
                linear(current).view(len(batch.row_ids), maximum, heads, head_dim),
                (head_dim,),
            )
            return source_math.apply_rot(torch, value, cosine, sine)

        attention = block.attn
        value = attention.c_v(current).view(len(batch.row_ids), maximum, heads, head_dim)
        v1 = capture[f"v1_before{boundary}"].view_as(value)
        effective = (1.0 - attention.lamb) * value + attention.lamb * v1
        query, key = qk(attention.c_q), qk(attention.c_k)
        query2, key2 = qk(attention.c_q2), qk(attention.c_k2)
        score1 = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
        score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
        pattern = (score1 * score2).masked_fill(~mask, 0.0)
        reconstructed = torch.einsum("bhqk,bkhd->bqhd", pattern, effective)
        error = float((reconstructed.float() - capture[f"head_output{boundary}"].float()).abs().max())
        return pattern.detach(), effective.detach(), error

    def role_positions(self, batch, donor_batch):
        banks = []
        for base_ids, donor_ids, query, donor_query in zip(
            batch.token_rows, donor_batch.token_rows,
            batch.semantic_positions, donor_batch.semantic_positions,
        ):
            if len(base_ids) != len(donor_ids) or query != donor_query:
                raise ExperimentError("paired alignment changed")
            differences = [
                position for position, pair in enumerate(zip(base_ids, donor_ids))
                if pair[0] != pair[1]
            ]
            if len(differences) != 1:
                raise ExperimentError("row does not have one aligned cue difference")
            cue = differences[0]
            singleton = {
                "cue": (cue,), "last": (cue + 1,), "period": (cue + 2,),
                "determiner": (cue + 3,), "self": (query,),
            }
            if len({positions[0] for positions in singleton.values()}) != 5:
                raise ExperimentError("source roles overlap")
            if any(not 0 <= positions[0] <= query for positions in singleton.values()):
                raise ExperimentError("source role is outside the causal prefix")
            occupied = {positions[0] for positions in singleton.values()}
            singleton["other"] = tuple(position for position in range(query + 1) if position not in occupied)
            flattened = [position for role in ROLES for position in singleton[role]]
            if len(flattened) != query + 1 or set(flattened) != set(range(query + 1)):
                raise ExperimentError("source roles do not partition the causal prefix")
            banks.append(singleton)
        return banks

    def projected_source_delta(
        self, batch, role_banks, base_terms, hybrid_terms, boundary, selected_roles
    ):
        if len(selected_roles) != len(set(selected_roles)) or any(role not in ROLES for role in selected_roles):
            raise ExperimentError("source subset changed")
        base_pattern, base_value = base_terms
        hybrid_pattern, hybrid_value = hybrid_terms
        head = HEAD_BY_BOUNDARY[boundary]
        width, heads = self.model.config.n_embd, self.model.config.n_head
        head_dim = width // heads
        delta_heads = self.torch.zeros(
            (len(batch.row_ids), base_value.shape[1], heads, head_dim),
            device=self.device, dtype=self.torch.float32,
        )
        for i, (query, banks) in enumerate(zip(batch.semantic_positions, role_banks)):
            sources = tuple(position for role in selected_roles for position in banks[role])
            delta = self.torch.zeros(head_dim, device=self.device, dtype=self.torch.float32)
            for source in sources:
                delta += (
                    hybrid_pattern[i, head, query, source].float() * hybrid_value[i, source, head].float()
                    - base_pattern[i, head, query, source].float() * base_value[i, source, head].float()
                )
            delta_heads[i, query, head] = delta
        weight = self.model.transformer.h[boundary].attn.c_proj.weight.float()
        return self.F.linear(delta_heads.reshape(len(batch.row_ids), base_value.shape[1], width), weight, None)

    def source_crossing(
        self, batch, role_banks, base_capture, hybrid_capture, base_terms, hybrid_terms,
        boundary, selected_roles,
    ):
        state = base_capture[f"resid{boundary + 1}"].clone()
        lambda0 = self.model.transformer.h[boundary].lambdas[0]
        projected = self.projected_source_delta(
            batch, role_banks, base_terms, hybrid_terms, boundary, selected_roles
        )
        for i, query in enumerate(batch.semantic_positions):
            delta = (
                lambda0.float() * (
                    hybrid_capture[f"resid{boundary}"][i, query].float()
                    - base_capture[f"resid{boundary}"][i, query].float()
                )
                + projected[i, query]
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
        "count": len(values), "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    selection, confirmation, spec, parent_result = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_attention11h3_15h5_source_compression_split_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "selection_rows": len(selection), "confirmation_rows": len(confirmation),
        "head_by_boundary": HEAD_BY_BOUNDARY, "source_roles": list(ROLES),
        "selection_arms_per_boundary": len(selection_arms()),
        "selected_width": SELECTED_WIDTH, "confirmation_arms_per_boundary": 3,
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
    backend = SourceCompressionBackend.load("cuda")
    native = {}
    captures = {"selection": [], "confirmation": []}
    writer_values = {phase: {family: [] for family in ("A1", "A2")} for phase in captures}
    selection_values = {
        boundary: {arm: {family: [] for family in ("A1", "A2")} for arm in selection_arms()}
        for boundary in BOUNDARIES
    }
    confirmation_values = {
        boundary: {arm: {family: [] for family in ("A1", "A2")} for arm in ("no_sources", "selected_three", "all_sources")}
        for boundary in BOUNDARIES
    }
    parent_logits = {
        (int(record["boundary"]), record["row_id"]): (record["answer_logit"], record["foil_logit"])
        for record in parent_result["intervention_logits"]
        if record["boundary"] in BOUNDARIES and record["arm_id"] == "dominant_single"
    }
    raw_records = []
    manual_base_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    term_reconstruction_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    source_projection_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    head_control_logit_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
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
                        base_batch, role_banks, terms[boundary][0], terms[boundary][1], boundary, ROLES
                    )
                    projected_head = backend.projected_head_delta(
                        base_capture, hybrid_capture, boundary, (HEAD_BY_BOUNDARY[boundary],)
                    )
                    source_projection_max_abs[boundary] = max(
                        source_projection_max_abs[boundary],
                        float((projected_sources - projected_head).abs().max()),
                    )
                captures[phase].append((family, chunk, base_batch, role_banks, base_capture, hybrid_capture, terms))

    for family, chunk, base_batch, role_banks, base_capture, hybrid_capture, terms in captures["selection"]:
        for boundary in BOUNDARIES:
            role_sets = {
                "no_sources": (), "all_sources": ROLES,
                **{f"only_{role}": (role,) for role in ROLES},
                **{f"all_except_{role}": tuple(other for other in ROLES if other != role) for role in ROLES},
            }
            for arm, selected_roles in role_sets.items():
                output = backend.source_crossing(
                    base_batch, role_banks, base_capture, hybrid_capture, *terms[boundary], boundary, selected_roles
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

    selection_summary, selected_roles = {}, {}
    for boundary in BOUNDARIES:
        arm_summaries, targets = {}, {}
        for arm in selection_arms():
            families = {family: summarize(selection_values[boundary][arm][family]) for family in ("A1", "A2")}
            target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
            arm_summaries[arm] = {"families": families, "mean_target_recovery": target}
            targets[arm] = target
        attributions = {}
        for role in ROLES:
            singleton = targets[f"only_{role}"] - targets["no_sources"]
            necessity = targets["all_sources"] - targets[f"all_except_{role}"]
            attributions[role] = {
                "singleton_increment": singleton, "full_minus_leave_one_out_increment": necessity,
                "selection_score": 0.5 * (singleton + necessity),
            }
        ranking = sorted(ROLES, key=lambda role: (-attributions[role]["selection_score"], ROLES.index(role)))
        selected_roles[boundary] = tuple(ranking[:SELECTED_WIDTH])
        selection_summary[str(boundary)] = {
            "arms": arm_summaries, "attributions": attributions, "ranking": ranking,
            "selected_roles": list(selected_roles[boundary]),
            "all_minus_none_increment": targets["all_sources"] - targets["no_sources"],
        }

    for family, chunk, base_batch, role_banks, base_capture, hybrid_capture, terms in captures["confirmation"]:
        for boundary in BOUNDARIES:
            for arm, selected in {
                "no_sources": (), "selected_three": selected_roles[boundary], "all_sources": ROLES,
            }.items():
                output = backend.source_crossing(
                    base_batch, role_banks, base_capture, hybrid_capture, *terms[boundary], boundary, selected
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
                    if arm == "all_sources":
                        control = parent_logits[(boundary, str(row["row_id"]))]
                        head_control_logit_max_abs[boundary] = max(
                            head_control_logit_max_abs[boundary], abs(answer - control[0]), abs(foil - control[1])
                        )

    confirmation_summary, compression_pass = {}, []
    for boundary in BOUNDARIES:
        arm_summaries, targets = {}, {}
        for arm in ("no_sources", "selected_three", "all_sources"):
            families = {family: summarize(confirmation_values[boundary][arm][family]) for family in ("A1", "A2")}
            target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
            arm_summaries[arm] = {"families": families, "mean_target_recovery": target}
            targets[arm] = target
        denominator = targets["all_sources"] - targets["no_sources"]
        numerator = targets["selected_three"] - targets["no_sources"]
        retained = numerator / denominator
        family_increments = {
            family: arm_summaries["selected_three"]["families"][family]["mean_recovery"]
            - arm_summaries["no_sources"]["families"][family]["mean_recovery"]
            for family in ("A1", "A2")
        }
        compression_pass.append(retained >= 0.70 and all(value > 0.0 for value in family_increments.values()))
        confirmation_summary[str(boundary)] = {
            "arms": arm_summaries, "selected_roles": list(selected_roles[boundary]),
            "selected_source_increment": numerator, "all_source_increment": denominator,
            "selected_to_all_source_fraction": retained,
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
        and all(value <= 0.04 for value in source_projection_max_abs.values())
        and all(value <= 0.125 for value in head_control_logit_max_abs.values())
    )
    pred_b = (
        abs(pooled_writer - 0.2835613798233539) <= 0.01
        and all(
            writer_summary[phase][family]["mean_recovery"] > 0.0
            and writer_summary[phase][family]["direction_fraction"] >= 0.75
            for phase in ("selection", "confirmation") for family in ("A1", "A2")
        )
        and all(value <= 0.125 for value in head_control_logit_max_abs.values())
    )
    pred_c = all(
        selection_summary[str(boundary)]["all_minus_none_increment"] > 0.0
        and len(set(selected_roles[boundary])) == SELECTED_WIDTH for boundary in BOUNDARIES
    )
    pred_d = all(compression_pass)
    pred_e = (
        len(raw_records) == 576
        and len({(record["phase"], str(record["boundary"]), record["arm_id"], record["row_id"]) for record in raw_records}) == 576
        and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else (
        "null" if pred_a and pred_b and pred_c and pred_e else "invalid"
    )
    reason = {
        "screen": "attention11h3_15h5_three_role_source_banks_transfer_disjointly",
        "null": "one_or_both_three_role_source_banks_failed_disjoint_compression",
        "invalid": "authority_split_capability_instrument_control_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_attention11h3_15h5_source_compression_split_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "single_head_result_sha256": EXPECTED_SINGLE_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_split_capability_and_exact_instrument": pred_a,
            "pred_b_writer_and_head_control_recurrence": pred_b,
            "pred_c_positive_selection_source_signal": pred_c,
            "pred_d_disjoint_three_role_compression": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "selection": selection_summary, "confirmation": confirmation_summary,
            "writer": writer_summary, "pooled_writer_mean_recovery": pooled_writer,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "attention_term_reconstruction_max_abs": {str(key): value for key, value in term_reconstruction_max_abs.items()},
            "all_source_to_selected_head_projection_max_abs": {str(key): value for key, value in source_projection_max_abs.items()},
            "all_source_to_parent_single_head_logit_max_abs": {str(key): value for key, value in head_control_logit_max_abs.items()},
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": (
            "compile the validated 11H3 and 15H5 source banks into the transparent suffix program"
            if terminal == "screen" else "retain the dominant heads without a three-role source compression claim"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"],
        "selection": {boundary: list(selected_roles[boundary]) for boundary in BOUNDARIES},
        "confirmation_fraction": {
            boundary: confirmation_summary[str(boundary)]["selected_to_all_source_fraction"]
            for boundary in BOUNDARIES
        },
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
