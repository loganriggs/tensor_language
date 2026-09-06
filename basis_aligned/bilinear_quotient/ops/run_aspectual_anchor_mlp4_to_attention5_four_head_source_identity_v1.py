#!/usr/bin/env python3
# BQGATE: frozen A-E attention5 source-identity predictions; CUDA is managed-queue only.
"""Exact source terms linking the MLP4 write to attention5 H7/H1/H6/H8."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_fast_screen_candidate_aspectual as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source_math
import run_aspectual_anchor_mlp4_induced_attention5_head_sweep_v1 as head_sweep
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_attention5_four_head_factorial_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_induced_attention5_four_head_factorial_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_to_attention5_four_head_source_identity_v1"
EXPECTED_PRIOR_SHA256 = "60adc338a560f9ddf01dc5f83d755121ce22df58b815281ef1f636d77f355971"
EXPECTED_PARENT_SHA256 = "18a88ee7142f18d98b1ffdb41deb338a81ff1b80be7b416fa68593b129532b96"
EXPECTED_PARENT_RUNNER_SHA256 = "8faca69c33ec457c04afff53ef604ebaaa40be9d28cb9e7037e04f78cf16cda7"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
HEADS = (7, 1, 6, 8)
WRITER_FACTORS = ("left_change", "right_change")
BANK = ("last", "period", "determiner")
ARMS = (
    "complete_four_heads",
    "all_sources",
    "last_period_determiner",
    "last",
    "period",
    "determiner",
    "cue",
    "self",
    "cue_self",
)
MODEL_FORWARDS_MAX = 28
EXAMPLE_EVALUATIONS_MAX = 896


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        PARENT: EXPECTED_PARENT_SHA256,
        PARENT_RUNNER: EXPECTED_PARENT_RUNNER_SHA256,
    }.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if parent.get("terminal") != "screen":
        raise ExperimentError("parent terminal changed")
    if parent["score"]["factor_ranking"] != ["h7", "h1", "h6", "h8"]:
        raise ExperimentError("licensed head set changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or len(ARMS) != 9:
        raise ExperimentError("population or arm inventory changed")
    return enriched, spec


class SourceIdentityBackend(head_sweep.Attention5Backend):
    def capture_attention5(self, batch: producer.ModelBatch):
        torch, F, model = self.torch, self.F, self.model
        attention = model.transformer.h[5].attn
        saved = {}

        def capture_inputs(_module, arguments):
            saved["current"] = arguments[0].detach().clone()
            saved["v1"] = arguments[1].detach().clone()

        def capture_heads(_module, arguments):
            flattened = arguments[0]
            head_dim = model.config.n_embd // model.config.n_head
            saved["head_output"] = flattened.view(
                len(batch.row_ids), flattened.shape[1], model.config.n_head, head_dim
            ).detach().clone()

        input_handle = attention.register_forward_pre_hook(capture_inputs)
        head_handle = attention.c_proj.register_forward_pre_hook(capture_heads)
        try:
            output = self.native(batch, capture=False)
        finally:
            head_handle.remove()
            input_handle.remove()
        if set(saved) != {"current", "v1", "head_output"}:
            raise ExperimentError("attention5 capture incomplete")

        current = saved["current"]
        batch_size, maximum, width = current.shape
        heads = model.config.n_head
        head_dim = width // heads
        cosine, sine = source_math.rope_tables(
            torch, maximum, head_dim, self.device, current.dtype
        )
        cosine = cosine[None, :, None, :]
        sine = sine[None, :, None, :]

        def qk(linear):
            value = F.rms_norm(
                linear(current).view(batch_size, maximum, heads, head_dim),
                (head_dim,),
            )
            return source_math.apply_rot(torch, value, cosine, sine)

        with torch.no_grad():
            value = attention.c_v(current).view(batch_size, maximum, heads, head_dim)
            effective_value = (
                (1.0 - attention.lamb) * value
                + attention.lamb * saved["v1"].view_as(value)
            )
            query, key = qk(attention.c_q), qk(attention.c_k)
            query2, key2 = qk(attention.c_q2), qk(attention.c_k2)
            score1 = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
            score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
            mask = torch.tril(
                torch.ones(maximum, maximum, device=self.device, dtype=torch.bool)
            )
            pattern = (score1 * score2).masked_fill(~mask, 0.0)
        reconstruction_error = 0.0
        for i, (query_position, token_row) in enumerate(
            zip(batch.semantic_positions, batch.token_rows)
        ):
            for head in HEADS:
                reconstructed = torch.stack([
                    pattern[i, head, query_position, position]
                    * effective_value[i, position, head]
                    for position in range(len(token_row))
                ]).sum(0)
                reconstruction_error = max(
                    reconstruction_error,
                    float((reconstructed.float() - saved["head_output"][i, query_position, head].float()).abs().max()),
                )
        capture = {
            "pattern": pattern.detach().clone(),
            "value": effective_value.detach().clone(),
            "head_output": saved["head_output"],
            "reconstruction_max_abs": reconstruction_error,
        }
        return output, capture

    def capture_writer_attention5(
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
                    changed[i, position] = (changed[i, position].float() + delta).to(changed.dtype)
            return changed

        handle = self.model.transformer.h[4].mlp.register_forward_hook(patch_mlp4)
        try:
            output, capture = self.capture_attention5(base_batch)
        finally:
            handle.remove()
        return output, capture, tensor_error

    def intervene_sources(
        self,
        base_batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_capture,
        hybrid_capture,
        arm: str,
    ):
        if arm not in ARMS:
            raise ExperimentError("source arm changed")
        head_dim = self.model.config.n_embd // self.model.config.n_head

        def patch_heads(_module, arguments):
            flattened = arguments[0]
            head_output = flattened.view(
                len(base_batch.row_ids),
                flattened.shape[1],
                self.model.config.n_head,
                head_dim,
            ).clone()
            for i, (base_ids, donor_ids, query_position) in enumerate(zip(
                base_batch.token_rows,
                donor_batch.token_rows,
                base_batch.semantic_positions,
            )):
                differences = [
                    position
                    for position, (base_id, donor_id) in enumerate(zip(base_ids, donor_ids))
                    if base_id != donor_id
                ]
                if len(base_ids) != len(donor_ids) or len(differences) != 1:
                    raise ExperimentError("cue alignment changed")
                cue = differences[0]
                positions = {
                    "cue": cue,
                    "last": cue + 1,
                    "period": cue + 2,
                    "determiner": cue + 3,
                    "self": query_position,
                }
                if arm == "complete_four_heads":
                    for head in HEADS:
                        head_output[i, query_position, head] = hybrid_capture["head_output"][i, query_position, head]
                    continue
                if arm == "all_sources":
                    selected_positions = tuple(range(query_position + 1))
                elif arm == "last_period_determiner":
                    selected_positions = tuple(positions[name] for name in BANK)
                elif arm == "cue_self":
                    selected_positions = (positions["cue"], positions["self"])
                else:
                    selected_positions = (positions[arm],)
                for position in selected_positions:
                    for head in HEADS:
                        base_term = (
                            base_capture["pattern"][i, head, query_position, position]
                            * base_capture["value"][i, position, head]
                        )
                        hybrid_term = (
                            hybrid_capture["pattern"][i, head, query_position, position]
                            * hybrid_capture["value"][i, position, head]
                        )
                        head_output[i, query_position, head] += hybrid_term - base_term
            return (head_output.reshape_as(flattened),) + tuple(arguments[1:])

        handle = self.model.transformer.h[5].attn.c_proj.register_forward_pre_hook(
            patch_heads
        )
        try:
            return self.native(base_batch, capture=False)
        finally:
            handle.remove()


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery missing or nonfinite")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_mlp4_to_attention5_four_head_source_identity_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "heads": list(HEADS),
        "source_bank": list(BANK),
        "arms": list(ARMS),
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
    backend = SourceIdentityBackend.load("cuda")
    native = {}
    arm_values = {arm: {"A1": [], "A2": []} for arm in ARMS}
    logits = {}
    raw_records = []
    native_capture_max_abs = 0.0
    vector_reconstruction_max_abs = 0.0
    tensor_error_max_abs = 0.0
    all_to_complete_max_abs = 0.0
    bank_to_complete_max_abs = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_native, base_bilinear = backend.capture_bilinear(base_batch)
            donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            base_output, base_attention = backend.capture_attention5(base_batch)
            _, hybrid_attention, tensor_error = backend.capture_writer_attention5(
                base_batch, donor_batch, base_bilinear, donor_bilinear
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            tensor_error_max_abs = max(tensor_error_max_abs, tensor_error)
            vector_reconstruction_max_abs = max(
                vector_reconstruction_max_abs,
                float(base_attention["reconstruction_max_abs"]),
                float(hybrid_attention["reconstruction_max_abs"]),
            )
            for reference, captured in zip(base_native.answer_foil, base_output.answer_foil):
                native_capture_max_abs = max(
                    native_capture_max_abs,
                    abs(reference[0] - captured[0]),
                    abs(reference[1] - captured[1]),
                )
            for side, output in (("base", base_native), ("donor", donor_native)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )
            outputs = {
                arm: backend.intervene_sources(
                    base_batch, donor_batch, base_attention, hybrid_attention, arm
                )
                for arm in ARMS
            }
            forward_calls += len(ARMS)
            evaluations += len(ARMS) * len(chunk)
            for complete, all_sources, bank in zip(
                outputs["complete_four_heads"].answer_foil,
                outputs["all_sources"].answer_foil,
                outputs["last_period_determiner"].answer_foil,
            ):
                all_to_complete_max_abs = max(
                    all_to_complete_max_abs,
                    abs(complete[0] - all_sources[0]),
                    abs(complete[1] - all_sources[1]),
                )
                bank_to_complete_max_abs = max(
                    bank_to_complete_max_abs,
                    abs(complete[0] - bank[0]),
                    abs(complete[1] - bank[1]),
                )
            for arm, output in outputs.items():
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    row_id = str(row["row_id"])
                    recovery = kernel.signed_pairwise_donor_recovery(
                        -native[(row_id, "base")].margin,
                        native[(row_id, "donor")].margin,
                        -(answer - foil),
                    )
                    arm_values[arm][family].append(recovery)
                    logits[(arm, row_id)] = (answer, foil)
                    raw_records.append({
                        "arm_id": arm, "family": family, "row_id": row_id,
                        "answer_logit": answer, "foil_logit": foil, "recovery": recovery,
                    })

    native_capability = True
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                native_capability = native_capability and accuracy >= 0.85
    summaries = {}
    targets = {}
    for arm in ARMS:
        families = {family: summarize(arm_values[arm][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm] = {"families": families, "mean_target_recovery": target}
        targets[arm] = target
    complete_target = targets["complete_four_heads"]
    bank_retained = targets["last_period_determiner"] / complete_target
    cue_self_absolute_fraction = abs(targets["cue_self"]) / statistics.fmean(
        summaries["complete_four_heads"]["families"][family]["mean_absolute_recovery"]
        for family in ("A1", "A2")
    )
    pred_a = native_capability and native_capture_max_abs <= 1.0e-4 and tensor_error_max_abs <= 2.0e-3 and vector_reconstruction_max_abs <= 1.0e-4 and all_to_complete_max_abs <= 0.125
    pred_b = abs(complete_target - 0.04557515628642643) <= 0.01 and all(summaries["complete_four_heads"]["families"][family]["mean_recovery"] > 0.0 and summaries["complete_four_heads"]["families"][family]["direction_fraction"] >= 0.80 for family in ("A1", "A2"))
    pred_c = bank_to_complete_max_abs <= 0.125 and bank_retained >= 0.95 and all(summaries["last_period_determiner"]["families"][family]["direction_fraction"] >= 0.80 for family in ("A1", "A2"))
    pred_d = cue_self_absolute_fraction <= 0.05
    expected_records = len(ARMS) * len(rows)
    pred_e = len(raw_records) == expected_records and len(logits) == expected_records and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if pred_a and pred_b and pred_c and pred_d and pred_e else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "exact_mlp4_bank_to_attention5_four_head_path", "null": "source_identity_or_control_failed", "invalid": "source_instrument_capability_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_mlp4_to_attention5_four_head_source_identity_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256, "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_attention_instrument": pred_a,
            "pred_b_four_head_recurrence": pred_b,
            "pred_c_source_bank_identity": pred_c,
            "pred_d_unchanged_source_control": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "native_capture_scored_logit_max_abs": native_capture_max_abs,
            "bilinear_tensor_reconstruction_max_abs": tensor_error_max_abs,
            "attention_vector_reconstruction_max_abs": vector_reconstruction_max_abs,
            "all_sources_to_complete_scored_logit_max_abs": all_to_complete_max_abs,
            "bank_to_complete_scored_logit_max_abs": bank_to_complete_max_abs,
            "bank_retained_fraction": bank_retained,
            "cue_self_absolute_complete_fraction": cue_self_absolute_fraction,
            "arms": summaries,
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": "compile the exact MLP4-bank-attention5 path and factor later accumulation" if terminal == "screen" else "retain four complete attention5 heads without source identity",
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "bank_retained": bank_retained, "cue_self_fraction": cue_self_absolute_fraction, "bank_closure": bank_to_complete_max_abs, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
