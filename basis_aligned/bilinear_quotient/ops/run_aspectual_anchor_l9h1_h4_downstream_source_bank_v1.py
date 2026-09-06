#!/usr/bin/env python3
# BQGATE: frozen A-E multi-source predictions; CUDA is managed-queue only.
"""Exact multi-source compression of the L9H1/H4 aspectual read."""

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
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_l9h1_h4_downstream_source_bank_v1.json"
SOURCE_RESULT = ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_source_term_factorial_v1_result.json"
SOURCE_RUNNER = ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_downstream_source_bank_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.l9h1_h4_downstream_source_bank_v1"
EXPECTED_PRIOR_SHA256 = "57453d4d9bc531976b429b2269f44d72434166d273512ae2b267095913d16954"
EXPECTED_SOURCE_RESULT_SHA256 = "92fec63632f91be57a8bd85322366e8e1d2d41367974ac5e85c897f6d3433d4f"
EXPECTED_SOURCE_RUNNER_SHA256 = "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
HEADS = (1, 4)
SOURCE_NAMES = ("cue", "last", "period", "determiner", "self")
ARMS = {
    "full_pair": None,
    "all_changed_sources": SOURCE_NAMES,
    "period_determiner": ("period", "determiner"),
    "last_period_determiner": ("last", "period", "determiner"),
    "cue_last_period_determiner": ("cue", "last", "period", "determiner"),
    "period": ("period",),
    "determiner": ("determiner",),
    "cue_self": ("cue", "self"),
}
MODEL_FORWARDS_MAX = 36
EXAMPLE_EVALUATIONS_MAX = 1152
IDENTITY_TOLERANCE = 1.0e-4


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    if sha256(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior-art hash changed")
    if sha256(SOURCE_RESULT) != EXPECTED_SOURCE_RESULT_SHA256:
        raise ExperimentError("source-factorial result hash changed")
    if sha256(SOURCE_RUNNER) != EXPECTED_SOURCE_RUNNER_SHA256:
        raise ExperimentError("source-factorial runner hash changed")
    prior = json.loads(PRIOR.read_text())
    source_result = json.loads(SOURCE_RESULT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if source_result.get("terminal") != "null" or not source_result["predictions"]["pred_a_exact_instrument"]:
        raise ExperimentError("source-factorial authority changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or tuple(ARMS) != (
        "full_pair", "all_changed_sources", "period_determiner",
        "last_period_determiner", "cue_last_period_determiner", "period",
        "determiner", "cue_self",
    ):
        raise ExperimentError("population or arm inventory changed")
    return enriched, spec


class SourceBankBackend(source.SourceBackend):
    def patched_sources(
        self,
        batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_capture: dict[str, object],
        donor_capture: dict[str, object],
        source_names: tuple[str, ...] | None,
    ) -> producer.BatchOutput:
        if source_names is not None and (
            not source_names
            or len(source_names) != len(set(source_names))
            or any(name not in SOURCE_NAMES for name in source_names)
        ):
            raise ExperimentError("source bank changed")
        torch = self.torch
        head_dim = self.model.config.n_embd // self.model.config.n_head

        def patch_preprojection(_module, arguments):
            flattened = arguments[0]
            head_output = flattened.view(
                len(batch.row_ids), flattened.shape[1], self.model.config.n_head, head_dim
            ).clone()
            base_pattern = base_capture["pattern"]
            base_value = base_capture["value"]
            donor_pattern = donor_capture["pattern"]
            donor_value = donor_capture["value"]
            donor_head_output = donor_capture["head_output"]
            for i, (base_ids, donor_ids, q, donor_q) in enumerate(zip(
                batch.token_rows,
                donor_batch.token_rows,
                batch.semantic_positions,
                donor_batch.semantic_positions,
            )):
                differences = [
                    position for position, (base_id, donor_id) in enumerate(
                        zip(base_ids, donor_ids)
                    ) if base_id != donor_id
                ]
                if len(base_ids) != len(donor_ids) or len(differences) != 1:
                    raise ExperimentError("cue alignment changed")
                cue = differences[0]
                positions = {
                    "cue": cue,
                    "last": cue + 1,
                    "period": cue + 2,
                    "determiner": cue + 3,
                    "self": q,
                }
                if positions["determiner"] >= len(base_ids) or positions["self"] != len(base_ids) - 1:
                    raise ExperimentError("source-position contract changed")
                if source_names is None:
                    for head in HEADS:
                        head_output[i, q, head] = donor_head_output[i, donor_q, head]
                    continue
                for name in source_names:
                    position = positions[name]
                    for head in HEADS:
                        base_term = base_pattern[i, head, q, position] * base_value[i, position, head]
                        donor_term = donor_pattern[i, head, donor_q, position] * donor_value[i, position, head]
                        head_output[i, q, head] += donor_term - base_term
            return (head_output.reshape_as(flattened),) + tuple(arguments[1:])

        handle = self.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(
            patch_preprojection
        )
        try:
            return self.native(batch, capture=False)
        finally:
            handle.remove()


def family_summary(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("family recovery missing or nonfinite")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_l9h1_h4_downstream_source_bank_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "source_factorial_sha256": EXPECTED_SOURCE_RESULT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
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
    backend = SourceBankBackend.load("cuda")
    native: dict[tuple[str, str], producer.NativeLogitEvidence] = {}
    batch_pairs = []
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_output, base_capture = backend.manual_forward(base_batch)
            donor_output, donor_capture = backend.manual_forward(donor_batch)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for side, batch, output in (
                ("base", base_batch, base_output), ("donor", donor_batch, donor_output)
            ):
                for row_id, pair in zip(batch.row_ids, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(row_id, side)] = producer.NativeLogitEvidence(
                        row_id, family, side, answer, foil  # type: ignore[arg-type]
                    )
            batch_pairs.append((family, tuple(chunk), base_batch, donor_batch, base_capture, donor_capture))

    native_capability = True
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            cell_rows = [
                row for row in rows
                if row["transform_id"] == family and row["direction_id"] == direction
            ]
            for side in ("base", "donor"):
                accuracy = sum(
                    native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows
                ) / len(cell_rows)
                native_capability = native_capability and accuracy >= 0.85

    recoveries = {arm: {"A1": [], "A2": []} for arm in ARMS}
    raw_records = []
    scored_logits: dict[tuple[str, str], tuple[float, float]] = {}
    for arm, source_names in ARMS.items():
        for family, chunk, base_batch, donor_batch, base_capture, donor_capture in batch_pairs:
            output = backend.patched_sources(
                base_batch, donor_batch, base_capture, donor_capture, source_names
            )
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                row_id = str(row["row_id"])
                base_score = -native[(row_id, "base")].margin
                donor_score = native[(row_id, "donor")].margin
                intervened_score = -(answer - foil)
                recovery = kernel.signed_pairwise_donor_recovery(
                    base_score, donor_score, intervened_score
                )
                recoveries[arm][family].append(recovery)
                scored_logits[(arm, row_id)] = (answer, foil)
                raw_records.append({
                    "arm_id": arm,
                    "family": family,
                    "row_id": row_id,
                    "answer_logit": answer,
                    "foil_logit": foil,
                    "recovery": recovery,
                })

    summaries = {
        arm: {family: family_summary(recoveries[arm][family]) for family in ("A1", "A2")}
        for arm in ARMS
    }
    for arm in ARMS:
        summaries[arm]["mean_target_recovery"] = statistics.fmean(
            summaries[arm][family]["mean_recovery"] for family in ("A1", "A2")
        )
    closure_max_abs = max(
        abs(value - reference)
        for row in rows
        for value, reference in zip(
            scored_logits[("all_changed_sources", str(row["row_id"]))],
            scored_logits[("full_pair", str(row["row_id"]))],
        )
    )
    full_mean = float(summaries["full_pair"]["mean_target_recovery"])
    pd_mean = float(summaries["period_determiner"]["mean_target_recovery"])
    lpd_mean = float(summaries["last_period_determiner"]["mean_target_recovery"])
    cue_self_mean = float(summaries["cue_self"]["mean_target_recovery"])
    pd_retained = pd_mean / full_mean
    lpd_retained = lpd_mean / full_mean
    cue_self_retained = cue_self_mean / full_mean
    pred_a = native_capability and closure_max_abs <= IDENTITY_TOLERANCE
    pred_b = bool(
        pd_retained >= 0.60
        and all(float(summaries["period_determiner"][family]["direction_fraction"]) >= 0.80 for family in ("A1", "A2"))
    )
    pred_c = bool(
        lpd_retained >= 0.80
        and all(float(summaries["last_period_determiner"][family]["direction_fraction"]) >= 0.80 for family in ("A1", "A2"))
    )
    pred_d = cue_self_retained <= 0.25
    expected_records = len(ARMS) * len(rows)
    pred_e = bool(
        len(raw_records) == expected_records
        and len(scored_logits) == expected_records
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if pred_a and pred_b and pred_c and pred_d and pred_e else (
        "null" if pred_a and pred_e else "invalid"
    )
    reason = {
        "screen": "l9h1_h4_contextual_source_bank_compression",
        "null": "contextual_source_compression_or_control_failed",
        "invalid": "source_closure_capability_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_l9h1_h4_downstream_source_bank_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "source_factorial_sha256": EXPECTED_SOURCE_RESULT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_source_closure": pred_a,
            "pred_b_period_determiner_compression": pred_b,
            "pred_c_last_extension": pred_c,
            "pred_d_cue_self_negative_control": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "source_closure_scored_logit_max_abs": closure_max_abs,
            "period_determiner_retained_fraction": pd_retained,
            "last_period_determiner_retained_fraction": lpd_retained,
            "cue_self_retained_fraction": cue_self_retained,
            "arms": summaries,
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "raw_record_count": len(raw_records),
            "model_backwards": 0,
            "model_updates": 0,
            "fit_parameters": 0,
        },
        "intervention_logits": raw_records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "trace which upstream writers place aspectual state in the period/determiner source bank"
            if terminal == "screen"
            else "retain only the full distributed-source closure and avoid source-bank promotion"
        ),
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "closure_max_abs": closure_max_abs,
        "period_determiner_retained": pd_retained,
        "last_period_determiner_retained": lpd_retained,
        "cue_self_retained": cue_self_retained,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
