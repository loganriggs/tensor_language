#!/usr/bin/env python3
# BQGATE: frozen A-E source-mechanism predictions; CUDA is managed-queue only.
"""Exact attention-source-term factorial for aspectual L9H1/H4."""

from __future__ import annotations

from dataclasses import asdict
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
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_l9h1_h4_source_term_factorial_v1.json"
PARENT = ROOT / "circuits/fast_screens/aspectual_anchor_has_vs_had_v1_result.json"
FACTORIAL = ROOT / "circuits/followups/aspectual_anchor_layer8_9_module_factorial_v2_result.json"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_aspectual.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_source_term_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.l9h1_h4_source_term_factorial_v1"
EXPECTED_PRIOR_SHA256 = "4f7704860c5e63106c04b2295f528ef18b1fb9edf0626a18430c324534bf7c6c"
EXPECTED_PARENT_SHA256 = "5ca2125e7d18bd6a377efcfa0c3a361b949e5a8fff4c053ae7481b4384c4fb94"
EXPECTED_FACTORIAL_SHA256 = "06918c951ab52c0b6082f440addba553bb994fec02e1a774772473917fd40050"
EXPECTED_BUILDER_SHA256 = "cca10e7f49f27ae49af62adbb0afb55d1d0b43b7174d0a5920db6f842fb1db20"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
HEADS = (1, 4)
ARMS = ("full_pair", "cue_joint", "last_joint", "period_joint", "self_joint", "cue_h1", "cue_h4")
MODEL_FORWARDS_MAX = 32
EXAMPLE_EVALUATIONS_MAX = 1024
IDENTITY_TOLERANCE = 1.0e-4


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    expected = {
        PRIOR: EXPECTED_PRIOR_SHA256,
        PARENT: EXPECTED_PARENT_SHA256,
        FACTORIAL: EXPECTED_FACTORIAL_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PARENT.read_text())
    factorial = json.loads(FACTORIAL.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if parent.get("terminal") != "screen" or factorial.get("terminal") != "screen":
        raise ExperimentError("a parent circuit is not a screen")
    if factorial["score"]["localized_attn09_singleton_heads"] != [
        "attn:09:head:01", "attn:09:head:04"
    ]:
        raise ExperimentError("frozen head set changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    if len(selected) != 64:
        raise ExperimentError("A1/A2 population changed")
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    return enriched, spec, parent


def rope_tables(torch, length: int, head_dim: int, device, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(length, dtype=torch.float32)
    frequencies = torch.outer(positions, inv)
    cosine, sine = frequencies.cos().bfloat16(), frequencies.sin().bfloat16()
    return cosine.to(device=device, dtype=dtype), sine.to(device=device, dtype=dtype)


def apply_rot(torch, value, cosine, sine):
    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    return torch.cat(
        [first * cosine + second * sine, -first * sine + second * cosine], dim=-1
    )


def source_positions(batch: producer.ModelBatch) -> list[dict[str, int]]:
    positions = []
    for base_ids, donor_ids, query in zip(
        batch.token_rows, getattr(batch, "donor_token_rows", batch.token_rows), batch.semantic_positions
    ):
        # This helper is replaced by explicit paired rows in manual_forward; the
        # fallback is unreachable and exists only to keep position logic local.
        differences = [i for i, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
        if len(differences) != 1:
            raise ExperimentError("row does not have one aligned cue difference")
        cue = differences[0]
        positions.append({"cue": cue, "last": cue + 1, "period": cue + 2, "self": query})
    return positions


class SourceBackend(producer.Bilin18TorchBackend):
    def manual_forward(
        self,
        batch: producer.ModelBatch,
        *,
        donor_batch: producer.ModelBatch | None = None,
        donor_capture: dict[str, object] | None = None,
        arm: str | None = None,
    ) -> tuple[producer.BatchOutput, dict[str, object]]:
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        width = model.config.n_embd
        heads = model.config.n_head
        head_dim = width // heads
        maximum = tokens.shape[1]
        mask = torch.tril(torch.ones(maximum, maximum, device=self.device, dtype=torch.bool))
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (width,))
            x0 = x
            cosine, sine = rope_tables(torch, maximum, head_dim, self.device, x.dtype)
            cosine = cosine[None, :, None, :]
            sine = sine[None, :, None, :]
            v1 = None
            l9_capture = None
            for layer, block in enumerate(model.transformer.h):
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                attention = block.attn
                current = F.rms_norm(x, (width,))

                def qk(linear):
                    value = F.rms_norm(
                        linear(current).view(len(batch.row_ids), maximum, heads, head_dim),
                        (head_dim,),
                    )
                    return apply_rot(torch, value, cosine, sine)

                value = attention.c_v(current).view(
                    len(batch.row_ids), maximum, heads, head_dim
                )
                if v1 is None:
                    v1 = value
                effective_value = (1.0 - attention.lamb) * value + attention.lamb * v1.view_as(value)
                query, key = qk(attention.c_q), qk(attention.c_k)
                query2, key2 = qk(attention.c_q2), qk(attention.c_k2)
                score1 = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
                score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
                pattern = (score1 * score2).masked_fill(~mask, 0.0)
                head_output = torch.einsum("bhqk,bkhd->bqhd", pattern, effective_value)
                if layer == 9:
                    reconstruction = torch.stack([
                        torch.stack([
                            torch.stack([
                                torch.stack([
                                    pattern[i, h, q, k] * effective_value[i, k, h]
                                    for k in range(lengths[i])
                                ]).sum(0)
                                for h in HEADS
                            ])
                            for q in [batch.semantic_positions[i]]
                        ])[0]
                        for i in range(len(batch.row_ids))
                    ])
                    selected_native = torch.stack([
                        torch.stack([head_output[i, batch.semantic_positions[i], h] for h in HEADS])
                        for i in range(len(batch.row_ids))
                    ])
                    reconstruction_error = float(
                        (reconstruction.float() - selected_native.float()).abs().max()
                    )
                    l9_capture = {
                        "pattern": pattern.detach().clone(),
                        "value": effective_value.detach().clone(),
                        "head_output": head_output.detach().clone(),
                        "reconstruction_max_abs": reconstruction_error,
                    }
                    if arm is not None:
                        if donor_batch is None or donor_capture is None:
                            raise ExperimentError("intervention lacks paired donor capture")
                        if batch.row_ids != donor_batch.row_ids:
                            raise ExperimentError("recipient/donor row order differs")
                        donor_pattern = donor_capture["pattern"]
                        donor_value = donor_capture["value"]
                        donor_head_output = donor_capture["head_output"]
                        changed = head_output.clone()
                        for i, (base_ids, donor_ids, q, donor_q) in enumerate(zip(
                            batch.token_rows,
                            donor_batch.token_rows,
                            batch.semantic_positions,
                            donor_batch.semantic_positions,
                        )):
                            if len(base_ids) != len(donor_ids):
                                raise ExperimentError("paired token lengths differ")
                            differences = [
                                position for position, (base_id, donor_id) in enumerate(
                                    zip(base_ids, donor_ids)
                                ) if base_id != donor_id
                            ]
                            if len(differences) != 1:
                                raise ExperimentError("row does not have one aligned cue difference")
                            cue = differences[0]
                            positions = {
                                "cue": cue,
                                "last": cue + 1,
                                "period": cue + 2,
                                "self": q,
                            }
                            if any(not 0 <= position < lengths[i] for position in positions.values()):
                                raise ExperimentError("registered source position is out of range")
                            if arm == "full_pair":
                                for h in HEADS:
                                    changed[i, q, h] = donor_head_output[i, donor_q, h]
                                continue
                            source_name = arm.split("_", 1)[0]
                            if source_name not in positions:
                                raise ExperimentError("unknown source arm")
                            source = positions[source_name]
                            selected_heads = HEADS
                            if arm == "cue_h1":
                                selected_heads = (1,)
                            elif arm == "cue_h4":
                                selected_heads = (4,)
                            for h in selected_heads:
                                base_term = pattern[i, h, q, source] * effective_value[i, source, h]
                                donor_term = donor_pattern[i, h, donor_q, source] * donor_value[i, source, h]
                                changed[i, q, h] += donor_term - base_term
                        head_output = changed
                attention_output = attention.c_proj(head_output.reshape(
                    len(batch.row_ids), maximum, width
                ))
                x = x + attention_output
                x = x + block.mlp(F.rms_norm(x, (width,)))
            logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (width,))) / 30.0)
            values = tuple(
                (
                    float(logits[i, length - 1, batch.answer_ids[i]].float()),
                    float(logits[i, length - 1, batch.foil_ids[i]].float()),
                )
                for i, length in enumerate(lengths)
            )
        if l9_capture is None:
            raise ExperimentError("L9 capture missing")
        return producer.BatchOutput(values, {}), l9_capture


def family_summary(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("family recovery is missing or nonfinite")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    rows, spec, parent = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_l9h1_h4_source_term_factorial_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
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
    backend = SourceBackend.load("cuda")
    native: dict[tuple[str, str], producer.NativeLogitEvidence] = {}
    captures: dict[tuple[str, str], dict[str, object]] = {}
    batch_pairs = []
    forward_calls = 0
    evaluations = 0
    manual_logit_max_abs = 0.0
    reconstruction_max_abs = 0.0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            pair_captures = {}
            for side, batch in (("base", base_batch), ("donor", donor_batch)):
                reference = backend.native(batch, capture=False)
                manual, capture = backend.manual_forward(batch)
                forward_calls += 2
                evaluations += 2 * len(chunk)
                for reference_pair, manual_pair in zip(reference.answer_foil, manual.answer_foil):
                    manual_logit_max_abs = max(
                        manual_logit_max_abs,
                        abs(reference_pair[0] - manual_pair[0]),
                        abs(reference_pair[1] - manual_pair[1]),
                    )
                reconstruction_max_abs = max(
                    reconstruction_max_abs, float(capture["reconstruction_max_abs"])
                )
                pair_captures[side] = capture
                for row_id, pair in zip(batch.row_ids, reference.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(row_id, side)] = producer.NativeLogitEvidence(
                        row_id, family, side, answer, foil  # type: ignore[arg-type]
                    )
                    captures[(row_id, side)] = capture
            batch_pairs.append((family, tuple(chunk), base_batch, donor_batch, pair_captures["donor"]))

    parent_cells = [
        cell for cell in parent["run"]["capability_cells"] if cell["family"] in {"A1", "A2"}
    ]
    native_capability = True
    for cell in parent_cells:
        family = cell["family"]
        direction = cell["cell_id"].rsplit("/", 1)[-1]
        cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
        for side in ("base", "donor"):
            accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
            native_capability = native_capability and accuracy >= 0.85

    raw_records = []
    recoveries: dict[str, dict[str, list[float]]] = {
        arm: {"A1": [], "A2": []} for arm in ARMS
    }
    for arm in ARMS:
        for family, chunk, base_batch, donor_batch, donor_capture in batch_pairs:
            output, _capture = backend.manual_forward(
                base_batch,
                donor_batch=donor_batch,
                donor_capture=donor_capture,
                arm=arm,
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
                raw_records.append({
                    "arm_id": arm,
                    "family": family,
                    "row_id": row_id,
                    "answer_logit": answer,
                    "foil_logit": foil,
                    "recovery": recovery,
                })

    summaries = {
        arm: {
            family: family_summary(recoveries[arm][family])
            for family in ("A1", "A2")
        }
        for arm in ARMS
    }
    for arm in ARMS:
        summaries[arm]["mean_target_recovery"] = statistics.fmean(
            summaries[arm][family]["mean_recovery"] for family in ("A1", "A2")
        )
    full = summaries["full_pair"]
    cue = summaries["cue_joint"]
    full_mean = float(full["mean_target_recovery"])
    cue_mean = float(cue["mean_target_recovery"])
    retained_fraction = cue_mean / full_mean if full_mean > 0.0 else float("nan")
    control_max = max(
        abs(float(summaries[arm]["mean_target_recovery"]))
        for arm in ("last_joint", "period_joint", "self_joint")
    )
    pred_a = bool(
        native_capability
        and manual_logit_max_abs <= IDENTITY_TOLERANCE
        and reconstruction_max_abs <= IDENTITY_TOLERANCE
    )
    pred_b = bool(
        float(full["A1"]["mean_recovery"]) > 0.0
        and float(full["A2"]["mean_recovery"]) > 0.0
        and float(full["A1"]["direction_fraction"]) >= 0.80
        and float(full["A2"]["direction_fraction"]) >= 0.80
    )
    pred_c = bool(
        math.isfinite(retained_fraction)
        and retained_fraction >= 0.70
        and float(cue["A1"]["direction_fraction"]) >= 0.80
        and float(cue["A2"]["direction_fraction"]) >= 0.80
    )
    pred_d = cue_mean >= 2.0 * control_max
    expected_records = len(ARMS) * len(rows)
    pred_e = bool(
        len(raw_records) == expected_records
        and len({(row["arm_id"], row["row_id"]) for row in raw_records}) == expected_records
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if pred_a and pred_b and pred_c and pred_d and pred_e else (
        "null" if pred_a and pred_e else "invalid"
    )
    reason = {
        "screen": "l9h1_h4_read_since_by_cue_term",
        "null": "head_pair_or_cue_source_prediction_failed",
        "invalid": "instrument_capability_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_l9h1_h4_source_term_factorial_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "module_factorial_sha256": EXPECTED_FACTORIAL_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_instrument": pred_a,
            "pred_b_full_pair_recurrence": pred_b,
            "pred_c_cue_source_sufficiency": pred_c,
            "pred_d_cue_source_specificity": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_scored_logit_max_abs": manual_logit_max_abs,
            "source_sum_reconstruction_max_abs": reconstruction_max_abs,
            "cue_retained_fraction_of_full_pair": retained_fraction,
            "registered_control_max_absolute_recovery": control_max,
            "cue_to_control_ratio": cue_mean / control_max if control_max > 0.0 else float("inf"),
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
            "compile the since/by cue reader with the L8/L9 writers and test a capability-qualified sealed population"
            if terminal == "screen"
            else "retain L9H1/H4 as causal writers without a cue-source read claim"
        ),
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "full_pair": full_mean,
        "cue": cue_mean,
        "retained_fraction": retained_fraction,
        "control_max": control_max,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
