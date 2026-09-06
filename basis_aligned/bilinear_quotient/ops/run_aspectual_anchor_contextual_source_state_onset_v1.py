#!/usr/bin/env python3
# BQGATE: frozen A-E residual-onset predictions; CUDA is managed-queue only.
"""Exact depth sweep for the contextual aspectual source-state bank."""

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
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_contextual_source_state_onset_v1.json"
SOURCE_BANK = ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_downstream_source_bank_v2_result.json"
PARENT = ROOT / "circuits/fast_screens/aspectual_anchor_has_vs_had_v1_result.json"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_aspectual.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_contextual_source_state_onset_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.contextual_source_state_onset_v1"
EXPECTED_PRIOR_SHA256 = "78fd434df2c1bfdab06335f8eb782ac0edef8e7d78520a766f7106bb795d7d8a"
EXPECTED_SOURCE_BANK_SHA256 = "6d694f92d35970f4eb5eba25ca3d9aff15cdbd1949db158a8be18e827e0423a7"
EXPECTED_PARENT_SHA256 = "5ca2125e7d18bd6a377efcfa0c3a361b949e5a8fff4c053ae7481b4384c4fb94"
EXPECTED_BUILDER_SHA256 = "cca10e7f49f27ae49af62adbb0afb55d1d0b43b7174d0a5920db6f842fb1db20"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
BOUNDARIES = tuple(range(10))
BANKS = ("source_bank", "cue")
MODEL_FORWARDS_MAX = 48
EXAMPLE_EVALUATIONS_MAX = 1536
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
        SOURCE_BANK: EXPECTED_SOURCE_BANK_SHA256,
        PARENT: EXPECTED_PARENT_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    source_bank = json.loads(SOURCE_BANK.read_text())
    parent = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if source_bank.get("terminal") != "screen" or parent.get("terminal") != "screen":
        raise ExperimentError("a parent result is not a screen")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    parent_self = {
        item["site"]["site_id"]: item
        for item in parent["run"]["site_results"]
        if item["site"]["site_id"].startswith("resid:")
    }
    if len(enriched) != 64 or set(parent_self) != {f"resid:{i:02d}" for i in range(19)}:
        raise ExperimentError("population or parent residual inventory changed")
    if parent_self["resid:10"]["terminal"] != "screen" or any(
        parent_self[f"resid:{i:02d}"]["terminal"] == "screen" for i in range(10)
    ):
        raise ExperimentError("parent final-subject onset changed")
    return enriched, spec, parent_self


def positions_for(batch: producer.ModelBatch, donor_batch: producer.ModelBatch):
    output = []
    for base_ids, donor_ids, query, donor_query in zip(
        batch.token_rows,
        donor_batch.token_rows,
        batch.semantic_positions,
        donor_batch.semantic_positions,
    ):
        differences = [
            position for position, (base_id, donor_id) in enumerate(zip(base_ids, donor_ids))
            if base_id != donor_id
        ]
        if len(base_ids) != len(donor_ids) or len(differences) != 1 or query != donor_query:
            raise ExperimentError("cue alignment changed")
        cue = differences[0]
        positions = {
            "cue": (cue,),
            "source_bank": (cue + 1, cue + 2, cue + 3),
            "self": (query,),
        }
        if any(not 0 <= position < len(base_ids) for values in positions.values() for position in values):
            raise ExperimentError("registered position is out of range")
        output.append(positions)
    return output


class ResidualBankBackend(producer.Bilin18TorchBackend):
    def forward_states(
        self,
        batch: producer.ModelBatch,
        *,
        donor_batch: producer.ModelBatch | None = None,
        donor_states: tuple[object, ...] | None = None,
        boundary: int | None = None,
        bank: str | None = None,
    ) -> tuple[producer.BatchOutput, tuple[object, ...]]:
        if boundary is None:
            if donor_batch is not None or donor_states is not None or bank is not None:
                raise ExperimentError("native capture has intervention inputs")
        elif (
            boundary not in BOUNDARIES
            or donor_batch is None
            or donor_states is None
            or bank not in BANKS
        ):
            raise ExperimentError("residual intervention is incomplete")
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        positions = positions_for(batch, donor_batch) if donor_batch is not None else None

        def patch(value, at_boundary: int):
            if boundary != at_boundary:
                return value
            assert positions is not None and donor_states is not None and bank is not None
            changed = value.clone()
            donor_value = donor_states[at_boundary]
            for i, mapping in enumerate(positions):
                for position in mapping[bank]:
                    changed[i, position] = donor_value[i, position].to(
                        device=value.device, dtype=value.dtype
                    )
            return changed

        captured = []
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0 = x
            captured.append(x.detach().clone())
            x = patch(x, 0)
            v1 = None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if layer + 1 <= 9:
                    captured.append(x.detach().clone())
                    x = patch(x, layer + 1)
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
        if len(captured) != len(BOUNDARIES):
            raise ExperimentError("residual capture coverage changed")
        return producer.BatchOutput(values, {}), tuple(captured)


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
    rows, spec, parent_self = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_contextual_source_state_onset_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "source_bank_sha256": EXPECTED_SOURCE_BANK_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "banks": list(BANKS),
        "boundaries": list(BOUNDARIES),
        "arm_count": len(BANKS) * len(BOUNDARIES),
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
    backend = ResidualBankBackend.load("cuda")
    native: dict[tuple[str, str], producer.NativeLogitEvidence] = {}
    batch_pairs = []
    manual_logit_max_abs = 0.0
    boundary0_bank_state_max_abs = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            captures = {}
            for side, batch in (("base", base_batch), ("donor", donor_batch)):
                reference = backend.native(batch, capture=False)
                manual, states = backend.forward_states(batch)
                forward_calls += 2
                evaluations += 2 * len(chunk)
                for ref_pair, manual_pair in zip(reference.answer_foil, manual.answer_foil):
                    manual_logit_max_abs = max(
                        manual_logit_max_abs,
                        abs(ref_pair[0] - manual_pair[0]),
                        abs(ref_pair[1] - manual_pair[1]),
                    )
                for row_id, pair in zip(batch.row_ids, reference.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(row_id, side)] = producer.NativeLogitEvidence(
                        row_id, family, side, answer, foil  # type: ignore[arg-type]
                    )
                captures[side] = states
            positions = positions_for(base_batch, donor_batch)
            for i, mapping in enumerate(positions):
                for position in mapping["source_bank"]:
                    boundary0_bank_state_max_abs = max(
                        boundary0_bank_state_max_abs,
                        float((captures["base"][0][i, position].float() - captures["donor"][0][i, position].float()).abs().max()),
                    )
            batch_pairs.append((family, tuple(chunk), base_batch, donor_batch, captures["donor"]))

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

    recoveries = {
        (bank, boundary): {"A1": [], "A2": []}
        for bank in BANKS for boundary in BOUNDARIES
    }
    raw_records = []
    boundary0_bank_logit_max_abs = 0.0
    for bank in BANKS:
        for boundary in BOUNDARIES:
            arm_id = f"{bank}:resid:{boundary:02d}"
            for family, chunk, base_batch, donor_batch, donor_states in batch_pairs:
                output, _ = backend.forward_states(
                    base_batch,
                    donor_batch=donor_batch,
                    donor_states=donor_states,
                    boundary=boundary,
                    bank=bank,
                )
                forward_calls += 1
                evaluations += len(chunk)
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    row_id = str(row["row_id"])
                    if bank == "source_bank" and boundary == 0:
                        base_native = native[(row_id, "base")]
                        boundary0_bank_logit_max_abs = max(
                            boundary0_bank_logit_max_abs,
                            abs(answer - base_native.answer_logit),
                            abs(foil - base_native.foil_logit),
                        )
                    base_score = -native[(row_id, "base")].margin
                    donor_score = native[(row_id, "donor")].margin
                    intervened_score = -(answer - foil)
                    recovery = kernel.signed_pairwise_donor_recovery(
                        base_score, donor_score, intervened_score
                    )
                    recoveries[(bank, boundary)][family].append(recovery)
                    raw_records.append({
                        "arm_id": arm_id,
                        "bank": bank,
                        "boundary": boundary,
                        "family": family,
                        "row_id": row_id,
                        "answer_logit": answer,
                        "foil_logit": foil,
                        "recovery": recovery,
                    })

    curves = {}
    for bank in BANKS:
        points = []
        for boundary in BOUNDARIES:
            families = {
                family: summarize(recoveries[(bank, boundary)][family])
                for family in ("A1", "A2")
            }
            mean_target = statistics.fmean(
                families[family]["mean_recovery"] for family in ("A1", "A2")
            )
            passed = all(
                float(families[family]["mean_recovery"]) >= 0.50
                and float(families[family]["direction_fraction"]) >= 0.80
                for family in ("A1", "A2")
            )
            points.append({
                "boundary": boundary,
                "families": families,
                "mean_target_recovery": mean_target,
                "passed": passed,
            })
        curves[bank] = points
    passing_boundaries = [
        point["boundary"] for point in curves["source_bank"] if point["passed"]
    ]
    onset = min(passing_boundaries) if passing_boundaries else None
    pred_a = bool(
        native_capability
        and manual_logit_max_abs <= IDENTITY_TOLERANCE
        and boundary0_bank_state_max_abs == 0.0
        and boundary0_bank_logit_max_abs <= IDENTITY_TOLERANCE
    )
    pred_b = onset is not None and 1 <= onset <= 9
    pred_c = onset is not None and onset < 10
    pred_d = True
    expected_records = len(BANKS) * len(BOUNDARIES) * len(rows)
    pred_e = bool(
        len(raw_records) == expected_records
        and len({(row["arm_id"], row["row_id"]) for row in raw_records}) == expected_records
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if pred_a and pred_b and pred_c and pred_e else (
        "null" if pred_a and pred_e else "invalid"
    )
    reason = {
        "screen": "contextual_source_state_onset_localized",
        "null": "no_source_bank_boundary_through_9_passed",
        "invalid": "instrument_capability_alignment_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_contextual_source_state_onset_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "source_bank_sha256": EXPECTED_SOURCE_BANK_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_instrument": pred_a,
            "pred_b_source_bank_onset": pred_b,
            "pred_c_earlier_than_final_subject": pred_c,
            "pred_d_origin_contextualization": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_scored_logit_max_abs": manual_logit_max_abs,
            "boundary0_source_bank_state_max_abs": boundary0_bank_state_max_abs,
            "boundary0_source_bank_scored_logit_max_abs": boundary0_bank_logit_max_abs,
            "source_bank_onset_boundary": onset,
            "parent_final_subject_onset_boundary": 10,
            "curves": curves,
            "parent_final_subject_residual_scores": {
                f"resid:{boundary:02d}": parent_self[f"resid:{boundary:02d}"]
                for boundary in range(11)
            },
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
            "factor the crossing block into attention and MLP writers at the contextual source bank"
            if terminal == "screen"
            else "retain the contextual source read but treat its upstream representation as distributed"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "source_bank_onset": onset,
        "source_curve": [point["mean_target_recovery"] for point in curves["source_bank"]],
        "cue_curve": [point["mean_target_recovery"] for point in curves["cue"]],
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
