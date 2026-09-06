#!/usr/bin/env python3
# BQGATE: frozen A-E block-4 writer predictions; CUDA is managed-queue only.
"""Exact carried/attention/MLP factorization of the aspectual resid4->5 crossing."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
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
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_block4_contextual_source_writer_factorial_v1.json"
ONSET = ROOT / "circuits/followups/aspectual_anchor_contextual_source_state_onset_v1_result.json"
ONSET_RUNNER = ROOT / "ops/run_aspectual_anchor_contextual_source_state_onset_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_block4_contextual_source_writer_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.block4_contextual_source_writer_factorial_v1"
EXPECTED_PRIOR_SHA256 = "1c63e22da8d6242bcfed77cf16ff6f43b31b34c9699dde3cd2dd4fb72f3aa515"
EXPECTED_ONSET_SHA256 = "5ffc93473ba0895bdfb184a8a3045e64438600e168b392ecac6dd58c33138c63"
EXPECTED_ONSET_RUNNER_SHA256 = "4f8b0bb90e176621c8e313f07b5529601abaef4812f9865f3ffaa760c19ff33c"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
FACTORS = ("carried", "attention4", "mlp4")
MODEL_FORWARDS_MAX = 28
EXAMPLE_EVALUATIONS_MAX = 896
IDENTITY_TOLERANCE = 0.125


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets():
    return tuple(
        subset for width in range(len(FACTORS) + 1)
        for subset in itertools.combinations(FACTORS, width)
    )


def arm_id(subset: tuple[str, ...]) -> str:
    return "empty" if not subset else "+".join(subset)


def validate_static():
    if sha256(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior-art hash changed")
    if sha256(ONSET) != EXPECTED_ONSET_SHA256:
        raise ExperimentError("onset result hash changed")
    if sha256(ONSET_RUNNER) != EXPECTED_ONSET_RUNNER_SHA256:
        raise ExperimentError("onset runner hash changed")
    prior = json.loads(PRIOR.read_text())
    onset = json.loads(ONSET.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if onset.get("terminal") != "screen" or onset["score"]["source_bank_onset_boundary"] != 5:
        raise ExperimentError("onset authority changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or len(subsets()) != 8:
        raise ExperimentError("population or factorial changed")
    return enriched, spec


def source_positions(batch: producer.ModelBatch, donor_batch: producer.ModelBatch):
    output = []
    for base_ids, donor_ids in zip(batch.token_rows, donor_batch.token_rows):
        differences = [
            i for i, (base_id, donor_id) in enumerate(zip(base_ids, donor_ids))
            if base_id != donor_id
        ]
        if len(base_ids) != len(donor_ids) or len(differences) != 1:
            raise ExperimentError("cue alignment changed")
        cue = differences[0]
        bank = (cue + 1, cue + 2, cue + 3)
        if any(position >= len(base_ids) for position in bank):
            raise ExperimentError("source bank is out of range")
        output.append(bank)
    return output


class ComponentBackend(producer.Bilin18TorchBackend):
    def capture_components(self, batch: producer.ModelBatch):
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0 = x
            v1 = None
            captured = {}
            for layer, block in enumerate(model.transformer.h):
                if layer == 4:
                    captured["resid4"] = x.detach().clone()
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                mlp = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                x = x + mlp
                if layer == 4:
                    captured.update({
                        "attention4": attention.detach().clone(),
                        "mlp4": mlp.detach().clone(),
                        "resid5": x.detach().clone(),
                        "x0": x0.detach().clone(),
                        "v1_after4": v1.detach().clone(),
                    })
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
        if set(captured) != {"resid4", "attention4", "mlp4", "resid5", "x0", "v1_after4"}:
            raise ExperimentError("component capture incomplete")
        return producer.BatchOutput(values, {}), captured

    def suffix_from_resid5(self, batch: producer.ModelBatch, state, x0, v1):
        torch, F, model = self.torch, self.F, self.model
        lengths = tuple(len(row) for row in batch.token_rows)
        x = state
        with torch.no_grad():
            for layer in range(5, 18):
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

    def intervene(
        self,
        base_batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_capture: dict[str, object],
        donor_capture: dict[str, object],
        subset: tuple[str, ...] | None,
    ) -> producer.BatchOutput:
        if subset is not None and subset not in subsets():
            raise ExperimentError("factorial arm changed")
        positions = source_positions(base_batch, donor_batch)
        state = base_capture["resid5"].clone()
        if subset is None:
            for i, bank in enumerate(positions):
                for position in bank:
                    state[i, position] = donor_capture["resid5"][i, position]
        else:
            lambda0 = self.model.transformer.h[4].lambdas[0]
            for i, bank in enumerate(positions):
                for position in bank:
                    delta = self.torch.zeros_like(state[i, position], dtype=self.torch.float32)
                    if "carried" in subset:
                        delta += lambda0.float() * (
                            donor_capture["resid4"][i, position].float()
                            - base_capture["resid4"][i, position].float()
                        )
                    if "attention4" in subset:
                        delta += (
                            donor_capture["attention4"][i, position].float()
                            - base_capture["attention4"][i, position].float()
                        )
                    if "mlp4" in subset:
                        delta += (
                            donor_capture["mlp4"][i, position].float()
                            - base_capture["mlp4"][i, position].float()
                        )
                    state[i, position] = (state[i, position].float() + delta).to(state.dtype)
        return self.suffix_from_resid5(
            base_batch, state, base_capture["x0"], base_capture["v1_after4"]
        )


def summary(values: list[float]) -> dict[str, object]:
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
        "schema": "aspectual_anchor_block4_contextual_source_writer_factorial_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "onset_result_sha256": EXPECTED_ONSET_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "factors": list(FACTORS),
        "factorial_arm_count": len(subsets()),
        "direct_ceiling_arms": 1,
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
    backend = ComponentBackend.load("cuda")
    native: dict[tuple[str, str], producer.NativeLogitEvidence] = {}
    batch_pairs = []
    manual_logit_max_abs = 0.0
    skip_source_max_abs = 0.0
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
                manual, capture = backend.capture_components(batch)
                forward_calls += 2
                evaluations += 2 * len(chunk)
                for reference_pair, manual_pair in zip(reference.answer_foil, manual.answer_foil):
                    manual_logit_max_abs = max(
                        manual_logit_max_abs,
                        abs(reference_pair[0] - manual_pair[0]),
                        abs(reference_pair[1] - manual_pair[1]),
                    )
                for row_id, pair in zip(batch.row_ids, reference.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(row_id, side)] = producer.NativeLogitEvidence(
                        row_id, family, side, answer, foil  # type: ignore[arg-type]
                    )
                captures[side] = capture
            for i, bank in enumerate(source_positions(base_batch, donor_batch)):
                for position in bank:
                    skip_source_max_abs = max(
                        skip_source_max_abs,
                        float((captures["base"]["x0"][i, position].float() - captures["donor"]["x0"][i, position].float()).abs().max()),
                    )
            batch_pairs.append((family, tuple(chunk), base_batch, donor_batch, captures["base"], captures["donor"]))

    native_capability = True
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                native_capability = native_capability and accuracy >= 0.85

    arms = {subset: {"A1": [], "A2": []} for subset in subsets()}
    ceiling = {"A1": [], "A2": []}
    logits = {}
    raw_records = []
    for subset in subsets():
        name = arm_id(subset)
        for family, chunk, base_batch, donor_batch, base_capture, donor_capture in batch_pairs:
            output = backend.intervene(base_batch, donor_batch, base_capture, donor_capture, subset)
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                row_id = str(row["row_id"])
                recovery = kernel.signed_pairwise_donor_recovery(
                    -native[(row_id, "base")].margin,
                    native[(row_id, "donor")].margin,
                    -(answer - foil),
                )
                arms[subset][family].append(recovery)
                logits[(name, row_id)] = (answer, foil)
                raw_records.append({"arm_id": name, "family": family, "row_id": row_id, "answer_logit": answer, "foil_logit": foil, "recovery": recovery})
    for family, chunk, base_batch, donor_batch, base_capture, donor_capture in batch_pairs:
        output = backend.intervene(base_batch, donor_batch, base_capture, donor_capture, None)
        forward_calls += 1
        evaluations += len(chunk)
        for row, pair in zip(chunk, output.answer_foil):
            answer, foil = producer._finite_pair(pair)
            row_id = str(row["row_id"])
            recovery = kernel.signed_pairwise_donor_recovery(
                -native[(row_id, "base")].margin,
                native[(row_id, "donor")].margin,
                -(answer - foil),
            )
            ceiling[family].append(recovery)
            logits[("direct_resid5_ceiling", row_id)] = (answer, foil)
            raw_records.append({"arm_id": "direct_resid5_ceiling", "family": family, "row_id": row_id, "answer_logit": answer, "foil_logit": foil, "recovery": recovery})

    summaries = {}
    values = {}
    for subset in subsets():
        families = {family: summary(arms[subset][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm_id(subset)] = {"factors": list(subset), "families": families, "mean_target_recovery": target}
        values[subset] = target
    ceiling_summary = {family: summary(ceiling[family]) for family in ("A1", "A2")}
    ceiling_target = statistics.fmean(ceiling_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    full = FACTORS
    full_name = arm_id(full)
    closure_max_abs = max(
        abs(value - reference)
        for row in rows
        for value, reference in zip(
            logits[(full_name, str(row["row_id"]))],
            logits[("direct_resid5_ceiling", str(row["row_id"]))],
        )
    )
    shapley = {}
    n = len(FACTORS)
    all_subsets = subsets()
    for factor in FACTORS:
        total = 0.0
        for subset in all_subsets:
            if factor in subset:
                continue
            extended = tuple(item for item in FACTORS if item in set(subset) | {factor})
            weight = math.factorial(len(subset)) * math.factorial(n - len(subset) - 1) / math.factorial(n)
            total += weight * (values[extended] - values[subset])
        shapley[factor] = total
    winner = max(("attention4", "mlp4"), key=lambda factor: shapley[factor])
    without_winner = tuple(factor for factor in FACTORS if factor != winner)
    family_drops = {
        family: summaries[full_name]["families"][family]["mean_recovery"]
        - summaries[arm_id(without_winner)]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    pred_a = native_capability and manual_logit_max_abs <= 1.0e-4 and skip_source_max_abs == 0.0 and closure_max_abs <= IDENTITY_TOLERANCE
    pred_b = all(
        summaries[full_name]["families"][family]["mean_recovery"] >= 0.50
        and summaries[full_name]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_c = shapley[winner] >= 0.05 and all(drop > 0.0 for drop in family_drops.values())
    pred_d = shapley["attention4"] + shapley["mlp4"] >= 0.10
    expected_records = (len(all_subsets) + 1) * len(rows)
    pred_e = len(raw_records) == expected_records and len(logits) == expected_records and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if pred_a and pred_b and pred_c and pred_d and pred_e else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "block4_contextual_writer_identified", "null": "block4_writer_factor_prediction_failed", "invalid": "component_closure_capability_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_block4_contextual_source_writer_factorial_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "onset_result_sha256": EXPECTED_ONSET_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_component_closure": pred_a,
            "pred_b_full_factor_passes": pred_b,
            "pred_c_block4_writer_identified": pred_c,
            "pred_d_writer_increment": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_scored_logit_max_abs": manual_logit_max_abs,
            "embedding_skip_source_max_abs": skip_source_max_abs,
            "component_closure_scored_logit_max_abs": closure_max_abs,
            "factorial_shapley_target_recovery": shapley,
            "block4_writer_winner": winner,
            "winner_full_removal_family_drops": family_drops,
            "factorial_arms": summaries,
            "direct_resid5_ceiling": {"families": ceiling_summary, "mean_target_recovery": ceiling_target},
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
            f"decompose {winner} at the three contextual source positions into its internal causal factors"
            if terminal == "screen" else "retain resid5 onset without a block4 writer claim"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "winner": winner, "shapley": shapley, "family_drops": family_drops, "closure_max_abs": closure_max_abs, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
