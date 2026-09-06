#!/usr/bin/env python3
"""Exact cross-task MLP4 contextual-source response factorial for is/was."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument pred_b_live_mlp4_contextual_source_writer pred_c_shared_left_right_two_term_program pred_d_shared_factor_structure pred_e_exact_zero_fit_price
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

import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as rows_builder
import circuit_das_subspace as das
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_mlp4_bilinear_response_factorial_v1 as inherited


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_mlp4_contextual_source_factorial_v1.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_contextual_source_factorial_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.mlp4_contextual_source_factorial_v1"
PATHS = {
    "matched_builder": ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2.py",
    "has_had_factorial": ROOT / "circuits/followups/aspectual_anchor_mlp4_bilinear_response_factorial_v2_result.json",
    "factorial_instrument": ROOT / "ops/run_aspectual_anchor_mlp4_bilinear_response_factorial_v1.py",
    "is_was_source": ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_source_term_factorial_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "6be828a1e756398075b5d8b486eeb05edcecc0c4f95690b69bfc48debcc01c70"
EXPECTED = {
    "matched_builder": "1f4b29bda3e26af3ee0102316ab0af166e317d1646e8b0b51332061245e606d6",
    "has_had_factorial": "359483cfb4807e9293e1f25f877db8d7303bc76333d83a6d237cf72a9c7e77e4",
    "factorial_instrument": "290ef0e8b071a487d0d4560094e49ecc75d8a7358fbb8ec28c58e37a68463a57",
    "is_was_source": "4c266158213edcda9f0c86b19064cabe6d673815167b69d9eff381ddadda9cf5",
}
EXPECTED_ROWS_SHA256 = "2efd47b9a89d0f092688a96d75bbc33e5b89991a8e5de28723c714319b9ccceb"
FACTORS = ("left_change", "right_change", "bilinear_interaction")


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsets():
    return tuple(subset for width in range(4) for subset in itertools.combinations(FACTORS, width))


def arm_id(subset):
    return "empty" if not subset else "+".join(subset)


def source_positions(base_batch, donor_batch):
    banks = []
    for base_ids, donor_ids in zip(base_batch.token_rows, donor_batch.token_rows):
        if len(base_ids) != len(donor_ids):
            raise ExperimentError("paired token lengths differ")
        differences = [i for i, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
        if len(differences) != 1:
            raise ExperimentError("row does not have one aligned cue difference")
        cue = differences[0]
        bank = (cue + 1, cue + 2)
        if any(position >= len(base_ids) for position in bank):
            raise ExperimentError("contextual source bank is out of range")
        banks.append(bank)
    return banks


class Backend(inherited.BilinearBackend):
    def intervene_factors(self, base_batch, donor_batch, base_capture, donor_capture, subset):
        if subset not in subsets():
            raise ExperimentError("factorial arm changed")
        projected, error = self.projected_terms(base_capture, donor_capture)
        state = base_capture["resid5"].clone()
        for i, bank in enumerate(source_positions(base_batch, donor_batch)):
            for position in bank:
                delta = self.torch.zeros_like(state[i, position], dtype=self.torch.float32)
                for factor in subset:
                    delta += projected[factor][i, position]
                state[i, position] = (state[i, position].float() + delta).to(state.dtype)
        return self.suffix_from_resid5(base_batch, state, base_capture["x0"], base_capture["v1_after4"]), error

    def intervene_direct(self, base_batch, donor_batch, base_capture, donor_capture):
        state = base_capture["resid5"].clone()
        delta = donor_capture["mlp4"].float() - base_capture["mlp4"].float()
        for i, bank in enumerate(source_positions(base_batch, donor_batch)):
            for position in bank:
                state[i, position] = (state[i, position].float() + delta[i, position]).to(state.dtype)
        return self.suffix_from_resid5(base_batch, state, base_capture["x0"], base_capture["v1_after4"])


def summary(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("missing or nonfinite recovery")
    return {"count": len(values), "mean_recovery": statistics.fmean(values), "mean_absolute_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    rows_by_bank = rows_builder.build_rows_by_bank()
    digests = rows_builder.validate_rows_by_bank(rows_by_bank)
    rows = [row for row in rows_by_bank["is_was"] if row["transform_id"] in {"A1", "A2"}]
    if prior.get("candidate_id") != CANDIDATE_ID or digests["is_was"] != EXPECTED_ROWS_SHA256 or len(rows) != 32 or len(subsets()) != 8:
        raise ExperimentError("candidate or row authority changed")
    if json.loads(PATHS["has_had_factorial"].read_text()).get("terminal") != "screen" or json.loads(PATHS["is_was_source"].read_text()).get("terminal") != "screen":
        raise ExperimentError("parent screen changed")
    return rows


def main():
    rows = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "rows": 32, "causal_arms": 9, "model_forwards_max": 26, "example_evaluations_max": 416, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = Backend.load("cuda")
    native, batch_pairs = {}, []
    forward_calls = evaluations = 0
    manual_error = reconstruction_error = 0.0
    capability_cells = []
    for family in ("A1", "A2"):
        chunk = [row for row in rows if row["transform_id"] == family]
        base_batch, donor_batch = das._batch(backend, chunk, side="base"), das._batch(backend, chunk, side="donor")
        captures = {}
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            reference = backend.native(batch, capture=False)
            manual, capture = backend.capture_bilinear(batch)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            manual_error = max(manual_error, max(abs(float(a) - float(b)) for pair_a, pair_b in zip(reference.answer_foil, manual.answer_foil) for a, b in zip(pair_a, pair_b)))
            for row, pair in zip(chunk, reference.answer_foil):
                native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(str(row["row_id"]), family, side, *producer._finite_pair(pair))
            captures[side] = capture
        _, error = backend.projected_terms(captures["base"], captures["donor"])
        reconstruction_error = max(reconstruction_error, error)
        batch_pairs.append((family, chunk, base_batch, donor_batch, captures["base"], captures["donor"]))
    for family in ("A1", "A2"):
        for direction in ("present_to_past", "past_to_present"):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                capability_cells.append({"family": family, "direction": direction, "side": side, "count": len(cell_rows), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    arm_values = {subset: {"A1": [], "A2": []} for subset in subsets()}
    logits, direct_logits, records = {}, {}, []
    for subset in subsets():
        for family, chunk, base_batch, donor_batch, base_capture, donor_capture in batch_pairs:
            output, _ = backend.intervene_factors(base_batch, donor_batch, base_capture, donor_capture, subset)
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                row_id = str(row["row_id"])
                recovery = kernel.signed_pairwise_donor_recovery(-native[(row_id, "base")].margin, native[(row_id, "donor")].margin, -(answer - foil))
                arm_values[subset][family].append(recovery)
                logits[(subset, row_id)] = (answer, foil)
                records.append({"arm": arm_id(subset), "family": family, "row_id": row_id, "recovery": recovery})
    direct_values = {"A1": [], "A2": []}
    for family, chunk, base_batch, donor_batch, base_capture, donor_capture in batch_pairs:
        output = backend.intervene_direct(base_batch, donor_batch, base_capture, donor_capture)
        forward_calls += 1
        evaluations += len(chunk)
        for row, pair in zip(chunk, output.answer_foil):
            answer, foil = producer._finite_pair(pair)
            row_id = str(row["row_id"])
            recovery = kernel.signed_pairwise_donor_recovery(-native[(row_id, "base")].margin, native[(row_id, "donor")].margin, -(answer - foil))
            direct_values[family].append(recovery)
            direct_logits[row_id] = (answer, foil)
            records.append({"arm": "direct_mlp4_ceiling", "family": family, "row_id": row_id, "recovery": recovery})
    summaries, values = {}, {}
    for subset in subsets():
        families = {family: summary(arm_values[subset][family]) for family in ("A1", "A2")}
        value = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm_id(subset)] = {"factors": list(subset), "families": families, "mean_target_recovery": value}
        values[subset] = value
    direct_summary = {family: summary(direct_values[family]) for family in ("A1", "A2")}
    full, two_term = FACTORS, ("left_change", "right_change")
    closure_error = max(abs(a - b) for row in rows for a, b in zip(logits[(full, str(row["row_id"]))], direct_logits[str(row["row_id"])]))
    shapley = {}
    for factor in FACTORS:
        total = 0.0
        for subset in subsets():
            if factor in subset:
                continue
            extended = tuple(item for item in FACTORS if item in set(subset) | {factor})
            total += math.factorial(len(subset)) * math.factorial(2 - len(subset)) / math.factorial(3) * (values[extended] - values[subset])
        shapley[factor] = total
    retained = values[two_term] / values[full] if values[full] != 0.0 else math.nan
    pred_a = all(cell["passed"] for cell in capability_cells) and manual_error <= 1e-4 and reconstruction_error <= 2e-3 and closure_error <= 1e-4 and len(records) == 288
    pred_b = values[full] >= 0.20 and all(summaries[arm_id(full)]["families"][family]["mean_recovery"] > 0.0 and summaries[arm_id(full)]["families"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_c = math.isfinite(retained) and retained >= 0.80 and all(summaries[arm_id(two_term)]["families"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = shapley["left_change"] > 0.0 and shapley["right_change"] > 0.0 and abs(shapley["bilinear_interaction"]) <= 0.25 * abs(values[full])
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "rows": len(rows), "causal_arms": 9, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 26, "example_evaluations": 416, "rows": 32, "causal_arms": 9, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_instrument": pred_a, "pred_b_live_mlp4_contextual_source_writer": pred_b, "pred_c_shared_left_right_two_term_program": pred_c, "pred_d_shared_factor_structure": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    result = {"schema": "tense_auxiliary_is_was_mlp4_contextual_source_factorial_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "rows_sha256": EXPECTED_ROWS_SHA256, "capability_cells": capability_cells, "instrument": {"manual_native_max_abs_logit_error": manual_error, "factor_reconstruction_max_abs_error": reconstruction_error, "full_factor_vs_direct_ceiling_max_abs_logit_error": closure_error}, "summaries": summaries, "direct_ceiling": direct_summary, "factorial_shapley_target_recovery": shapley, "two_term_retained_fraction": retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "shared_mlp4_two_term_contextual_source_writer", "null": "mlp4_writer_magnitude_or_factor_structure_does_not_reuse", "invalid": "authority_capability_factor_closure_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "test exact MLP4-to-L9H1/H4 mediation on fresh is/was rows" if terminal == "screen" else "retain shared downstream readers without a shared MLP4 writer"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells", "instrument", "summaries", "direct_ceiling", "factorial_shapley_target_recovery", "two_term_retained_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
