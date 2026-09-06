#!/usr/bin/env python3
"""Exact MLP4 -> contextual source bank -> L9H1/H4 mediation for is/was."""

# BQGATE: EXPERIMENT pred_a_exact_path_instrument pred_b_writer_recurrence pred_c_moment_determiner_mediation pred_d_reader_specificity pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as inherited
import run_tense_auxiliary_is_was_mlp4_contextual_source_factorial_v1 as writer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.mlp4_to_l9h1_h4_path_mediation_v1"
PATHS = {
    "writer_result": ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_contextual_source_factorial_v1_result.json",
    "path_instrument": ROOT / "ops/run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.py",
    "source_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py",
    "matched_builder": ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2.py",
}
EXPECTED_PRIOR_SHA256 = "1ff7168fd9784e0d824f656f8c03234e73ec454e88038a91bd3e321ae6cfcf57"
EXPECTED = {
    "writer_result": "18cbd0496ec79d1a30c41d12b620e5b9bd2d79eabd627b98b7db9016d4c63510",
    "path_instrument": "5449d45505267f2ebc92c9285b7a984fce773f834c2071e1d7e6d1a9f1949372",
    "source_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01",
    "matched_builder": "1f4b29bda3e26af3ee0102316ab0af166e317d1646e8b0b51332061245e606d6",
}
EXPECTED_ROWS_SHA256 = "2efd47b9a89d0f092688a96d75bbc33e5b89991a8e5de28723c714319b9ccceb"
FACTORS, HEADS = ("left_change", "right_change"), (1, 4)
BANK = ("moment", "determiner")
ARMS = ("writer_two_term", "h1h4_complete", "h1h4_all_sources", "h1h4_moment_determiner", "h1h4_cue_self")


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Backend(inherited.PathBackend):
    def capture_writer(self, base_batch, donor_batch, base_capture, donor_capture, factors):
        if factors not in ((), FACTORS):
            raise ExperimentError("writer factor set changed")
        projected, tensor_error = self.projected_terms(base_capture, donor_capture)
        positions = writer.source_positions(base_batch, donor_batch)

        def patch_mlp4(_module, _arguments, output):
            changed = output.clone()
            for i, bank in enumerate(positions):
                for position in bank:
                    delta = self.torch.zeros_like(changed[i, position], dtype=self.torch.float32)
                    for factor in factors:
                        delta += projected[factor][i, position]
                    changed[i, position] = (changed[i, position].float() + delta).to(changed.dtype)
            return changed

        handle = self.model.transformer.h[4].mlp.register_forward_hook(patch_mlp4)
        try:
            output, attention_capture = self.manual_forward(base_batch)
        finally:
            handle.remove()
        return output, attention_capture, tensor_error

    def mediate(self, base_batch, donor_batch, base_attention, hybrid_attention, source_names):
        if source_names not in (None, BANK, ("cue", "self"), ("all",)):
            raise ExperimentError("reader arm changed")
        head_dim = self.model.config.n_embd // self.model.config.n_head

        def patch_heads(_module, arguments):
            flattened = arguments[0]
            head_output = flattened.view(len(base_batch.row_ids), flattened.shape[1], self.model.config.n_head, head_dim).clone()
            for i, (base_ids, donor_ids, q) in enumerate(zip(base_batch.token_rows, donor_batch.token_rows, base_batch.semantic_positions)):
                differences = [position for position, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
                if len(base_ids) != len(donor_ids) or len(differences) != 1:
                    raise ExperimentError("cue alignment changed")
                cue = differences[0]
                positions = {"cue": cue, "moment": cue + 1, "determiner": cue + 2, "self": q}
                if source_names is None:
                    for head in HEADS:
                        head_output[i, q, head] = hybrid_attention["head_output"][i, q, head]
                    continue
                selected = tuple(range(q + 1)) if source_names == ("all",) else tuple(positions[name] for name in source_names)
                for position in selected:
                    for head in HEADS:
                        base_term = base_attention["pattern"][i, head, q, position] * base_attention["value"][i, position, head]
                        hybrid_term = hybrid_attention["pattern"][i, head, q, position] * hybrid_attention["value"][i, position, head]
                        head_output[i, q, head] += hybrid_term - base_term
            return (head_output.reshape_as(flattened),) + tuple(arguments[1:])

        handle = self.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
        try:
            output, _ = self.manual_forward(base_batch)
        finally:
            handle.remove()
        return output


def summary(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("missing or nonfinite recovery")
    return {"count": len(values), "mean_recovery": statistics.fmean(values), "mean_absolute_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    rows_by_bank = rows_builder.build_rows_by_bank()
    digests = rows_builder.validate_rows_by_bank(rows_by_bank)
    rows = [row for row in rows_by_bank["is_was"] if row["transform_id"] in {"A1", "A2"}]
    writer_result = json.loads(PATHS["writer_result"].read_text())
    if json.loads(PRIOR.read_text()).get("candidate_id") != CANDIDATE_ID or digests["is_was"] != EXPECTED_ROWS_SHA256 or len(rows) != 32 or writer_result.get("terminal") != "screen" or writer_result["summaries"]["left_change+right_change"]["mean_target_recovery"] != 0.31871069327054274:
        raise ExperimentError("candidate, rows, or writer authority changed")
    return rows


def main():
    rows = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "rows": 32, "scored_arms": 5, "intervention_records": 160, "model_forwards": 16, "example_evaluations": 256, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = Backend.load("cuda")
    native, arm_values = {}, {arm: {"A1": [], "A2": []} for arm in ARMS}
    capability_cells, records = [], []
    forward_calls = evaluations = 0
    empty_error = tensor_error = source_error = route_error = 0.0
    for family in ("A1", "A2"):
        chunk = [row for row in rows if row["transform_id"] == family]
        base_batch, donor_batch = das._batch(backend, chunk, side="base"), das._batch(backend, chunk, side="donor")
        base_output, base_capture = backend.capture_bilinear(base_batch)
        donor_output, donor_capture = backend.capture_bilinear(donor_batch)
        forward_calls += 2
        evaluations += 2 * len(chunk)
        for side, output in (("base", base_output), ("donor", donor_output)):
            for row, pair in zip(chunk, output.answer_foil):
                native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(str(row["row_id"]), family, side, *producer._finite_pair(pair))
        empty_output, base_attention, error_a = backend.capture_writer(base_batch, donor_batch, base_capture, donor_capture, ())
        writer_output, hybrid_attention, error_b = backend.capture_writer(base_batch, donor_batch, base_capture, donor_capture, FACTORS)
        forward_calls += 2
        evaluations += 2 * len(chunk)
        tensor_error = max(tensor_error, error_a, error_b)
        source_error = max(source_error, float(base_attention["reconstruction_max_abs"]), float(hybrid_attention["reconstruction_max_abs"]))
        empty_error = max(empty_error, max(abs(a - b) for pair_a, pair_b in zip(base_output.answer_foil, empty_output.answer_foil) for a, b in zip(pair_a, pair_b)))
        outputs = {
            "writer_two_term": writer_output,
            "h1h4_complete": backend.mediate(base_batch, donor_batch, base_attention, hybrid_attention, None),
            "h1h4_all_sources": backend.mediate(base_batch, donor_batch, base_attention, hybrid_attention, ("all",)),
            "h1h4_moment_determiner": backend.mediate(base_batch, donor_batch, base_attention, hybrid_attention, BANK),
            "h1h4_cue_self": backend.mediate(base_batch, donor_batch, base_attention, hybrid_attention, ("cue", "self")),
        }
        forward_calls += 4
        evaluations += 4 * len(chunk)
        route_error = max(route_error, max(abs(a - b) for pair_a, pair_b in zip(outputs["h1h4_complete"].answer_foil, outputs["h1h4_all_sources"].answer_foil) for a, b in zip(pair_a, pair_b)))
        for arm, output in outputs.items():
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                row_id = str(row["row_id"])
                recovery = kernel.signed_pairwise_donor_recovery(-native[(row_id, "base")].margin, native[(row_id, "donor")].margin, -(answer - foil))
                arm_values[arm][family].append(recovery)
                records.append({"arm": arm, "family": family, "row_id": row_id, "recovery": recovery})
    for family in ("A1", "A2"):
        for direction in ("present_to_past", "past_to_present"):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                capability_cells.append({"family": family, "direction": direction, "side": side, "count": len(cell_rows), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    summaries, targets = {}, {}
    for arm in ARMS:
        families = {family: summary(arm_values[arm][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm] = {"families": families, "mean_target_recovery": target}
        targets[arm] = target
    writer_recovery, all_recovery, bank_recovery, cue_self = targets["writer_two_term"], targets["h1h4_all_sources"], targets["h1h4_moment_determiner"], targets["h1h4_cue_self"]
    bank_writer = bank_recovery / writer_recovery
    bank_all = bank_recovery / all_recovery
    cue_self_fraction = abs(cue_self) / abs(all_recovery)
    pred_a = all(cell["passed"] for cell in capability_cells) and empty_error <= 1e-4 and tensor_error <= 2e-3 and source_error <= 1e-4 and route_error <= 1e-4 and len(records) == 160
    pred_b = abs(writer_recovery - 0.31871069327054274) <= 0.02 and all(summaries["writer_two_term"]["families"][family]["mean_recovery"] > 0.0 and summaries["writer_two_term"]["families"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_c = bank_writer >= 0.40 and all(summaries["h1h4_moment_determiner"]["families"][family]["mean_recovery"] > 0.0 and summaries["h1h4_moment_determiner"]["families"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = bank_all >= 0.80 and cue_self_fraction <= 0.25
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "rows": len(rows), "scored_arms": 5, "intervention_records": len(records), "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 16, "example_evaluations": 256, "rows": 32, "scored_arms": 5, "intervention_records": 160, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_exact_path_instrument": pred_a, "pred_b_writer_recurrence": pred_b, "pred_c_moment_determiner_mediation": pred_c, "pred_d_reader_specificity": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "rows_sha256": EXPECTED_ROWS_SHA256, "capability_cells": capability_cells, "instrument": {"empty_writer_hook_max_abs_logit_error": empty_error, "bilinear_tensor_reconstruction_max_abs_error": tensor_error, "attention_source_reconstruction_max_abs_error": source_error, "all_source_vs_complete_head_max_abs_logit_error": route_error}, "summaries": summaries, "bank_to_writer_retained_fraction": bank_writer, "bank_to_all_h1h4_retained_fraction": bank_all, "cue_self_absolute_all_h1h4_fraction": cue_self_fraction, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "shared_mlp4_to_contextual_l9h1_h4_path", "null": "writer_reader_mediation_or_specificity_failed", "invalid": "path_instrument_writer_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "compile shared MLP4 contextualizer and H1/H4 reader path into the typed dual program" if terminal == "screen" else "retain separately localized shared components without this mediation edge"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries", "bank_to_writer_retained_fraction", "bank_to_all_h1h4_retained_fraction", "cue_self_absolute_all_h1h4_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
