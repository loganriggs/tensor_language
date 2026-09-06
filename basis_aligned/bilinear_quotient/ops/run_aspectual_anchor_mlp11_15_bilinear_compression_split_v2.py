#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_authority_split_capability_and_exact_instrument pred_b_writer_and_source_control_recurrence pred_c_positive_selection_mlp_signal pred_d_disjoint_two_term_compression pred_e_exact_coverage
"""Corrected three-role-context execution of the suffix-MLP split factorial."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v1 as v1


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp11_15_bilinear_compression_split_v2.json"
AUDIT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_v2_result.json"
OLD_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp11_15_bilinear_compression_split_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_split_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_split_v2"
EXPECTED_PRIOR_SHA256 = "7097bec6a49e38f70f5c5dc5b9a2bbc29aaa7a0290510685a151965a7672c017"
EXPECTED_AUDIT_SHA256 = "7933cf1e4adae814fc3d2251692803ce1233f6eab61343cce026748b11b4127a"
EXPECTED_OLD_RUNNER_SHA256 = "4114c79ca08e1fb5e293481221a4a4249bc18c7f3dc7018b393d751be5f18b84"
SOURCE_BANK_BY_BOUNDARY = {
    11: ("determiner", "period", "self"),
    15: ("period", "determiner", "self"),
}
PREDICTION_SCHEMA = {
    "pred_a_authority_split_capability_and_exact_instrument": None,
    "pred_b_writer_and_source_control_recurrence": None,
    "pred_c_positive_selection_mlp_signal": None,
    "pred_d_disjoint_two_term_compression": None,
    "pred_e_exact_coverage": None,
}


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorrectedSuffixMlpBackend(v1.SuffixMlpBackend):
    def factor_crossing(
        self, batch, role_banks, base_capture, hybrid_capture, base_terms,
        hybrid_terms, boundary, selected_factors,
    ):
        if selected_factors not in v1.subsets():
            raise ExperimentError("MLP factor subset changed")
        state = base_capture[f"resid{boundary + 1}"].clone()
        lambda0 = self.model.transformer.h[boundary].lambdas[0]
        projected_attention = self.projected_source_delta(
            batch, role_banks, base_terms, hybrid_terms, boundary,
            SOURCE_BANK_BY_BOUNDARY[boundary],
        )
        projected_mlp, _error = self.projected_mlp_terms(
            base_capture, hybrid_capture, boundary
        )
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


def main() -> None:
    if sha256(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("corrected prior hash changed")
    if sha256(AUDIT) != EXPECTED_AUDIT_SHA256:
        raise ExperimentError("design-audit hash changed")
    if sha256(OLD_RUNNER) != EXPECTED_OLD_RUNNER_SHA256:
        raise ExperimentError("v1 engine hash changed")
    prior = json.loads(PRIOR.read_text())
    audit = json.loads(AUDIT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or audit.get("terminal") != "screen"
        or audit.get("scientific_disposition") != "v1_superseded_as_invalid"
    ):
        raise ExperimentError("corrected authority changed")

    original_validate = v1.validate_static
    original_atomic = managed.atomic_create_json

    def corrected_validate():
        selection, confirmation, spec, source_release, source_result = original_validate()
        corrected_records = []
        for record in source_result["intervention_logits"]:
            if record.get("phase") == "confirmation" and record.get("boundary") in v1.BOUNDARIES:
                if record.get("arm_id") == "all_sources":
                    continue
                if record.get("arm_id") == "selected_three":
                    changed = dict(record)
                    changed["arm_id"] = "all_sources"
                    corrected_records.append(changed)
                    continue
            corrected_records.append(record)
        changed_result = dict(source_result)
        changed_result["intervention_logits"] = corrected_records
        return selection, confirmation, spec, source_release, changed_result

    def corrected_atomic(path, value):
        if Path(path) != OUT:
            raise ExperimentError("corrected output path changed")
        changed = dict(value)
        changed["schema"] = "aspectual_anchor_mlp11_15_bilinear_compression_split_result_v2"
        changed["correction"] = "factor_crossing uses the released boundary-specific three-role source bank"
        changed["v1_design_audit_sha256"] = EXPECTED_AUDIT_SHA256
        changed["next_action"] = (
            "compile the corrected MLP11 and MLP15 bilinear terms into transparent program v5"
            if changed.get("terminal") == "screen"
            else "retain full native MLP11 and MLP15 deltas"
        )
        if isinstance(changed.get("dryrun"), dict):
            changed["dryrun"] = dict(changed["dryrun"])
            changed["dryrun"]["schema"] = "aspectual_anchor_mlp11_15_bilinear_compression_split_dryrun_v2"
            changed["dryrun"]["sole_correction"] = "boundary-specific released three-role source bank"
        return original_atomic(path, changed)

    v1.PRIOR = PRIOR
    v1.OUT = OUT
    v1.CANDIDATE_ID = CANDIDATE_ID
    v1.EXPECTED_PRIOR_SHA256 = EXPECTED_PRIOR_SHA256
    v1.SuffixMlpBackend = CorrectedSuffixMlpBackend
    v1.validate_static = corrected_validate
    managed.atomic_create_json = corrected_atomic
    try:
        v1.main()
    finally:
        managed.atomic_create_json = original_atomic


if __name__ == "__main__":
    main()
