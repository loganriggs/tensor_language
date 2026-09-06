#!/usr/bin/env python3
"""Zero-forward conformance and persisted-causal audit for the dual text program."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_authority_release_and_rows pred_b_exhaustive_raw_text_selector pred_c_compiled_dual_program_conformance pred_d_persisted_causal_preservation pred_e_control_abstention pred_f_zero_forward_price_and_scope
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import torch

import aspectual_anchor_transparent_path_program_v12 as has_program
import aspectual_tense_raw_text_dual_program_v1 as dual
import circuit_candidate_aspectual_fresh_lexicon_v5 as has_builder
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6 as is_builder
from circuit_fast_screen_managed_runner import atomic_create_json
import tense_auxiliary_is_was_transparent_path_program_v1 as is_program


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_raw_text_dual_program_v1.json"
PROGRAM = ROOT / "ops/aspectual_tense_raw_text_dual_program_v1.py"
ALIGNED = ROOT / "circuits/followups/aspectual_tense_joint_upstream_program_composition_v1_result.json"
HAS_RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json"
IS_RELEASE = ROOT / "circuits/followups/tense_auxiliary_is_was_transparent_path_program_release_v1_result.json"
HAS_PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v12.py"
IS_PROGRAM = ROOT / "ops/tense_auxiliary_is_was_transparent_path_program_v1.py"
HAS_BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v5.py"
IS_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6.py"
OUT = ROOT / "circuits/followups/aspectual_tense_raw_text_dual_program_v1_result.json"
CANDIDATE_ID = "aspectual_tense.raw_text_dual_program_v1"
EXPECTED_PRIOR_SHA256 = "eef86d7a4aff30eb62bd0979b1049f790a932115db40a1e25c712d64b16c4312"
EXPECTED_PROGRAM_SHA256 = "a756bfbeddaad7db2bb0c7feec1f3a6bd976b05fec36d231865b69e4813a976c"
EXPECTED = {
    HAS_PROGRAM: "359953a4ec747fd83c1db9a4874655699e3580eda1d5d5d97b6c51c003ce22b1",
    IS_PROGRAM: "be5c7ea1dceb850ab125fbeb3b5f4814e571e678dbb6929fc89bd74e39588307",
    HAS_RELEASE: "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
    IS_RELEASE: "9804a0d0f047f194f6cce3490828c3a6e9525940f8c2b822467bc52176957e98",
    HAS_BUILDER: "ae624913c5adfe07cf028acf6549cd5fe2debd4b090c71659218fe158089fe2c",
    IS_BUILDER: "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
    ALIGNED: "46479986f81751af6141e8fcbaf19d4413198b119171711715414d2869f43e08",
}
EXPECTED_ROWS = {"has_had": "296c2186f477a6d450bbbb87fda5ba89b999eb4d3ac0dc18e31496ca47d5caf7", "is_was": "4eee90d9f39f6997c4926a0e7f6baecc4134c06535fe307d0a38f936b75defd5"}


class AuditError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    values = list(values)
    if not values or any(not math.isfinite(value) for value in values):
        raise AuditError("empty or nonfinite causal values")
    return {"count": len(values), "mean": statistics.fmean(values), "mean_absolute": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def expected_command(bank, row, side):
    if row["family"] == "C":
        return "abstain", None
    answer = str(row[f"{side}_answer"]).strip()
    if bank == "has_had":
        return bank, "present_to_past" if answer == "has" else "past_to_present"
    return bank, "present_to_past" if answer == "is" else "past_to_present"


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_forwards": 0, "selector_decisions": 256, "synthetic_conformance_cases": 3, "fitted_scalars": 0}, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or sha(PROGRAM) != EXPECTED_PROGRAM_SHA256 or any(sha(path) != digest for path, digest in EXPECTED.items()):
        raise AuditError("prior, program, or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    aligned = json.loads(ALIGNED.read_text())
    has_release = json.loads(HAS_RELEASE.read_text())
    is_release = json.loads(IS_RELEASE.read_text())
    rows_by_bank = {"has_had": has_builder.build_rows(), "is_was": is_builder.build_rows()}
    rows_sha = {"has_had": has_builder.validate_rows(rows_by_bank["has_had"]), "is_was": is_builder.validate_rows(rows_by_bank["is_was"])}
    row_maps = {bank: {str(row["row_id"]): row for row in rows} for bank, rows in rows_by_bank.items()}

    selector_records = []
    for bank, rows in rows_by_bank.items():
        for row in rows:
            for side in ("base", "donor"):
                expected_bank, expected_direction = expected_command(bank, row, side)
                observed = dual.select_command(row[f"{side}_text"])
                selector_records.append({
                    "bank": bank, "family": row["family"], "row_id": str(row["row_id"]), "side": side,
                    "expected_bank": expected_bank, "expected_direction": expected_direction,
                    "observed_bank": observed["bank"], "observed_direction": observed["direction"],
                    "correct": observed["bank"] == expected_bank and observed["direction"] == expected_direction,
                })

    width, vocabulary = 8, 600
    resid10 = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)
    base18 = torch.linspace(0.75, -0.25, width, dtype=torch.float64)
    q_has = torch.zeros(width, dtype=torch.float64); q_has[0] = 1.0
    q_is = torch.zeros(width, dtype=torch.float64); q_is[1] = 1.0
    lm_head = torch.nn.Linear(width, vocabulary, bias=False, dtype=torch.float64)
    with torch.no_grad():
        lm_head.weight.copy_(torch.arange(vocabulary * width, dtype=torch.float64).reshape(vocabulary, width) / 10000.0 - 0.2)
    bases = {"has_had": q_has, "is_was": q_is}
    wrapper_has = dual.actuate(resid10, base18, bases, lm_head, text="  Since   LAST week the pilot ")
    direct_has = has_program.upstream_carrier_actuation(resid10, base18, q_has, lm_head, direction="present_to_past")
    wrapper_is = dual.actuate(resid10, base18, bases, lm_head, text="At the previous moment the pilot")
    direct_is = is_program.upstream_writer_actuation(resid10, base18, q_is, lm_head, direction="past_to_present")
    wrapper_abstain = dual.actuate(resid10, base18, bases, lm_head, text="Beside the harbor the pilot finished")
    conformance = {
        "has_exact": wrapper_has["bank"] == "has_had" and wrapper_has["direction"] == "present_to_past" and torch.equal(wrapper_has["patched_resid18"], direct_has["patched_resid18"]) and torch.equal(wrapper_has["alpha"], direct_has["alpha"]) and torch.equal(wrapper_has["resid10_unembedding_contrast"], direct_has["resid10_unembedding_contrast"]),
        "is_exact": wrapper_is["bank"] == "is_was" and wrapper_is["direction"] == "past_to_present" and torch.equal(wrapper_is["patched_resid18"], direct_is["patched_resid18"]) and torch.equal(wrapper_is["alpha"], direct_is["alpha"]) and torch.equal(wrapper_is["resid10_unembedding_contrast"], direct_is["resid10_unembedding_contrast"]),
        "abstain_exact": wrapper_abstain["bank"] == "abstain" and wrapper_abstain["direction"] is None and wrapper_abstain["patched_resid18"] is base18 and wrapper_abstain["alpha"] == 0.0,
    }
    rejection_inputs = (None, "", "since last and by last", "since last at this moment")
    rejection = []
    for value in rejection_inputs:
        try:
            dual.select_command(value)
            rejection.append(False)
        except dual.SelectorInputError:
            rejection.append(True)

    causal_values = {bank: {family: [] for family in ("A1", "A2", "P")} for bank in ("has_had", "is_was")}
    causal_route_ok = True
    c_effects = []
    for record in aligned["intervention_records"]:
        bank, family = record["bank"], record["family"]
        row = row_maps[bank][record["row_id"]]
        side = "donor" if family == "P" else "base"
        command = dual.select_command(row[f"{side}_text"])
        if family == "C":
            causal_route_ok = causal_route_ok and command["bank"] == "abstain" and command["direction"] is None
            c_effects.append(0.0)
        else:
            causal_route_ok = causal_route_ok and command["bank"] == bank and command["direction"] == record["direction"]
            key = "own_margin_reflection_fraction" if family == "P" else "own_recovery"
            causal_values[bank][family].append(record[key])
    causal = {bank: {family: summarize(values) for family, values in families.items()} for bank, families in causal_values.items()}
    causal["C"] = {"count": len(c_effects), "mean_normalized_effect": statistics.fmean(c_effects), "max_normalized_effect": max(c_effects)}

    expected_authorities = {
        "q_has_program_sha256": EXPECTED[HAS_PROGRAM], "q_is_program_sha256": EXPECTED[IS_PROGRAM],
        "q_has_release_sha256": EXPECTED[HAS_RELEASE], "q_is_release_sha256": EXPECTED[IS_RELEASE],
        "q_has_builder_sha256": EXPECTED[HAS_BUILDER], "q_is_builder_sha256": EXPECTED[IS_BUILDER],
        "aligned_joint_result_sha256": EXPECTED[ALIGNED],
    }
    manifest = dual.program_manifest()
    pred_a = prior.get("candidate_id") == CANDIDATE_ID and prior.get("authorities") == expected_authorities and rows_sha == EXPECTED_ROWS and has_release.get("terminal") == "release" and is_release.get("terminal") == "release" and aligned.get("terminal") == "null" and len(aligned["intervention_records"]) == 128
    pred_b = len(selector_records) == 256 and all(record["correct"] for record in selector_records)
    pred_c = all(conformance.values()) and all(rejection) and manifest["branches"] == {"has_had": has_program.PROGRAM_ID, "is_was": is_program.PROGRAM_ID, "abstain": "identity at resid:18"}
    pred_d = causal_route_ok and all(causal[bank][family]["mean"] >= 0.75 and causal[bank][family]["direction_fraction"] >= 0.75 for bank in ("has_had", "is_was") for family in ("A1", "A2", "P"))
    pred_e = causal["C"]["count"] == 32 and causal["C"]["mean_normalized_effect"] == causal["C"]["max_normalized_effect"] == 0.0
    pred_f = manifest["selector"]["fitted_values"] == 0 and manifest["internal_task_circuit_claimed"] is False and "registered" in manifest["scope"]
    predictions = {
        "pred_a_authority_release_and_rows": pred_a,
        "pred_b_exhaustive_raw_text_selector": pred_b,
        "pred_c_compiled_dual_program_conformance": pred_c,
        "pred_d_persisted_causal_preservation": pred_d,
        "pred_e_control_abstention": pred_e,
        "pred_f_zero_forward_price_and_scope": pred_f,
    }
    terminal = "release" if all(predictions.values()) else ("null" if pred_a and pred_c and pred_f else "invalid")
    reason = {"release": "fit_free_raw_text_dual_program_released", "null": "selector_or_causal_preservation_misses", "invalid": "authority_conformance_coverage_rejection_or_scope_invalid"}[terminal]
    result = {
        "schema": "aspectual_tense_raw_text_dual_program_result_v1", "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "program_sha256": EXPECTED_PROGRAM_SHA256, "authority_sha256": expected_authorities, "rows_sha256": rows_sha,
        "manifest": manifest, "selector_summary": {"decisions": len(selector_records), "correct": sum(record["correct"] for record in selector_records), "ambiguities": 0},
        "conformance": conformance, "rejection": {"cases": len(rejection), "passed": sum(rejection)}, "causal": causal,
        "predictions": predictions,
        "price": {"model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "registered_cue_strings": 6, "selector_decisions": 256, "synthetic_conformance_cases": 3, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason,
        "scope_boundary": "Literal raw-text selection is released only for registered grammars and is not evidence for an internal task-gating circuit.",
        "next_action": "prospectively validate the unchanged selector and dual program on a disjoint lexical population, then localize an internal task branch" if terminal == "release" else "retain separate released programs and redesign the branch interface without modifying either writer",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "selector_summary", "conformance", "rejection", "causal", "predictions", "price", "terminal", "reason", "scope_boundary", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
