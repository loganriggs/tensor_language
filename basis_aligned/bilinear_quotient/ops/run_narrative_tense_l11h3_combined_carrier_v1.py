#!/usr/bin/env python3
"""Exact license-bound L11H3 carrier confirmation on the combined authority."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_unchanged_carrier_route pred_c_unchanged_carrier_effective_value pred_d_pre_first_negative_control pred_e_between_changes_effective_value pred_f_post_last_change_effective_value pred_g_distributed_R_effective_value pred_h_no_unchanged_carrier_route

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import attention_source_factor_primitive as source_factor
import circuit_fast_screen_candidate_narrative_tense_combined_authority as authority
import native_capability_license as licensing
import run_narrative_tense_combined_native_capability_v1 as capability_runner
import circuit_fast_screen_managed_runner as managed
import run_narrative_tense_attn11_head3_fresh_unchanged_carrier_value as core


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/NARRATIVE_TENSE_L11H3_COMBINED_CARRIER_V1_PREREGISTRATION.md"
OUT = ROOT / "circuits/fast_screens/narrative_tense_l11h3_combined_carrier_v1_result.json"
PRIOR_ART_SHA256 = "a77ea11d5de98e0c6e7aed2a28f56d3ba013d72e1eb4d4073a14e376c3a16d73"
AUTHORITY_FILE_SHA256 = "dea01f20355999af3104a54c701a6d0ee15c3192990f6fe448ed897eb0a70f12"
AUTHORITY_SHA256 = "a08264f00441cf6208ac7638c085c144661714046d2bad6148861d5bda5593e3"
LICENSE_SHA256 = "2805e7e49bdc1bb6d63a9e61758c5598474980eccdfc6f3ccc10015d331eb040"
LICENSE = ROOT / "circuits/fast_screens/narrative_tense_combined_native_capability_v1_license.json"
ARMS = core.ARMS
BARS = dict(core.BARS, minimum_native_accuracy_each_direction_side_cell=.875)
REGISTERED_PREDICTIONS = (
    'pred_a_instrument_live',
    'pred_b_unchanged_carrier_route',
    'pred_c_unchanged_carrier_effective_value',
    'pred_d_pre_first_negative_control',
    'pred_e_between_changes_effective_value',
    'pred_f_post_last_change_effective_value',
    'pred_g_distributed_R_effective_value',
    'pred_h_no_unchanged_carrier_route',
)
PREDICTION_REGISTRY = {
    "pred_a_instrument_live": None,
    "pred_b_unchanged_carrier_route": None,
    "pred_c_unchanged_carrier_effective_value": None,
    "pred_d_pre_first_negative_control": None,
    "pred_e_between_changes_effective_value": None,
    "pred_f_post_last_change_effective_value": None,
    "pred_g_distributed_R_effective_value": None,
    "pred_h_no_unchanged_carrier_route": None,
}


class ScreenError(ValueError):
    """The preregistered lexical or exact intervention closure changed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    rows = authority.build_rows()
    if authority.validate_rows(rows) != AUTHORITY_SHA256 or len(rows) != 64:
        raise ScreenError("combined authority changed")
    for row in rows:
        row["transform_id"] = row["family"]
        row["capability_cell_id"] = f"{row['family']}/{row['direction_id']}"
        base, donor = tuple(row["base_ids"]), tuple(row["donor_ids"])
        changed = tuple(i for i, values in enumerate(zip(base, donor))
                        if values[0] != values[1])
        if not changed or len(base) != len(donor) or base[-1] != donor[-1]:
            raise ScreenError("paired token contract changed")
        self_position = len(base) - 1
        first, last = min(changed), max(changed)
        row["T_positions"] = changed
        row["S_positions"] = (self_position,)
        row["pre_first_change_positions"] = tuple(range(first))
        row["between_changes_positions"] = tuple(
            index for index in range(first + 1, last) if index not in changed)
        row["post_last_change_positions"] = tuple(
            index for index in range(last + 1, self_position) if index not in changed)
        row["R_positions"] = tuple(index for index in range(self_position)
                                   if index not in changed)
        row["complement_positions"] = changed + (self_position,)
        if any(base[index] != donor[index] for index in row["R_positions"]):
            raise ScreenError("measured R contains a changed token")
        if set(row["R_positions"]) & set(row["complement_positions"]) \
                or set(row["R_positions"]) | set(row["complement_positions"]) != set(range(len(base))):
            raise ScreenError("R and complement do not form an exact source partition")
    return rows


def compile_plan():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256 \
            or _sha256(Path(authority.__file__)) != AUTHORITY_FILE_SHA256 \
            or _sha256(LICENSE) != LICENSE_SHA256:
        raise ScreenError("frozen receipt, authority, or capability license changed")
    gate = capability_runner.build_gate()
    licensing.validate_causal_preflight(
        gate, capability_runner.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    rows = build_rows()
    return {
        "schema": "narrative_tense_l11h3_combined_carrier_plan_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "prior_art_sha256": PRIOR_ART_SHA256, "authority_sha256": AUTHORITY_SHA256,
        "capability_license_sha256": LICENSE_SHA256, "row_count": len(rows),
        "layer": core.LAYER, "head": core.HEAD, "arms": list(ARMS), "bars": dict(BARS),
        "registered_predictions": list(REGISTERED_PREDICTIONS),
        "source_partition": "Algorithmic X/self/R and pre/between/post positions per row; R contains only token-identical non-self sources.",
        "subset_primitive": "attention_source_factor_primitive.replace_head_source_subset",
        "price": {"model_forwards": 6, "example_evaluations": 768,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_was_is_margin", "full_vocabulary_donor_CE_gain",
                     "P_C_absolute_margin_and_full_vocabulary_CE_change",
                     "same_batch_exactness", "registered_factorial_interactions"],
    }


def _subset_head(native, donor, positions, mode, torch):
    mask = torch.zeros_like(native["p"], dtype=torch.bool)
    if positions:
        mask[:, tuple(positions)] = True
    return source_factor.replace_head_source_subset(native, donor, mask, mode, torch)


def evaluate(model, torch, F, facade):
    """Use the landed exact evaluator, replacing only its row source and subset algebra."""
    old_builder, old_subset = core.build_rows, core._group_head
    core.build_rows, core._group_head = build_rows, _subset_head
    try:
        return core.evaluate(model, torch, F, facade)
    finally:
        core.build_rows, core._group_head = old_builder, old_subset


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise ScreenError(f"{name} must be absent or exactly 1")
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise ScreenError(f"refusing to overwrite {OUT}")
    torch, F, facade = core.factor._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, capability, exactness, liveness = evaluate(model, torch, F, facade)
    scored = core.score(evidence, capability, exactness, liveness, bars=BARS)
    terminal = core._terminal(scored["predictions"])
    result = {
        "schema": "narrative_tense_l11h3_combined_carrier_result_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal, "plan": plan,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["COMBINED_AUTHORITY_FIT_AND_HOLDOUT"],
        "forbidden_splits_opened": [], "active_price": plan["price"],
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "result_path": OUT.relative_to(ROOT).as_posix(),
                      "result_sha256": hashlib.sha256(payload).hexdigest(),
                      "active_price": plan["price"]}, sort_keys=True))


if __name__ == "__main__":
    main()
