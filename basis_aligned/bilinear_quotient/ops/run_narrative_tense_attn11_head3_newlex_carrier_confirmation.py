#!/usr/bin/env python3
"""Exact L11H3 carrier confirmation on a genuinely new lexical authority."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_unchanged_carrier_route pred_c_unchanged_carrier_effective_value pred_d_pre_first_negative_control pred_e_between_changes_effective_value pred_f_post_last_change_effective_value pred_g_distributed_R_effective_value pred_h_no_unchanged_carrier_route

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import attention_source_factor_primitive as source_factor
import circuit_fast_screen_candidate_narrative_tense_newlex_carrier_confirmation as authority
import circuit_fast_screen_managed_runner as managed
import run_narrative_tense_attn11_head3_fresh_unchanged_carrier_value as core


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/narrative_tense_attn11_head3_newlex_carrier_confirmation_v1.json"
OUT = ROOT / "circuits/fast_screens/narrative_tense_attn11_head3_newlex_carrier_confirmation_v1_result.json"
PRIOR_ART_SHA256 = "e442d7e006615b7d3dd66687c13766cba1725cef44b808d5e30286ec4e8ac5ab"
AUTHORITY_FILE_SHA256 = "fba93e9f228ea5e59c97090ad45805b36b788b2d9a82c95a3d4cd9d192503c3d"
AUTHORITY_SHA256 = "e8dc35550aa73a1380e5965c86d0fec420ea61b278dec71ad70a6ab55fd0ef06"
LICENSE_SHA256 = "db08a3d313330058f31f971c800c9481f485e368fd023554910e9cc30bc359ee"
LICENSE = ROOT / "circuits/fast_screens/narrative_tense_a1_direct_template_capability_select_holdout_v1_result.json"
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


class ScreenError(ValueError):
    """The preregistered lexical or exact intervention closure changed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    rows = authority.build_rows()
    if authority.validate_rows(rows) != AUTHORITY_SHA256 or len(rows) != 64:
        raise ScreenError("new lexical authority changed")
    for row in rows:
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
    rows = build_rows()
    return {
        "schema": "narrative_tense_attn11_head3_newlex_carrier_confirmation_plan_v1",
        "candidate_id": "narrative_tense.attn11_head3_newlex_carrier_confirmation_v1",
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
        "schema": "narrative_tense_attn11_head3_newlex_carrier_confirmation_result_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal, "plan": plan,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["NEW_LEXICAL_CONFIRMATION_BASIC"],
        "forbidden_splits_opened": [], "active_price": plan["price"],
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "result_path": OUT.relative_to(ROOT).as_posix(),
                      "result_sha256": hashlib.sha256(payload).hexdigest(),
                      "active_price": plan["price"]}, sort_keys=True))


if __name__ == "__main__":
    main()
