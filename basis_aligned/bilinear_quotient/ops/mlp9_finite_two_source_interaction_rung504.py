#!/usr/bin/env python3
"""RUNG504 -- finite two-source interactions through MLP9 and the real suffix.

The CPU-testable interaction algebra and frozen authorities are implemented here first. Real
execution remains fail-closed until the vectorized MLP9-plus-layers10--17 collector and its
literal call accounting are complete.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mlp9_attention8_finite_partner_screen_rung503 as parent


PREREG = POLY / "MLP9_FINITE_TWO_SOURCE_INTERACTION_RUNG504_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp9_attention8_finite_partner_screen_rung503.py"
PARENT_RESULT = ROOT / "mlp9_attention8_finite_partner_screen_rung503_results.json"
PARENT_BUNDLE = ROOT / "mlp9_attention8_finite_partner_screen_rung503_bundle.pt"
OUT = ROOT / "mlp9_finite_two_source_interaction_rung504_results.json"
BUNDLE = ROOT / "mlp9_finite_two_source_interaction_rung504_bundle.pt"
HASHES = {
    PREREG: "18a40b00be16acee3ac01f448a2d0ef68cbb2d91dd92a8a06ff21824859d0870",
    PARENT_SOURCE: "dd792fa67be3b8a14b8f552b356d0cac1bd424da3c9899701f85015922413e17",
    PARENT_RESULT: "b320e9706e1de230620a4da98b2c2a4e9e2b811bb7db577849e46a634b94f966",
    PARENT_BUNDLE: "7c59452743345eb29a32bebc30d1a51ae69e2dd8bf92674b98e1b26e33203c3f",
}
PARTNERS = parent.PARTNERS
PAIR_INDICES = tuple(itertools.combinations(range(len(PARTNERS)), 2))
PAIR_NAMES = tuple(f"{PARTNERS[left]}+{PARTNERS[right]}" for left, right in PAIR_INDICES)
SOURCE_SETS = tuple((index,) for index in range(len(PARTNERS))) + PAIR_INDICES
SINGLETON_COUNT = len(PARTNERS)
PAIR_COUNT = len(PAIR_INDICES)
DISCOVERY_BATCHES = 62
CONFIRMATION_BATCHES = 63
ORDINARY_EVALUATIONS_SELECTION = DISCOVERY_BATCHES * 2 * 3 * len(SOURCE_SETS)
ORDINARY_EVALUATIONS_TOTAL = (
    DISCOVERY_BATCHES + CONFIRMATION_BATCHES) * 2 * 3 * len(SOURCE_SETS)
POSITION_EVALUATIONS_PER_SELECTED_PAIR = CONFIRMATION_BATCHES * 2 * 3 * 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    required = {
        "status": "complete",
        "rung": 503,
        "pred_a_finite_source_instrument_and_parent_valid": True,
        "pred_b_compact_partner_set_selected": False,
        "pred_c_partner_identity_and_group_confirm": False,
        "pred_d_selective_downstream_use_confirms": False,
        "pred_e_candidate_for_suffix_intervention": False,
        "strong_null": True,
        "next_step": "finite_pair_removal_screen_or_float32_control",
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("rung503 does not license rung504")
    rows, circuit_masks, tags, metadata = parent.validate_inputs()
    if PARTNERS != parent.PARTNERS or len(PAIR_INDICES) != 153 \
            or len(set(PAIR_INDICES)) != 153 or len(SOURCE_SETS) != 171:
        raise RuntimeError("rung504 pair vocabulary changed")
    return rows, circuit_masks, tags, {
        "parent": metadata,
        "rung503_pair_outcomes_loaded_for_selection": False,
        "validation_documents_or_tags_opened": False,
    }


def finite_effect(native_absent, native_other, removed_absent, removed_other):
    """Return complete state difference and the part lost under every removal."""
    complete = native_absent - native_other
    after_removal = removed_absent - removed_other
    contribution = complete.unsqueeze(0) - after_removal
    return complete, contribution


def finite_mixed(pair_contribution, singleton_contribution):
    """Exact inclusion--exclusion interaction for the frozen unordered pairs."""
    if pair_contribution.shape[0] != PAIR_COUNT \
            or singleton_contribution.shape[0] != SINGLETON_COUNT:
        raise ValueError("candidate leading dimension changed")
    left = torch.tensor([pair[0] for pair in PAIR_INDICES], device=pair_contribution.device)
    right = torch.tensor([pair[1] for pair in PAIR_INDICES], device=pair_contribution.device)
    return pair_contribution - singleton_contribution[left] - singleton_contribution[right]


def source_sums(partner_sources, source_sets=SOURCE_SETS, *, shift=0):
    """Construct the exact raw contribution removed for each registered source set."""
    if partner_sources.shape[2] != SINGLETON_COUNT:
        raise ValueError("partner-source axis changed")
    values = [partner_sources[:, :, indices].sum(2) for indices in source_sets]
    stacked = torch.stack(values, dim=0)
    return torch.roll(stacked, shift, dims=2) if shift else stacked


def split_candidates(values):
    if values.shape[0] != len(SOURCE_SETS):
        raise ValueError("candidate leading dimension changed")
    return values[:SINGLETON_COUNT], values[SINGLETON_COUNT:]


def expected_price(selected_pair_count: int, confirmation_opened: bool):
    if selected_pair_count < 0 or selected_pair_count > 10:
        raise ValueError("selected pair count is outside the registered compact range")
    if not confirmation_opened:
        if selected_pair_count:
            raise ValueError("a nonempty selection must open confirmation")
        return {
            "full_model_forwards": 496,
            "mlp9_plus_suffix_evaluations": ORDINARY_EVALUATIONS_SELECTION,
            "backwards": 0,
        }
    return {
        "full_model_forwards": 1000,
        "mlp9_plus_suffix_evaluations": (
            ORDINARY_EVALUATIONS_TOTAL
            + POSITION_EVALUATIONS_PER_SELECTED_PAIR * selected_pair_count),
        "backwards": 0,
    }


def _dry_run():
    validate_inputs()
    torch.manual_seed(504)
    single = torch.randn(SINGLETON_COUNT, 2, 3)
    synergy = torch.randn(PAIR_COUNT, 2, 3)
    pair = torch.stack([
        single[left] + single[right] for left, right in PAIR_INDICES]) + synergy
    torch.testing.assert_close(finite_mixed(pair, single), synergy)
    sources = torch.randn(2, 3, SINGLETON_COUNT, 5)
    sums = source_sums(sources)
    assert tuple(sums.shape) == (171, 2, 3, 5)
    print(json.dumps({
        "status": "dry_run_core_passed",
        "rung": 504,
        "model_loaded": False,
        "pair_outcomes_opened": False,
        "partner_count": SINGLETON_COUNT,
        "pair_count": PAIR_COUNT,
        "selection_price": expected_price(0, False),
        "conditional_price_at_one_pair": expected_price(1, True),
        "real_execution_enabled": False,
    }, indent=2))


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv[1:]:
        _dry_run()
        return
    raise RuntimeError(
        "rung504 real execution is fail-closed until the vectorized suffix collector "
        "and literal call audit are complete")


if __name__ == "__main__":
    main()
