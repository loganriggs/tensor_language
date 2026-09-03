#!/usr/bin/env python3

import torch

import bilin18_observed_model_facade as facade
import equality_factor_companion_causal_equivalence_rung532 as rung532


def test_replacement_patterns_are_the_registered_factorial():
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    b = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])
    c = torch.tensor([[[4.0, 1.0], [-2.0, 2.0]]])
    d = torch.tensor([[[0.5, -3.0], [2.0, 1.0]]])
    assert torch.equal(rung532.replacement_pattern("native", a, b, c, d), c * d)
    assert torch.equal(rung532.replacement_pattern("absent", a, b, c, d), torch.zeros_like(c))
    assert torch.allclose(
        rung532.replacement_pattern("swapped_first", a, b, c, d),
        rung532.ALPHA * b * d)
    assert torch.allclose(
        rung532.replacement_pattern("swapped_second", a, b, c, d),
        c * rung532.BETA * a)
    assert torch.allclose(
        rung532.replacement_pattern("swapped_both", a, b, c, d),
        rung532.ALPHA * rung532.BETA * a * b)
    assert torch.allclose(
        rung532.replacement_pattern("product_control", a, b, c, d),
        rung532.GAMMA * a * b)


def test_permuted_patterns_reverse_only_the_donor_key_prefix():
    a = torch.arange(16, dtype=torch.float64).view(1, 4, 4) + 1
    b = 2 * a
    c = 3 * a
    d = 4 * a
    first = rung532.replacement_pattern("permuted_first", a, b, c, d)
    expected = rung532.ALPHA * rung532.factor_screen._key_prefix_reverse(b) * d
    assert torch.equal(first, expected)
    second = rung532.replacement_pattern("permuted_second", a, b, c, d)
    expected = c * rung532.BETA * rung532.factor_screen._key_prefix_reverse(a)
    assert torch.allclose(second, expected)


def test_native_branch_product_multiplies_before_float_cast():
    first = torch.tensor([0.1, 0.3, -0.7], dtype=torch.bfloat16)
    second = torch.tensor([0.2, -0.4, 0.9], dtype=torch.bfloat16)
    deployed = rung532.native_branch_product(first, second)
    assert torch.equal(deployed, (first * second).float())
    assert not torch.equal(deployed, first.float() * second.float())


def _planted_collection():
    tags = (tuple(f"d{i}" for i in range(32)), tuple(f"v{i}" for i in range(30)))
    collection = rung532.empty_collection(tags)
    for counts in collection["circuit_counts"].values():
        counts.fill_(10)
    collection["task_counts"].fill_(10)
    for tag_set_name in rung532.TAG_SET_NAMES:
        sums = collection["circuit_sums"][tag_set_name]
        n = sums.shape[-1]
        reference = torch.linspace(0.2, 1.0, n, dtype=torch.float64)
        bad = reference.flip(0)
        for background in range(len(rung532.BACKGROUNDS)):
            for half in range(2):
                for arm_index, arm in enumerate(rung532.ARMS):
                    if arm == "absent":
                        effect = torch.zeros_like(reference)
                    elif arm in {"native", "swapped_first", "swapped_both", "product_control"}:
                        effect = reference
                    elif arm in {"permuted_first", "direct_first"}:
                        effect = -reference
                    else:
                        effect = bad
                    member_ce = 5.0 - effect
                    sums[background, arm_index, half, rung532.MASK_TYPES.index("member")] = (
                        10 * member_ce)
                    native_slice = torch.full_like(reference, 3.0)
                    slice_ce = native_slice if arm == "swapped_first" else native_slice + 0.02
                    if arm == "native":
                        slice_ce = native_slice
                    sums[background, arm_index, half,
                         rung532.MASK_TYPES.index("slice_control")] = 10 * slice_ce
    for background in range(len(rung532.BACKGROUNDS)):
        for half in range(2):
            for arm_index, arm in enumerate(rung532.ARMS):
                copy_ce = 5.0 if arm == "absent" else (
                    4.0 if arm in {"native", "swapped_first", "swapped_both", "product_control"}
                    else 4.8)
                collection["task_sums"][background, arm_index, half, 0] = 10 * copy_ce
                collection["task_sums"][background, arm_index, half, 1] = 40.0
    return collection


def test_scoring_can_identify_one_downstream_factor_without_claiming_the_other():
    collection = _planted_collection()
    reports, contexts = rung532.analyze(collection)
    diagnostics = {
        "native_replay_logit_max_abs": 0.0,
        "factor_reconstruction_max": 0.0,
        "branch_product_max_abs": 0.0,
        "minimum_donor_edit_rms": 1.0,
        "minimum_target_edit_rms": 1.0,
        "zero_intended_edits": 0,
        "calls_exact": True,
        "all_circuit_supports_live": True,
        "all_task_supports_live": True,
    }
    predictions, checks = rung532.score(
        reports, contexts, diagnostics, collection, facade.WEIGHTS_SHA256)
    assert predictions["pred_a_exact_live_interaction_instrument"]
    assert predictions["pred_b_product_control_transfers"]
    assert predictions["pred_c_source_second_replaces_target_first"]
    assert not predictions["pred_d_source_first_replaces_target_second"]
    assert predictions["pred_e_heldout_interaction_defined_factor"]
    assert checks["swapped_first_contexts_passing"] == 8


def test_price_and_registered_axes_are_exact():
    assert len(rung532.ARMS) == 10
    assert len(rung532.BACKGROUNDS) == 2
    assert rung532.FORWARDS_PER_BATCH == 21
    assert rung532.BATCHES == 125
    assert rung532.FORWARDS == 2625
