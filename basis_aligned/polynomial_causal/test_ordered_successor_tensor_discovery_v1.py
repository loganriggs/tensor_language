from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import circuit_campaign_runtime as campaign
from ordered_successor_masks_v1 import OrderedLexicon, build_ordered_successor_masks
import ordered_successor_tensor_discovery_v1 as discovery


HEX = "a" * 64


def _closure(arm: str, documents: int = 4) -> campaign.ForwardClosure:
    sites = []
    for site in range(discovery.SITE_COUNT):
        replace_attention = arm != "native" and site == discovery.TARGET_SITE
        sites.append(campaign.SiteCallLedger(
            site=site,
            native_attention_calls=0 if replace_attention else 1,
            replacement_attention_calls=1 if replace_attention else 0,
            native_mlp_calls=1,
            replacement_mlp_calls=0,
        ))
    return campaign.ForwardClosure(
        circuit=discovery.SCHEMA,
        arm=arm,
        arm_kind=(campaign.ArmKind.NATIVE if arm == "native" else campaign.ArmKind.CANDIDATE),
        attempted_outer_forwards=1,
        completed_outer_forwards=1,
        outer_returns=1,
        document_count=documents,
        sites=tuple(sites),
        candidate_native_call_prohibition_passed=True,
        closed=True,
    )


def _authority() -> discovery.DiscoveryAuthority:
    return discovery.DiscoveryAuthority(
        source_commit="b" * 40,
        source_files=tuple(
            (path, format(index + 1, "064x"))
            for index, path in enumerate(discovery.SOURCE_CLOSURE)
        ),
        select_rows_sha256="c" * 64,
        select_support_sha256="d" * 64,
        select_documents=96,
        lexicon_registry_sha256="e" * 64,
        model_config_sha256="f" * 64,
        model_weights_sha256="0" * 64,
        programs=tuple(discovery.ProgramBinding(
            arm, format(index + 100, "064x"), discovery.arm_stored_parameters(arm),
        ) for index, arm in enumerate(discovery.ARM_NAMES[1:])),
    )


def test_arm_registry_is_exact_complete_priced_and_replaces_only_attention8() -> None:
    assert discovery.RANK_LADDER == (8, 16, 32, 64, 96, 128)
    assert len(discovery.ARM_NAMES) == 17
    assert discovery.arm_stored_parameters("native") == 1_032_192
    assert discovery.arm_stored_parameters("head8_7_both_r64_true") == 811_008
    assert discovery.arm_stored_parameters("head8_7_both_r64_spectral_null") == 811_008
    assert discovery.arm_stored_parameters(discovery.CURRENT_ONLY) == 884_736
    plan = discovery.build_circuit_plan()
    assert tuple(arm.name for arm in plan.arms) == discovery.ARM_NAMES
    for arm in plan.arms:
        replaced = tuple(
            item.site for item in arm.attention
            if item.action is campaign.ComponentAction.REPLACE
        )
        assert replaced == (() if arm.name == "native" else (8,))
        assert all(item.action is campaign.ComponentAction.NATIVE for item in arm.mlp)


def test_authority_is_select_only_exact_order_and_price_fail_closed() -> None:
    authority = _authority()
    assert authority.select_documents == 96
    with pytest.raises(ValueError, match="frozen formula"):
        replace(
            authority,
            programs=(replace(authority.programs[0], stored_parameters=7),)
            + authority.programs[1:],
        )
    with pytest.raises(ValueError, match="arm order"):
        replace(authority, programs=authority.programs[::-1])
    with pytest.raises(ValueError, match="git hash"):
        replace(authority, source_commit="short")


def test_closure_requires_zero_native_attention8_and_literal_other_components() -> None:
    for arm in discovery.ARM_NAMES:
        discovery.validate_forward_closure(_closure(arm), arm, 4)
    bad = _closure("head8_7_both_r32_true")
    sites = list(bad.sites)
    sites[8] = replace(sites[8], native_attention_calls=1)
    with pytest.raises(ValueError, match="attention call ledger"):
        discovery.validate_forward_closure(replace(bad, sites=tuple(sites)), bad.arm, 4)
    sites = list(_closure("native").sites)
    sites[7] = replace(sites[7], native_mlp_calls=0)
    with pytest.raises(ValueError, match="MLP call ledger"):
        discovery.validate_forward_closure(
            replace(_closure("native"), sites=tuple(sites)), "native", 4,
        )


def test_registry_and_support_hash_bind_order_rows_cells_and_pair_indices() -> None:
    first = OrderedLexicon("first", ((10,), (11,), (12,)))
    second = OrderedLexicon("second", ((20,), (21,)))
    lexicons = (first, second)
    rows = torch.full((2, discovery.ROW_LENGTH), 90, dtype=torch.long)
    rows[0, :5] = torch.tensor([10, 90, 91, 92, 11])
    rows[1, :5] = torch.tensor([20, 80, 81, 82, 21])
    rows = rows.contiguous()
    masks = {
        lexicon.name: build_ordered_successor_masks(
            rows, lexicon, window=8, first_prediction=0,
        ) for lexicon in lexicons
    }
    digest = discovery.support_sha256(rows, lexicons, masks)
    assert len(digest) == 64
    changed = rows.clone(); changed[0, 1] = 99
    changed_masks = {
        lexicon.name: build_ordered_successor_masks(
            changed, lexicon, window=8, first_prediction=0,
        ) for lexicon in lexicons
    }
    assert discovery.support_sha256(changed, lexicons, changed_masks) != digest
    with pytest.raises(ValueError, match="lexicon order"):
        discovery.support_sha256(rows, lexicons, dict(reversed(tuple(masks.items()))))
    with pytest.raises(ValueError, match="two discovery lexicons"):
        discovery.validate_lexicon_registry((first, OrderedLexicon("overlap", ((12,), (30,)))))


def test_document_statistics_match_direct_ce_kl_top1_and_margin_sums() -> None:
    native = torch.tensor([
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ])
    arm = native.clone()
    arm[0, 0] = torch.tensor([0.0, 2.0, -1.0])
    targets = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    mask = torch.tensor([[True, False], [True, True]])
    source_logits = torch.tensor([[0.5, 0.0], [0.25, 0.75]])
    target_logits = torch.tensor([[1.5, 0.0], [1.25, 2.75]])
    stats = discovery.document_cell_statistics(
        native, arm, targets, mask, source_logits, target_logits,
    )
    direct_ce = torch.nn.functional.cross_entropy(
        arm.transpose(1, 2), targets, reduction="none",
    ).double()
    native_lp = torch.log_softmax(native, -1)
    arm_lp = torch.log_softmax(arm, -1)
    direct_kl = (native_lp.exp() * (native_lp - arm_lp)).sum(-1).double()
    weight = mask.double()
    assert torch.equal(stats.count, torch.tensor([1, 2]))
    torch.testing.assert_close(stats.ce_sum, (direct_ce * weight).sum(-1))
    torch.testing.assert_close(stats.native_kl_sum, (direct_kl * weight).sum(-1))
    torch.testing.assert_close(stats.top1_change_sum, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(stats.successor_margin_sum, torch.tensor([1.0, 3.0]))


def test_item_margin_is_recomputed_from_pair_index_and_multitoken_items() -> None:
    lexicon = OrderedLexicon("multi", ((0, 1), (2,), (3, 4)))
    rows = torch.tensor([[0, 9, 2, 9, 3]], dtype=torch.long)
    masks = build_ordered_successor_masks(rows, lexicon, window=4, first_prediction=0)
    logits = torch.arange(1 * 4 * 10, dtype=torch.float32).reshape(1, 4, 10)
    source, target = discovery.ordered_item_margin_logits(logits, masks, lexicon)
    # Pair index 0 targets item (2,) and sources the mean of (0,1).
    selected0 = masks.pair_index == 0
    torch.testing.assert_close(source[selected0], logits[..., [0, 1]].mean(-1)[selected0])
    torch.testing.assert_close(target[selected0], logits[..., [2]].mean(-1)[selected0])
    # Pair index 1 targets mean(3,4) and sources item (2,).
    selected1 = masks.pair_index == 1
    torch.testing.assert_close(source[selected1], logits[..., [2]].mean(-1)[selected1])
    torch.testing.assert_close(target[selected1], logits[..., [3, 4]].mean(-1)[selected1])


def test_production_batch_rejects_missing_arm_and_wrong_unsliced_vocab() -> None:
    rows = torch.zeros((1, discovery.ROW_LENGTH), dtype=torch.long).contiguous()
    # A missing arm fails before large production tensors need to be allocated.
    with pytest.raises(ValueError, match="arm order"):
        discovery.validate_production_batch(rows, {})
    malformed = {arm: torch.zeros((1, 256, 7)) for arm in discovery.ARM_NAMES}
    with pytest.raises(ValueError, match="malformed"):
        discovery.validate_production_batch(rows, malformed)
