from __future__ import annotations

import torch

import previous_token_tensor_discovery as subject


def test_fit_bigram_set_uses_ordered_previous_query_pairs():
    rows = torch.arange(2 * 257).reshape(2, 257)
    pairs = subject.fit_bigram_set(rows)
    assert (0, 1) in pairs
    assert (255, 256) not in pairs
    assert (257, 258) in pairs
    assert len(pairs) == 2 * 255


def test_cell_masks_are_disjoint_and_unseen_is_subset_of_previous():
    tokens = torch.arange(256).repeat(2, 1)
    pattern = torch.zeros(2, 256, 256)
    query = torch.arange(256)
    pattern[0, query[1:], query[:-1]] = 2
    pattern[1, query, query] = 3
    fit = {(int(i), int(i + 1)) for i in range(100)}
    masks = subject.cell_masks(tokens, pattern, fit)
    assert torch.equal(
        masks["previous_top_unseen_bigram"] | masks["previous_top_seen_bigram"],
        masks["previous_top"],
    )
    assert not bool((masks["previous_top"] & masks["self_top"]).any())
    assert torch.equal(
        masks["previous_top"] | masks["self_top"] | masks["other_top"],
        masks["all"],
    )


def test_bootstrap_effects_recovers_planted_document_constant_effects():
    ledger = subject.empty_ledger()
    for document in range(40):
        for arm in subject.ARMS:
            for cell in subject.CELLS:
                count = 5
                native = 2.0
                if arm == "native" or arm == "full_replay":
                    ce = native
                elif arm == "head_deleted":
                    ce = 3.0
                elif arm == "extract_previous":
                    ce = 2.2
                elif arm == "remove_previous":
                    ce = 2.5 if cell.startswith("previous_top") else 2.01
                else:
                    ce = 2.9
                ledger[arm][cell]["count"].append(count)
                ledger[arm][cell]["loss_sum"].append(count * ce)
                ledger[arm][cell]["kl_sum"].append(0.0)
                ledger[arm][cell]["top1_changes"].append(0.0)
    result = subject.bootstrap_effects(ledger)["effects"]
    assert abs(result["removal_previous_top"]["mean"] - 0.5) < 1e-12
    assert abs(result["specificity_previous_minus_self"]["mean"] - 0.49) < 1e-12
    assert abs(result["extraction_recovery_previous_top"]["mean"] - 0.8) < 1e-12
