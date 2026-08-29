import pytest
import torch

import terminal_copy_induction_v1 as contract


def _rows_for_matching():
    rows = torch.arange(3 * contract.ROW_WIDTH, dtype=torch.long).reshape(
        3, contract.ROW_WIDTH,
    )
    # One positive at row0,p=64: prior 10,11 equals query,target 7,8.
    rows[0, 10:12] = torch.tensor([7, 8])
    rows[0, 64:66] = torch.tensor([7, 8])
    # Same matching stratum negative: prior query 7 has successor 9, current target 8.
    rows[1, 10:12] = torch.tensor([7, 9])
    rows[1, 64:66] = torch.tensor([7, 8])
    # A second positive without a matched negative exercises explicit exclusion.
    rows[2, 20:22] = torch.tensor([17, 18])
    rows[2, 80:82] = torch.tensor([17, 18])
    counts = torch.ones(int(rows.max()) + 1, dtype=torch.long)
    counts[7] = counts[17] = 8
    counts[8] = counts[18] = 4
    return rows, counts


def test_registered_candidate_sets_and_specificity_negative_are_frozen():
    assert contract.NAMED_SIX_HEAD_FAMILY == (
        "L5H5", "L7H3", "L8H3", "L8H4", "L13H0", "L14H7",
    )
    assert contract.REGISTERED_FOUR_HEAD_SET == ("L5H5", "L7H3", "L8H3", "L8H4")
    assert contract.REGISTERED_LATE_PAIR == ("L13H0", "L14H7")
    assert contract.CAPITALIZATION_SPECIFICITY_NEGATIVE == {
        "name": "boundary_capitalization_generic_booster",
        "prior_boundary_over_proper_noun_ratio": 1.0,
        "eligible_for_selection": False,
    }


def test_copy_cells_are_exact_disjoint_deterministic_and_one_to_one():
    rows, counts = _rows_for_matching()
    first = contract.build_copy_cells(rows, counts, ("d0", "d1", "d2"))
    second = contract.build_copy_cells(rows, counts, ("d0", "d1", "d2"))
    assert torch.equal(first.all_positive, second.all_positive)
    assert first.pair_indices == second.pair_indices == ((0, 64, 1, 64),)
    assert first.all_positive[0, 64] and first.positive[0, 64]
    assert first.matched_negative[1, 64]
    assert first.all_positive[2, 80] and not first.positive[2, 80]
    assert first.unmatched_positive_count == 1
    assert not bool((first.all_positive & first.off_target).any())
    assert not bool((first.matched_negative & first.off_target).any())
    assert int(first.all_positive.sum() + first.matched_negative.sum() + first.off_target.sum()) == (
        len(rows) * (contract.SCORE_STOP - contract.SCORE_START)
    )


def test_copy_matching_rejects_bad_rows_counts_and_document_ids():
    rows, counts = _rows_for_matching()
    with pytest.raises(ValueError, match=r"CPU long\[n,257\]"):
        contract.build_copy_cells(rows[:, :-1], counts, ("d0", "d1", "d2"))
    with pytest.raises(ValueError, match="token counts"):
        contract.build_copy_cells(rows, counts[:-1], ("d0", "d1", "d2"))
    with pytest.raises(ValueError, match="document IDs"):
        contract.build_copy_cells(rows, counts, ("same", "same", "d2"))


def test_synthetic_control_preserves_support_and_breaks_only_prior_successor():
    sequence = (100, 101, 102, 103, 104)
    cut = 3
    stem = tuple(range(1000, 1000 + contract.ROW_WIDTH - len(sequence) - cut - 1))
    positive, control = contract.build_synthetic_copy_pair(stem, sequence, cut)
    assert positive.shape == control.shape == (contract.ROW_WIDTH,)
    assert torch.equal(torch.sort(positive).values, torch.sort(control).values)
    query_position = contract.ROW_WIDTH - 2
    assert int(positive[query_position]) == int(control[query_position]) == sequence[cut - 1]
    assert int(positive[-1]) == int(control[-1]) == sequence[cut]
    first_query = len(stem) + cut - 1
    assert int(positive[first_query + 1]) == sequence[cut]
    assert int(control[first_query + 1]) != sequence[cut]


def test_synthetic_pair_requires_exact_width_unique_bank_and_interior_cut():
    sequence = (10, 11, 12, 13)
    stem = tuple(range(1000, 1000 + contract.ROW_WIDTH - len(sequence) - 2 - 1))
    with pytest.raises(ValueError, match="exact 257"):
        contract.build_synthetic_copy_pair(stem[:-1], sequence, 2)
    with pytest.raises(ValueError, match="malformed"):
        contract.build_synthetic_copy_pair(stem, (10, 11, 11, 13), 2)
    with pytest.raises(ValueError, match="malformed"):
        contract.build_synthetic_copy_pair(stem, sequence, 0)
    with pytest.raises(ValueError, match="malformed"):
        contract.build_synthetic_copy_pair((10,) + stem[1:], sequence, 2)


def test_behavior_reductions_keep_ce_top1_and_kl_separate():
    rows, counts = _rows_for_matching()
    cells = contract.build_copy_cells(rows, counts, ("d0", "d1", "d2"))
    vocab = int(rows.max()) + 2
    native = torch.zeros(len(rows), contract.MODEL_WIDTH, vocab)
    candidate = native.clone()
    targets = rows[:, 1:]
    native.scatter_(2, targets.unsqueeze(-1), 2.0)
    candidate.scatter_(2, targets.unsqueeze(-1), 1.0)
    reduced = contract.reduce_behavior(candidate, rows, cells, native_logits=native)
    assert set(reduced) == {"positive", "matched_negative", "off_target"}
    assert reduced["positive"].count == reduced["matched_negative"].count == 1
    assert reduced["positive"].ce == pytest.approx(-reduced["positive"].target_logprob)
    assert reduced["positive"].top1_accuracy == 1.0
    assert reduced["positive"].native_to_candidate_kl > 0
    assert reduced["off_target"].count > 0


def test_behavior_reduction_refuses_empty_confirmatory_support():
    rows, counts = _rows_for_matching()
    cells = contract.build_copy_cells(rows, counts, ("d0", "d1", "d2"))
    empty = contract.CopyCells(
        all_positive=cells.all_positive,
        positive=torch.zeros_like(cells.positive),
        matched_negative=torch.zeros_like(cells.matched_negative),
        off_target=(torch.zeros_like(cells.off_target).index_fill_(1, torch.arange(64, 256), True)
                    & ~cells.all_positive),
        pair_indices=(), unmatched_positive_count=int(cells.all_positive.sum()),
        negative_candidate_count=cells.negative_candidate_count,
    )
    logits = torch.zeros(len(rows), contract.MODEL_WIDTH, int(rows.max()) + 2)
    with pytest.raises(ValueError, match="positive support"):
        contract.reduce_behavior(logits, rows, empty)


def test_extraction_recovery_has_exact_denominator_and_no_zero_stake_fallback():
    assert contract.extraction_recovery(
        native_positive_ce=2.0, ablated_positive_ce=3.0, extracted_positive_ce=2.25,
    ) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="stake is not positive"):
        contract.extraction_recovery(
            native_positive_ce=3.0, ablated_positive_ce=3.0, extracted_positive_ce=2.5,
        )


def test_launch_gate_is_explicit_no_go_until_every_binding_is_true():
    bindings = {name: False for name in contract.REQUIRED_LAUNCH_BINDINGS}
    with pytest.raises(RuntimeError, match="launch NO-GO") as error:
        contract.assert_launch_ready(bindings)
    assert "per_head_attention_adapter" in str(error.value)
    ready = {name: True for name in contract.REQUIRED_LAUNCH_BINDINGS}
    contract.assert_launch_ready(ready)
    with pytest.raises(ValueError, match="exact frozen boolean schema"):
        contract.assert_launch_ready({**ready, "extra": True})
