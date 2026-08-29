import pytest
import torch

import terminal_copy_induction_v1 as contract


def _rows_for_matching():
    rows = torch.arange(1000, 1000 + 5 * contract.ROW_WIDTH, dtype=torch.long).reshape(
        5, contract.ROW_WIDTH,
    )
    # Two positive documents and two negative documents in one eligible stratum.
    for row_index in (0, 1):
        rows[row_index, 10:12] = torch.tensor([7, 8])
        rows[row_index, 64:66] = torch.tensor([7, 8])
    for row_index in (2, 3):
        rows[row_index, 10:12] = torch.tensor([7, 9])
        rows[row_index, 64:66] = torch.tensor([7, 8])
    # A positive in a different stratum has no supported negative polarity.
    rows[4, 20:22] = torch.tensor([17, 18])
    rows[4, 80:82] = torch.tensor([17, 18])
    size = int(rows.max()) + 1
    query = torch.ones(size, dtype=torch.long)
    target = torch.ones(size, dtype=torch.long)
    query[7] = query[17] = 8
    target[8] = target[18] = 4
    return rows, contract.FitTokenFrequencies(query=query, target=target)


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
    rows, frequencies = _rows_for_matching()
    document_ids = tuple(f"d{i}" for i in range(len(rows)))
    first = contract.build_copy_cells(rows, frequencies, document_ids)
    second = contract.build_copy_cells(rows, frequencies, document_ids)
    assert torch.equal(first.all_positive, second.all_positive)
    assert first.pair_indices == second.pair_indices
    assert len(first.pair_indices) == 2
    assert {pair[0] for pair in first.pair_indices} == {0, 1}
    assert {pair[2] for pair in first.pair_indices} == {2, 3}
    assert first.all_positive[0, 64] and first.positive[0, 64]
    assert first.matched_negative[2, 64] or first.matched_negative[3, 64]
    assert first.all_positive[4, 80] and not first.positive[4, 80]
    assert first.unmatched_positive_count == 1
    assert first.eligible_stratum_count == 1
    assert not bool((first.all_positive & first.off_target).any())
    assert not bool((first.matched_negative & first.off_target).any())
    assert int(first.all_positive.sum() + first.matched_negative.sum() + first.off_target.sum()) == (
        len(rows) * (contract.SCORE_STOP - contract.SCORE_START)
    )


def test_copy_matching_rejects_bad_rows_counts_and_document_ids():
    rows, frequencies = _rows_for_matching()
    document_ids = tuple(f"d{i}" for i in range(len(rows)))
    with pytest.raises(ValueError, match=r"CPU long\[n,257\]"):
        contract.build_copy_cells(rows[:, :-1], frequencies, document_ids)
    with pytest.raises(ValueError, match="token counts"):
        contract.build_copy_cells(
            rows,
            contract.FitTokenFrequencies(
                query=frequencies.query[:-1], target=frequencies.target,
            ),
            document_ids,
        )
    with pytest.raises(ValueError, match="document IDs"):
        contract.build_copy_cells(rows, frequencies, ("same",) * len(rows))


def test_frequency_bins_keep_unseen_distinct_from_count_one():
    assert contract._frequency_bin(0) == -1
    assert contract._frequency_bin(1) == 0
    assert contract._frequency_bin(2) == 1


def test_nearest_prior_query_controls_the_copy_label():
    rows = torch.arange(2000, 2000 + 4 * contract.ROW_WIDTH, dtype=torch.long).reshape(
        4, contract.ROW_WIDTH,
    )
    # Positive rows: an older contradiction is superseded by nearest q->y.
    for row_index in (0, 1):
        rows[row_index, 5:7] = torch.tensor([7, 9])
        rows[row_index, 10:12] = torch.tensor([7, 8])
        rows[row_index, 64:66] = torch.tensor([7, 8])
    # Negative rows: an older q->y witness is superseded by nearest q->z.
    for row_index in (2, 3):
        rows[row_index, 5:7] = torch.tensor([7, 8])
        rows[row_index, 10:12] = torch.tensor([7, 9])
        rows[row_index, 64:66] = torch.tensor([7, 8])
    size = int(rows.max()) + 1
    frequencies = contract.FitTokenFrequencies(
        query=torch.ones(size, dtype=torch.long),
        target=torch.ones(size, dtype=torch.long),
    )
    cells = contract.build_copy_cells(
        rows, frequencies, tuple(f"nearest-{i}" for i in range(4)),
    )
    assert cells.all_positive[:, 64].tolist() == [True, True, False, False]
    assert int(cells.positive[:, 64].sum()) == 2
    assert int(cells.matched_negative[:, 64].sum()) == 2


def test_retained_matches_remain_document_balanced_with_many_positions_in_one_document():
    rows = torch.arange(3000, 3000 + 4 * contract.ROW_WIDTH, dtype=torch.long).reshape(
        4, contract.ROW_WIDTH,
    )
    # Each polarity has one prolific document and one single-record document.  All
    # records share a position/distance/frequency/multiplicity stratum.
    positive_positions = {0: range(64, 80, 4), 1: (64,)}
    negative_positions = {2: range(64, 80, 4), 3: (64,)}
    for row_index, positions in positive_positions.items():
        for position in positions:
            query = 7 + 20 * (position - 64)
            target = query + 1
            rows[row_index, position - 16:position - 14] = torch.tensor([query, target])
            rows[row_index, position:position + 2] = torch.tensor([query, target])
    for row_index, positions in negative_positions.items():
        for position in positions:
            query = 7 + 20 * (position - 64)
            target = query + 1
            rows[row_index, position - 16:position - 14] = torch.tensor([query, target + 1])
            rows[row_index, position:position + 2] = torch.tensor([query, target])
    size = int(rows.max()) + 1
    frequencies = contract.FitTokenFrequencies(
        query=torch.ones(size, dtype=torch.long) * 4,
        target=torch.ones(size, dtype=torch.long) * 4,
    )
    cells = contract.build_copy_cells(
        rows, frequencies, tuple(f"imbalanced-{i}" for i in range(4)),
    )
    assert {pair[0] for pair in cells.pair_indices} == {0, 1}
    retained_negative_documents = {pair[2] for pair in cells.pair_indices}
    assert {2, 3}.issubset(retained_negative_documents)
    assert len(retained_negative_documents) >= contract.MIN_DOCUMENTS_PER_POLARITY_STRATUM


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


def test_reciprocal_synthetic_crossover_preserves_multiset_and_scores_did():
    base = tuple(range(1000, 1000 + contract.ROW_WIDTH))
    crossover = contract.build_synthetic_association_crossover(
        base, first_query_position=10, reciprocal_position=30, query_position=80,
        query_token=7, reciprocal_query=11, successor_y=8, successor_z=12,
    )
    assert torch.equal(
        torch.sort(crossover.query_to_y).values,
        torch.sort(crossover.query_to_z).values,
    )
    assert crossover.query_to_y[10:12].tolist() == [7, 8]
    assert crossover.query_to_y[30:32].tolist() == [11, 12]
    assert crossover.query_to_z[10:12].tolist() == [7, 12]
    assert crossover.query_to_z[30:32].tolist() == [11, 8]
    logits = torch.zeros(2, contract.MODEL_WIDTH, 20)
    logits[0, 80, 8] = 2.0
    logits[0, 80, 12] = -1.0
    logits[1, 80, 8] = -2.0
    logits[1, 80, 12] = 1.0
    assert contract.synthetic_association_did(logits, crossover) == pytest.approx(6.0)


def test_behavior_reductions_keep_ce_top1_and_kl_separate():
    rows, frequencies = _rows_for_matching()
    cells = contract.build_copy_cells(
        rows, frequencies, tuple(f"d{i}" for i in range(len(rows))),
    )
    vocab = int(rows.max()) + 2
    native = torch.zeros(len(rows), contract.MODEL_WIDTH, vocab)
    candidate = native.clone()
    targets = rows[:, 1:]
    native.scatter_(2, targets.unsqueeze(-1), 2.0)
    candidate.scatter_(2, targets.unsqueeze(-1), 1.0)
    reduced = contract.reduce_behavior(candidate, rows, cells, native_logits=native)
    assert set(reduced) == {"positive", "matched_negative", "off_target"}
    assert reduced["positive"].count == reduced["matched_negative"].count == 2
    assert reduced["positive"].ce == pytest.approx(-reduced["positive"].target_logprob)
    assert reduced["positive"].top1_accuracy == 1.0
    assert reduced["positive"].native_to_candidate_kl > 0
    assert reduced["off_target"].count > 0


def test_behavior_reduction_refuses_empty_confirmatory_support():
    rows, frequencies = _rows_for_matching()
    cells = contract.build_copy_cells(
        rows, frequencies, tuple(f"d{i}" for i in range(len(rows))),
    )
    empty = contract.CopyCells(
        all_positive=cells.all_positive,
        positive=torch.zeros_like(cells.positive),
        matched_negative=torch.zeros_like(cells.matched_negative),
        off_target=(torch.zeros_like(cells.off_target).index_fill_(1, torch.arange(64, 256), True)
                    & ~cells.all_positive),
        pair_indices=(), unmatched_positive_count=int(cells.all_positive.sum()),
        negative_candidate_count=cells.negative_candidate_count,
        eligible_stratum_count=cells.eligible_stratum_count,
        excluded_low_document_stratum_count=cells.excluded_low_document_stratum_count,
    )
    logits = torch.zeros(len(rows), contract.MODEL_WIDTH, int(rows.max()) + 2)
    with pytest.raises(ValueError, match="positive support"):
        contract.reduce_behavior(logits, rows, empty)


def test_causal_contrast_uses_within_input_effect_then_specificity_subtraction():
    def reduction(count, ce):
        return contract.CellReduction(
            count=count, ce=ce, target_logprob=-ce, top1_accuracy=0.5,
            native_to_candidate_kl=None, support_sha256=f"support-{count}",
        )

    native = {
        "positive": reduction(20, 2.0),
        "matched_negative": reduction(20, 2.5),
        "off_target": reduction(100, 3.0),
    }
    ablated = {
        "positive": reduction(20, 2.8),
        "matched_negative": reduction(20, 2.7),
        "off_target": reduction(100, 3.1),
    }
    contrast = contract.causal_copy_contrast(native, ablated)
    assert contrast.positive_ce_effect == pytest.approx(0.8)
    assert contrast.matched_negative_ce_effect == pytest.approx(0.2)
    assert contrast.specificity_ce_effect == pytest.approx(0.6)

    wrong_support = dict(ablated)
    wrong_support["positive"] = contract.CellReduction(
        count=20, ce=2.8, target_logprob=-2.8, top1_accuracy=0.5,
        native_to_candidate_kl=None, support_sha256="different-support",
    )
    with pytest.raises(ValueError, match="unequal cell support"):
        contract.causal_copy_contrast(native, wrong_support)


def test_reduction_support_digest_binds_ordered_row_bytes_not_only_coordinates():
    rows, frequencies = _rows_for_matching()
    document_ids = tuple(f"support-{i}" for i in range(len(rows)))
    cells = contract.build_copy_cells(rows, frequencies, document_ids)
    vocab = int(rows.max()) + 3
    logits = torch.zeros(len(rows), contract.MODEL_WIDTH, vocab)
    native_reduction = contract.reduce_behavior(logits, rows, cells)
    changed_rows = rows.clone()
    changed_rows[0, 0] += 1
    changed_reduction = contract.reduce_behavior(logits, changed_rows, cells)
    assert native_reduction["positive"].count == changed_reduction["positive"].count
    assert (
        native_reduction["positive"].support_sha256
        != changed_reduction["positive"].support_sha256
    )
    with pytest.raises(ValueError, match="unequal cell support"):
        contract.causal_copy_contrast(native_reduction, changed_reduction)


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
