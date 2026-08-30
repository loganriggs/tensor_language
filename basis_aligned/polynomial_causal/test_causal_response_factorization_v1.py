import torch
import json
from pathlib import Path

from causal_response_factorization_v1 import (
    block_d_optimal_anchor_mask,
    fit_shared_private_program,
    infer_document_codes,
    make_program_from_factors,
    predict_from_codes,
    prospective_anchor_mask,
    prospective_anchor_arm_mask,
    prospective_document_split,
    score_program_on_validation,
    signed_response_from_sums,
)
from causal_response_block_design_planted_toy_v1 import build_receipt


def test_signed_response_uses_member_minus_off_and_masks_unsupported():
    shape = (2, 3, 2, 4)
    member_count = torch.tensor([[2, 0, 4, 1], [1, 2, 1, 3]], dtype=torch.int64)
    off_count = torch.full((2, 4), 2, dtype=torch.int64)
    member = torch.full(shape, 6.0, dtype=torch.float64)
    member[:, :, 0, 1] = 0
    off = torch.full(shape, 2.0, dtype=torch.float64)
    statistics = {
        "member_signed_sum": member.contiguous(),
        "member_abs_sum": member.abs().contiguous(),
        "off_signed_sum": off.contiguous(),
        "off_abs_sum": off.abs().contiguous(),
    }
    response, valid = signed_response_from_sums(statistics, member_count, off_count)
    assert response[0, 0, 0, 0] == 2.0  # 6/2 - 2/2
    assert not valid[0, 0, 0, 1]
    assert response[0, 0, 0, 1] == 0
    assert response[1, 2, 1, 3] == 1.0  # 6/3 - 2/2


def test_prospective_splits_are_deterministic_and_outcome_blind():
    ids = torch.tensor([19, 2, 91, 7, 33, 48], dtype=torch.int64)
    train1, validation1 = prospective_document_split(ids, train_documents=4)
    train2, validation2 = prospective_document_split(ids, train_documents=4)
    assert torch.equal(train1, train2)
    assert torch.equal(validation1, validation2)
    assert set(train1.tolist()).isdisjoint(validation1.tolist())
    assert sorted(train1.tolist() + validation1.tolist()) == list(range(6))
    anchors = prospective_anchor_mask(2, 3, 4, anchors=5)
    assert anchors.dtype == torch.bool and int(anchors.sum()) == 5
    block_anchors, selected = prospective_anchor_arm_mask(2, 3, 4, arms=2)
    assert len(selected) == 2 and int(block_anchors.sum()) == 8
    assert bool((block_anchors.reshape(6, 4).all(1)
                 | (~block_anchors.reshape(6, 4)).all(1)).all())


def test_shared_private_program_has_exact_tensor_basis_and_literal_price():
    groups = torch.tensor([0, 0, 1], dtype=torch.int64)
    global_factors = (
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[1.0], [3.0], [5.0]]),
        torch.tensor([[2.0], [7.0]]),
    )
    private = (
        (torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [4.0]])),
        (torch.tensor([[2.0], [3.0]]), torch.tensor([[5.0]]), torch.tensor([[1.0], [2.0]])),
    )
    program = make_program_from_factors(global_factors, private, groups)
    basis = program.basis().reshape(2, 3, 2, 3)
    # Source zero receives global plus group-zero private, never group-one private.
    assert torch.equal(basis[0, 0, 0], torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64))
    # Source two receives global plus group-one private, never group-zero private.
    assert torch.equal(basis[0, 2, 0], torch.tensor([10.0, 0.0, 10.0], dtype=torch.float64))
    assert program.code_dimension == 3
    # global: 2+3+2; private: (2+2+2)+(2+1+2)
    assert program.persistent_values == 18


def test_toy_recovers_unseen_cells_from_anchor_responses():
    generator = torch.Generator().manual_seed(17)
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    global_factors = tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
                           for shape in ((2, 2), (4, 2), (3, 2)))
    private = tuple(
        tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
              for shape in ((2, 1), (2, 1), (3, 1)))
        for _ in range(2)
    )
    program = make_program_from_factors(global_factors, private, groups)
    basis = program.basis()
    true_codes = torch.randn((7, program.code_dimension), generator=generator,
                             dtype=torch.float64)
    responses = basis @ true_codes.T
    valid = torch.ones_like(responses, dtype=torch.bool)
    # Use a deterministic well-conditioned anchor panel for this algebraic known answer.
    anchor_indices = []
    rank = program.code_dimension
    for index in range(basis.shape[0]):
        candidate = anchor_indices + [index]
        if torch.linalg.matrix_rank(basis[candidate]) > len(anchor_indices):
            anchor_indices.append(index)
        if len(anchor_indices) == rank:
            break
    anchors = torch.zeros(basis.shape[0], dtype=torch.bool)
    anchors[anchor_indices] = True
    codes, supported = infer_document_codes(
        basis, responses, valid, anchors, ridge=0.0, minimum_anchor_ratio=1
    )
    predicted = predict_from_codes(basis, codes)
    assert bool(supported.all())
    assert torch.allclose(codes, true_codes, atol=1e-9, rtol=1e-9)
    assert torch.allclose(predicted[~anchors], responses[~anchors], atol=1e-9, rtol=1e-9)


def test_missing_anchor_support_fails_closed_per_document():
    basis = torch.eye(3, dtype=torch.float64)
    response = torch.ones((3, 2), dtype=torch.float64)
    valid = torch.ones((3, 2), dtype=torch.bool)
    valid[1:, 1] = False
    anchors = torch.ones(3, dtype=torch.bool)
    codes, supported = infer_document_codes(
        basis, response, valid, anchors, minimum_anchor_ratio=1
    )
    assert supported.tolist() == [True, False]
    assert torch.equal(codes[1], torch.zeros(3, dtype=torch.float64))


def test_optimizer_recovers_planted_shared_plus_private_toy_and_replays_canonical_form():
    generator = torch.Generator().manual_seed(88)
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    planted = make_program_from_factors(
        tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
              for shape in ((2, 1), (4, 1), (3, 1))),
        tuple(
            tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
                  for shape in ((2, 1), (2, 1), (3, 1)))
            for _ in range(2)
        ),
        groups,
    )
    codes = torch.randn((12, planted.code_dimension), generator=generator,
                        dtype=torch.float64)
    response = predict_from_codes(planted.basis(), codes).reshape(2, 4, 3, 12)
    valid = torch.ones_like(response, dtype=torch.bool)
    fitted = fit_shared_private_program(
        response, valid, groups, global_rank=1, private_rank=1,
        seed=2026083001, steps=2_000, learning_rate=0.04,
    )
    replay = predict_from_codes(
        fitted.program.basis(), fitted.document_codes
    ).reshape_as(response)
    assert fitted.improvement_fraction > 0.9999
    assert fitted.final_mse < 1e-8
    assert torch.allclose(replay, response, atol=5e-4, rtol=5e-4)


def test_validation_scorer_separates_unconditional_from_calibrated_and_reports_blocks():
    generator = torch.Generator().manual_seed(123)
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    program = make_program_from_factors(
        tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
              for shape in ((2, 1), (4, 1), (4, 1))),
        tuple(
            tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
                  for shape in ((2, 1), (2, 1), (4, 1)))
            for _ in range(2)
        ),
        groups,
    )
    train_codes = torch.randn((20, 3), generator=generator, dtype=torch.float64)
    validation_codes = torch.randn((8, 3), generator=generator, dtype=torch.float64)
    truth = predict_from_codes(program.basis(), validation_codes).reshape(2, 4, 4, 8)
    valid = torch.ones_like(truth, dtype=torch.bool)
    # Select two complete physical arms, giving eight scalar cells for a 3-D code.
    basis = program.basis()
    anchors, selected, path = block_d_optimal_anchor_mask(
        basis, shape=(2, 4, 4), arms=2
    )
    assert len(selected) == 2 and path[1] >= path[0]
    report = score_program_on_validation(
        program, train_codes, truth, valid, anchors,
        training_rms=float(truth.square().mean().sqrt()),
    )
    assert report["support_gate_passes"] is True
    assert report["supported_documents"] == 8
    assert report["calibrated"]["pooled"]["mse"] < 1e-10
    assert report["unconditional"]["pooled"]["mse"] > 1e-4
    assert len(report["calibrated"]["owner_pairs"]) == 4
    assert report["calibrated"]["uses_pooled_only"] is False
    assert report["claim_boundary"]["calibrated_is_zero_shot_ood"] is False
    assert report["anchor_source_arms"] == 2


def test_block_d_optimal_selection_is_invariant_to_invertible_code_gauge():
    generator = torch.Generator().manual_seed(991)
    basis = torch.randn((2 * 7 * 5, 4), generator=generator, dtype=torch.float64)
    gauge = torch.randn((4, 4), generator=generator, dtype=torch.float64)
    gauge = gauge + 3.0 * torch.eye(4, dtype=torch.float64)
    mask1, arms1, path1 = block_d_optimal_anchor_mask(
        basis, shape=(2, 7, 5), arms=4
    )
    mask2, arms2, path2 = block_d_optimal_anchor_mask(
        basis @ gauge, shape=(2, 7, 5), arms=4
    )
    assert arms1 == arms2
    assert torch.equal(mask1, mask2)
    assert torch.allclose(torch.tensor(path1), torch.tensor(path2), atol=1e-10, rtol=0)
    assert int(mask1.sum()) == 4 * 5


def test_validation_scorer_rejects_scattered_cell_mask_as_false_price():
    groups = torch.tensor([0, 1], dtype=torch.int64)
    program = make_program_from_factors(
        (torch.ones((1, 1)), torch.ones((2, 1)), torch.ones((2, 1))),
        (
            (torch.empty((1, 0)), torch.empty((1, 0)), torch.empty((2, 0))),
            (torch.empty((1, 0)), torch.empty((1, 0)), torch.empty((2, 0))),
        ),
        groups,
    )
    codes = torch.ones((3, 1), dtype=torch.float64)
    truth = predict_from_codes(program.basis(), codes).reshape(1, 2, 2, 3)
    scattered = torch.tensor([True, False, True, False])
    try:
        score_program_on_validation(
            program, codes, truth, torch.ones_like(truth, dtype=torch.bool), scattered,
            training_rms=1.0, minimum_anchor_ratio=1,
        )
    except ValueError as error:
        assert "complete physical target blocks" in str(error)
    else:
        raise AssertionError("scattered cell pricing must fail closed")


def test_planted_block_design_receipt_replays_exactly():
    path = Path(__file__).with_name("causal_response_block_design_planted_toy_v1.json")
    assert build_receipt() == json.loads(path.read_text())
