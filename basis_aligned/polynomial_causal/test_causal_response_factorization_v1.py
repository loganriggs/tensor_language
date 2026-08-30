import torch

from causal_response_factorization_v1 import (
    fit_shared_private_program,
    infer_document_codes,
    make_program_from_factors,
    predict_from_codes,
    prospective_anchor_mask,
    prospective_document_split,
    signed_response_from_sums,
)


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
