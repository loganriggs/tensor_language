import torch

import evaluate_mlp0_native_down_hierarchy_v1 as evaluate


def test_chunk_split_covers_both_256_prediction_windows_and_preserves_document():
    rows = torch.arange(2 * 513).view(2, 513)
    records = [{"source_document_ordinal": 7}, {"source_document_ordinal": 9}]
    windows, ordinals = evaluate.split_chunks(rows, records, 0, 2)
    assert windows.shape == (4, 257)
    assert torch.equal(windows[0], rows[0, :257])
    assert torch.equal(windows[2], rows[0, 256:513])
    assert ordinals.tolist() == [7, 9, 7, 9]


def test_program_proxy_uses_token_codebook_and_zero_sentinel():
    program = {
        "intercept": torch.tensor([1., 2.]),
        "left": torch.zeros(2, 1), "right": torch.zeros(1, 3),
        "centroids": torch.tensor([[4., 5.], [6., 7.]]),
        "assignments": torch.tensor([0, 1, 2]),
    }
    proxy = evaluate.ProgramDown(program, "test")
    evaluate.STATE["idx"] = torch.tensor([[0, 1, 2]])
    observed = proxy(torch.zeros(1, 3, 3))
    assert torch.equal(observed[0, 0], torch.tensor([5., 7.]))
    assert torch.equal(observed[0, 1], torch.tensor([7., 9.]))
    assert torch.equal(observed[0, 2], torch.tensor([1., 2.]))
    assert proxy.calls == 1


def test_document_cell_aggregation_combines_two_windows_of_same_document():
    ledger = {"sums": torch.zeros(3, 16, dtype=torch.float64),
              "counts": torch.zeros(3, 16, dtype=torch.float64)}
    ordinals = torch.tensor([1, 1])
    cell = torch.tensor([[0, 0, 1], [0, 1, 1]])
    valid = torch.ones_like(cell, dtype=torch.bool)
    effects = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    evaluate.add_document_cells(ledger, ordinals, cell, valid, effects)
    assert ledger["counts"][1, :2].tolist() == [3, 3]
    assert ledger["sums"][1, :2].tolist() == [7, 14]
