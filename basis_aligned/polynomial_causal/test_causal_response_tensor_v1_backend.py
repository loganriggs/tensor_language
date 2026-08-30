import torch

import bilin18_observed_model_facade as facade
from causal_response_tensor_v1_backend import CircuitSpec, ObservedResponseCollector
from test_bilin18_observed_model_facade import tiny_model


def _spec(tag: str, component: str, members: list[int], slice_positions: list[int], n: int):
    member = torch.zeros(n, dtype=torch.bool)
    member[members] = True
    slice_mask = torch.zeros(n, dtype=torch.bool)
    slice_mask[slice_positions] = True
    return CircuitSpec(tag, component, member, slice_mask)


def test_typed_backend_collects_complete_signed_grid_without_hooks() -> None:
    torch.manual_seed(7)
    model = tiny_model()
    # The shared facade fixture is intentionally zero-initialized.  Give this
    # collector test non-degenerate writes so FIT directions and their
    # component-wise residuals are actually exercised.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.03)
    positions = 4
    rows = torch.randint(0, 32, (8, positions + 1), dtype=torch.long)
    documents = torch.arange(8, dtype=torch.int64)
    grid = rows.shape[0] * positions
    a_slice = [position for position in range(grid) if position % positions in (0, 1)]
    m_slice = [position for position in range(grid) if position % positions in (2, 3)]
    specs = (
        _spec("a.one", "a1", list(range(0, grid, positions)), a_slice, grid),
        _spec("a.two", "a1", list(range(1, grid, positions)), a_slice, grid),
        _spec("m.one", "m2", list(range(2, grid, positions)), m_slice, grid),
        _spec("m.two", "m2", list(range(3, grid, positions)), m_slice, grid),
    )
    collector = ObservedResponseCollector(
        model,
        rows,
        documents,
        specs,
        require_production=False,
    )
    before_hooks = sum(len(module._forward_hooks) for module in model.modules())
    payload = collector.collect(torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7]))
    after_hooks = sum(len(module._forward_hooks) for module in model.modules())
    assert before_hooks == after_hooks == 0
    assert payload["phases"] == ["full", "residual"]
    assert payload["source_tags"] == [spec.tag for spec in specs]
    assert payload["eval_document_ids"].tolist() == [4, 5, 6, 7]
    assert payload["statistics"]["member_signed_sum"].shape == (2, 4, 4, 4)
    assert payload["validation"]["valid_cells"] == 2 * 4 * 4 * 4
    # One FIT batch + one native EVAL batch + eight intervention EVAL batches.
    assert payload["call_ledger"]["outer_forwards"] == 10
    assert payload["call_ledger"]["attention_native_calls"] == 40
    assert payload["call_ledger"]["mlp_native_calls"] == 40
    assert sum(payload["call_ledger"]["projection_calls"].values()) == 8


def test_backend_rejects_document_leakage_even_when_rows_differ() -> None:
    model = tiny_model()
    rows = torch.randint(0, 32, (8, 5), dtype=torch.long)
    documents = torch.tensor([0, 1, 2, 3, 0, 4, 5, 6])
    grid = 8 * 4
    specs = (
        _spec("a.one", "a1", [0, 4], list(range(16)), grid),
        _spec("a.two", "a1", [1, 5], list(range(16)), grid),
    )
    collector = ObservedResponseCollector(
        model, rows, documents, specs, require_production=False
    )
    try:
        collector.collect(torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7]))
    except ValueError as error:
        assert "source documents overlap" in str(error)
    else:
        raise AssertionError("document leakage was accepted")
