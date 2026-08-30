import pytest
import torch

import bilin18_observed_model_facade as facade
from causal_response_tensor_v1_backend import (
    CircuitSpec,
    ObservedResponseCollector,
    leading_shared_direction,
)
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
    fit_collector = ObservedResponseCollector(
        model,
        rows,
        documents,
        specs,
        require_production=False,
    )
    before_hooks = sum(len(module._forward_hooks) for module in model.modules())
    fit_payload = fit_collector.fit_stage(torch.tensor([0, 1, 2, 3]))
    eval_collector = ObservedResponseCollector(
        model, rows, documents, specs, require_production=False
    )
    eval_payload = eval_collector._evaluate_stage_preimage(
        torch.tensor([4, 5, 6, 7]),
        direction_preimage=fit_payload["_direction_preimage"],
        fit_document_ids=fit_payload["fit_response"]["document_ids"],
    )
    after_hooks = sum(len(module._forward_hooks) for module in model.modules())
    assert before_hooks == after_hooks == 0
    assert fit_payload["phases"] == ["full", "residual"]
    assert fit_payload["source_tags"] == [spec.tag for spec in specs]
    assert fit_payload["fit_response"]["document_ids"].tolist() == [0, 1, 2, 3]
    assert eval_payload["eval_response"]["document_ids"].tolist() == [4, 5, 6, 7]
    assert fit_payload["fit_response"]["statistics"]["member_signed_sum"].shape == (
        2, 4, 4, 4,
    )
    assert eval_payload["eval_response"]["validation"]["valid_cells"] == 2 * 4 * 4 * 4
    # FIT: capture + native response + eight interventions. EVAL: native + eight.
    assert fit_payload["call_ledger"]["outer_forwards"] == 10
    assert fit_payload["call_ledger"]["attention_native_calls"] == 40
    assert eval_payload["call_ledger"]["outer_forwards"] == 9
    assert eval_payload["call_ledger"]["attention_native_calls"] == 36
    assert sum(fit_payload["call_ledger"]["projection_calls"].values()) == 8
    assert sum(eval_payload["call_ledger"]["projection_calls"].values()) == 8


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
    direction = torch.zeros(model.config.n_embd, dtype=torch.float32)
    direction[0] = 1
    directions = {
        phase: {spec.tag: direction.clone() for spec in specs}
        for phase in ("full", "residual")
    }
    try:
        collector._evaluate_stage_preimage(
            torch.tensor([4, 5, 6, 7]),
            direction_preimage=directions,
            fit_document_ids=torch.tensor([0, 1, 2, 3]),
        )
    except ValueError as error:
        assert "source documents overlap" in str(error)
    else:
        raise AssertionError("document leakage was accepted")


def test_backend_owns_inputs_rejects_coercion_and_is_one_use() -> None:
    torch.manual_seed(11)
    model = tiny_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.03)
    rows = torch.randint(0, 32, (8, 5), dtype=torch.int64)
    documents = torch.arange(8, dtype=torch.int64)
    member_one = list(range(0, 32, 4))
    member_two = list(range(1, 32, 4))
    slice_positions = [
        position for position in range(32) if position % 4 in (0, 1)
    ]
    specs = (
        _spec("a.one", "a1", member_one, slice_positions, 32),
        _spec("a.two", "a1", member_two, slice_positions, 32),
    )
    collector = ObservedResponseCollector(
        model, rows, documents, specs, require_production=False
    )
    rows.fill_(31)
    documents.fill_(99)
    specs[0].member_mask.zero_()
    assert not torch.all(collector.rows == 31)
    assert collector.row_document_ids.tolist() == list(range(8))
    assert collector.specs[0].member_mask.any()

    fit = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    evaluate = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
    collector.fit_stage(fit)
    with pytest.raises(RuntimeError, match="spent"):
        collector.fit_stage(fit)

    with pytest.raises(TypeError, match="int64"):
        ObservedResponseCollector(
            model, rows.int(), torch.arange(8), specs, require_production=False
        )


def test_backend_rejects_bad_roles_and_preexisting_hooks() -> None:
    model = tiny_model()
    rows = torch.randint(0, 32, (8, 5), dtype=torch.int64)
    documents = torch.arange(8, dtype=torch.int64)
    specs = (
        _spec(
            "a.one", "a1", list(range(0, 32, 4)),
            [position for position in range(32) if position % 4 in (0, 1)], 32,
        ),
        _spec(
            "a.two", "a1", list(range(1, 32, 4)),
            [position for position in range(32) if position % 4 in (0, 1)], 32,
        ),
    )
    handle = model.transformer.h[0].register_forward_pre_hook(
        lambda _module, _inputs: None
    )
    try:
        with pytest.raises(RuntimeError, match="hook"):
            ObservedResponseCollector(
                model, rows, documents, specs, require_production=False
            )
    finally:
        handle.remove()

    for bad, message in (
        (torch.tensor([0, 0, 1, 2]), "duplicate"),
        (torch.tensor([-1, 0, 1, 2]), "out-of-range"),
        (torch.tensor([0, 1, 2, 8]), "out-of-range"),
    ):
        collector = ObservedResponseCollector(
            model, rows, documents, specs, require_production=False
        )
        with pytest.raises(ValueError, match=message):
            collector.fit_stage(bad)


def test_shared_direction_rejects_tie_and_accepts_above_frozen_gap() -> None:
    with pytest.raises(RuntimeError, match="tied"):
        leading_shared_direction(torch.eye(2, dtype=torch.float64))
    matrix = torch.diag(torch.tensor([1.0 + 1.1e-6, 1.0], dtype=torch.float64))
    direction, spectrum = leading_shared_direction(matrix)
    assert direction.dtype == torch.float32
    assert direction.tolist() == [1.0, 0.0]
    assert float((spectrum[0] - spectrum[1]) / spectrum[0]) > 1e-6
