from types import SimpleNamespace

import torch

import attention15_head_module_mediation_executor as executor


class Parent:
    @staticmethod
    def manual_forward(backend, batch, counters, **kwargs):
        counters["calls"] += 1
        source = torch.full((1, 2, 6), float(bool(kwargs.get("raw_by_site"))))
        return {"value": backend.model.transformer.h[15].attn.c_proj(source)}


def backend():
    blocks = [SimpleNamespace(attn=SimpleNamespace(c_proj=torch.nn.Identity())) for _ in range(16)]
    return SimpleNamespace(
        torch=torch,
        model=SimpleNamespace(config=SimpleNamespace(n_head=3),
                              transformer=SimpleNamespace(h=blocks)),
    )


def test_capture_then_singleton_reset_rescue_executes_four_cells():
    state, counters = backend(), {"calls": 0}
    context = {"base_batch": SimpleNamespace(semantic_positions=(1,))}
    out0, cap0 = executor.capture_live_response(Parent, state, context, counters, {})
    out1, cap1 = executor.capture_live_response(Parent, state, context, counters, {"x": 1})
    cells = executor.execute_mediator_cells(
        Parent, state, context, counters, {"x": 1},
        {"00": cap0, "11": cap1, "00_output": out0, "11_output": out1}, (1,))
    assert counters["calls"] == 4
    assert torch.equal(cells["00"]["value"], torch.zeros(1, 2, 6))
    assert torch.equal(cells["11"]["value"], torch.ones(1, 2, 6))
    rescued = cells["01"]["value"].reshape(1, 2, 3, 2)
    reset = cells["10"]["value"].reshape(1, 2, 3, 2)
    assert torch.equal(rescued[:, :, 1], torch.ones(1, 2, 2))
    assert torch.equal(rescued[:, :, (0, 2)], torch.zeros(1, 2, 2, 2))
    assert torch.equal(reset[:, :, 1], torch.zeros(1, 2, 2))
    assert torch.equal(reset[:, :, (0, 2)], torch.ones(1, 2, 2, 2))
