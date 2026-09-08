from types import SimpleNamespace

import numpy as np
import torch

import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as run


class _Block(torch.nn.Module):
    def forward(self, x, v1, x0):
        return x + 0.1 * x0, v1


def test_full_forward_calls_block_boundaries_and_scores_requested_tokens():
    width, vocab = 4, 7
    model = SimpleNamespace(
        config=SimpleNamespace(n_embd=width),
        transformer=SimpleNamespace(
            wte=torch.nn.Embedding(vocab, width),
            h=torch.nn.ModuleList([_Block(), _Block()]),
        ),
        lm_head=torch.nn.Linear(width, vocab, bias=False),
    )
    token_rows = ((1, 2), (3,))
    backend = SimpleNamespace(
        torch=torch,
        F=torch.nn.functional,
        model=model,
        _tensor_batch=lambda batch: (
            torch.tensor([[1, 2], [3, 0]], dtype=torch.long),
            tuple(len(row) for row in batch.token_rows),
        ),
    )
    batch = SimpleNamespace(token_rows=token_rows, answer_ids=(1, 2), foil_ids=(2, 1))
    calls = {"count": 0}
    handle = model.transformer.h[1].register_forward_pre_hook(
        lambda _module, _arguments: calls.__setitem__("count", calls["count"] + 1)
    )
    try:
        output = run.full_forward(backend, batch)
    finally:
        handle.remove()
    assert calls["count"] == 1
    assert len(output.answer_foil) == 2
    assert output.captured == {}


def test_postcue_rows_require_aligned_contiguous_a2_cue():
    base = SimpleNamespace(token_rows=((1, 2, 3, 8, 9),), semantic_positions=(4,))
    donor = SimpleNamespace(token_rows=((4, 5, 6, 8, 9),), semantic_positions=(4,))
    assert run.postcue_rows(base, donor) == ((3, 4),)


def test_effect_metrics_and_rescue_fraction_are_exact():
    reference = np.asarray([1.0, 2.0])
    half = .5 * reference
    full = reference.copy()
    report = run.metrics(half, reference)
    assert report["signed_recovery"] == .5
    assert report["cosine"] == 1.0
    assert run.rescue_fraction(half, full, reference) == 1.0
    assert run.rescue_fraction(1.1 * reference, full, reference) == 0.0


def test_unbound_runner_fails_closed_before_model_access(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "BINDING", tmp_path / "missing_binding.json")
    eligible, status, binding = run.eligibility()
    assert eligible is False
    assert status == "awaiting_binding"
    assert binding is None


def test_price_and_arm_inventory_are_exact():
    assert run.PRICE["model_forwards"] == 3 + 2 + len(run.ARM_SPECS) == 17
    assert run.PRICE["sequence_evaluations"] == 17 * 16
    assert run.PRICE["scored_token_positions"] == 2 * 17 * 16
    assert set(run.SITES) == {"A11", "M11", "A12_control", "M16_control"}
