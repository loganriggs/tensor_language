from __future__ import annotations

import copy

import pytest
import torch

import early_mlp_suffix_transport_v1_consumer_norms as consumer
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_runtime as runtime


def _sha(payload) -> str:
    return runtime.logical_identity_sha256(payload)


class _Projection(torch.nn.Module):
    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Expansion keeps the synthetic production-shaped output inexpensive until
        # the capture computes the real 1152-wide norm.
        return value.expand(-1, -1, consumer.MODEL_WIDTH) * self.scale + self.bias


class _Attention(torch.nn.Module):
    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.c_proj = _Projection(scale, bias)


class _Block(torch.nn.Module):
    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.attn = _Attention(scale, bias)


class _Transformer(torch.nn.Module):
    def __init__(self, bias: float) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([
            _Block(float(layer + 1), bias)
            for layer in range(consumer.MODEL_LAYER_COUNT)
        ])


class _Model(torch.nn.Module):
    def __init__(self, bias: float = 0.25) -> None:
        super().__init__()
        self.transformer = _Transformer(bias)

    def forward(
        self, value: torch.Tensor, *, skip: int | None = None,
        duplicate: int | None = None,
    ) -> None:
        for layer, block in enumerate(self.transformer.h):
            if layer == skip:
                continue
            block.attn.c_proj(value)
            if layer == duplicate:
                block.attn.c_proj(value)


def _value(scored: float, unscored: float = 0.0) -> torch.Tensor:
    value = torch.full(
        (consumer.BATCH_SIZE, consumer.SEQUENCE_LENGTH, 1),
        scored, dtype=torch.float32,
    )
    value[:, :consumer.SCORE_START] = unscored
    return value


def _capture(
    model: _Model, action: final_capability.FinalAction, value: torch.Tensor,
    *, batch: int = 0, state: dict[str, str] | None = None,
    forward=None,
):
    state = state if state is not None else {
        "action": _sha([action.key, batch]), "model": _sha("model"),
    }
    capture = consumer.AttentionConsumerOutputCapture(
        model=model, action=action, batch_ordinal=batch,
        common_support_sha256=_sha("support"),
        expected_action_identity_sha256=state["action"],
        action_identity_reader=lambda: state["action"],
        expected_model_identity_sha256=state["model"],
        model_identity_reader=lambda: state["model"],
    )
    with capture:
        (forward or model)(value)
    return capture.finish()


def test_exact_output_bias_scored_slice_and_paired_ratio() -> None:
    model = _Model(bias=0.25)
    action = final_capability.FinalAction("qq", "N")
    native = final_capability.FinalAction("o_o", "N")
    action_capture = _capture(model, action, _value(2.0, unscored=10000.0))
    native_capture = _capture(model, native, _value(1.0, unscored=-10000.0))
    result = consumer.reduce_consumer_norm_batch(
        action_capture=action_capture, native_capture=native_capture,
    )

    assert len(result.row_ratios) == 18
    for layer, ratios in enumerate(result.row_ratios):
        scale = float(layer + 1)
        expected = (2.0 * scale + 0.25) / (scale + 0.25)
        assert torch.allclose(
            ratios, torch.full((4,), expected, dtype=torch.float64), atol=2e-6,
        )
    assert result.receipt.metric_role == "integrity_only"
    assert result.receipt.authorized_for_selection is False
    assert all(not block.attn.c_proj._forward_hooks for block in model.transformer.h)
    assert not hasattr(result, "writes") and not hasattr(result.receipt, "model")
    second_action = _capture(model, action, _value(3.0))
    assert consumer.reduce_consumer_norm_batch(
        action_capture=second_action, native_capture=native_capture,
    ).receipt.native_capture_sha256 == native_capture.receipt.sha256
    with pytest.raises(RuntimeError, match="replayed"):
        consumer.reduce_consumer_norm_batch(
            action_capture=action_capture, native_capture=native_capture,
        )


@pytest.mark.parametrize("mode", ["skip", "duplicate"])
def test_missing_or_duplicate_output_capture_poisoned_and_hooks_removed(mode: str) -> None:
    model = _Model()
    action = final_capability.FinalAction("qq", "E")
    state = {"action": _sha("action"), "model": _sha("model")}
    capture = consumer.AttentionConsumerOutputCapture(
        model=model, action=action, batch_ordinal=0,
        common_support_sha256=_sha("support"),
        expected_action_identity_sha256=state["action"],
        action_identity_reader=lambda: state["action"],
        expected_model_identity_sha256=state["model"],
        model_identity_reader=lambda: state["model"],
    )
    with pytest.raises(RuntimeError, match="duplicated|close exactly"):
        with capture:
            model(
                _value(1.0),
                skip=(17 if mode == "skip" else None),
                duplicate=(7 if mode == "duplicate" else None),
            )
    assert all(not block.attn.c_proj._forward_hooks for block in model.transformer.h)
    with pytest.raises(RuntimeError, match="incomplete or failed"):
        capture.finish()


@pytest.mark.parametrize("identity", ["action", "model"])
def test_action_or_model_identity_drift_fails_after_forward(identity: str) -> None:
    model = _Model()
    action = final_capability.FinalAction("qq", "N")
    state = {"action": _sha("action"), "model": _sha("model")}

    def forward(value):
        model(value)
        state[identity] = _sha([identity, "drift"])

    with pytest.raises(RuntimeError, match="did not close exactly"):
        _capture(model, action, _value(1.0), state=state, forward=forward)
    assert all(not block.attn.c_proj._forward_hooks for block in model.transformer.h)


def test_zero_native_denominator_and_wrong_background_fail_closed() -> None:
    model = _Model(bias=0.0)
    action = final_capability.FinalAction("qq", "N")
    wrong_native = final_capability.FinalAction("o_o", "E")
    action_capture = _capture(model, action, _value(1.0))
    zero_native = _capture(
        model, final_capability.FinalAction("o_o", "N"), _value(0.0),
    )
    with pytest.raises(RuntimeError, match="zero or numerically empty"):
        consumer.reduce_consumer_norm_batch(
            action_capture=action_capture, native_capture=zero_native,
        )

    action_capture = _capture(model, action, _value(1.0))
    native_capture = _capture(model, wrong_native, _value(1.0))
    with pytest.raises(RuntimeError, match="identical support"):
        consumer.reduce_consumer_norm_batch(
            action_capture=action_capture, native_capture=native_capture,
        )


def test_native_baseline_must_be_one_and_self_denominator_is_exact() -> None:
    model = _Model()
    native = final_capability.FinalAction("o_o", "E")
    changed = _capture(model, native, _value(2.0))
    reference = _capture(model, native, _value(1.0))
    with pytest.raises(RuntimeError, match="not one"):
        consumer.reduce_consumer_norm_batch(
            action_capture=changed, native_capture=reference,
        )

    self_capture = _capture(model, native, _value(1.0))
    result = consumer.reduce_consumer_norm_batch(
        action_capture=self_capture, native_capture=self_capture,
    )
    assert all(torch.equal(value, torch.ones(4, dtype=torch.float64))
               for value in result.row_ratios)


def _batch_result(
    action: final_capability.FinalAction, ordinal: int,
) -> consumer.ConsumerNormBatchResult:
    ratios = tuple(
        torch.full((4,), layer + ordinal / 100.0, dtype=torch.float64)
        for layer in range(18)
    )
    receipt = consumer.ConsumerNormBatchReceipt(
        action_key=action.key, background=action.background, batch_ordinal=ordinal,
        action_capture_sha256=_sha(["action", action.key, ordinal]),
        native_capture_sha256=_sha(["native", action.background, ordinal]),
        ratio_sha256s=tuple(runtime.tensor_identity_sha256(value) for value in ratios),
    )
    return consumer.ConsumerNormBatchResult(
        action=action, batch_ordinal=ordinal, row_ratios=ratios, receipt=receipt,
    )


def test_48_batch_action_aggregation_emits_exact_row_reductions() -> None:
    action = final_capability.FinalAction("qq", "N")
    batches = tuple(_batch_result(action, ordinal) for ordinal in range(48))
    result = consumer.aggregate_consumer_norm_action(action, batches)
    assert len(result.reductions) == 18
    for layer, reduction in enumerate(result.reductions):
        expected = torch.cat(tuple(batch.row_ratios[layer] for batch in batches))
        assert torch.equal(reduction.row_sum, expected)
        assert torch.equal(reduction.row_count, torch.ones(192, dtype=torch.long))
    with pytest.raises(RuntimeError, match="skipped, reordered, or mixed"):
        consumer.aggregate_consumer_norm_action(
            action, (batches[1], batches[0], *batches[2:]),
        )
    with pytest.raises(RuntimeError, match="skipped, reordered, or mixed"):
        consumer.aggregate_consumer_norm_action(
            action, (*batches[:-1], batches[-2]),
        )


def test_private_capture_is_noncopiable_and_nonserializable() -> None:
    capture = _capture(
        _Model(), final_capability.FinalAction("qq", "N"), _value(1.0),
    )
    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.copy(capture)
    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.deepcopy(capture)
    with pytest.raises(RuntimeError, match="cannot be serialized"):
        capture.__reduce__()
