from __future__ import annotations

import copy

import pytest
import torch

import bilin18_observed_model_facade as facade
import jacclust.tt_model as TT


def tiny_model() -> TT.GPT:
    config = TT.GPTConfig(
        vocab_size=facade.LOGIT_VOCAB,
        n_layer=4,
        n_head=1,
        n_embd=8,
        bilinear=True,
        expansion_factor=1,
        squared_attn=True,
        bilinear_attn=True,
    )
    model = TT.GPT(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_config_separates_tokenizer_support_from_logit_width() -> None:
    assert facade.TOKENIZER_VOCAB == 50_257
    assert facade.LOGIT_VOCAB == 50_304
    assert facade.validate_config(facade.EXPECTED_CONFIG)["vocab_size"] == 50_304
    changed = dict(facade.EXPECTED_CONFIG, vocab_size=50_257)
    with pytest.raises(RuntimeError, match="config differs"):
        facade.validate_config(changed)


def test_token_ids_must_be_reachable_but_padded_logits_are_not_sliced() -> None:
    model = tiny_model()
    tokens = torch.tensor([[0, facade.TOKENIZER_VOCAB - 1]], dtype=torch.long)
    calls = []

    def native(event):
        calls.append(event)
        return event.block.mlp(event.state)

    logits = facade.forward_with_early_dispatch(
        model, tokens, native, require_production=False,
    )
    assert logits.shape == (1, 2, facade.LOGIT_VOCAB)
    assert [event.site for event in calls] == [0, 1, 2]
    assert all(event.block is model.transformer.h[event.site] for event in calls)
    assert [len(event.prior_writes) for event in calls] == [0, 1, 2]
    assert all(event.tokens is tokens for event in calls)
    torch.testing.assert_close(calls[1].prior_writes[0], calls[0].block.mlp(calls[0].state))

    invalid = tokens.clone()
    invalid[0, 1] = facade.TOKENIZER_VOCAB
    with pytest.raises(RuntimeError, match="outside"):
        facade.forward_with_early_dispatch(
            model, invalid, native, require_production=False,
        )


def test_dispatcher_must_return_exact_live_write_contract() -> None:
    model = tiny_model()
    tokens = torch.zeros((1, 2), dtype=torch.long)

    def wrong_shape(event):
        return event.state[..., :-1]

    with pytest.raises(RuntimeError, match="dispatcher write is malformed"):
        facade.forward_with_early_dispatch(
            model, tokens, wrong_shape, require_production=False,
        )


def test_full_dispatch_surface_observes_all_sites_and_rejects_bad_attention() -> None:
    model = tiny_model()
    tokens = torch.zeros((1, 2), dtype=torch.long)
    attention_sites = []
    mlp_sites = []

    def attention(event):
        attention_sites.append(event.site)
        return event.block.attn(event.state, event.first_value)

    def mlp(event):
        mlp_sites.append((event.site, len(event.prior_writes)))
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False,
    )
    assert logits.shape[-1] == facade.LOGIT_VOCAB
    assert attention_sites == [0, 1, 2, 3]
    assert mlp_sites == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def bad_attention(event):
        return event.state

    with pytest.raises(RuntimeError, match="result is malformed"):
        facade.forward_with_dispatch(
            model, tokens, bad_attention, mlp, require_production=False,
        )


def test_production_validator_rejects_training_and_wrong_topology() -> None:
    model = tiny_model()
    with pytest.raises(RuntimeError, match="topology"):
        facade.validate_production_model(model)

    class ExactTypeSubstitute(TT.GPT):
        pass

    substitute = ExactTypeSubstitute(copy.copy(model.config))
    with pytest.raises(RuntimeError, match="exact TT.GPT"):
        facade.validate_production_model(substitute)


def test_snapshot_revision_fails_closed_before_file_access(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="revision"):
        facade.validate_snapshot(tmp_path, verify_weights_sha256=False)
