import pytest

import cached_response_program_eval as subject


def backend():
    config = type("Config", (), {"n_head": 2, "n_embd": 4})()
    blocks = [type("Block", (), {})() for _ in range(3)]
    return type("Backend", (), {"model": type("Model", (), {
        "config": config, "transformer": type("Transformer", (), {"h": blocks})()
    })()})()


def test_rejects_noncausal_order_and_duplicate_module():
    with pytest.raises(subject.CachedResponseProgramError):
        subject.intervene_cached_response_program(
            backend(), None, {}, {}, (("mlp", 2, ()), ("attn", 1, (0,))))
    with pytest.raises(subject.CachedResponseProgramError):
        subject.intervene_cached_response_program(
            backend(), None, {}, {}, (("attn", 1, (0,)), ("attn", 1, (1,))))


def test_rejects_bad_typed_components_before_hooking():
    with pytest.raises(subject.CachedResponseProgramError):
        subject.intervene_cached_response_program(
            backend(), None, {}, {}, (("mlp", 1, (0,)),))
    with pytest.raises(subject.CachedResponseProgramError):
        subject.intervene_cached_response_program(
            backend(), None, {}, {}, (("attn", 1, (2,)),))
