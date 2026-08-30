from __future__ import annotations

import copy

import pytest
import tiktoken

import newline_l12h6_token_registry_v1 as subject


def test_pinned_gpt2_registry_is_exact_and_disjoint() -> None:
    registry = subject.build_registry(tiktoken.get_encoding("gpt2"))
    assert tuple(registry) == subject.CLASS_ORDER
    assert {name: len(values) for name, values in registry.items()} == subject.EXPECTED_COUNTS
    assert registry["newline"] == (198, 628, 44320)
    assert not any(
        set(registry[left]) & set(registry[right])
        for index, left in enumerate(subject.CLASS_ORDER)
        for right in subject.CLASS_ORDER[index + 1:]
    )


def test_registry_rejects_value_order_and_cross_class_mutation() -> None:
    registry = subject.build_registry(tiktoken.get_encoding("gpt2"))
    wrong = {name: list(values) for name, values in registry.items()}
    wrong["newline"] = list(reversed(wrong["newline"]))
    with pytest.raises(RuntimeError, match="malformed"):
        subject.validate_registry(wrong)
    wrong = copy.deepcopy({name: list(values) for name, values in registry.items()})
    wrong["punctuation"].append(wrong["newline"][0]); wrong["punctuation"].sort()
    with pytest.raises(RuntimeError, match="overlap"):
        subject.validate_registry(wrong)


def test_tokenizer_fingerprint_is_frozen() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    assert subject.encoding_fingerprint(encoding) == subject.ENCODING_SHA256
