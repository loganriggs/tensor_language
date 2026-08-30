from __future__ import annotations

import tiktoken
import torch

import ordered_successor_digit_lexicon_v2 as digits
from ordered_successor_masks_v1 import build_ordered_successor_masks
import ordered_successor_tensor_discovery_v1 as v1
import ordered_successor_tensor_select_registry_v2 as protocol


def test_exact_gpt2_digit_forms_ids_and_hashes() -> None:
    lexicon, encoding = digits.load_pinned_lexicon()
    assert encoding.name == "gpt2"
    assert digits.registry_sha256() == digits.REGISTRY_SHA256 == (
        "e59c912c542d4477a222487086fcdfe02e2bee1d3b1176bb87bc137e3627cff3"
    )
    assert lexicon.items == digits.DIGIT_TOKEN_IDS
    assert digits.DIGIT_SURFACE_FORMS == tuple(
        (str(value), f" {value}") for value in range(10)
    )
    for forms, token_ids in zip(digits.DIGIT_SURFACE_FORMS, digits.DIGIT_TOKEN_IDS):
        assert tuple(encoding.encode_ordinary(form)[0] for form in forms) == token_ids
        assert all(len(encoding.encode_ordinary(form)) == 1 for form in forms)
        assert tuple(encoding.decode([token_id]) for token_id in token_ids) == forms


def test_changed_single_item_tokenization_fails_closed(monkeypatch) -> None:
    real = tiktoken.get_encoding("gpt2")

    class ChangedEncoding:
        name = "gpt2"
        n_vocab = 50_257

        @staticmethod
        def encode_ordinary(value):
            if value == "9":
                return [digits.DIGIT_TOKEN_IDS[9][0] + 1]
            return real.encode_ordinary(value)

        @staticmethod
        def decode(values):
            return real.decode(values)

    monkeypatch.setattr(digits, "encoding_fingerprint", lambda _encoding: digits.ENCODING_SHA256)
    try:
        digits.validate_encoding(ChangedEncoding())
    except RuntimeError as error:
        assert "token IDs" in str(error) or "one exact" in str(error)
    else:
        raise AssertionError("changed digit tokenization was accepted")


def test_decimal_order_is_noncyclic() -> None:
    lexicon, _ = digits.load_pinned_lexicon()
    rows = torch.full((1, 257), 1000, dtype=torch.long)
    prediction = 100
    rows[0, prediction - 5] = digits.DIGIT_TOKEN_IDS[9][0]
    rows[0, prediction + 1] = digits.DIGIT_TOKEN_IDS[0][0]
    masks = build_ordered_successor_masks(rows, lexicon, window=128, first_prediction=64)
    assert not bool(masks.eligible_target[0, prediction])


def test_v2_protocol_registry_is_exact_15_arm_projection() -> None:
    protocol.validate_registry()
    assert protocol.registry_sha256() == protocol.REGISTRY_SHA256 == (
        "38e9775c8a30e9ed9ac1278ca3940a0b46699527e958b24d145d0978932be7d5"
    )
    assert protocol.ARM_NAMES == v1.ARM_NAMES[:-2]
    assert len(protocol.ARM_NAMES) == 15
    assert protocol.OMITTED_V1_DIAGNOSTICS == (v1.CURRENT_ONLY, v1.V1_ONLY)
    assert protocol.PROMOTIVE_ARMS == v1.PROMOTIVE_ARMS
    payload = protocol.registry_payload()
    assert payload["statistical_procedure"] == {
        "source": "ordered_successor_tensor_select_statistics_v1.py",
        "bootstrap_draws": 20_000,
        "bootstrap_seed": 2_026_083_013,
        "order_index": 18_999,
        "powered_cells": ["positive_clean", "wrong_source_clean", "no_source_clean"],
    }
