"""Prospective GPT-2 token classes for the newline L12H6 canary.

The module is pure: callers supply the pinned ``tiktoken`` encoding.  Exact counts
and ordered-ID hashes make a package/tokenizer drift fail before rows are selected.
The priority order makes the four score-only classes disjoint.
"""

from __future__ import annotations

import hashlib
import json
import string
from typing import Any, Mapping


ENCODING_NAME = "gpt2"
ENCODING_SHA256 = "0be287937901b1baae837369293dd6f63da1bece9609006e6485b57a3de37335"
CLASS_ORDER = ("newline", "quote_bracket", "punctuation", "capitalized")
EXPECTED_COUNTS = {
    "newline": 3,
    "quote_bracket": 123,
    "punctuation": 634,
    "capitalized": 16_777,
}
EXPECTED_ID_SHA256 = {
    "newline": "c3d73b237cc75488fa7d0cb8a513900784a6de406935e3dd906c4f6a7c29d3ca",
    "quote_bracket": "409aa84e8c02a38e98f619df6a9b723c2a2a200362f6e3f12695ca3091cd861d",
    "punctuation": "e3d8e80469f4152a7ac6ba39e5560a6cd7610eb8fb7cd0fe2c20d777a324b04d",
    "capitalized": "481180085053ca3d858bd57d1f001aa5198ab5a2b02e1250137d08b647a52b7c",
}
REGISTRY_SHA256 = "0621e173f7b58728bd8436727ab076b4de9eb1f75afd25529978cd569c974984"


def _logical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def encoding_fingerprint(encoding: Any) -> str:
    digest = hashlib.sha256()
    for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda item: item[1]):
        digest.update(len(token).to_bytes(8, "little")); digest.update(token)
        digest.update(int(rank).to_bytes(8, "little"))
    for token, rank in sorted(encoding._special_tokens.items()):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little")); digest.update(encoded)
        digest.update(int(rank).to_bytes(8, "little"))
    return digest.hexdigest()


def build_registry(encoding: Any) -> dict[str, tuple[int, ...]]:
    """Classify exact decoded token bytes with frozen priority and ASCII rules."""

    if getattr(encoding, "name", None) != ENCODING_NAME or (
        encoding_fingerprint(encoding) != ENCODING_SHA256
    ):
        raise RuntimeError("newline tokenizer identity changed")
    groups: dict[str, list[int]] = {name: [] for name in CLASS_ORDER}
    quote_bracket = set(b"'\"()[]{}<>")
    punctuation = set(string.punctuation.encode("ascii"))
    for token_id in range(encoding.n_vocab):
        try:
            raw = encoding.decode_single_token_bytes(token_id)
        except KeyError:
            continue
        stripped = raw.strip(b" \t\r\n")
        if b"\n" in raw:
            groups["newline"].append(token_id)
        elif stripped and all(value in quote_bracket for value in stripped):
            groups["quote_bracket"].append(token_id)
        elif stripped and all(value in punctuation for value in stripped):
            groups["punctuation"].append(token_id)
        else:
            first = raw.lstrip(b" \t\r\n")[:1]
            if first and b"A" <= first <= b"Z":
                groups["capitalized"].append(token_id)
    result = {name: tuple(groups[name]) for name in CLASS_ORDER}
    validate_registry(result)
    return result


def validate_registry(registry: Mapping[str, object]) -> None:
    if tuple(registry) != CLASS_ORDER:
        raise RuntimeError("newline token-class order changed")
    seen: set[int] = set()
    serializable: dict[str, list[int]] = {}
    for name in CLASS_ORDER:
        values = registry[name]
        if type(values) not in (tuple, list) or any(
            type(value) is not int or value < 0 for value in values
        ) or list(values) != sorted(set(values)):
            raise RuntimeError(f"newline token class {name} is malformed")
        if seen.intersection(values):
            raise RuntimeError("newline token classes overlap")
        seen.update(values); serializable[name] = list(values)
        if len(values) != EXPECTED_COUNTS[name] or _logical_sha(list(values)) != (
            EXPECTED_ID_SHA256[name]
        ):
            raise RuntimeError(f"newline token class {name} differs from frozen GPT-2 IDs")
    if _logical_sha(serializable) != REGISTRY_SHA256:
        raise RuntimeError("newline token registry hash changed")


__all__ = (
    "CLASS_ORDER", "ENCODING_NAME", "ENCODING_SHA256", "EXPECTED_COUNTS",
    "EXPECTED_ID_SHA256", "REGISTRY_SHA256", "build_registry",
    "encoding_fingerprint", "validate_registry",
)
