"""Exact GPT-2 decimal successor registry for the prospective v2 SELECT assay."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from local_fineweb_harvest import encoding_fingerprint
from ordered_successor_masks_v1 import OrderedLexicon


SCHEMA = "ordered_successor_digit_lexicon_v2"
ENCODING_NAME = "gpt2"
ENCODING_SHA256 = "0be287937901b1baae837369293dd6f63da1bece9609006e6485b57a3de37335"
CONTEXT_RULE = (
    "encode_ordinary of the exact complete surface string; include only forms yielding "
    "exactly one token and decoding exactly to that string"
)
DIGIT_SURFACE_FORMS = tuple((str(value), f" {value}") for value in range(10))
DIGIT_TOKEN_IDS = (
    (15, 657), (16, 352), (17, 362), (18, 513), (19, 604),
    (20, 642), (21, 718), (22, 767), (23, 807), (24, 860),
)
REGISTRY_SHA256 = "e59c912c542d4477a222487086fcdfe02e2bee1d3b1176bb87bc137e3627cff3"


def registry_payload() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "encoding": ENCODING_NAME,
        "context_rule": CONTEXT_RULE,
        "items": [
            {
                "digit": str(index),
                "surface_forms": list(DIGIT_SURFACE_FORMS[index]),
                "token_ids": list(DIGIT_TOKEN_IDS[index]),
            }
            for index in range(10)
        ],
    }


def registry_sha256() -> str:
    return hashlib.sha256(json.dumps(
        registry_payload(), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def validate_encoding(encoding: Any) -> OrderedLexicon:
    if getattr(encoding, "name", None) != ENCODING_NAME or getattr(
        encoding, "n_vocab", None
    ) != 50_257 or encoding_fingerprint(encoding) != ENCODING_SHA256:
        raise RuntimeError("pinned GPT-2 tokenizer identity changed")
    observed = []
    for forms in DIGIT_SURFACE_FORMS:
        item = []
        for surface in forms:
            tokens = encoding.encode_ordinary(surface)
            if len(tokens) != 1 or encoding.decode(tokens) != surface:
                raise RuntimeError("digit surface is no longer one exact GPT-2 item")
            item.append(tokens[0])
        observed.append(tuple(item))
    if tuple(observed) != DIGIT_TOKEN_IDS or registry_sha256() != REGISTRY_SHA256:
        raise RuntimeError("frozen digit token IDs or registry hash changed")
    return OrderedLexicon("decimal_digits_v2", DIGIT_TOKEN_IDS)


def load_pinned_lexicon() -> tuple[OrderedLexicon, Any]:
    import tiktoken

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return validate_encoding(encoding), encoding


__all__ = (
    "CONTEXT_RULE", "DIGIT_SURFACE_FORMS", "DIGIT_TOKEN_IDS", "ENCODING_NAME",
    "ENCODING_SHA256", "REGISTRY_SHA256", "SCHEMA", "load_pinned_lexicon",
    "registry_payload", "registry_sha256", "validate_encoding",
)
