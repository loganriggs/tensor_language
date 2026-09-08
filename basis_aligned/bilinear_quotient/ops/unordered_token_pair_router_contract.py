"""Transparent direction-invariant token-difference signatures for finite routing."""

from __future__ import annotations


class TokenRouterError(ValueError):
    pass


def signature(base_tokens, source_tokens, semantic_position):
    if len(base_tokens) != len(source_tokens):
        raise TokenRouterError("token rows must be exactly aligned")
    stop = int(semantic_position)
    if not 0 <= stop < len(base_tokens):
        raise TokenRouterError("semantic position out of range")
    return tuple(tuple(sorted((int(left), int(right))))
                 for left, right in zip(base_tokens[:stop + 1], source_tokens[:stop + 1])
                 if int(left) != int(right))


def signatures(batch_base, batch_source):
    if batch_base.row_ids != batch_source.row_ids or len(batch_base.row_ids) != len(batch_base.semantic_positions):
        raise TokenRouterError("base/source batch rows must align")
    if batch_base.semantic_positions != batch_source.semantic_positions:
        raise TokenRouterError("semantic positions must align")
    return tuple(signature(left, right, stop) for left, right, stop in zip(
        batch_base.token_rows, batch_source.token_rows, batch_base.semantic_positions))


def fit(train_signatures, labels):
    if len(train_signatures) != len(labels):
        raise TokenRouterError("signature/label count mismatch")
    target_map = {}
    off = set()
    for item, label in zip(train_signatures, labels):
        label = int(label)
        if label == 2:
            off.add(item)
        elif label in (0, 1):
            if item in target_map and target_map[item] != label:
                raise TokenRouterError("target signature collision")
            target_map[item] = label
        else:
            raise TokenRouterError("label out of range")
    if not target_map or set(target_map) & off:
        raise TokenRouterError("target signature overlaps off")
    return target_map


def predict(train_map, held_signatures):
    return tuple(int(train_map.get(item, 2)) for item in held_signatures)
