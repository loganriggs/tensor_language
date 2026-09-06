#!/usr/bin/env python3
"""Exact controlled-domain recipient-state and L13H8 source-token selector."""

from __future__ import annotations

import transparent_margin_program as v7
from transparent_margin_program_v9 import pending_closer_id

OPENER_TOKEN_BY_CLOSER = {1: 366, 8: 357, 60: 685}


def select(text: str, token_ids: list[int]) -> dict:
    recipient = pending_closer_id(text)
    if not isinstance(token_ids, list) or not token_ids or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids):
        raise v7.ProgramError("token_ids must be a nonempty integer list")
    opener = OPENER_TOKEN_BY_CLOSER[recipient]
    positions = [index for index, token in enumerate(token_ids) if token == opener]
    if not positions:
        raise v7.ProgramError("token_ids do not contain the inferred semantic opener")
    source = positions[-1]
    return {"recipient_closer_id": recipient, "semantic_open_token_id": opener, "semantic_open_position": source}
