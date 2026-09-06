#!/usr/bin/env python3
"""v7 margin program with an exact controlled-text pending-delimiter selector."""

from __future__ import annotations

import json
import sys

import transparent_margin_program as v7

CLOSER_IDS = {'"': 1, ')': 8, ']': 60}


def pending_closer_id(text: str) -> int:
    if not isinstance(text, str):
        raise v7.ProgramError("text must be a string")
    stack: list[str] = []
    for character in text:
        if character == '"':
            if stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append('"')
        elif character == '(':
            stack.append(')')
        elif character == '[':
            stack.append(']')
        elif character in ')]':
            if not stack or stack[-1] != character:
                raise v7.ProgramError("mismatched closing delimiter")
            stack.pop()
    if len(stack) != 1:
        raise v7.ProgramError("text must contain exactly one pending supported opener")
    return CLOSER_IDS[stack[0]]


def bracket_text(artifact: dict, *, text: str, native_unedited_donorward_margin: float, donor_closer_id: int) -> dict:
    recipient = pending_closer_id(text)
    answer = v7.bracket(artifact, native_unedited_donorward_margin=native_unedited_donorward_margin, recipient_closer_id=recipient, donor_closer_id=donor_closer_id)
    answer["behavior"] = "bracket_text"
    answer["inferred_recipient_closer_id"] = recipient
    return answer


def dispatch(artifact: dict, request: dict) -> dict:
    if not isinstance(request, dict):
        raise v7.ProgramError("request must be an object")
    if request.get("behavior") != "bracket_text":
        return v7.dispatch(artifact, request)
    if set(request) != {"behavior", "text", "native_unedited_donorward_margin", "donor_closer_id"}:
        raise v7.ProgramError("unexpected bracket_text fields")
    return bracket_text(artifact, text=request["text"], native_unedited_donorward_margin=request["native_unedited_donorward_margin"], donor_closer_id=request["donor_closer_id"])


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        print(json.dumps(dispatch(v7.load_artifact(), request), sort_keys=True))
    except (v7.ProgramError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
