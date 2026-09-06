#!/usr/bin/env python3
"""v9 transparent margin program with controlled-text Task14 direction inference."""

from __future__ import annotations

import json
import re
import sys

import transparent_margin_program as v7
import transparent_margin_program_v9 as v9

IRREGULAR_SINGULAR = {"child", "person", "mouse", "woman", "man", "fisherman", "glass"}
IRREGULAR_PLURAL = {"children", "people", "mice", "women", "men", "fishermen", "glasses"}


def subject_number(text: str) -> str:
    if not isinstance(text, str):
        raise v7.ProgramError("text must be a string")
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        raise v7.ProgramError("text must end in an alphabetic controlled-domain subject")
    subject = words[-1].lower()
    if subject in IRREGULAR_SINGULAR:
        return "singular"
    if subject in IRREGULAR_PLURAL:
        return "plural"
    if subject.endswith("ss"):
        raise v7.ProgramError("unsupported ambiguous final subject")
    return "plural" if subject.endswith("s") else "singular"


def task14_text(artifact: dict, *, text: str, background: str, edit: bool) -> dict:
    number = subject_number(text)
    direction = "singular_to_plural" if number == "singular" else "plural_to_singular"
    answer = v7.task14(artifact, direction=direction, background=background, edit=edit)
    answer["behavior"] = "task14_text"
    answer["inferred_subject_number"] = number
    answer["inferred_direction"] = direction
    return answer


def dispatch(artifact: dict, request: dict) -> dict:
    if not isinstance(request, dict):
        raise v7.ProgramError("request must be an object")
    if request.get("behavior") != "task14_text":
        return v9.dispatch(artifact, request)
    if set(request) != {"behavior", "text", "background", "edit"}:
        raise v7.ProgramError("unexpected task14_text fields")
    return task14_text(artifact, text=request["text"], background=request["background"], edit=request["edit"])


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        print(json.dumps(dispatch(v7.load_artifact(), request), sort_keys=True))
    except (v7.ProgramError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
