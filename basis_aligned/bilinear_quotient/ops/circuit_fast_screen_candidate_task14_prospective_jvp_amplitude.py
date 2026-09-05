#!/usr/bin/env python3
"""Frozen new-text authority for prospective downstream-JVP amplitude validation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import circuit_battery_task14 as old_task14
import circuit_fast_screen_candidate_task14_fresh_fronted_natural_qk_number_specificity as prior_fronted
import circuit_fast_screen_candidate_task14_fresh_matched_natural_native_capability as prior_matched
import circuit_fast_screen_candidate_task14_pristine_split_mlp6_7_absolute_composition as prior_pristine


SCHEMA = "task14_prospective_jvp_amplitude_authority_v1"
TASK_ID = "subject_verb.number_agreement"
CAPABILITY_ID = "subject_verb.number_agreement.prospective_jvp_amplitude_capability_v1"
CAUSAL_CANDIDATE_ID = "subject_verb.number_agreement.prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1"
SUBJECT_POSITION = 8
ROLES = ("recipient", "opposite_same_lemma", "same_number_different_lemma")
TEMPLATES = (
    ("above_below", "Above the {a1} below the {a2}, the {subject}"),
    ("below_above", "Below the {a1} above the {a2}, the {subject}"),
)
NOUN_PAIRS = (
    ("brewer", "brewers"), ("monk", "monks"),
    ("saint", "saints"), ("prince", "princes"),
    ("lord", "lords"), ("chief", "chiefs"),
    ("speaker", "speakers"), ("buyer", "buyers"),
    ("seller", "sellers"), ("miner", "miners"),
    ("maker", "makers"), ("viewer", "viewers"),
    ("voter", "voters"), ("guest", "guests"),
    ("host", "hosts"), ("cousin", "cousins"),
)
EXPECTED_AUTHORITY_SHA256 = "d9f16f8066e48607ddcf23fd7d99c2d1d6025fd6ccc541c3f140a95370a4c79f"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _render(template, a1, a2, subject):
    text = template.format(a1=a1, a2=a2, subject=subject)
    ids = old_task14.ENCODING.encode(text)
    if len(ids) != 9:
        raise ValueError(f"prompt is not nine GPT-2 tokens: {text!r} -> {ids}")
    return text, ids


def _build_unvalidated():
    rows = []
    for group in range(len(NOUN_PAIRS)):
        recipient_number = 0 if group < len(NOUN_PAIRS)//2 else 1
        direction = "singular_to_plural" if recipient_number == 0 else "plural_to_singular"
        within_direction = group % (len(NOUN_PAIRS)//2)
        lexical_group = (0 if recipient_number == 0 else len(NOUN_PAIRS)//2) \
            + (within_direction + 3) % (len(NOUN_PAIRS)//2)
        a1_group = (group + 1) % len(NOUN_PAIRS)
        a2_group = (group + 5) % len(NOUN_PAIRS)
        attractor_state = ((within_direction % 4)//2, within_direction % 2)
        a1 = NOUN_PAIRS[a1_group][attractor_state[0]]
        a2 = NOUN_PAIRS[a2_group][attractor_state[1]]
        for template_id, template in TEMPLATES:
            subjects = {"recipient": NOUN_PAIRS[group][recipient_number],
                "opposite_same_lemma": NOUN_PAIRS[group][1-recipient_number],
                "same_number_different_lemma": NOUN_PAIRS[lexical_group][recipient_number]}
            numbers = {"recipient": recipient_number,
                "opposite_same_lemma": 1-recipient_number,
                "same_number_different_lemma": recipient_number}
            endpoints = {}
            for role in ROLES:
                text, ids = _render(template, a1, a2, subjects[role])
                answer = 318 if numbers[role] == 0 else 389
                endpoints[role] = {"text": text, "ids": ids, "subject": subjects[role],
                    "subject_number": "singular" if numbers[role] == 0 else "plural",
                    "answer_id": answer, "foil_id": 389 if answer == 318 else 318}
            identity = [SCHEMA, group, template_id, attractor_state, endpoints]
            rows.append({"schema": SCHEMA, "task_id": TASK_ID,
                "capability_id": CAPABILITY_ID, "causal_candidate_id": CAUSAL_CANDIDATE_ID,
                "phase": "PROSPECTIVE", "group_number": group,
                "row_id": canonical_sha256(identity), "template_id": template_id,
                "direction_id": direction, "attractor_state": list(attractor_state),
                "subject_position": SUBJECT_POSITION, "endpoints": endpoints})
    return rows


def _prior_material():
    vocabulary, prompts, tokens = set(), set(), set()
    old, _ = old_task14.build_authority()
    for row in old:
        for key in ("head_pair", "attractor_pair", "second_head_pair",
                    "second_attractor_pair", "surface_attractor_pair"):
            vocabulary.update(row[key])
        for side in ("base", "donor"):
            prompts.add(row[f"{side}_text"]); tokens.add(tuple(row[f"{side}_ids"]))
    for module, roles in ((prior_fronted, ("base", "same", "opposite")),
                          (prior_matched, None), (prior_pristine, None)):
        vocabulary.update(word for pair in module.NOUN_PAIRS for word in pair)
        for row in module.build_rows():
            endpoints = row.get("endpoints")
            values = endpoints.values() if endpoints else (
                {"text": row[f"{role}_text"], "ids": row[f"{role}_ids"]} for role in roles)
            for endpoint in values:
                prompts.add(endpoint["text"]); tokens.add(tuple(endpoint["ids"]))
    return vocabulary, prompts, tokens


def validate_rows(rows, *, verify_hash=True):
    rows = list(rows)
    if rows != _build_unvalidated() or len(rows) != 32 or len({r["row_id"] for r in rows}) != 32:
        raise ValueError("authority differs from frozen 32-row design")
    counts = Counter((r["direction_id"], r["template_id"]) for r in rows)
    if sorted(counts.values()) != [8, 8, 8, 8]:
        raise ValueError(f"direction/template balance changed: {counts}")
    forms = {word for pair in NOUN_PAIRS for word in pair}
    prior_vocabulary, prior_prompts, prior_tokens = _prior_material()
    if len(forms) != 32 or forms & prior_vocabulary:
        raise ValueError("prospective noun forms overlap prior Task14 material")
    if any(len(old_task14.ENCODING.encode(" " + word)) != 1 for word in forms):
        raise ValueError("prospective noun form is not one GPT-2 token")
    prompts, tokens = [], []
    for row in rows:
        endpoints = row["endpoints"]
        if tuple(endpoints) != ROLES or row["subject_position"] != SUBJECT_POSITION:
            raise ValueError("endpoint interface changed")
        recipient, opposite, lexical = (endpoints[x] for x in ROLES)
        if recipient["subject_number"] == opposite["subject_number"] or \
                recipient["subject_number"] != lexical["subject_number"] or \
                recipient["subject"] == lexical["subject"]:
            raise ValueError("endpoint counterfactual relation changed")
        for alternate in (opposite, lexical):
            if [i for i, pair in enumerate(zip(recipient["ids"], alternate["ids"]))
                    if pair[0] != pair[1]] != [SUBJECT_POSITION]:
                raise ValueError("endpoint changes outside subject position")
        for endpoint in endpoints.values():
            answer = " is" if endpoint["answer_id"] == 318 else " are"
            if old_task14.ENCODING.encode(endpoint["text"] + answer) != endpoint["ids"] + [endpoint["answer_id"]]:
                raise ValueError("answer continuation tokenization changed")
            prompts.append(endpoint["text"]); tokens.append(tuple(endpoint["ids"]))
    if len(prompts) != 96 or len(set(prompts)) != 96 or len(set(tokens)) != 96:
        raise ValueError("authority endpoints are not unique")
    if set(prompts) & prior_prompts or set(tokens) & prior_tokens:
        raise ValueError("prospective prompts overlap prior Task14 material")
    digest = canonical_sha256(rows)
    if verify_hash and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError(f"authority hash changed: {digest}")
    return digest


def build_rows():
    rows = _build_unvalidated(); validate_rows(rows); return rows


def compile_plan():
    rows = build_rows()
    return {"schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
        "causal_candidate_id": CAUSAL_CANDIDATE_ID, "row_count": len(rows),
        "prompt_endpoints": len(rows)*3, "templates": [x[0] for x in TEMPLATES],
        "authority_sha256": canonical_sha256(rows)}


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
