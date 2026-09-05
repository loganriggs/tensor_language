#!/usr/bin/env python3
"""Frozen pristine FIT/HOLDOUT authority for absolute MLP6--7 composition transfer."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import circuit_battery_task14 as old_task14
import circuit_fast_screen_candidate_task14_fresh_fronted_natural_qk_number_specificity as prior_fronted
import circuit_fast_screen_candidate_task14_fresh_matched_natural_native_capability as prior_matched


SCHEMA = "task14_pristine_split_mlp6_7_absolute_composition_authority_v1"
TASK_ID = "subject_verb.number_agreement"
CAPABILITY_ID = "subject_verb.number_agreement.pristine_split_mlp6_7_absolute_composition_capability_v1"
CAUSAL_CANDIDATE_ID = "subject_verb.number_agreement.pristine_split_mlp6_7_absolute_composition_transfer_v1"
SUBJECT_POSITION = 8
ROLES = ("recipient", "opposite_same_lemma", "same_number_different_lemma")
FIT_GROUPS = tuple(range(16))
HOLDOUT_GROUPS = tuple(range(16, 24))
FIT_TEMPLATES = (
    ("near_behind", "Near the {a1} behind the {a2}, the {subject}"),
    ("outside_past", "Outside the {a1} past the {a2}, the {subject}"),
)
HOLDOUT_TEMPLATES = (
    ("past_outside", "Past the {a1} outside the {a2}, the {subject}"),
)
NOUN_PAIRS = (
    ("banker", "bankers"), ("clerk", "clerks"),
    ("dancer", "dancers"), ("editor", "editors"),
    ("hunter", "hunters"), ("lawyer", "lawyers"),
    ("mayor", "mayors"), ("poet", "poets"),
    ("priest", "priests"), ("scholar", "scholars"),
    ("soldier", "soldiers"), ("vendor", "vendors"),
    ("captain", "captains"), ("chef", "chefs"),
    ("coach", "coaches"), ("manager", "managers"),
    ("officer", "officers"), ("professor", "professors"),
    ("reporter", "reporters"), ("musician", "musicians"),
    ("engineer", "engineers"), ("architect", "architects"),
    ("surgeon", "surgeons"), ("director", "directors"),
)
EXPECTED_AUTHORITY_SHA256 = "7d8220ed50c7b2ec526e32d81c791735027ba012b5034bdef837caf8a95f1b8f"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _phase(group):
    return "FIT" if group in FIT_GROUPS else "HOLDOUT"


def _templates(group):
    return FIT_TEMPLATES if group in FIT_GROUPS else HOLDOUT_TEMPLATES


def _phase_bounds(group):
    return (0, 16) if group in FIT_GROUPS else (16, 24)


def _render(template, a1, a2, subject):
    text = template.format(a1=a1, a2=a2, subject=subject)
    ids = old_task14.ENCODING.encode(text)
    if len(ids) != 9:
        raise ValueError(f"prompt is not nine GPT-2 tokens: {text!r} -> {ids}")
    return text, ids


def _build_unvalidated():
    rows = []
    for group in range(24):
        start, stop = _phase_bounds(group); width = stop - start; half = width // 2
        local = group - start
        recipient_number = 0 if local < half else 1
        direction = "singular_to_plural" if recipient_number == 0 else "plural_to_singular"
        within_direction = local % half
        lexical_group = start + (0 if recipient_number == 0 else half) + (within_direction + 2) % half
        a1_group = start + (local + 1) % width
        a2_group = start + (local + 3) % width
        attractor_state = ((within_direction % 4) // 2, within_direction % 2)
        a1 = NOUN_PAIRS[a1_group][attractor_state[0]]
        a2 = NOUN_PAIRS[a2_group][attractor_state[1]]
        for template_id, template in _templates(group):
            endpoints = {}
            subjects = {"recipient": NOUN_PAIRS[group][recipient_number],
                "opposite_same_lemma": NOUN_PAIRS[group][1-recipient_number],
                "same_number_different_lemma": NOUN_PAIRS[lexical_group][recipient_number]}
            numbers = {"recipient": recipient_number,
                "opposite_same_lemma": 1-recipient_number,
                "same_number_different_lemma": recipient_number}
            for role in ROLES:
                text, ids = _render(template, a1, a2, subjects[role])
                answer = 318 if numbers[role] == 0 else 389
                endpoints[role] = {"text": text, "ids": ids,
                    "subject": subjects[role],
                    "subject_number": "singular" if numbers[role] == 0 else "plural",
                    "answer_id": answer, "foil_id": 389 if answer == 318 else 318}
            identity = [SCHEMA, group, template_id, attractor_state, endpoints]
            rows.append({"schema": SCHEMA, "task_id": TASK_ID,
                "capability_id": CAPABILITY_ID, "causal_candidate_id": CAUSAL_CANDIDATE_ID,
                "phase": _phase(group), "group_number": group,
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
    vocabulary.update(word for pair in prior_fronted.NOUN_PAIRS for word in pair)
    for row in prior_fronted.build_rows():
        for role in ("base", "same", "opposite"):
            prompts.add(row[f"{role}_text"]); tokens.add(tuple(row[f"{role}_ids"]))
    vocabulary.update(word for pair in prior_matched.NOUN_PAIRS for word in pair)
    for row in prior_matched.build_rows():
        for endpoint in row["endpoints"].values():
            prompts.add(endpoint["text"]); tokens.add(tuple(endpoint["ids"]))
    return vocabulary, prompts, tokens


def validate_rows(rows, *, verify_hash=True):
    rows = list(rows)
    if rows != _build_unvalidated() or len(rows) != 40 or len({r["row_id"] for r in rows}) != 40:
        raise ValueError("authority differs from the frozen 40-row design")
    counts = Counter((r["phase"], r["direction_id"], r["template_id"]) for r in rows)
    if sorted(counts.values()) != [4, 4, 8, 8, 8, 8]:
        raise ValueError(f"phase/direction/template balance changed: {counts}")
    forms = {word for pair in NOUN_PAIRS for word in pair}
    prior_vocabulary, prior_prompts, prior_tokens = _prior_material()
    if len(forms) != 48 or forms & prior_vocabulary:
        raise ValueError("new noun forms overlap prior Task14 material")
    if any(len(old_task14.ENCODING.encode(" " + word)) != 1 for word in forms):
        raise ValueError("new noun form is not one GPT-2 token")
    prompts, tokens, phase_forms = [], [], {"FIT": set(), "HOLDOUT": set()}
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
            phase_forms[row["phase"]].add(endpoint["subject"])
    if len(prompts) != 120 or len(set(prompts)) != 120 or len(set(tokens)) != 120:
        raise ValueError("authority endpoints are not 120 unique prompts")
    if set(prompts) & prior_prompts or set(tokens) & prior_tokens:
        raise ValueError("new prompts overlap prior Task14 material")
    if phase_forms["FIT"] & phase_forms["HOLDOUT"]:
        raise ValueError("FIT and HOLDOUT subject vocabularies overlap")
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
        "prompt_endpoints": len(rows) * 3, "fit_groups": list(FIT_GROUPS),
        "holdout_groups": list(HOLDOUT_GROUPS), "fit_templates": [x[0] for x in FIT_TEMPLATES],
        "holdout_templates": [x[0] for x in HOLDOUT_TEMPLATES],
        "authority_sha256": canonical_sha256(rows)}


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
