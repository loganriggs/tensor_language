#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen fresh-text authority for Task14 fronted natural-QK specificity."""

from __future__ import annotations

from collections import Counter
import hashlib
import json

import circuit_battery_task14 as old_task14


SCHEMA = "task14_fresh_fronted_natural_qk_number_specificity_authority_v1"
TASK_ID = "subject_verb.number_agreement"
SPLIT = "FRESH_TEXT"
TEMPLATES = (
    ("behind_near", "Behind the {a1} near the {a2}, the {subject}"),
    ("beyond_under", "Beyond the {a1} under the {a2}, the {subject}"),
)
NOUN_PAIRS = (
    ("doctor", "doctors"), ("teacher", "teachers"), ("king", "kings"),
    ("queen", "queens"), ("apple", "apples"), ("orange", "oranges"),
    ("cup", "cups"), ("beach", "beaches"), ("city", "cities"),
    ("village", "villages"), ("worker", "workers"), ("student", "students"),
    ("parent", "parents"), ("brother", "brothers"), ("sister", "sisters"),
    ("actor", "actors"),
)
EXPECTED_ROWS_SHA256 = "c5a0753f8a03a77b88c7226808ec2053a19aaa19056682e2f67a9a3da146ae59"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _old_material():
    authority, _digest = old_task14.build_authority()
    vocabulary, prompts, token_tuples = set(), set(), set()
    for row in authority:
        for key in ("head_pair", "attractor_pair", "second_head_pair",
                    "second_attractor_pair", "surface_attractor_pair"):
            vocabulary.update(row[key])
        for side in ("base", "donor"):
            prompts.add(row[f"{side}_text"])
            token_tuples.add(tuple(row[f"{side}_ids"]))
    return vocabulary, prompts, token_tuples


def _render(template_index, subject_index, subject_number, state, donor=False):
    template_id, template = TEMPLATES[template_index]
    if donor:
        a1_index, a2_index = (subject_index + 5) % 16, (subject_index + 7) % 16
    else:
        a1_index, a2_index = (subject_index + 1) % 16, (subject_index + 2) % 16
    a1 = NOUN_PAIRS[a1_index][state[0]]
    a2 = NOUN_PAIRS[a2_index][state[1]]
    subject = NOUN_PAIRS[subject_index][subject_number]
    text = template.format(a1=a1, a2=a2, subject=subject)
    ids = old_task14.ENCODING.encode(text)
    if len(ids) != 9:
        raise ValueError(f"fresh prompt is not nine GPT-2 tokens: {text!r} -> {ids}")
    return template_id, text, ids


def _build_unvalidated():
    output = []
    for subject_index in range(16):
        subject_number = 0 if subject_index < 8 else 1
        direction = "singular_to_plural" if subject_number == 0 else "plural_to_singular"
        local = subject_index % 8
        state = ((local % 4) // 2, local % 2)
        foreign_subject_index = (subject_index // 8) * 8 + ((local + 4) % 8)
        for recipient_template in range(2):
            donor_template = 1 - recipient_template
            template_id, base_text, base_ids = _render(
                recipient_template, subject_index, subject_number, state)
            donor_template_id, same_text, same_ids = _render(
                donor_template, foreign_subject_index, subject_number, state, donor=True)
            _, opposite_text, opposite_ids = _render(
                donor_template, foreign_subject_index, 1-subject_number, state, donor=True)
            cell_id = f"{direction}__{template_id}"
            identity = [SCHEMA, subject_index, template_id, foreign_subject_index, donor_template_id]
            output.append({
                "schema": SCHEMA, "task_id": TASK_ID, "split": SPLIT,
                "row_id": canonical_sha256(identity), "group_id": f"fresh:{subject_index:02d}:{template_id}",
                "foreign_group_id": f"fresh:{foreign_subject_index:02d}:{donor_template_id}:donor",
                "cell_id": cell_id, "atlas_cell_id": cell_id,
                "diagnostic_cell_id": f"{direction}__a{state[0]}{state[1]}",
                "recipient_template": template_id, "donor_template": donor_template_id,
                "attractor_state": list(state), "subject_pair_index": subject_index,
                "foreign_subject_pair_index": foreign_subject_index,
                "base_ids": base_ids, "same_ids": same_ids, "opposite_ids": opposite_ids,
                "base_text": base_text, "same_text": same_text, "opposite_text": opposite_text,
                "base_subject_number": "singular" if subject_number == 0 else "plural",
                "same_subject_number": "singular" if subject_number == 0 else "plural",
                "opposite_subject_number": "plural" if subject_number == 0 else "singular",
                "base_answer_id": 318 if subject_number == 0 else 389,
                "base_foil_id": 389 if subject_number == 0 else 318,
                "same_answer_id": 318 if subject_number == 0 else 389,
                "opposite_answer_id": 389 if subject_number == 0 else 318,
                "donor_answer_id": 389 if subject_number == 0 else 318,
                "donor_foil_id": 318 if subject_number == 0 else 389,
                "subject_position": 8,
            })
    return output


def validate_rows(rows, *, verify_frozen_hash=True):
    rows = list(rows)
    if len(rows) != 32 or len({row["row_id"] for row in rows}) != 32:
        raise ValueError("fresh authority must have 32 unique recipients")
    cells = Counter(row["cell_id"] for row in rows)
    diagnostics = Counter(row["diagnostic_cell_id"] for row in rows)
    if len(cells) != 4 or set(cells.values()) != {8}:
        raise ValueError(f"direction-template cells are not 4 x 8: {cells}")
    if len(diagnostics) != 8 or set(diagnostics.values()) != {4}:
        raise ValueError(f"direction-attractor cells are not 8 x 4: {diagnostics}")
    old_vocabulary, old_prompts, old_tokens = _old_material()
    fresh_forms = {word for pair in NOUN_PAIRS for word in pair}
    if fresh_forms & old_vocabulary or len(fresh_forms) != 32:
        raise ValueError("fresh noun forms overlap the old Task14 vocabulary")
    if any(len(old_task14.ENCODING.encode(" " + word)) != 1 for word in fresh_forms):
        raise ValueError("a fresh noun form is not one GPT-2 token after a space")
    prompts, tokens = [], []
    for row in rows:
        if row["schema"] != SCHEMA or row["task_id"] != TASK_ID or row["split"] != SPLIT:
            raise ValueError("fresh authority identity changed")
        if row["recipient_template"] == row["donor_template"] \
                or row["subject_pair_index"] == row["foreign_subject_pair_index"]:
            raise ValueError("recipient and donor are not foreign in template and noun")
        differences = [i for i, pair in enumerate(zip(row["same_ids"], row["opposite_ids"]))
                       if pair[0] != pair[1]]
        if differences != [8] or row["base_ids"][:8] == row["same_ids"][:8]:
            raise ValueError("donor counterfactual or foreign prefix changed")
        for role in ("base", "same", "opposite"):
            prompts.append(row[f"{role}_text"]); tokens.append(tuple(row[f"{role}_ids"]))
            answer_id = row[f"{role}_answer_id"]
            answer = " is" if answer_id == 318 else " are" if answer_id == 389 else None
            if answer is None:
                raise ValueError("fresh authority answer token changed")
            joint = old_task14.ENCODING.encode(row[f"{role}_text"] + answer)
            if joint[:-1] != row[f"{role}_ids"] or joint[-1:] != [answer_id]:
                raise ValueError("fresh prompt is not stable under prompt-plus-answer tokenization")
    if len(set(prompts)) != 96 or len(set(tokens)) != 96:
        raise ValueError("all 96 fresh prompts and token tuples must be unique")
    if set(prompts) & old_prompts or set(tokens) & old_tokens:
        raise ValueError("fresh prompt or token tuple overlaps old Task14")
    digest = canonical_sha256(rows)
    if verify_frozen_hash and digest != EXPECTED_ROWS_SHA256:
        raise ValueError(f"fresh authority digest changed: {digest}")
    return digest


def build_rows():
    rows = _build_unvalidated()
    validate_rows(rows)
    return rows


def compile_plan():
    rows = build_rows()
    return {"schema": SCHEMA, "task_id": TASK_ID, "split": SPLIT,
            "row_count": len(rows), "authority_sha256": canonical_sha256(rows),
            "direction_template_cells": dict(Counter(row["cell_id"] for row in rows)),
            "direction_attractor_cells": dict(Counter(row["diagnostic_cell_id"] for row in rows))}


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
