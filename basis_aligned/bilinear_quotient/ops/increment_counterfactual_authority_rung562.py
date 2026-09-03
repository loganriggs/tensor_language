#!/usr/bin/env python3
"""Build R562's fresh increment/successor counterfactual authority; CPU only."""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
OUT = BQ / "increment_counterfactual_authority_rung562.json"
RECEIPT = BQ / "increment_counterfactual_authority_rung562_receipt.json"
PREREG = ROOT / "polynomial_causal" / "INCREMENT_COUNTERFACTUAL_AUTHORITY_RUNG562_PREREGISTRATION.md"
ENC = tiktoken.get_encoding("gpt2")

NUMBER_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}

SPLITS = {
    "FIT": {
        "count": 64, "seed": 56201, "starts": [1, 2], "control_starts": list(range(30, 38)),
        "leads": ["Inventory count", "Workshop order", "Garden sequence", "Daily entries"],
        "words": ["apple", "book", "chair", "door", "field", "glass", "horse", "island",
                  "jacket", "key", "lamp", "map", "needle", "orange", "plant", "road"],
    },
    "SELECT": {
        "count": 32, "seed": 56202, "starts": [6, 7], "control_starts": list(range(50, 58)),
        "leads": ["Archive order", "Kitchen count", "Market sequence", "Travel entries"],
        "words": ["basket", "cloud", "drawer", "engine", "forest", "guitar", "helmet", "ladder",
                  "mirror", "pocket", "rabbit", "shovel", "ticket", "valley", "window", "yarn"],
    },
    "FINAL_TEST": {
        "count": 32, "seed": 56203, "starts": [11, 12], "control_starts": list(range(70, 78)),
        "leads": ["Museum order", "Clinic count", "Library sequence", "Harbor entries"],
        "words": ["barrel", "candle", "diamond", "feather", "glove", "hotel", "lemon", "ocean",
                  "quilt", "ribbon", "statue", "trumpet", "umbrella", "wagon", "xylophone", "zebra"],
    },
    "OOD": {
        "count": 32, "seed": 56204, "starts": [16, 17], "control_starts": list(range(90, 98)),
        "leads": ["Calibration states", "Theorem indices", "Protocol stages", "Estimator passes"],
        "words": ["axiom", "buffer", "covariance", "derivative", "estimator", "frequency", "gradient", "kernel",
                  "likelihood", "manifold", "parameter", "regularizer", "statistic", "tensor", "variable", "weight"],
    },
}

FAMILIES = {
    "digit_coherent_shift": "interchange",
    "word_coherent_shift": "interchange",
    "cross_format_coherent_shift": "interchange",
    "incoherent_middle_number_edit": "necessity",
    "operation_preserved_surface_edit": "invariance",
    "repeated_number_numeric_control": "invariance",
    "step_two_numeric_control": "invariance",
}


def encode(text: str) -> list[int]:
    ids = ENC.encode(text)
    assert ENC.decode(ids) == text
    return ids


def token_id(text: str) -> int:
    ids = encode(text)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def content_id(value: object) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def digit_prompt(lead: str, values: tuple[int, int, int], words: tuple[str, str, str], style: int) -> str:
    a, b, c = values
    w0, w1, w2 = words
    if style == 0:
        return f"{lead}: {a} {w0}; {b} {w1}; {c} {w2}. Next:"
    if style == 1:
        return f"{lead} -- {a} {w0}, then {b} {w1}, then {c} {w2}. Next:"
    if style == 2:
        return f"{lead}: item {a} is {w0}; item {b} is {w1}; item {c} is {w2}. Next item:"
    return f"{lead}. We recorded {a} {w0} / {b} {w1} / {c} {w2}. Continue with:"


def word_prompt(lead: str, values: tuple[int, int, int], words: tuple[str, str, str], style: int) -> str:
    rendered = tuple(NUMBER_WORD[value] for value in values)
    return digit_prompt(lead + " in words", rendered, words, style)  # type: ignore[arg-type]


def make_row(split: str, group_id: str, family: str, lead: str, words: tuple[str, str, str],
             base_start: int, donor_start: int, control_start: int, style: int) -> dict:
    base_values = (base_start, base_start + 1, base_start + 2)
    donor_values = (donor_start, donor_start + 1, donor_start + 2)
    base_answer_value, donor_answer_value = base_start + 3, donor_start + 3
    base_operation = donor_operation = "+1"

    if family == "digit_coherent_shift":
        base = digit_prompt(lead + " [digits]", base_values, words, style)
        donor = digit_prompt(lead + " [shifted digits]", donor_values, words, style)
        base_answer, donor_answer = f" {base_answer_value}", f" {donor_answer_value}"
    elif family == "word_coherent_shift":
        base = word_prompt(lead + " [base]", base_values, words, style)
        donor = word_prompt(lead + " [shifted]", donor_values, words, style)
        base_answer = " " + NUMBER_WORD[base_answer_value]
        donor_answer = " " + NUMBER_WORD[donor_answer_value]
    elif family == "cross_format_coherent_shift":
        base = digit_prompt(lead + " [cross-format base]", base_values, words, style)
        donor = word_prompt(lead + " [cross-format donor]", donor_values, words, style)
        base_answer, donor_answer = f" {base_answer_value}", " " + NUMBER_WORD[donor_answer_value]
    elif family == "incoherent_middle_number_edit":
        broken = (base_start, base_start + 5, base_start + 2)
        base = digit_prompt(lead + " [coherent]", base_values, words, style)
        donor = digit_prompt(lead + " [middle changed]", broken, words, style)
        base_answer = donor_answer = f" {base_answer_value}"
        donor_operation = "broken +1 evidence; expected endpoint retained"
        donor_values = broken
    elif family == "operation_preserved_surface_edit":
        base = digit_prompt(lead + " [surface A]", base_values, words, style)
        donor = digit_prompt(lead + " [surface B]", base_values, tuple(reversed(words)), (style + 1) % 4)
        base_answer = donor_answer = f" {base_answer_value}"
        donor_values = base_values
    elif family == "repeated_number_numeric_control":
        values = (base_start, base_start, base_start)
        base = digit_prompt(lead + " [repeat A]", values, words, style)
        donor = digit_prompt(lead + " [repeat B]", values, tuple(reversed(words)), (style + 1) % 4)
        base_answer = donor_answer = f" {base_start}"
        base_operation = donor_operation = "copy/repeat"
        base_values = donor_values = values
    elif family == "step_two_numeric_control":
        values = (control_start, control_start + 2, control_start + 4)
        base = digit_prompt(lead + " [step two A]", values, words, style)
        donor = digit_prompt(lead + " [step two B]", values, tuple(reversed(words)), (style + 1) % 4)
        base_answer = donor_answer = f" {control_start + 6}"
        base_operation = donor_operation = "+2"
        base_values = donor_values = values
    else:
        raise KeyError(family)

    base_ids, donor_ids = encode(base), encode(donor)
    role = FAMILIES[family]
    return {
        "row_id": content_id({"group_id": group_id, "family_id": family}),
        "group_id": group_id,
        "split": split,
        "family_id": family,
        "role": role,
        "prompt_lead": lead,
        "content_words": list(words),
        "surface_style": style,
        "base_text": base,
        "donor_text": donor,
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer": base_answer,
        "donor_answer": donor_answer,
        "base_answer_id": token_id(base_answer),
        "donor_answer_id": token_id(donor_answer),
        "base_operation": base_operation,
        "donor_operation": donor_operation,
        "base_state": list(base_values),
        "donor_state": list(donor_values),
        "answer_changes": base_answer != donor_answer,
        "evaluation_directions": ["base_to_donor", "donor_to_base"],
        "construction_checks": {
            "base_roundtrip": ENC.decode(base_ids) == base,
            "donor_roundtrip": ENC.decode(donor_ids) == donor,
            "single_token_base_answer": len(encode(base_answer)) == 1,
            "single_token_donor_answer": len(encode(donor_answer)) == 1,
        },
    }


def make_groups(split: str, spec: dict) -> list[tuple]:
    rng = random.Random(spec["seed"])
    orientations = [(spec["starts"][0], spec["starts"][1]), (spec["starts"][1], spec["starts"][0])]
    selected, seen = [], set()
    for index in range(spec["count"]):
        base_start, donor_start = orientations[index % 2]
        while True:
            words = tuple(rng.sample(spec["words"], 3))
            lead = rng.choice(spec["leads"])
            style = rng.randrange(4)
            identity = (lead, words, base_start, donor_start, style)
            if identity not in seen:
                seen.add(identity)
                selected.append((lead, words, base_start, donor_start,
                                 spec["control_starts"][index % len(spec["control_starts"])], style))
                break
    return selected


def main() -> None:
    rows = []
    for split, spec in SPLITS.items():
        for lead, words, base_start, donor_start, control_start, style in make_groups(split, spec):
            coordinates = {
                "rung": 562, "split": split, "lead": lead, "words": words,
                "base_start": base_start, "donor_start": donor_start,
                "control_start": control_start, "style": style,
            }
            group_id = content_id(coordinates)
            rows.extend(make_row(split, group_id, family, lead, words, base_start, donor_start, control_start, style)
                        for family in FAMILIES)

    group_families: dict[str, set[str]] = collections.defaultdict(set)
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        group_families[row["group_id"]].add(row["family_id"])
        group_splits[row["group_id"]].add(row["split"])
        assert all(row["construction_checks"].values())
    expected_groups = sum(spec["count"] for spec in SPLITS.values())
    assert len(rows) == expected_groups * len(FAMILIES)
    assert len(group_families) == expected_groups
    assert all(families == set(FAMILIES) for families in group_families.values())
    assert all(len(splits) == 1 for splits in group_splits.values())
    pair_ids = [content_id({"base": row["base_ids"], "donor": row["donor_ids"],
                            "base_answer": row["base_answer_id"], "donor_answer": row["donor_answer_id"]})
                for row in rows]
    assert len(pair_ids) == len(set(pair_ids))
    orientation_counts = collections.Counter((row["split"], row["family_id"], row["base_state"][0],
                                               row["donor_state"][0]) for row in rows
                                              if row["family_id"] in {"digit_coherent_shift", "word_coherent_shift",
                                                                      "cross_format_coherent_shift"})
    for split, spec in SPLITS.items():
        expected = spec["count"] // 2
        for family in ("digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift"):
            cells = [count for (s, f, _base, _donor), count in orientation_counts.items() if s == split and f == family]
            assert len(cells) == 2 and set(cells) == {expected}

    result = {
        "schema": "increment_counterfactual_authority_rung562_v1",
        "status": "rows_frozen_outcomes_unopened",
        "causal_variable": "numeric state plus the operation used to predict the next value",
        "target_operation": "+1",
        "families": FAMILIES,
        "split_policy": {
            "unit": "content-addressed semantic group shared across all seven families",
            "group_counts": {split: spec["count"] for split, spec in SPLITS.items()},
            "lexical_pools_disjoint": True,
            "prompt_leads_disjoint": True,
            "plus_one_start_pools_disjoint": True,
            "step_two_number_pools_disjoint": True,
            "final_test_used_for_selection": False,
            "ood_used_for_selection": False,
        },
        "row_count": len(rows),
        "group_count": len(group_families),
        "rows": rows,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    OUT.write_bytes(payload)
    receipt = {
        "schema": "increment_counterfactual_authority_rung562_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT)),
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "group_count": len(group_families),
        "family_counts": dict(sorted(collections.Counter(row["family_id"] for row in rows).items())),
        "split_counts": dict(sorted(collections.Counter(row["split"] for row in rows).items())),
        "unique_prompt_pair_count": len(set(pair_ids)),
        "all_groups_have_all_families": True,
        "all_groups_belong_to_one_split": True,
        "all_answer_endpoints_single_token": True,
        "balanced_target_orientations": True,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
