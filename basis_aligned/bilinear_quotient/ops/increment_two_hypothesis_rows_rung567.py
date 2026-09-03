#!/usr/bin/env python3
"""Build fresh list-successor and numeric-sequence counterfactual rows; CPU only."""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
OUT = BQ / "increment_two_hypothesis_rows_rung567.json"
RECEIPT = BQ / "increment_two_hypothesis_rows_rung567_receipt.json"
PREREG = ROOT / "polynomial_causal" / "INCREMENT_TWO_HYPOTHESIS_FRESH_FREEZE_RUNG567.md"
DEV = BQ / "increment_format_exploration_rung566_results.json"
ENC = tiktoken.get_encoding("gpt2")

NUMBER_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}

SPLITS = {
    "FIT": {
        "count": 32, "seed": 56701, "list_starts": [21, 22], "sequence_starts": [8, 9],
        "words": ["acorn", "beacon", "cabin", "drum", "ember", "flask", "grove", "harp",
                  "inlet", "kettle", "meadow", "orchard", "pebble", "reef", "saddle", "violin"],
    },
    "SELECT": {
        "count": 16, "seed": 56702, "list_starts": [31, 32], "sequence_starts": [11, 12],
        "words": ["alcove", "bonnet", "cradle", "dune", "easel", "fountain", "granite", "hinge",
                  "igloo", "lantern", "mosaic", "oar", "pillow", "ridge", "silo", "vase"],
    },
    "FINAL_TEST": {
        "count": 16, "seed": 56703, "list_starts": [41, 42], "sequence_starts": [14, 15],
        "words": ["archway", "blossom", "compass", "delta", "elm", "fossil", "geyser", "hammock",
                  "inkwell", "lighthouse", "marble", "obelisk", "parchment", "raft", "summit", "velvet"],
    },
    "OOD": {
        "count": 16, "seed": 56704, "list_starts": [51, 52], "sequence_starts": [16, 17],
        "words": ["activation", "basis", "circuit", "decoder", "eigenvalue", "feature", "graph", "hessian",
                  "intervention", "jacobian", "matrix", "operator", "projector", "residual", "subspace", "vector"],
    },
}

LIST_FAMILIES = {
    "list_two_line_state_shift": "interchange",
    "list_three_line_state_shift": "interchange",
    "list_surface_preserved": "invariance",
    "list_middle_index_break": "necessity",
    "list_repeated_index_control": "invariance",
    "list_step_two_conflict": "invariance",
}
SEQUENCE_FAMILIES = {
    "sequence_digit_state_shift": "interchange",
    "sequence_word_state_shift": "interchange",
    "sequence_cross_format_shift": "interchange",
    "sequence_digit_surface_preserved": "invariance",
    "sequence_word_surface_preserved": "invariance",
    "sequence_middle_value_break": "necessity",
    "sequence_digit_copy_control": "invariance",
    "sequence_word_copy_control": "invariance",
    "sequence_step_two_conflict": "invariance",
}


def encode(text: str) -> list[int]:
    ids = ENC.encode(text)
    assert ENC.decode(ids) == text
    return ids


def token_id(text: str) -> int:
    ids = encode(text)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def list_prompt(values: tuple[int, ...], words: tuple[str, str, str]) -> str:
    return "".join(f"{value}. {word}\n" for value, word in zip(values, words, strict=True))


def digit_sequence(values: tuple[int, int, int], word: str, variant: int) -> str:
    a, b, c = values
    if variant % 2 == 0:
        return f"The {word} number sequence is {a}, {b}, {c},"
    return f"For the {word}, the numbers are {a}, {b}, {c},"


def word_sequence(values: tuple[int, int, int], word: str, variant: int) -> str:
    rendered = tuple(NUMBER_WORD[value] for value in values)
    return digit_sequence(rendered, word, variant)  # type: ignore[arg-type]


def row(group_id: str, split: str, hypothesis: str, family: str, role: str, base: str, donor: str,
        base_answer: str | None, donor_answer: str | None, details: dict) -> dict:
    base_ids, donor_ids = encode(base), encode(donor)
    answer_changes = base_answer != donor_answer if base_answer is not None and donor_answer is not None else False
    result = {
        "row_id": digest({"group_id": group_id, "family_id": family}),
        "group_id": group_id, "split": split, "hypothesis_id": hypothesis,
        "family_id": family, "role": role, "base_text": base, "donor_text": donor,
        "base_ids": base_ids, "donor_ids": donor_ids, "base_answer": base_answer,
        "donor_answer": donor_answer, "answer_changes": answer_changes,
        "evaluation_directions": ["base_to_donor", "donor_to_base"], "semantic_details": details,
        "construction_checks": {
            "base_roundtrip": ENC.decode(base_ids) == base,
            "donor_roundtrip": ENC.decode(donor_ids) == donor,
            "distinct_prompts": base_ids != donor_ids,
        },
    }
    if base_answer is not None:
        result["base_answer_id"] = token_id(base_answer)
        result["donor_answer_id"] = token_id(donor_answer)  # type: ignore[arg-type]
        result["construction_checks"]["single_token_answers"] = True
    else:
        result["base_answer_id"] = result["donor_answer_id"] = None
        result["construction_checks"]["single_token_answers"] = True
    return result


def make_list_rows(split: str, group_id: str, words: tuple[str, str, str], base_start: int,
                   donor_start: int) -> list[dict]:
    base2, donor2 = (base_start, base_start + 1), (donor_start, donor_start + 1)
    base3, donor3 = (*base2, base_start + 2), (*donor2, donor_start + 2)
    reversed_words = tuple(reversed(words))
    common = {"target_computation": "next list index = final observed label + 1"}
    rows = [
        row(group_id, split, "numbered_list_index_successor", "list_two_line_state_shift", "interchange",
            list_prompt(base2, words[:2]), list_prompt(donor2, words[:2]), str(base_start + 2), str(donor_start + 2),
            {**common, "base_labels": base2, "donor_labels": donor2}),
        row(group_id, split, "numbered_list_index_successor", "list_three_line_state_shift", "interchange",
            list_prompt(base3, words), list_prompt(donor3, words), str(base_start + 3), str(donor_start + 3),
            {**common, "base_labels": base3, "donor_labels": donor3}),
        row(group_id, split, "numbered_list_index_successor", "list_surface_preserved", "invariance",
            list_prompt(base3, words), list_prompt(base3, reversed_words), str(base_start + 3), str(base_start + 3),
            {**common, "labels": base3, "changes": "content nouns and their order"}),
        row(group_id, split, "numbered_list_index_successor", "list_middle_index_break", "necessity",
            list_prompt(base3, words), list_prompt((base_start, base_start + 7, base_start + 2), words),
            str(base_start + 3), str(base_start + 3),
            {**common, "held_fixed": "first/final labels and expected successor", "changed": "middle label"}),
        row(group_id, split, "numbered_list_index_successor", "list_repeated_index_control", "invariance",
            list_prompt((base_start, base_start, base_start), words),
            list_prompt((base_start, base_start, base_start), reversed_words), str(base_start), str(base_start),
            {"control_rule": "repeat/copy final label", "labels": (base_start,) * 3}),
    ]
    step_values = (base_start, base_start + 2, base_start + 4)
    rows.append(row(
        group_id, split, "numbered_list_index_successor", "list_step_two_conflict", "invariance",
        list_prompt(step_values, words), list_prompt(step_values, reversed_words), str(base_start + 5), str(base_start + 5),
        {"structural_successor_answer": base_start + 5, "arithmetic_step_two_answer": base_start + 6,
         "conflict": "last list label + 1 versus continuing +2 arithmetic"},
    ))
    return rows


def make_sequence_rows(split: str, group_id: str, words: tuple[str, str, str], base_start: int,
                       donor_start: int, variant: int) -> list[dict]:
    base = (base_start, base_start + 1, base_start + 2)
    donor = (donor_start, donor_start + 1, donor_start + 2)
    word, alternate = words[0], words[1]
    ba, da = base_start + 3, donor_start + 3
    common = {"target_computation": "continue a +1 numeric sequence"}
    rows = [
        row(group_id, split, "numeric_sequence_continuation", "sequence_digit_state_shift", "interchange",
            digit_sequence(base, word, variant), digit_sequence(donor, word, variant), f" {ba}", f" {da}",
            {**common, "representation": "digits", "base_values": base, "donor_values": donor}),
        row(group_id, split, "numeric_sequence_continuation", "sequence_word_state_shift", "interchange",
            word_sequence(base, word, variant), word_sequence(donor, word, variant),
            " " + NUMBER_WORD[ba], " " + NUMBER_WORD[da],
            {**common, "representation": "number words", "base_values": base, "donor_values": donor}),
        row(group_id, split, "numeric_sequence_continuation", "sequence_cross_format_shift", "interchange",
            digit_sequence(base, word, variant), word_sequence(donor, word, variant), f" {ba}", " " + NUMBER_WORD[da],
            {**common, "representation_change": "digits to number words", "base_values": base, "donor_values": donor}),
        row(group_id, split, "numeric_sequence_continuation", "sequence_digit_surface_preserved", "invariance",
            digit_sequence(base, word, 0), digit_sequence(base, alternate, 1), f" {ba}", f" {ba}",
            {**common, "representation": "digits", "values": base}),
        row(group_id, split, "numeric_sequence_continuation", "sequence_word_surface_preserved", "invariance",
            word_sequence(base, word, 0), word_sequence(base, alternate, 1),
            " " + NUMBER_WORD[ba], " " + NUMBER_WORD[ba],
            {**common, "representation": "number words", "values": base}),
        row(group_id, split, "numeric_sequence_continuation", "sequence_middle_value_break", "necessity",
            digit_sequence(base, word, variant), digit_sequence((base_start, base_start + 6, base_start + 2), word, variant),
            f" {ba}", f" {ba}",
            {**common, "held_fixed": "first/final values and expected answer", "changed": "middle value"}),
        row(group_id, split, "numeric_sequence_continuation", "sequence_digit_copy_control", "invariance",
            digit_sequence((base_start,) * 3, word, 0), digit_sequence((base_start,) * 3, alternate, 1),
            f" {base_start}", f" {base_start}", {"control_rule": "copy repeated digit"}),
        row(group_id, split, "numeric_sequence_continuation", "sequence_word_copy_control", "invariance",
            word_sequence((base_start,) * 3, word, 0), word_sequence((base_start,) * 3, alternate, 1),
            " " + NUMBER_WORD[base_start], " " + NUMBER_WORD[base_start],
            {"control_rule": "copy repeated number word"}),
    ]
    step = (base_start, base_start + 2, base_start + 4)
    rows.append(row(
        group_id, split, "numeric_sequence_continuation", "sequence_step_two_conflict", "invariance",
        digit_sequence(step, word, 0), digit_sequence(step, alternate, 1), None, None,
        {"arithmetic_step_two_answer": base_start + 6, "last_value_successor_answer": base_start + 5,
         "purpose": "intervention selectivity only; native behavior need not choose either candidate"},
    ))
    return rows


def group_coordinates(split: str, spec: dict, hypothesis: str) -> list[tuple]:
    rng = random.Random(spec["seed"] + (0 if hypothesis == "list" else 100))
    starts = spec["list_starts"] if hypothesis == "list" else spec["sequence_starts"]
    selected, seen = [], set()
    for index in range(spec["count"]):
        base_start, donor_start = (starts if index % 2 == 0 else list(reversed(starts)))
        while True:
            words = tuple(rng.sample(spec["words"], 3))
            variant = rng.randrange(2)
            # Uniqueness is keyed by the words actually visible to the least detailed
            # family, not by the hidden third word in the semantic-group coordinates.
            identity = ((words[:2], base_start, donor_start) if hypothesis == "list" else
                        (words[0], base_start, donor_start, variant))
            if identity not in seen:
                seen.add(identity)
                selected.append((words, base_start, donor_start, variant))
                break
    return selected


def main() -> None:
    old_sequences = {tuple(row["ids"]) for row in json.loads(DEV.read_text())["rows"]}
    rows = []
    for split, spec in SPLITS.items():
        for hypothesis in ("list", "sequence"):
            for words, base_start, donor_start, variant in group_coordinates(split, spec, hypothesis):
                coordinates = {"rung": 567, "split": split, "hypothesis": hypothesis, "words": words,
                               "base_start": base_start, "donor_start": donor_start, "variant": variant}
                group_id = digest(coordinates)
                if hypothesis == "list":
                    rows.extend(make_list_rows(split, group_id, words, base_start, donor_start))
                else:
                    rows.extend(make_sequence_rows(split, group_id, words, base_start, donor_start, variant))

    expected = {"numbered_list_index_successor": set(LIST_FAMILIES),
                "numeric_sequence_continuation": set(SEQUENCE_FAMILIES)}
    by_group, group_splits, group_hypotheses = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)
    prompt_pairs = []
    new_sequences = set()
    for item in rows:
        by_group[item["group_id"]].add(item["family_id"])
        group_splits[item["group_id"]].add(item["split"])
        group_hypotheses[item["group_id"]].add(item["hypothesis_id"])
        assert all(item["construction_checks"].values())
        prompt_pairs.append(digest({"base": item["base_ids"], "donor": item["donor_ids"],
                                    "base_answer": item["base_answer_id"], "donor_answer": item["donor_answer_id"]}))
        new_sequences.update((tuple(item["base_ids"]), tuple(item["donor_ids"])))
    assert len(rows) == 1200 and len(by_group) == 160
    assert len(prompt_pairs) == len(set(prompt_pairs))
    assert not (new_sequences & old_sequences)
    assert all(len(value) == 1 for value in group_splits.values())
    assert all(len(value) == 1 for value in group_hypotheses.values())
    for group_id, families in by_group.items():
        hypothesis = next(iter(group_hypotheses[group_id]))
        assert families == expected[hypothesis]

    result = {
        "schema": "increment_two_hypothesis_rows_rung567_v1",
        "status": "rows_frozen_outcomes_unopened",
        "hypotheses": {
            "numbered_list_index_successor": {"families": LIST_FAMILIES, "group_count": 80},
            "numeric_sequence_continuation": {"families": SEQUENCE_FAMILIES, "group_count": 80},
        },
        "split_policy": {
            "group_counts_per_hypothesis": {split: spec["count"] for split, spec in SPLITS.items()},
            "lexical_pools_disjoint": True, "starting_value_pools_disjoint": True,
            "no_exact_r566_prompt_reused": True, "all_derived_rows_share_group_split": True,
            "final_test_used_for_selection": False, "ood_used_for_selection": False,
        },
        "row_count": len(rows), "group_count": len(by_group), "rows": rows,
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0, "outcomes_opened": [],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    OUT.write_bytes(payload)
    receipt = {
        "schema": "increment_two_hypothesis_rows_rung567_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT)), "rows_sha256": hashlib.sha256(payload).hexdigest(),
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "development_result_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
        "row_count": len(rows), "group_count": len(by_group),
        "hypothesis_counts": dict(collections.Counter(item["hypothesis_id"] for item in rows)),
        "family_counts": dict(collections.Counter(item["family_id"] for item in rows)),
        "split_row_counts": dict(collections.Counter(item["split"] for item in rows)),
        "unique_prompt_pair_count": len(set(prompt_pairs)), "unique_token_sequence_count": len(new_sequences),
        "development_sequence_overlap": len(new_sequences & old_sequences),
        "all_groups_complete": True, "all_groups_one_split": True,
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0, "outcomes_opened": [],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
