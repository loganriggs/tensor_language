#!/usr/bin/env python3
"""Build fresh balanced three-value pending-opener counterfactual rows; CPU only."""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
OUT = BQ / "pending_opener_three_value_fresh_rows_rung545.json"
RECEIPT = BQ / "pending_opener_three_value_fresh_rows_rung545_receipt.json"
PREREG = ROOT / "polynomial_causal" / "PENDING_OPENER_THREE_VALUE_FRESH_ROWS_RUNG545_PREREGISTRATION.md"
ENC = tiktoken.get_encoding("gpt2")

SPLITS = {
    "FIT": {
        "count": 72, "seed": 54501,
        "prefixes": ["The baker", "A gardener", "The mechanic", "One dancer", "Our cousin", "The sailor"],
        "words": ["apple", "cloud", "desk", "feather", "garage", "hammer", "jacket", "lake", "mirror",
                  "needle", "orange", "paper", "queen", "rope", "spoon", "tower", "umbrella", "wagon"],
    },
    "SELECT": {
        "count": 36, "seed": 54502,
        "prefixes": ["The pilot", "A doctor", "The farmer", "One actor"],
        "words": ["beach", "candle", "drawer", "fence", "guitar", "helmet", "ladder", "pocket", "rabbit",
                  "shovel", "temple", "whistle"],
    },
    "FINAL_TEST": {
        "count": 36, "seed": 54503,
        "prefixes": ["The judge", "A nurse", "The merchant", "One athlete"],
        "words": ["barrel", "chimney", "diamond", "factory", "glove", "hotel", "lemon", "ocean", "ribbon",
                  "statue", "trumpet", "wallet"],
    },
    "OOD": {
        "count": 36, "seed": 54504,
        "prefixes": ["In theorem C the reviewer", "During calibration the engineer", "The clinical record", "A legal memorandum"],
        "words": ["axiom", "buffer", "covariance", "derivative", "estimator", "frequency", "likelihood",
                  "manifold", "protocol", "regularizer", "statistic", "variable"],
    },
}

DELIMITERS = (
    {"name": "parenthesis", "open": "(", "close": ")"},
    {"name": "square", "open": "[", "close": "]"},
    {"name": "quote", "open": '"', "close": '"'},
)
ORDERED_PAIRS = tuple((left, right) for left in range(3) for right in range(3) if left != right)
FAMILIES = {
    "direct_three_value_type_substitution": "interchange",
    "completed_then_reopened_three_value_order": "interchange",
    "pending_type_preserved_surface_rewrite": "invariance",
    "pending_type_preserved_distance_extension": "invariance",
    "pending_type_preserved_nonopener_punctuation": "invariance",
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scaffold(template: int, prefix: str, verb0: str, verb1: str) -> str:
    choices = (
        f"{prefix} carefully {verb0}",
        f"After lunch, {prefix.lower()} quietly {verb1}",
        f"Without finishing, {prefix.lower()} deliberately {verb0}",
    )
    return choices[template]


def make_row(split: str, group_id: str, family: str, prefix: str, words: tuple[str, ...],
             left: dict, right: dict, template: int) -> dict:
    w0, w1, w2, w3, w4 = words
    lead = scaffold(template, prefix, "wrote", "listed")
    if family == "direct_three_value_type_substitution":
        body = f"the {w0}, the {w1}, the {w2}, the {w3}, and the {w4} remained together"
        base = f"{lead} {left['open']} {body}"
        donor = f"{lead} {right['open']} {body}"
        base_answer, donor_answer = left["close"], right["close"]
    elif family == "completed_then_reopened_three_value_order":
        lead = scaffold(template, prefix, "marked", "copied")
        base = (f"{lead} {left['open']} the {w0} and the {w1} {left['close']}, then continued "
                f"{right['open']} the {w2}, the {w3}, and the {w4}")
        donor = (f"{lead} {right['open']} the {w0} and the {w1} {right['close']}, then continued "
                 f"{left['open']} the {w2}, the {w3}, and the {w4}")
        base_answer, donor_answer = right["close"], left["close"]
    elif family == "pending_type_preserved_surface_rewrite":
        base = f"{lead} {left['open']} the {w0} before the {w1}, the {w2}, the {w3}, and the {w4}"
        donor = f"{lead} {left['open']} the {w4} after the {w3}, the {w2}, the {w1}, and the {w0}"
        base_answer = donor_answer = left["close"]
    elif family == "pending_type_preserved_distance_extension":
        base = f"{lead} {left['open']} the {w0} beside the {w1} and the {w2}"
        donor = (f"{lead} {left['open']} after mentioning the {w3} and the {w4} in a separate clause, "
                 f"the {w0} stayed beside the {w1} and the {w2}")
        base_answer = donor_answer = left["close"]
    elif family == "pending_type_preserved_nonopener_punctuation":
        base = f"{prefix} paused, then carefully wrote {left['open']} the {w0}, the {w1}, the {w2}, the {w3}, and the {w4}"
        donor = f"{prefix} paused: then carefully wrote {left['open']} the {w0}, the {w1}, the {w2}, the {w3}, and the {w4}"
        base_answer = donor_answer = left["close"]
    else:
        raise KeyError(family)

    base_ids, donor_ids = encode(base), encode(donor)
    differences = [index for index, values in enumerate(zip(base_ids, donor_ids)) if values[0] != values[1]]
    direct_single = len(base_ids) == len(donor_ids) and len(differences) == 1
    if family in {"direct_three_value_type_substitution", "pending_type_preserved_nonopener_punctuation"}:
        assert direct_single
    role = FAMILIES[family]
    assert (base_answer != donor_answer) == (role == "interchange")
    return {
        "row_id": content_id({"group_id": group_id, "family": family}),
        "group_id": group_id, "split": split, "family_id": family, "role": role, "template": template,
        "base_text": base, "donor_text": donor, "base_ids": base_ids, "donor_ids": donor_ids,
        "base_answer": base_answer, "donor_answer": donor_answer,
        "base_answer_id": token_id(base_answer), "donor_answer_id": token_id(donor_answer),
        "answer_changes": base_answer != donor_answer,
        "proposed_variable_base": f"pending_{left['name'] if family != 'completed_then_reopened_three_value_order' else right['name']}",
        "proposed_variable_donor": f"pending_{right['name'] if role == 'interchange' else left['name']}",
        "evaluation_directions": ["base_to_donor", "donor_to_base"],
        "construction_checks": {
            "base_roundtrip": ENC.decode(base_ids) == base,
            "donor_roundtrip": ENC.decode(donor_ids) == donor,
            "equal_token_length": len(base_ids) == len(donor_ids),
            "single_token_difference": direct_single,
        },
    }


def groups(spec: dict) -> list[tuple]:
    per_pair = spec["count"] // len(ORDERED_PAIRS)
    assert per_pair * len(ORDERED_PAIRS) == spec["count"]
    generator = random.Random(spec["seed"])
    selected, seen = [], set()
    for pair in ORDERED_PAIRS:
        while sum(item[2] == pair for item in selected) < per_pair:
            candidate = (generator.choice(spec["prefixes"]), tuple(generator.sample(spec["words"], 5)),
                         pair, generator.randrange(3))
            invariant_identity = (candidate[0], candidate[1], pair[0], candidate[3])
            if invariant_identity not in seen:
                seen.add(invariant_identity)
                selected.append(candidate)
    return selected


def main() -> None:
    rows = []
    for split, spec in SPLITS.items():
        for prefix, words, pair, template in groups(spec):
            coordinates = {"split": split, "prefix": prefix, "words": words,
                           "delimiter_pair": pair, "template": template, "rung": 545}
            group_id = content_id(coordinates)
            left, right = DELIMITERS[pair[0]], DELIMITERS[pair[1]]
            rows.extend(make_row(split, group_id, family, prefix, words, left, right, template)
                        for family in FAMILIES)

    expected_groups = sum(spec["count"] for spec in SPLITS.values())
    assert len(rows) == expected_groups * len(FAMILIES)
    pair_keys = [content_id({"base_ids": row["base_ids"], "donor_ids": row["donor_ids"],
                             "base_answer_id": row["base_answer_id"], "donor_answer_id": row["donor_answer_id"]})
                 for row in rows]
    sequence_keys = [content_id(ids) for row in rows for ids in (row["base_ids"], row["donor_ids"])]
    assert len(pair_keys) == len(set(pair_keys))
    assert len(sequence_keys) == len(set(sequence_keys))
    group_families: dict[str, set[str]] = collections.defaultdict(set)
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        group_families[row["group_id"]].add(row["family_id"])
        group_splits[row["group_id"]].add(row["split"])
    assert all(value == set(FAMILIES) for value in group_families.values())
    assert all(len(value) == 1 for value in group_splits.values())
    counts = collections.Counter((row["split"], row["family_id"], row["base_answer"], row["donor_answer"])
                                 for row in rows if row["role"] == "interchange")
    for split, spec in SPLITS.items():
        expected = spec["count"] // len(ORDERED_PAIRS)
        for family in tuple(FAMILIES)[:2]:
            cells = [count for (s, f, _base, _donor), count in counts.items() if s == split and f == family]
            assert len(cells) == 6 and set(cells) == {expected}

    result = {
        "schema": "pending_opener_three_value_fresh_rows_rung545_v1",
        "status": "rows_frozen_outcomes_unopened",
        "causal_variable": "candidate pending delimiter type over parenthesis, square bracket, and quote",
        "delimiter_types": list(DELIMITERS), "families": FAMILIES,
        "split_policy": {
            "unit": "content-addressed semantic group shared across all five families",
            "group_counts": {split: spec["count"] for split, spec in SPLITS.items()},
            "ordered_pair_counts": {split: spec["count"] // 6 for split, spec in SPLITS.items()},
            "prefix_and_word_pools_disjoint": True,
            "fresh_from_r543_templates_prefixes_words_and_seeds": True,
            "exact_prompt_pair_unique_globally": True,
            "exact_token_sequence_unique_globally": True,
            "final_test_used_for_selection": False,
        },
        "row_count": len(rows), "group_count": len(group_families), "rows": rows,
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0, "outcomes_opened": [],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    OUT.write_bytes(payload)
    receipt = {
        "schema": "pending_opener_three_value_fresh_rows_rung545_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT)), "rows_sha256": hashlib.sha256(payload).hexdigest(),
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "row_count": len(rows), "group_count": len(group_families),
        "family_counts": dict(sorted(collections.Counter(row["family_id"] for row in rows).items())),
        "split_counts": dict(sorted(collections.Counter(row["split"] for row in rows).items())),
        "unique_prompt_pair_count": len(set(pair_keys)), "unique_token_sequence_count": len(set(sequence_keys)),
        "all_groups_have_all_families": True, "all_groups_belong_to_one_split": True,
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0, "outcomes_opened": [],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

