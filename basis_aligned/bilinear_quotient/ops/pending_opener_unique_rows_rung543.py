#!/usr/bin/env python3
"""Build unique, content-addressed four-closer counterfactual rows (CPU only)."""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
OUT = BQ / "pending_opener_unique_rows_rung543.json"
RECEIPT = BQ / "pending_opener_unique_rows_rung543_receipt.json"
PREREG = ROOT / "polynomial_causal" / "PENDING_OPENER_UNIQUE_FOUR_CLOSER_ROWS_RUNG543_PREREGISTRATION.md"
ENC = tiktoken.get_encoding("gpt2")

SPLITS = {
    "FIT": {
        "count": 96,
        "seed": 54301,
        "prefixes": ["The editor", "A teacher", "My neighbor", "The guide", "Our friend", "A reporter"],
        "words": ["lantern", "river", "garden", "window", "basket", "forest", "letter", "market",
                  "planet", "harbor", "camera", "bridge", "pencil", "meadow", "ticket", "castle",
                  "singer", "valley", "painter", "station"],
    },
    "SELECT": {
        "count": 48,
        "seed": 54302,
        "prefixes": ["The curator", "One visitor", "The captain", "A student"],
        "words": ["anchor", "museum", "island", "journal", "violin", "kitchen", "tunnel", "blanket",
                  "engine", "portrait", "shelter", "cabinet"],
    },
    "FINAL_TEST": {
        "count": 48,
        "seed": 54303,
        "prefixes": ["The architect", "One witness", "The librarian", "A musician"],
        "words": ["orchard", "compass", "theater", "notebook", "fountain", "village", "gallery", "package",
                  "workshop", "stairway", "calendar", "envelope"],
    },
    "OOD": {
        "count": 48,
        "seed": 54304,
        "prefixes": ["During the audit the analyst", "In appendix B the author", "The laboratory note", "A field report"],
        "words": ["spectrum", "enzyme", "matrix", "voltage", "isotope", "kernel", "vector", "circuit",
                  "tensor", "gradient", "sample", "coefficient"],
    },
}

DELIMITERS = (
    {"name": "parenthesis", "open": "(", "close": ")"},
    {"name": "square", "open": "[", "close": "]"},
    {"name": "curly", "open": "{", "close": "}"},
    {"name": "quote", "open": '"', "close": '"'},
)
ORDERED_TYPE_PAIRS = tuple((left, right) for left in range(4) for right in range(4) if left != right)
FAMILIES = {
    "direct_type_substitution": "interchange",
    "completed_then_reopened_order": "interchange",
    "surface_paraphrase": "invariance",
    "distance_shift": "invariance",
    "nonopener_punctuation_substitution": "invariance",
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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def make_row(split: str, group_id: str, family: str, prefix: str, words: tuple[str, ...],
             left: dict, right: dict, template: int) -> dict:
    w0, w1, w2, w3, w4 = words
    style = ("briefly", "quietly", "carefully")[template]
    if family == "direct_type_substitution":
        body = f"the {w0} was near the {w1}, the {w2}, the {w3}, and the {w4}"
        base = f"{prefix} {style} said {left['open']} {body}"
        donor = f"{prefix} {style} said {right['open']} {body}"
        base_answer, donor_answer = left["close"], right["close"]
    elif family == "completed_then_reopened_order":
        base = (f"{prefix} {style} completed {left['open']} the {w0} {left['close']} and opened "
                f"{right['open']} the {w1} beside the {w2}, the {w3}, and the {w4}")
        donor = (f"{prefix} {style} completed {right['open']} the {w0} {right['close']} and opened "
                 f"{left['open']} the {w1} beside the {w2}, the {w3}, and the {w4}")
        base_answer, donor_answer = right["close"], left["close"]
    elif family == "surface_paraphrase":
        base = f"{prefix} {style} described {left['open']} the {w0} beside the {w1}, the {w2}, the {w3}, and the {w4}"
        donor = f"{prefix} {style} recorded {left['open']} the {w4} beyond the {w3}, the {w2}, the {w1}, and the {w0}"
        base_answer = donor_answer = left["close"]
    elif family == "distance_shift":
        base = f"{prefix} {style} opened {left['open']} the {w0} near the {w1}, the {w2}, the {w3}, and the {w4}"
        donor = (f"{prefix} {style} opened {left['open']} after noting the {w2}, the {w3}, and the {w4}, "
                 f"the {w0} remained near the {w1}")
        base_answer = donor_answer = left["close"]
    elif family == "nonopener_punctuation_substitution":
        base = f"{prefix} {style} noted, then opened {left['open']} the {w0} near the {w1}, the {w2}, the {w3}, and the {w4}"
        donor = f"{prefix} {style} noted: then opened {left['open']} the {w0} near the {w1}, the {w2}, the {w3}, and the {w4}"
        base_answer = donor_answer = left["close"]
    else:
        raise KeyError(family)

    base_ids, donor_ids = encode(base), encode(donor)
    answer_changes = base_answer != donor_answer
    differences = [i for i, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
    direct_single = len(base_ids) == len(donor_ids) and len(differences) == 1
    if family in {"direct_type_substitution", "nonopener_punctuation_substitution"}:
        assert direct_single
    assert answer_changes == (FAMILIES[family] == "interchange")
    return {
        "row_id": content_id({"group_id": group_id, "family": family}),
        "group_id": group_id,
        "split": split,
        "family_id": family,
        "role": FAMILIES[family],
        "template": template,
        "base_text": base,
        "donor_text": donor,
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer": base_answer,
        "donor_answer": donor_answer,
        "base_answer_id": token_id(base_answer),
        "donor_answer_id": token_id(donor_answer),
        "answer_changes": answer_changes,
        "proposed_variable_base": f"pending_{left['name'] if family != 'completed_then_reopened_order' else right['name']}",
        "proposed_variable_donor": f"pending_{right['name'] if family in {'direct_type_substitution', 'completed_then_reopened_order'} else left['name']}",
        "evaluation_directions": ["base_to_donor", "donor_to_base"],
        "construction_checks": {
            "base_roundtrip": ENC.decode(base_ids) == base,
            "donor_roundtrip": ENC.decode(donor_ids) == donor,
            "equal_token_length": len(base_ids) == len(donor_ids),
            "single_token_difference": direct_single,
        },
    }


def semantic_groups(split: str, spec: dict) -> list[tuple]:
    generator = random.Random(spec["seed"])
    selected, seen = [], set()
    while len(selected) < spec["count"]:
        candidate = (
            generator.choice(spec["prefixes"]),
            tuple(generator.sample(spec["words"], 5)),
            generator.choice(ORDERED_TYPE_PAIRS),
            generator.randrange(3),
        )
        # Invariance rows use only the first delimiter.  Bind uniqueness to
        # exactly the coordinates that those prompts expose, not to an unused
        # second-delimiter coordinate.
        invariant_identity = (candidate[0], candidate[1], candidate[2][0], candidate[3])
        if invariant_identity not in seen:
            seen.add(invariant_identity)
            selected.append(candidate)
    return selected


def main() -> None:
    rows = []
    for split, spec in SPLITS.items():
        for prefix, words, pair, template in semantic_groups(split, spec):
            coordinates = {
                "prefix": prefix, "words": words, "delimiter_pair": pair, "template": template,
            }
            group_id = content_id(coordinates)
            left, right = DELIMITERS[pair[0]], DELIMITERS[pair[1]]
            for family in FAMILIES:
                rows.append(make_row(split, group_id, family, prefix, words, left, right, template))

    expected_rows = sum(spec["count"] for spec in SPLITS.values()) * len(FAMILIES)
    assert len(rows) == expected_rows
    family_split_counts = collections.Counter((row["family_id"], row["split"]) for row in rows)
    for family in FAMILIES:
        for split, spec in SPLITS.items():
            assert family_split_counts[(family, split)] == spec["count"]

    pair_keys = [content_id({
        "base_ids": row["base_ids"], "donor_ids": row["donor_ids"],
        "base_answer_id": row["base_answer_id"], "donor_answer_id": row["donor_answer_id"],
    }) for row in rows]
    sequence_keys = [content_id(ids) for row in rows for ids in (row["base_ids"], row["donor_ids"])]
    assert len(set(pair_keys)) == len(pair_keys)
    assert len(set(sequence_keys)) == len(sequence_keys)

    group_families: dict[str, set[str]] = collections.defaultdict(set)
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        group_families[row["group_id"]].add(row["family_id"])
        group_splits[row["group_id"]].add(row["split"])
    assert all(value == set(FAMILIES) for value in group_families.values())
    assert all(len(value) == 1 for value in group_splits.values())

    result = {
        "schema": "pending_opener_unique_four_closer_rows_rung543_v1",
        "status": "rows_frozen_outcomes_unopened",
        "causal_variable": "pending delimiter type at the final prediction position",
        "delimiter_types": list(DELIMITERS),
        "families": FAMILIES,
        "split_policy": {
            "unit": "content-addressed semantic group shared across all five families",
            "group_counts": {split: spec["count"] for split, spec in SPLITS.items()},
            "prefix_and_word_pools_disjoint": True,
            "exact_prompt_pair_unique_globally": True,
            "exact_token_sequence_unique_globally": True,
            "final_test_used_for_selection": False,
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
        "schema": "pending_opener_unique_four_closer_rows_rung543_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT)),
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "group_count": len(group_families),
        "family_counts": dict(sorted(collections.Counter(row["family_id"] for row in rows).items())),
        "split_counts": dict(sorted(collections.Counter(row["split"] for row in rows).items())),
        "unique_prompt_pair_count": len(set(pair_keys)),
        "unique_token_sequence_count": len(set(sequence_keys)),
        "all_groups_have_all_families": True,
        "all_groups_belong_to_one_split": True,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
