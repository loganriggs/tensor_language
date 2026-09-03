#!/usr/bin/env python3
"""Freeze CPU-only multi-family pending-opener counterfactual rows.

This builder never loads the model and contains no behavioral outcomes.  Group-level
splits are shared across all families, preventing a lexical/template unit from being
training data in one family and final-test data in another.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
OUT = BQ / "pending_opener_multifamily_rows_rung537.json"
RECEIPT = BQ / "pending_opener_multifamily_rows_rung537_receipt.json"
ENC = tiktoken.get_encoding("gpt2")

SPLITS = {
    "FIT": {
        "count": 48,
        "prefixes": ["The editor", "A teacher", "My neighbor", "The guide", "Our friend", "A reporter"],
        "words": ["lantern", "river", "garden", "window", "basket", "forest", "letter", "market", "planet", "harbor", "camera", "bridge", "pencil", "meadow", "ticket", "castle", "singer", "valley"],
    },
    "SELECT": {
        "count": 16,
        "prefixes": ["The curator", "One visitor", "The captain", "A student"],
        "words": ["anchor", "museum", "island", "journal", "violin", "kitchen", "tunnel", "blanket"],
    },
    "FINAL_TEST": {
        "count": 16,
        "prefixes": ["The architect", "One witness", "The librarian", "A musician"],
        "words": ["orchard", "compass", "theater", "notebook", "fountain", "village", "gallery", "package"],
    },
    "OOD": {
        "count": 16,
        "prefixes": ["During the audit the analyst", "In appendix B the author", "The laboratory note", "A field report"],
        "words": ["spectrum", "enzyme", "matrix", "voltage", "isotope", "kernel", "vector", "circuit"],
    },
}

FAMILY_ROLES = {
    "opener_type_substitution": "interchange",
    "closed_then_reopened_type": "interchange",
    "pending_state_preserved_surface_edit": "invariance",
}


def encode(text: str) -> list[int]:
    return ENC.encode(text)


def lexical_counter(ids: list[int]) -> collections.Counter:
    punctuation = {
        encode(" (")[0], encode(" )")[0], encode(' "')[0], encode("(")[0],
        encode(")")[0], encode('"')[0],
    }
    return collections.Counter(token for token in ids if token not in punctuation)


def make_pair(split: str, index: int, family_id: str, prefix: str, words: list[str]) -> dict:
    w0, w1, w2, w3 = words
    if family_id == "opener_type_substitution":
        body = f"the {w0} was near the {w1} and the {w2}"
        base = f"{prefix} said ( {body}"
        donor = f'{prefix} said " {body}'
        base_answer, donor_answer = ")", '"'
        checks = {
            "equal_token_length": True,
            "single_token_difference": True,
            "same_lexical_token_multiset": True,
        }
    elif family_id == "closed_then_reopened_type":
        base = f'{prefix} added ( the {w0} ) and said " the {w1} was {w2}'
        donor = f'{prefix} said " the {w0} " and added ( the {w1} was {w2}'
        base_answer, donor_answer = '"', ")"
        checks = {
            "equal_token_length": True,
            "single_token_difference": False,
            "same_lexical_token_multiset": True,
        }
    elif family_id == "pending_state_preserved_surface_edit":
        base = f"{prefix} described ( the {w0} beside the {w1}"
        donor = f"{prefix} carefully described ( the {w2} beyond the {w3} and the {w1}"
        base_answer = donor_answer = ")"
        checks = {
            "equal_token_length": False,
            "single_token_difference": False,
            "same_lexical_token_multiset": False,
        }
    else:
        raise KeyError(family_id)

    base_ids, donor_ids = encode(base), encode(donor)
    actual = {
        "equal_token_length": len(base_ids) == len(donor_ids),
        "single_token_difference": (
            len(base_ids) == len(donor_ids)
            and sum(left != right for left, right in zip(base_ids, donor_ids)) == 1
        ),
        "same_lexical_token_multiset": lexical_counter(base_ids) == lexical_counter(donor_ids),
    }
    assert all(actual[key] == expected for key, expected in checks.items())
    answer_changes = base_answer != donor_answer
    role = FAMILY_ROLES[family_id]
    assert answer_changes == (role == "interchange")
    return {
        "row_id": f"{split.lower()}-{index:03d}-{family_id}",
        "group_id": f"{split.lower()}-{index:03d}",
        "split": split,
        "family_id": family_id,
        "role": role,
        "base_text": base,
        "donor_text": donor,
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer": base_answer,
        "donor_answer": donor_answer,
        "base_answer_id": encode(base_answer)[0],
        "donor_answer_id": encode(donor_answer)[0],
        "answer_changes": answer_changes,
        "proposed_variable_base": "pending_" + ("paren" if base_answer == ")" else "quote"),
        "proposed_variable_donor": "pending_" + ("paren" if donor_answer == ")" else "quote"),
        "evaluation_directions": ["base_to_donor", "donor_to_base"] if answer_changes else ["base", "donor"],
        "construction_checks": actual,
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    rows = []
    group_cursor = 0
    for split, spec in SPLITS.items():
        words, prefixes = spec["words"], spec["prefixes"]
        for local_index in range(spec["count"]):
            selected = [words[(local_index * 5 + offset * 3) % len(words)] for offset in range(4)]
            assert len(set(selected)) == 4
            prefix = prefixes[local_index % len(prefixes)]
            for family_id in FAMILY_ROLES:
                rows.append(make_pair(split, group_cursor, family_id, prefix, selected))
            group_cursor += 1

    group_to_split = {}
    text_to_split = {}
    for row in rows:
        old = group_to_split.setdefault(row["group_id"], row["split"])
        assert old == row["split"]
        for text in (row["base_text"], row["donor_text"]):
            previous = text_to_split.setdefault(text, row["split"])
            assert previous == row["split"]

    result = {
        "schema": "pending_opener_multifamily_rows_rung537_v1",
        "status": "rows_frozen_outcomes_unopened",
        "causal_variable": "pending opener type at the final prediction position",
        "families": FAMILY_ROLES,
        "split_policy": {
            "unit": "group_id shared across every counterfactual family",
            "counts": {split: spec["count"] for split, spec in SPLITS.items()},
            "lexical_pools_disjoint_across_splits": True,
            "final_test_used_for_selection": False,
            "ood_difference": "held-out technical/report prefixes and held-out lexical pool",
        },
        "row_count": len(rows),
        "group_count": len(group_to_split),
        "rows": rows,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    OUT.write_bytes(payload)
    receipt = {
        "schema": "pending_opener_multifamily_rows_rung537_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT)),
        "rows_sha256": sha256_bytes(payload),
        "row_count": len(rows),
        "group_count": len(group_to_split),
        "family_counts": dict(sorted(collections.Counter(row["family_id"] for row in rows).items())),
        "split_row_counts": dict(sorted(collections.Counter(row["split"] for row in rows).items())),
        "answer_changing_rows": sum(row["answer_changes"] for row in rows),
        "single_token_interchange_rows": sum(row["construction_checks"]["single_token_difference"] for row in rows),
        "lexical_multiset_matched_interchange_rows": sum(
            row["answer_changes"] and row["construction_checks"]["same_lexical_token_multiset"] for row in rows
        ),
        "outcomes_opened": False,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
