#!/usr/bin/env python3
"""R557: CPU-only exact semantics check for induction score/payload swaps."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
ROWS = ROOT / "induction_selector_payload_factorial_rows_rung552.json"
PREREG = POLY / "INDUCTION_FACTOR_INTERVENTION_SEMANTICS_RUNG557_PREREGISTRATION.md"
OUT = ROOT / "induction_factor_intervention_semantics_rung557_results.json"
HASHES = {
    ROWS: "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460",
    PREREG: "7292716ea21401830ce4fd523da01d5e2923cc16ac6d8db48c0abf1dc1207042",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equality_score(ids: list[int]) -> tuple[int, ...]:
    query = ids[-1]
    return tuple(1 if position >= 1 and ids[position - 1] == query else 0
                 for position in range(len(ids)))


def payload_values(ids: list[int]) -> tuple[int, ...]:
    return tuple(ids)


def fetch(score: tuple[int, ...], payload: tuple[int, ...]) -> int:
    if len(score) != len(payload):
        raise AssertionError("score and payload positions are not aligned")
    output: dict[int, int] = defaultdict(int)
    for weight, token in zip(score, payload, strict=True):
        output[token] += weight
    positive = [(value, token) for token, value in output.items() if value > 0]
    if len(positive) != 1 or positive[0][0] != 1:
        raise AssertionError(f"planted fetch is not one-hot: {positive}")
    return positive[0][1]


def factors(ids: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return equality_score(ids), payload_values(ids)


def oriented(row: dict):
    yield row["base_ids"], row["donor_ids"], row["donor_answer_id"]
    yield row["donor_ids"], row["base_ids"], row["base_answer_id"]


def main() -> None:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    document = json.loads(ROWS.read_text())
    assert document["model_loaded"] is False and document["outcomes_opened"] == []
    groups, rows = document["groups"], document["rows"]
    assert len(groups) == 180 and len(rows) == 1800

    factorial_checks = 0
    for group in groups:
        assert len(group["factorial_conditions"]) == 4
        for condition in group["factorial_conditions"].values():
            score, payload = factors(condition["ids"])
            assert fetch(score, payload) == condition["answer_id"]
            factorial_checks += 1

    family_rows = defaultdict(int)
    direction_checks = defaultdict(int)
    for row in rows:
        family = row["family_id"]
        family_rows[family] += 1
        if family == "two_valid_sources_selector_swap":
            for target_ids, donor_ids, donor_answer in oriented(row):
                target_score, target_payload = factors(target_ids)
                donor_score, _ = factors(donor_ids)
                assert target_score != donor_score
                assert fetch(donor_score, target_payload) == donor_answer
                direction_checks[family] += 1
        elif family == "payload_swap_match_preserved":
            for target_ids, donor_ids, donor_answer in oriented(row):
                target_score, _ = factors(target_ids)
                donor_score, donor_payload = factors(donor_ids)
                assert target_score == donor_score
                assert fetch(target_score, donor_payload) == donor_answer
                direction_checks[family] += 1
        elif family == "selector_payload_joint_answer_preserved":
            for target_ids, donor_ids, donor_answer in oriented(row):
                target_score, target_payload = factors(target_ids)
                donor_score, donor_payload = factors(donor_ids)
                assert target_score != donor_score and target_payload != donor_payload
                assert fetch(donor_score, donor_payload) == donor_answer
                direction_checks[family] += 1
        elif family == "match_break_payload_preserved":
            base_score, _ = factors(row["base_ids"])
            donor_score, donor_payload = factors(row["donor_ids"])
            assert sum(base_score) == 1 and sum(donor_score) == 0
            assert fetch(base_score, donor_payload) == row["donor_answer_id"]
            direction_checks[family] += 1
        elif family == "irrelevant_source_edit":
            base_score, _ = factors(row["base_ids"])
            donor_score, _ = factors(row["donor_ids"])
            assert base_score == donor_score
            direction_checks[family] += 1

    expected_family_rows = {
        "two_valid_sources_selector_swap": 360,
        "payload_swap_match_preserved": 360,
        "selector_payload_joint_answer_preserved": 360,
        "match_break_payload_preserved": 180,
        "irrelevant_source_edit": 180,
        "copy_relation_preserved_nuisance_change": 360,
    }
    assert dict(family_rows) == expected_family_rows
    assert factorial_checks == 720
    assert direction_checks == {
        "two_valid_sources_selector_swap": 720,
        "payload_swap_match_preserved": 720,
        "selector_payload_joint_answer_preserved": 720,
        "match_break_payload_preserved": 180,
        "irrelevant_source_edit": 180,
    }
    result = {
        "rung": 557,
        "stage": "induction_factor_intervention_semantics",
        "pred_a_factorial_fetch_exact": True,
        "pred_b_selector_score_transplant_exact": True,
        "pred_c_payload_value_transplant_exact": True,
        "pred_d_joint_transplant_exact": True,
        "pred_e_match_break_score_restore_exact": True,
        "pred_f_irrelevant_source_score_invariant": True,
        "all_checks_pass": True,
        "group_count": len(groups),
        "row_count": len(rows),
        "factorial_condition_checks": factorial_checks,
        "direction_checks": dict(direction_checks),
        "family_row_counts": dict(family_rows),
        "input_sha256": {str(path): sha256(path) for path in HASHES},
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
        "decision": (
            "The discrete score-versus-payload intervention is semantically valid for a separately preregistered "
            "bilin18 factor-site screen, conditional on R554 native capability."
        ),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
