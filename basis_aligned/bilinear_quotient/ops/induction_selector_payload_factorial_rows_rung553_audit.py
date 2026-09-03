#!/usr/bin/env python3
"""Independent CPU audit of R552's token-level selector x payload factorial rows."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "induction_selector_payload_factorial_rows_rung552.json"
RECEIPT = ROOT / "induction_selector_payload_factorial_rows_rung552_receipt.json"
PREREG = ROOT.parent / "polynomial_causal" / "INDUCTION_SELECTOR_PAYLOAD_FACTORIAL_ROWS_RUNG552_PREREGISTRATION.md"
OUT = ROOT / "induction_selector_payload_factorial_rows_rung553_audit.json"
ENC = tiktoken.get_encoding("gpt2")
EXPECTED_GROUPS = {"FIT": 72, "SELECT": 36, "FINAL_TEST": 36, "OOD": 36}
EXPECTED_FAMILIES = {
    "two_valid_sources_selector_swap": 2,
    "payload_swap_match_preserved": 2,
    "selector_payload_joint_answer_preserved": 2,
    "match_break_payload_preserved": 1,
    "copy_relation_preserved_nuisance_change": 2,
    "irrelevant_source_edit": 1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_structure(prompt_ids: list[int], structure: dict, *, expect_selected_match: bool = True) -> None:
    source_positions = structure["source_positions"]
    payload_positions = structure["payload_positions"]
    source_ids = structure["source_ids"]
    payload_ids = structure["payload_ids"]
    selector = 0 if structure["query_id"] == source_ids[0] else 1
    if structure["query_id"] not in source_ids:
        raise AssertionError("query is not either declared source")
    for source_position, payload_position, source_id, payload_id in zip(
        source_positions, payload_positions, source_ids, payload_ids
    ):
        if payload_position != source_position + 1:
            raise AssertionError("payload is not immediately after source")
        if prompt_ids[source_position] != source_id or prompt_ids[payload_position] != payload_id:
            raise AssertionError("declared source/payload token does not match prompt")
    if prompt_ids[structure["query_position"]] != structure["query_id"]:
        raise AssertionError("query position does not hold query token")
    if expect_selected_match and prompt_ids.count(structure["query_id"]) != 2:
        raise AssertionError("selected source must occur exactly once before its repeated query")
    if expect_selected_match and prompt_ids.count(source_ids[1 - selector]) != 1:
        raise AssertionError("unselected source must occur exactly once")
    if any(prompt_ids.count(payload_id) != 1 for payload_id in payload_ids):
        raise AssertionError("payload token appears outside its one declared follower position")


def verify_factorial_group(group: dict) -> None:
    cells = group["factorial_conditions"]
    if set(cells) != {"s0p0", "s1p0", "s0p1", "s1p1"}:
        raise AssertionError("factorial group does not have exactly four cells")
    variables = group["variable_token_ids"]
    if len(set(variables.values())) != 4:
        raise AssertionError("A, B, C, D are not distinct tokens")
    condition_ids = set()
    for name, cell in cells.items():
        if cell["condition_id"] in condition_ids:
            raise AssertionError("condition ID repeats")
        condition_ids.add(cell["condition_id"])
        if ENC.encode(cell["text"]) != cell["ids"]:
            raise AssertionError("condition text does not round-trip")
        check_structure(cell["ids"], cell)
        selector, assignment = int(name[1]), int(name[3])
        if cell["selector"] != selector or cell["payload_assignment"] != assignment:
            raise AssertionError("condition name and factors disagree")
        observed_layout = [
            cell["source_positions"][1] - cell["source_positions"][0] - 2,
            cell["query_position"] - cell["source_positions"][1] - 2,
        ]
        if observed_layout != group["layout"]:
            raise AssertionError("declared filler layout does not match token positions")
        expected_answer = variables["B"] if selector == assignment else variables["D"]
        if cell["answer_id"] != expected_answer:
            raise AssertionError("factorial XOR answer rule failed")
        selected_position = cell["payload_positions"][selector]
        if cell["ids"][selected_position] != cell["answer_id"]:
            raise AssertionError("answer is not the follower of the selected source")


def only_differences(left: list[int], right: list[int]) -> list[int]:
    if len(left) != len(right):
        raise AssertionError("equal-length counterfactual expected")
    return [index for index, (a, b) in enumerate(zip(left, right)) if a != b]


def verify_row(row: dict, group: dict) -> None:
    if row["split"] != group["split"]:
        raise AssertionError("row crosses its group split")
    if ENC.encode(row["base_text"]) != row["base_ids"] or ENC.encode(row["donor_text"]) != row["donor_ids"]:
        raise AssertionError("row text does not round-trip")
    family = row["family_id"]
    base, donor = row["base_structure"], row["donor_structure"]
    if family == "two_valid_sources_selector_swap":
        if row["base_selector"] == row["donor_selector"]:
            raise AssertionError("selector edge did not change selector")
        if row["base_payload_assignment"] != row["donor_payload_assignment"] or not row["answer_changes"]:
            raise AssertionError("selector edge changed payload assignment or preserved answer")
        differences = only_differences(row["base_ids"], row["donor_ids"])
        if differences != [base["query_position"]]:
            raise AssertionError("selector edge changes more than the final query token")
    elif family == "payload_swap_match_preserved":
        if row["base_selector"] != row["donor_selector"]:
            raise AssertionError("payload edge changed selector")
        if row["base_payload_assignment"] == row["donor_payload_assignment"] or not row["answer_changes"]:
            raise AssertionError("payload edge failed to change assignment and answer")
        differences = only_differences(row["base_ids"], row["donor_ids"])
        if differences != base["payload_positions"]:
            raise AssertionError("payload swap changes tokens outside the two follower positions")
    elif family == "selector_payload_joint_answer_preserved":
        if row["base_selector"] == row["donor_selector"]:
            raise AssertionError("joint diagonal did not change selector")
        if row["base_payload_assignment"] == row["donor_payload_assignment"]:
            raise AssertionError("joint diagonal did not change payload assignment")
        if row["answer_changes"] or row["base_answer_id"] != row["donor_answer_id"]:
            raise AssertionError("joint diagonal did not preserve the answer")
    elif family == "match_break_payload_preserved":
        if row["answer_changes"] or row["base_answer_id"] != row["donor_answer_id"]:
            raise AssertionError("necessity edit changed the registered answer")
        differences = only_differences(row["base_ids"], row["donor_ids"])
        selected = row["base_selector"]
        if differences != [base["source_positions"][selected]]:
            raise AssertionError("match break changes anything except the selected earlier source")
        selected_id = base["query_id"]
        if row["donor_ids"].count(selected_id) != 1:
            raise AssertionError("match break left an extra selected-source occurrence")
        check_structure(row["base_ids"], base)
        # The donor deliberately has no selected equality edge; validate the unchanged query/payload separately.
        if row["donor_ids"][donor["query_position"]] != selected_id:
            raise AssertionError("match break changed the query")
        if row["donor_ids"][donor["payload_positions"][selected]] != row["donor_answer_id"]:
            raise AssertionError("match break changed the original payload token")
        return
    elif family == "irrelevant_source_edit":
        if row["answer_changes"]:
            raise AssertionError("irrelevant-source edit changed answer")
        differences = only_differences(row["base_ids"], row["donor_ids"])
        unselected = 1 - row["base_selector"]
        if differences != [base["source_positions"][unselected]]:
            raise AssertionError("irrelevant-source edit changes the selected relation or other token")
    elif family == "copy_relation_preserved_nuisance_change":
        if row["answer_changes"]:
            raise AssertionError("nuisance edit changed answer")
        for key in ("source_ids", "payload_ids", "query_id"):
            if base[key] != donor[key]:
                raise AssertionError(f"nuisance edit changed {key}")
        if row["family_variant"] == "filler_change":
            if len(row["base_ids"]) != len(row["donor_ids"]):
                raise AssertionError("filler substitution changed length")
            protected = set(base["source_positions"] + base["payload_positions"] + [base["query_position"]])
            if any(index in protected for index in only_differences(row["base_ids"], row["donor_ids"])):
                raise AssertionError("filler substitution changed source, payload, or query")
        elif row["family_variant"] == "lag_extension":
            if len(row["donor_ids"]) <= len(row["base_ids"]):
                raise AssertionError("lag extension did not increase length")
            if donor["source_positions"] != base["source_positions"] or donor["payload_positions"] != base["payload_positions"]:
                raise AssertionError("lag extension moved either earlier source pair")
            if donor["query_position"] <= base["query_position"]:
                raise AssertionError("lag extension did not move query later")
        else:
            raise AssertionError("unknown nuisance variant")
    else:
        raise AssertionError(f"unknown family {family}")
    check_structure(row["base_ids"], base)
    check_structure(row["donor_ids"], donor)


def main() -> None:
    document = json.loads(ROWS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    if receipt["rows_sha256"] != sha256(ROWS) or receipt["preregistration_sha256"] != sha256(PREREG):
        raise AssertionError("receipt does not bind rows and preregistration")
    if document["model_loaded"] is not False or document["outcomes_opened"] != []:
        raise AssertionError("dataset construction was not outcome blind")
    if document["group_count"] != 180 or document["row_count"] != 1800:
        raise AssertionError("global group/row count changed")

    groups = {group["group_id"]: group for group in document["groups"]}
    if len(groups) != 180:
        raise AssertionError("group IDs are not unique")
    all_condition_ids = set()
    actual_variable_tokens = collections.defaultdict(set)
    for group in groups.values():
        verify_factorial_group(group)
        condition_ids = {cell["condition_id"] for cell in group["factorial_conditions"].values()}
        if all_condition_ids & condition_ids:
            raise AssertionError("condition ID crosses semantic groups")
        all_condition_ids |= condition_ids
        actual_variable_tokens[group["split"]].update(group["variable_token_ids"].values())
    for left_index, left in enumerate(EXPECTED_GROUPS):
        for right in tuple(EXPECTED_GROUPS)[left_index + 1:]:
            if actual_variable_tokens[left] & actual_variable_tokens[right]:
                raise AssertionError("actual variable token identity crosses splits")

    counts = collections.defaultdict(collections.Counter)
    row_ids, sequence_owners, pair_keys = [], collections.defaultdict(set), set()
    for row in document["rows"]:
        if row["group_id"] not in groups:
            raise AssertionError("row references missing group")
        verify_row(row, groups[row["group_id"]])
        counts[row["group_id"]][row["family_id"]] += 1
        row_ids.append(row["row_id"])
        sequence_owners[tuple(row["base_ids"])].add(row["group_id"])
        sequence_owners[tuple(row["donor_ids"])].add(row["group_id"])
        key = (tuple(row["base_ids"]), tuple(row["donor_ids"]), row["base_answer_id"], row["donor_answer_id"])
        if key in pair_keys:
            raise AssertionError("exact prompt/answer pair repeats")
        pair_keys.add(key)
    if len(row_ids) != len(set(row_ids)):
        raise AssertionError("row IDs repeat")
    if not all(value == EXPECTED_FAMILIES for value in counts.values()):
        raise AssertionError("a group lacks part of its factorial/control family")
    if any(len(owners) != 1 for owners in sequence_owners.values()):
        raise AssertionError("prompt sequence crosses semantic groups")
    split_counts = collections.Counter(group["split"] for group in groups.values())
    if dict(split_counts) != EXPECTED_GROUPS:
        raise AssertionError("split group counts changed")

    reuse_counts = collections.Counter()
    for row in document["rows"]:
        reuse_counts[(row["group_id"], tuple(row["base_ids"]))] += 1
        reuse_counts[(row["group_id"], tuple(row["donor_ids"]))] += 1
    audit = {
        "rung": 553,
        "audited_rung": 552,
        "status": "terminal_audit_complete",
        "rows_sha256": sha256(ROWS),
        "receipt_sha256": sha256(RECEIPT),
        "preregistration_sha256": sha256(PREREG),
        "group_count": len(groups),
        "row_count": len(document["rows"]),
        "factorial_condition_count": len(all_condition_ids),
        "unique_prompt_sequence_count": len(sequence_owners),
        "maximum_declared_within_group_sequence_reuse": max(reuse_counts.values()),
        "all_token_level_factorial_checks_pass": True,
        "no_accidental_selected_or_unselected_source_matches": True,
        "payloads_are_immediate_and_unique": True,
        "single_factor_and_joint_diagonal_semantics_exact": True,
        "necessity_and_invariance_edits_exact": True,
        "actual_variable_tokens_disjoint_across_splits": True,
        "prompt_sequences_never_cross_groups": True,
        "exact_prompt_answer_pairs_unique": True,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
        "decision": (
            "R552 is authorized for a separately preregistered FIT/SELECT native-capability and complete-state "
            "selector/payload site screen. FINAL_TEST/OOD remain unopened."
        ),
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
