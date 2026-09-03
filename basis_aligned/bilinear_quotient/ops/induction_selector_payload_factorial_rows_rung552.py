#!/usr/bin/env python3
"""Build grouped induction selector x payload counterfactual rows without loading a model."""

from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
OUT = BQ / "induction_selector_payload_factorial_rows_rung552.json"
RECEIPT = BQ / "induction_selector_payload_factorial_rows_rung552_receipt.json"
PREREG = ROOT / "polynomial_causal" / "INDUCTION_SELECTOR_PAYLOAD_FACTORIAL_ROWS_RUNG552_PREREGISTRATION.md"
ENC = tiktoken.get_encoding("gpt2")
SPLITS = {
    "FIT": {
        "count": 72, "seed": 55201, "token_range": (1000, 14000),
        "prefixes": ("Items:", "Sequence:", "Record:"), "layouts": ((2, 3), (3, 4), (4, 5)),
    },
    "SELECT": {
        "count": 36, "seed": 55202, "token_range": (14000, 24000),
        "prefixes": ("Entries:", "Series:", "Sample:"), "layouts": ((2, 4), (3, 5), (4, 6)),
    },
    "FINAL_TEST": {
        "count": 36, "seed": 55203, "token_range": (24000, 34000),
        "prefixes": ("Catalog:", "Register:", "Inventory:"), "layouts": ((3, 4), (4, 5), (5, 6)),
    },
    "OOD": {
        "count": 36, "seed": 55204, "token_range": (34000, 50000),
        "prefixes": ("trace =", "cache =", "tokens ="), "layouts": ((4, 7), (5, 8), (6, 9)),
    },
}
FAMILY_ROLES = {
    "two_valid_sources_selector_swap": "interchange",
    "payload_swap_match_preserved": "interchange",
    "selector_payload_joint_answer_preserved": "invariance",
    "match_break_payload_preserved": "necessity",
    "copy_relation_preserved_nuisance_change": "invariance",
    "irrelevant_source_edit": "invariance",
}


def content_id(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_bank(low: int, high: int) -> list[tuple[int, str]]:
    values = []
    for token in range(low, high):
        piece = ENC.decode([token])
        if re.fullmatch(r" [A-Za-z]{2,12}", piece) and ENC.encode(piece) == [token]:
            values.append((token, piece))
    if len(values) < 100:
        raise RuntimeError(f"too few single-token alphabetic pieces in [{low},{high})")
    return values


def prompt(prefix: str, body: list[int], *, source0: int, source1: int, query: int) -> dict:
    prefix_ids = ENC.encode(prefix)
    ids = prefix_ids + body
    text = ENC.decode(ids)
    if ENC.encode(text) != ids:
        raise RuntimeError("prompt token sequence does not round-trip")
    return {
        "text": text,
        "ids": ids,
        "source_positions": [len(prefix_ids) + source0, len(prefix_ids) + source1],
        "payload_positions": [len(prefix_ids) + source0 + 1, len(prefix_ids) + source1 + 1],
        "query_position": len(prefix_ids) + query,
        "source_to_query_lags": [query - source0, query - source1],
    }


def condition(prefix: str, variables: dict, fillers: list[int], layout: tuple[int, int], selector: int,
              assignment: int) -> dict:
    a, c, b, d = (variables[key] for key in ("A", "C", "B", "D"))
    payload0, payload1 = ((b, d) if assignment == 0 else (d, b))
    n0, n1 = layout
    body = [a, payload0] + fillers[:n0] + [c, payload1] + fillers[n0:n0+n1]
    source0, source1 = 0, 2 + n0
    query_token = a if selector == 0 else c
    body.append(query_token)
    packed = prompt(prefix, body, source0=source0, source1=source1, query=len(body)-1)
    packed.update({
        "selector": selector,
        "payload_assignment": assignment,
        "answer_id": payload0 if selector == 0 else payload1,
        "query_id": query_token,
        "source_ids": [a, c],
        "payload_ids": [payload0, payload1],
    })
    for source_position, payload_position, source_id, payload_id in zip(
        packed["source_positions"], packed["payload_positions"], packed["source_ids"], packed["payload_ids"]
    ):
        assert packed["ids"][source_position] == source_id
        assert packed["ids"][payload_position] == payload_id
        assert payload_position == source_position + 1
    assert packed["ids"][packed["query_position"]] == query_token
    return packed


def edited_condition(cell: dict, *, position: int, replacement: int) -> dict:
    ids = list(cell["ids"])
    ids[position] = replacement
    text = ENC.decode(ids)
    if ENC.encode(text) != ids:
        raise RuntimeError("edited prompt does not round-trip")
    result = dict(cell)
    result.update({"text": text, "ids": ids})
    return result


def nuisance_condition(prefix: str, variables: dict, fillers: list[int], layout: tuple[int, int], selector: int,
                       assignment: int, *, extension: list[int] | None = None) -> dict:
    cell = condition(prefix, variables, fillers, layout, selector, assignment)
    if not extension:
        return cell
    ids = list(cell["ids"])
    query_position = cell["query_position"]
    ids[query_position:query_position] = extension
    text = ENC.decode(ids)
    if ENC.encode(text) != ids:
        raise RuntimeError("lag-extended prompt does not round-trip")
    result = dict(cell)
    result.update({
        "text": text,
        "ids": ids,
        "query_position": query_position + len(extension),
        "source_to_query_lags": [lag + len(extension) for lag in cell["source_to_query_lags"]],
    })
    assert result["ids"][result["query_position"]] == result["query_id"]
    return result


def pair_row(group_id: str, split: str, family: str, variant: str, base_id: str, donor_id: str,
             base: dict, donor: dict, changes: list[str], holds_fixed: list[str]) -> dict:
    role = FAMILY_ROLES[family]
    return {
        "row_id": content_id({"group": group_id, "family": family, "variant": variant}),
        "group_id": group_id,
        "split": split,
        "family_id": family,
        "family_variant": variant,
        "role": role,
        "base_condition_id": base_id,
        "donor_condition_id": donor_id,
        "base_text": base["text"],
        "donor_text": donor["text"],
        "base_ids": base["ids"],
        "donor_ids": donor["ids"],
        "base_answer_id": base["answer_id"],
        "donor_answer_id": donor["answer_id"],
        "base_selector": base["selector"],
        "donor_selector": donor["selector"],
        "base_payload_assignment": base["payload_assignment"],
        "donor_payload_assignment": donor["payload_assignment"],
        "changes": changes,
        "holds_fixed": holds_fixed,
        "answer_changes": base["answer_id"] != donor["answer_id"],
        "base_structure": {key: base[key] for key in (
            "source_positions", "payload_positions", "query_position", "source_to_query_lags",
            "source_ids", "payload_ids", "query_id",
        )},
        "donor_structure": {key: donor[key] for key in (
            "source_positions", "payload_positions", "query_position", "source_to_query_lags",
            "source_ids", "payload_ids", "query_id",
        )},
        "evaluation_directions": ["base_to_donor", "donor_to_base"],
    }


def make_group(split: str, index: int, spec: dict, bank: list[tuple[int, str]]) -> tuple[dict, list[dict]]:
    generator = random.Random(spec["seed"] + 1009 * index)
    chosen = generator.sample(bank, 48)
    ids = [item[0] for item in chosen]
    pieces = {item[0]: item[1] for item in chosen}
    variables = dict(zip(("A", "C", "B", "D"), ids[:4]))
    fillers, alternate, extension = ids[4:19], ids[19:34], ids[34:40]
    decoy_selected, decoy_irrelevant = ids[40:42]
    prefix = spec["prefixes"][index % len(spec["prefixes"])]
    layout = spec["layouts"][index % len(spec["layouts"])]
    coordinates = {
        "rung": 552, "split": split, "index": index, "prefix": prefix, "layout": layout,
        "variables": variables, "fillers": fillers, "alternate": alternate, "extension": extension,
    }
    group_id = content_id(coordinates)
    cells = {}
    for selector in (0, 1):
        for assignment in (0, 1):
            name = f"s{selector}p{assignment}"
            cell = condition(prefix, variables, fillers, layout, selector, assignment)
            cell["condition_id"] = content_id({"group": group_id, "condition": name})
            cells[name] = cell

    rows = []
    fixed_sources = [variables["A"], variables["C"]]
    for assignment in (0, 1):
        base_name, donor_name = f"s0p{assignment}", f"s1p{assignment}"
        rows.append(pair_row(
            group_id, split, "two_valid_sources_selector_swap", f"payload_assignment_{assignment}",
            cells[base_name]["condition_id"], cells[donor_name]["condition_id"],
            cells[base_name], cells[donor_name],
            ["final query token", "selected earlier source", "correct payload answer"],
            ["both source tokens", "both payload tokens", "payload assignment", "positions", "filler"],
        ))
    for selector in (0, 1):
        base_name, donor_name = f"s{selector}p0", f"s{selector}p1"
        rows.append(pair_row(
            group_id, split, "payload_swap_match_preserved", f"selector_{selector}",
            cells[base_name]["condition_id"], cells[donor_name]["condition_id"],
            cells[base_name], cells[donor_name],
            ["which payload immediately follows each fixed source", "correct payload answer"],
            ["query token", "source tokens", "source/query equality", "positions", "filler"],
        ))
    for answer_name, left, right in (("payload_B", "s0p0", "s1p1"), ("payload_D", "s1p0", "s0p1")):
        rows.append(pair_row(
            group_id, split, "selector_payload_joint_answer_preserved", answer_name,
            cells[left]["condition_id"], cells[right]["condition_id"], cells[left], cells[right],
            ["selector", "payload assignment"],
            ["correct answer", "source-token set", "payload-token set", "positions", "filler"],
        ))

    control_selector = index % 2
    control_assignment = (index // 2) % 2
    control_name = f"s{control_selector}p{control_assignment}"
    control = cells[control_name]
    selected_source_slot = control_selector
    irrelevant_source_slot = 1 - control_selector
    broken = edited_condition(
        control, position=control["source_positions"][selected_source_slot], replacement=decoy_selected,
    )
    broken["source_ids"] = list(control["source_ids"])
    broken["source_ids"][selected_source_slot] = decoy_selected
    rows.append(pair_row(
        group_id, split, "match_break_payload_preserved", f"s{control_selector}p{control_assignment}",
        control["condition_id"], content_id({"group": group_id, "edit": "match_break"}), control, broken,
        ["earlier selected source token", "availability of the selected equality edge"],
        ["final query", "original payload token and answer", "unselected source pair", "length", "positions"],
    ))

    irrelevant = edited_condition(
        control, position=control["source_positions"][irrelevant_source_slot], replacement=decoy_irrelevant,
    )
    irrelevant["source_ids"] = list(control["source_ids"])
    irrelevant["source_ids"][irrelevant_source_slot] = decoy_irrelevant
    rows.append(pair_row(
        group_id, split, "irrelevant_source_edit", f"s{control_selector}p{control_assignment}",
        control["condition_id"], content_id({"group": group_id, "edit": "irrelevant_source"}),
        control, irrelevant,
        ["unselected earlier source token"],
        ["selected source/query equality", "selected payload and answer", "length", "positions"],
    ))

    filler_variant = nuisance_condition(
        prefix, variables, alternate, layout, control_selector, control_assignment,
    )
    rows.append(pair_row(
        group_id, split, "copy_relation_preserved_nuisance_change", "filler_change",
        control["condition_id"], content_id({"group": group_id, "edit": "filler"}), control, filler_variant,
        ["filler token identities"],
        ["both source/query relations", "selector", "payload assignment", "correct answer", "positions"],
    ))
    lag_variant = nuisance_condition(
        prefix, variables, fillers, layout, control_selector, control_assignment, extension=extension,
    )
    rows.append(pair_row(
        group_id, split, "copy_relation_preserved_nuisance_change", "lag_extension",
        control["condition_id"], content_id({"group": group_id, "edit": "lag"}), control, lag_variant,
        ["query lag", "additional filler before query", "sequence length"],
        ["both source/query relations", "selector", "payload assignment", "correct answer"],
    ))

    assert len(rows) == 10
    assert all(row["answer_changes"] for row in rows[:4])
    assert all(not row["answer_changes"] for row in rows[4:])
    group = {
        "group_id": group_id,
        "split": split,
        "prefix": prefix,
        "layout": list(layout),
        "variable_token_ids": variables,
        "variable_token_text": {key: pieces[value] for key, value in variables.items()},
        "all_sampled_token_ids": ids,
        "factorial_conditions": cells,
        "control_condition": control_name,
        "within_group_condition_reuse_is_intentional": True,
    }
    assert cells["s0p0"]["answer_id"] == cells["s1p1"]["answer_id"] == variables["B"]
    assert cells["s1p0"]["answer_id"] == cells["s0p1"]["answer_id"] == variables["D"]
    assert cells["s0p0"]["source_ids"] == fixed_sources
    return group, rows


def main() -> None:
    all_groups, all_rows, split_token_ids = [], [], {}
    for split, spec in SPLITS.items():
        bank = token_bank(*spec["token_range"])
        split_token_ids[split] = {token for token, _piece in bank}
        seen_group_ids = set()
        for index in range(spec["count"]):
            group, rows = make_group(split, index, spec, bank)
            if group["group_id"] in seen_group_ids:
                raise RuntimeError("duplicate semantic group")
            seen_group_ids.add(group["group_id"])
            all_groups.append(group)
            all_rows.extend(rows)

    for left_index, left in enumerate(SPLITS):
        for right in tuple(SPLITS)[left_index + 1:]:
            if split_token_ids[left] & split_token_ids[right]:
                raise RuntimeError("variable token banks overlap across splits")
    row_ids = [row["row_id"] for row in all_rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("duplicate row identity")
    sequence_owner = collections.defaultdict(set)
    for row in all_rows:
        sequence_owner[tuple(row["base_ids"])].add(row["group_id"])
        sequence_owner[tuple(row["donor_ids"])].add(row["group_id"])
    if any(len(owners) != 1 for owners in sequence_owner.values()):
        raise RuntimeError("a prompt sequence is reused across semantic groups")
    group_splits = collections.defaultdict(set)
    group_family_counts = collections.defaultdict(collections.Counter)
    for row in all_rows:
        group_splits[row["group_id"]].add(row["split"])
        group_family_counts[row["group_id"]][row["family_id"]] += 1
    expected_per_group = {
        "two_valid_sources_selector_swap": 2,
        "payload_swap_match_preserved": 2,
        "selector_payload_joint_answer_preserved": 2,
        "match_break_payload_preserved": 1,
        "copy_relation_preserved_nuisance_change": 2,
        "irrelevant_source_edit": 1,
    }
    if not all(counts == expected_per_group for counts in group_family_counts.values()):
        raise RuntimeError("a group does not contain the complete factorial/control family")
    if not all(len(splits) == 1 for splits in group_splits.values()):
        raise RuntimeError("a factorial group crosses splits")

    payload = {
        "schema": "induction_selector_payload_factorial_rows_rung552_v1",
        "status": "rows_frozen_outcomes_unopened",
        "causal_variables": {
            "selector": "which of two earlier source tokens is repeated as the final query",
            "payload_assignment": "which of two payload tokens immediately follows each fixed source",
        },
        "factorial_answer_rule": "answer(S,P) is B when S==P and D otherwise",
        "family_roles": FAMILY_ROLES,
        "split_policy": {
            "unit": "semantic factorial group containing all conditions and derived counterfactuals",
            "group_counts": {split: spec["count"] for split, spec in SPLITS.items()},
            "token_id_ranges": {split: list(spec["token_range"]) for split, spec in SPLITS.items()},
            "variable_token_banks_disjoint_across_splits": True,
            "prefix_templates_disjoint_across_splits": True,
            "all_derived_pairs_stay_with_group": True,
            "sequence_reuse_allowed_only_within_group": True,
            "final_test_used_for_selection": False,
        },
        "groups": all_groups,
        "rows": all_rows,
        "group_count": len(all_groups),
        "row_count": len(all_rows),
        "unique_prompt_sequence_count": len(sequence_owner),
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }
    encoded = (json.dumps(payload, indent=1) + "\n").encode()
    OUT.write_bytes(encoded)
    receipt = {
        "schema": "induction_selector_payload_factorial_rows_rung552_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT.parent)),
        "rows_sha256": hashlib.sha256(encoded).hexdigest(),
        "preregistration_sha256": sha256(PREREG),
        "group_count": len(all_groups),
        "row_count": len(all_rows),
        "factorial_condition_count": 4 * len(all_groups),
        "unique_prompt_sequence_count": len(sequence_owner),
        "split_group_counts": dict(collections.Counter(group["split"] for group in all_groups)),
        "split_row_counts": dict(collections.Counter(row["split"] for row in all_rows)),
        "family_row_counts": dict(collections.Counter(row["family_id"] for row in all_rows)),
        "every_group_has_complete_factorial_and_controls": True,
        "every_group_belongs_to_one_split": True,
        "prompt_sequences_never_cross_groups": True,
        "variable_token_banks_disjoint_across_splits": True,
        "within_group_condition_reuse_declared": True,
        "tokenizer": "tiktoken:gpt2",
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
