#!/usr/bin/env python3
"""Build outcome-blind selector x payload rows with a neutral third pair.

The module is import-safe and CPU-only.  It loads no model and reads no prior
model outcome while constructing rows.  See the frozen rung-578 preregistration
for the behavioral decision licensed by this artifact.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Mapping

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
OUT = BQ / "induction_selector_payload_three_source_rows_rung578.json"
RECEIPT = BQ / "induction_selector_payload_three_source_rows_rung578_receipt.json"
PREREG = (
    ROOT
    / "polynomial_causal"
    / "INDUCTION_SELECTOR_PAYLOAD_THREE_SOURCE_ROWS_RUNG578_PREREGISTRATION.md"
)
ENC = tiktoken.get_encoding("gpt2")
TOKENS_PER_GROUP = 68
SPLITS = {
    "FIT": {
        "count": 72,
        "seed": 57801,
        "token_range": (1000, 14000),
        "prefixes": ("Mapping:", "Pairs:", "Associations:"),
        "layouts": ((2, 3, 4), (3, 4, 2), (4, 2, 3)),
    },
    "SELECT": {
        "count": 36,
        "seed": 57802,
        "token_range": (14000, 24000),
        "prefixes": ("Lookup:", "Links:", "Relations:"),
        "layouts": ((3, 4, 5), (4, 5, 3), (5, 3, 4)),
    },
    "FINAL_TEST": {
        "count": 36,
        "seed": 57803,
        "token_range": (24000, 34000),
        "prefixes": ("Index:", "Table:", "Directory:"),
        "layouts": ((4, 5, 6), (5, 6, 4), (6, 4, 5)),
    },
    "OOD": {
        "count": 36,
        "seed": 57804,
        "token_range": (34000, 50000),
        "prefixes": ("trace_map =", "cache_links =", "symbol_table ="),
        "layouts": ((5, 7, 9), (7, 9, 5), (9, 5, 7)),
    },
}
FAMILY_ROLES = {
    "two_valid_sources_selector_swap": "interchange",
    "payload_swap_match_preserved": "interchange",
    "selector_payload_joint_answer_preserved": "interaction_invariance",
    "match_break_payload_preserved": "necessity",
    "irrelevant_source_edit": "endpoint_neutral_control",
    "irrelevant_payload_edit": "endpoint_neutral_control",
    "contrast_target_source_edit": "competition_diagnostic",
    "copy_relation_preserved_nuisance_change": "invariance",
}
EXPECTED_PER_GROUP = {
    "two_valid_sources_selector_swap": 2,
    "payload_swap_match_preserved": 2,
    "selector_payload_joint_answer_preserved": 2,
    "match_break_payload_preserved": 4,
    "irrelevant_source_edit": 4,
    "irrelevant_payload_edit": 4,
    "contrast_target_source_edit": 4,
    "copy_relation_preserved_nuisance_change": 8,
}


def content_id(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_bank(low: int, high: int) -> list[tuple[int, str]]:
    values = []
    for token in range(low, high):
        piece = ENC.decode([token])
        if re.fullmatch(r" [A-Za-z]{2,12}", piece) and ENC.encode(piece) == [token]:
            values.append((token, piece))
    return values


def _packed_prompt(prefix: str, body: list[int], positions: Mapping[str, int]) -> dict:
    prefix_ids = ENC.encode(prefix)
    ids = prefix_ids + body
    text = ENC.decode(ids)
    if ENC.encode(text) != ids:
        raise RuntimeError("prompt token sequence does not round-trip")
    absolute = {name: len(prefix_ids) + index for name, index in positions.items()}
    return {"text": text, "ids": ids, **absolute}


def condition(
    prefix: str,
    variables: Mapping[str, int],
    fillers: list[int],
    layout: tuple[int, int, int],
    pair_order: tuple[str, str, str],
    selector: int,
    assignment: int,
) -> dict:
    payload_by_role = {
        "A": variables["B"] if assignment == 0 else variables["D"],
        "C": variables["D"] if assignment == 0 else variables["B"],
        "N": variables["E"],
    }
    source_by_role = {"A": variables["A"], "C": variables["C"], "N": variables["X"]}
    body: list[int] = []
    positions: dict[str, int] = {}
    cursor = 0
    for slot, role in enumerate(pair_order):
        positions[f"{role}_source_position"] = len(body)
        body.extend((source_by_role[role], payload_by_role[role]))
        positions[f"{role}_payload_position"] = len(body) - 1
        width = layout[slot]
        body.extend(fillers[cursor : cursor + width])
        cursor += width
    query_role = "A" if selector == 0 else "C"
    positions["query_position"] = len(body)
    body.append(source_by_role[query_role])
    packed = _packed_prompt(prefix, body, positions)
    target_sources = [variables["A"], variables["C"]]
    target_payloads = [payload_by_role["A"], payload_by_role["C"]]
    target_source_positions = [packed["A_source_position"], packed["C_source_position"]]
    target_payload_positions = [packed["A_payload_position"], packed["C_payload_position"]]
    packed.update(
        {
            "selector": selector,
            "payload_assignment": assignment,
            "answer_id": target_payloads[selector],
            "other_answer_id": target_payloads[1 - selector],
            "query_id": target_sources[selector],
            "source_ids": target_sources,
            "payload_ids": target_payloads,
            "source_positions": target_source_positions,
            "payload_positions": target_payload_positions,
            "source_to_query_lags": [
                packed["query_position"] - position for position in target_source_positions
            ],
            "neutral_source_id": variables["X"],
            "neutral_payload_id": variables["E"],
            "neutral_source_to_query_lag": (
                packed["query_position"] - packed["N_source_position"]
            ),
            "pair_order": list(pair_order),
        }
    )
    return packed


def edited_condition(
    cell: Mapping[str, object], *, position: int, replacement: int, role: str
) -> dict:
    ids = list(cell["ids"])
    if ids[position] == replacement:
        raise RuntimeError("registered edit is a no-op")
    ids[position] = replacement
    text = ENC.decode(ids)
    if ENC.encode(text) != ids:
        raise RuntimeError("edited prompt does not round-trip")
    result = dict(cell)
    result.update(
        {
            "text": text,
            "ids": ids,
            "edit_position": position,
            "edit_replacement_id": replacement,
            "edit_role": role,
        }
    )
    if role == "selected_target_source":
        source_ids = list(result["source_ids"])
        source_ids[int(result["selector"])] = replacement
        result["source_ids"] = source_ids
    elif role == "contrast_target_source":
        source_ids = list(result["source_ids"])
        source_ids[1 - int(result["selector"])] = replacement
        result["source_ids"] = source_ids
    elif role == "neutral_source":
        result["neutral_source_id"] = replacement
    elif role == "neutral_payload":
        result["neutral_payload_id"] = replacement
    else:
        raise ValueError(f"unknown edit role {role!r}")
    return result


def lag_condition(cell: Mapping[str, object], extension: list[int]) -> dict:
    ids = list(cell["ids"])
    query_position = int(cell["query_position"])
    ids[query_position:query_position] = extension
    text = ENC.decode(ids)
    if ENC.encode(text) != ids:
        raise RuntimeError("lag extension does not round-trip")
    result = dict(cell)
    result.update(
        {
            "text": text,
            "ids": ids,
            "query_position": query_position + len(extension),
            "source_to_query_lags": [
                int(lag) + len(extension) for lag in cell["source_to_query_lags"]
            ],
            "neutral_source_to_query_lag": (
                int(cell["neutral_source_to_query_lag"]) + len(extension)
            ),
        }
    )
    return result


def pair_row(
    group_id: str,
    split: str,
    family: str,
    variant: str,
    base_id: str,
    donor_id: str,
    base: Mapping[str, object],
    donor: Mapping[str, object],
    changes: list[str],
    holds_fixed: list[str],
) -> dict:
    structure_keys = (
        "source_positions",
        "payload_positions",
        "query_position",
        "source_to_query_lags",
        "source_ids",
        "payload_ids",
        "query_id",
        "neutral_source_id",
        "neutral_payload_id",
        "N_source_position",
        "N_payload_position",
        "neutral_source_to_query_lag",
        "pair_order",
    )
    return {
        "row_id": content_id({"group": group_id, "family": family, "variant": variant}),
        "group_id": group_id,
        "split": split,
        "family_id": family,
        "family_variant": variant,
        "role": FAMILY_ROLES[family],
        "base_condition_id": base_id,
        "donor_condition_id": donor_id,
        "base_text": base["text"],
        "donor_text": donor["text"],
        "base_ids": base["ids"],
        "donor_ids": donor["ids"],
        "base_answer_id": base["answer_id"],
        "donor_answer_id": donor["answer_id"],
        "base_other_answer_id": base["other_answer_id"],
        "donor_other_answer_id": donor["other_answer_id"],
        "base_selector": base["selector"],
        "donor_selector": donor["selector"],
        "base_payload_assignment": base["payload_assignment"],
        "donor_payload_assignment": donor["payload_assignment"],
        "changes": changes,
        "holds_fixed": holds_fixed,
        "answer_changes": base["answer_id"] != donor["answer_id"],
        "base_structure": {key: base[key] for key in structure_keys},
        "donor_structure": {key: donor[key] for key in structure_keys},
        "edit_role": donor.get("edit_role"),
        "edit_position": donor.get("edit_position"),
        "evaluation_directions": ["base_to_donor", "donor_to_base"],
    }


def make_group(
    split: str,
    index: int,
    spec: Mapping[str, object],
    chosen: list[tuple[int, str]],
) -> tuple[dict, list[dict]]:
    ids = [item[0] for item in chosen]
    pieces = {item[0]: item[1] for item in chosen}
    variables = dict(zip(("A", "C", "B", "D", "X", "E"), ids[:6], strict=True))
    fillers = ids[6:30]
    alternate = ids[30:54]
    extension = ids[54:62]
    decoys = dict(zip(("selected", "neutral_source", "neutral_payload", "contrast"), ids[62:66], strict=True))
    prefix = spec["prefixes"][index % len(spec["prefixes"])]
    layout = spec["layouts"][index % len(spec["layouts"])]
    base_order = ("A", "C", "N")
    rotation = index % 3
    pair_order = base_order[rotation:] + base_order[:rotation]
    coordinates = {
        "rung": 578,
        "split": split,
        "index": index,
        "prefix": prefix,
        "layout": layout,
        "pair_order": pair_order,
        "variables": variables,
        "sampled_token_ids": ids,
    }
    group_id = content_id(coordinates)
    cells = {}
    for selector in (0, 1):
        for assignment in (0, 1):
            name = f"s{selector}p{assignment}"
            cell = condition(
                prefix, variables, fillers, layout, pair_order, selector, assignment
            )
            cell["condition_id"] = content_id({"group": group_id, "condition": name})
            cells[name] = cell

    rows = []
    for assignment in (0, 1):
        left, right = f"s0p{assignment}", f"s1p{assignment}"
        rows.append(
            pair_row(
                group_id,
                split,
                "two_valid_sources_selector_swap",
                f"payload_assignment_{assignment}",
                cells[left]["condition_id"],
                cells[right]["condition_id"],
                cells[left],
                cells[right],
                ["final query token", "selected target source", "correct payload answer"],
                ["all three source-payload pairs", "payload assignment", "positions", "filler"],
            )
        )
    for selector in (0, 1):
        left, right = f"s{selector}p0", f"s{selector}p1"
        rows.append(
            pair_row(
                group_id,
                split,
                "payload_swap_match_preserved",
                f"selector_{selector}",
                cells[left]["condition_id"],
                cells[right]["condition_id"],
                cells[left],
                cells[right],
                ["payloads assigned to A and C", "correct payload answer"],
                ["query", "all source identities", "neutral pair", "positions", "filler"],
            )
        )
    for answer_name, left, right in (
        ("payload_B", "s0p0", "s1p1"),
        ("payload_D", "s1p0", "s0p1"),
    ):
        rows.append(
            pair_row(
                group_id,
                split,
                "selector_payload_joint_answer_preserved",
                answer_name,
                cells[left]["condition_id"],
                cells[right]["condition_id"],
                cells[left],
                cells[right],
                ["selector", "target payload assignment"],
                ["correct answer", "source set", "payload set", "neutral pair", "positions", "filler"],
            )
        )

    # Cross every control with all four selector x payload cells.  This avoids
    # R552's nine-group SELECT subcells and permits within-group paired effects.
    for selector in (0, 1):
        for assignment in (0, 1):
            control_name = f"s{selector}p{assignment}"
            control = cells[control_name]
            selected_position = control["source_positions"][selector]
            contrast_position = control["source_positions"][1 - selector]
            edited_specs = (
                (
                    "match_break_payload_preserved",
                    "selected_target_source",
                    selected_position,
                    decoys["selected"],
                    ["earlier selected target source", "selected equality edge"],
                    ["query", "all payloads", "contrast and neutral pairs", "length", "positions"],
                ),
                (
                    "irrelevant_source_edit",
                    "neutral_source",
                    control["N_source_position"],
                    decoys["neutral_source"],
                    ["endpoint-neutral third source token"],
                    ["both target pairs", "query", "both endpoint tokens", "neutral payload", "length", "positions"],
                ),
                (
                    "irrelevant_payload_edit",
                    "neutral_payload",
                    control["N_payload_position"],
                    decoys["neutral_payload"],
                    ["endpoint-neutral third payload token"],
                    ["both target pairs", "query", "both endpoint tokens", "neutral source", "length", "positions"],
                ),
                (
                    "contrast_target_source_edit",
                    "contrast_target_source",
                    contrast_position,
                    decoys["contrast"],
                    ["unselected target source immediately before the contrast payload"],
                    ["selected target pair", "query", "both endpoint token identities", "length", "positions"],
                ),
            )
            for family, role, position, replacement, changes, holds in edited_specs:
                donor = edited_condition(
                    control, position=position, replacement=replacement, role=role
                )
                rows.append(
                    pair_row(
                        group_id,
                        split,
                        family,
                        control_name,
                        control["condition_id"],
                        content_id({"group": group_id, "condition": control_name, "edit": role}),
                        control,
                        donor,
                        changes,
                        holds,
                    )
                )

            filler_variant = condition(
                prefix, variables, alternate, layout, pair_order, selector, assignment
            )
            rows.append(
                pair_row(
                    group_id,
                    split,
                    "copy_relation_preserved_nuisance_change",
                    f"{control_name}:filler_change",
                    control["condition_id"],
                    content_id({"group": group_id, "condition": control_name, "edit": "filler"}),
                    control,
                    filler_variant,
                    ["filler token identities"],
                    ["all three pairs", "selector", "payload assignment", "answer", "positions"],
                )
            )
            lag_variant = lag_condition(control, extension)
            rows.append(
                pair_row(
                    group_id,
                    split,
                    "copy_relation_preserved_nuisance_change",
                    f"{control_name}:lag_extension",
                    control["condition_id"],
                    content_id({"group": group_id, "condition": control_name, "edit": "lag"}),
                    control,
                    lag_variant,
                    ["query lag", "additional filler before query", "sequence length"],
                    ["all three pairs", "selector", "payload assignment", "answer"],
                )
            )
    group = {
        "group_id": group_id,
        "split": split,
        "index": index,
        "prefix": prefix,
        "layout": list(layout),
        "pair_order": list(pair_order),
        "variable_token_ids": variables,
        "variable_token_text": {key: pieces[value] for key, value in variables.items()},
        "sampled_token_ids": ids,
        "factorial_conditions": cells,
        "control_conditions": ["s0p0", "s0p1", "s1p0", "s1p1"],
        "within_group_condition_reuse_is_intentional": True,
    }
    return group, rows


def _one_token_difference(row: Mapping[str, object]) -> int:
    base = row["base_ids"]
    donor = row["donor_ids"]
    if len(base) != len(donor):
        return -1
    return sum(left != right for left, right in zip(base, donor, strict=True))


def validate_dataset(payload: Mapping[str, object]) -> dict:
    groups = payload["groups"]
    rows = payload["rows"]
    if len(groups) != 180 or len(rows) != 5400:
        raise RuntimeError("registered group or row count changed")
    group_by_id = {group["group_id"]: group for group in groups}
    if len(group_by_id) != len(groups):
        raise RuntimeError("duplicate semantic group")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate row identity")

    group_family_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    sequence_owner: dict[tuple[int, ...], set[str]] = collections.defaultdict(set)
    prompt_answer_owner: dict[tuple[tuple[int, ...], int], set[str]] = collections.defaultdict(set)
    for row in rows:
        group_id = row["group_id"]
        if group_id not in group_by_id or row["split"] != group_by_id[group_id]["split"]:
            raise RuntimeError("row is detached from its semantic group or split")
        group_family_counts[group_id][row["family_id"]] += 1
        for side in ("base", "donor"):
            ids = tuple(row[f"{side}_ids"])
            sequence_owner[ids].add(group_id)
            prompt_answer_owner[(ids, row[f"{side}_answer_id"])].add(group_id)
    if any(counts != EXPECTED_PER_GROUP for counts in group_family_counts.values()):
        raise RuntimeError("a group lacks the complete factorial/control family")
    if any(len(owners) != 1 for owners in sequence_owner.values()):
        raise RuntimeError("a prompt sequence crosses semantic groups")
    if any(len(owners) != 1 for owners in prompt_answer_owner.values()):
        raise RuntimeError("a prompt-answer pair crosses semantic groups")

    token_owner: dict[int, set[str]] = collections.defaultdict(set)
    split_tokens: dict[str, set[int]] = collections.defaultdict(set)
    for group in groups:
        for token in group["sampled_token_ids"]:
            token_owner[token].add(group["group_id"])
            split_tokens[group["split"]].add(token)
    if any(len(owners) != 1 for owners in token_owner.values()):
        raise RuntimeError("a sampled token crosses semantic groups")
    split_names = tuple(SPLITS)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if split_tokens[left] & split_tokens[right]:
                raise RuntimeError("sampled token banks overlap across splits")

    for group in groups:
        cells = group["factorial_conditions"]
        variables = group["variable_token_ids"]
        expected_answers = {
            "s0p0": variables["B"],
            "s1p0": variables["D"],
            "s0p1": variables["D"],
            "s1p1": variables["B"],
        }
        if {name: cell["answer_id"] for name, cell in cells.items()} != expected_answers:
            raise RuntimeError("selector-by-payload factorial answer rule changed")
        for cell in cells.values():
            ids = cell["ids"]
            query_position = cell["query_position"]
            matches = [
                index for index, token in enumerate(ids[:query_position]) if token == cell["query_id"]
            ]
            selected_position = cell["source_positions"][cell["selector"]]
            if matches != [selected_position]:
                raise RuntimeError("factorial prompt does not have exactly one selected earlier match")
            for source_position, payload_position, source_id, payload_id in zip(
                cell["source_positions"],
                cell["payload_positions"],
                cell["source_ids"],
                cell["payload_ids"],
                strict=True,
            ):
                if payload_position != source_position + 1:
                    raise RuntimeError("target payload is not immediate")
                if ids[source_position] != source_id or ids[payload_position] != payload_id:
                    raise RuntimeError("target structure metadata differs from tokens")
            if cell["N_payload_position"] != cell["N_source_position"] + 1:
                raise RuntimeError("neutral payload is not immediate")
            if ids[cell["N_source_position"]] != cell["neutral_source_id"]:
                raise RuntimeError("neutral source metadata differs from tokens")
            if ids[cell["N_payload_position"]] != cell["neutral_payload_id"]:
                raise RuntimeError("neutral payload metadata differs from tokens")
            if cell["neutral_source_id"] in (cell["query_id"], *cell["source_ids"]):
                raise RuntimeError("neutral source is not distinct")
            if cell["neutral_payload_id"] in (
                cell["answer_id"], cell["other_answer_id"], *cell["payload_ids"]
            ):
                raise RuntimeError("neutral payload enters the endpoint")

    one_token_families = {
        "match_break_payload_preserved": "selected_target_source",
        "irrelevant_source_edit": "neutral_source",
        "irrelevant_payload_edit": "neutral_payload",
        "contrast_target_source_edit": "contrast_target_source",
    }
    for row in rows:
        family = row["family_id"]
        if family in one_token_families:
            if _one_token_difference(row) != 1 or row["edit_role"] != one_token_families[family]:
                raise RuntimeError("one-token control edit semantics changed")
            if row["answer_changes"]:
                raise RuntimeError("one-token control changed the registered endpoint")
        elif family in {
            "two_valid_sources_selector_swap",
            "payload_swap_match_preserved",
        }:
            if not row["answer_changes"]:
                raise RuntimeError("single-factor edge failed to change its answer")
        elif family == "selector_payload_joint_answer_preserved":
            if row["answer_changes"]:
                raise RuntimeError("joint diagonal failed to preserve its answer")
            if row["base_selector"] == row["donor_selector"]:
                raise RuntimeError("joint diagonal failed to change selector")
            if row["base_payload_assignment"] == row["donor_payload_assignment"]:
                raise RuntimeError("joint diagonal failed to change payload assignment")
        elif family == "copy_relation_preserved_nuisance_change" and row["answer_changes"]:
            raise RuntimeError("nuisance control changed its answer")

    split_group_counts = collections.Counter(group["split"] for group in groups)
    split_row_counts = collections.Counter(row["split"] for row in rows)
    family_row_counts = collections.Counter(row["family_id"] for row in rows)
    if split_group_counts != collections.Counter(
        {split: spec["count"] for split, spec in SPLITS.items()}
    ):
        raise RuntimeError("split group counts changed")
    return {
        "group_count": len(groups),
        "row_count": len(rows),
        "factorial_condition_count": 4 * len(groups),
        "unique_prompt_sequence_count": len(sequence_owner),
        "split_group_counts": dict(split_group_counts),
        "split_row_counts": dict(split_row_counts),
        "family_row_counts": dict(family_row_counts),
        "every_group_has_complete_factorial_and_controls": True,
        "every_group_belongs_to_one_split": True,
        "prompt_sequences_never_cross_groups": True,
        "prompt_answer_pairs_never_cross_groups": True,
        "sampled_tokens_never_cross_groups": True,
        "variable_token_banks_disjoint_across_splits": True,
        "factorial_and_interaction_semantics_exact": True,
        "selected_neutral_and_contrast_edits_exact": True,
        "exactly_one_earlier_query_match": True,
        "payloads_are_immediate": True,
    }


def build_dataset() -> dict:
    all_groups = []
    all_rows = []
    for split, spec in SPLITS.items():
        bank = token_bank(*spec["token_range"])
        generator = random.Random(spec["seed"])
        generator.shuffle(bank)
        required = spec["count"] * TOKENS_PER_GROUP
        if len(bank) < required:
            raise RuntimeError(f"too few distinct tokens for {split}: {len(bank)} < {required}")
        for index in range(spec["count"]):
            start = index * TOKENS_PER_GROUP
            chosen = bank[start : start + TOKENS_PER_GROUP]
            group, rows = make_group(split, index, spec, chosen)
            all_groups.append(group)
            all_rows.extend(rows)
    payload = {
        "schema": "induction_selector_payload_three_source_rows_rung578_v1",
        "status": "rows_frozen_outcomes_unopened",
        "causal_variables": {
            "selector": "which target source token is repeated as the final query",
            "payload_assignment": "whether B,D or D,B immediately follow A,C",
            "interaction": "answer is B exactly when selector equals payload assignment",
        },
        "control_scope": {
            "endpoint_neutral_pair": "X->E; X and E are absent from both endpoint logits",
            "old_r552_irrelevant_edit": "retained as contrast_target_source_edit, not an invariance gate",
        },
        "factorial_answer_rule": "answer(S,P) is B when S==P and D otherwise",
        "family_roles": FAMILY_ROLES,
        "split_policy": {
            "unit": "complete three-pair semantic group",
            "group_counts": {split: spec["count"] for split, spec in SPLITS.items()},
            "token_id_ranges": {split: list(spec["token_range"]) for split, spec in SPLITS.items()},
            "sampled_tokens_disjoint_across_groups": True,
            "prefix_and_layout_families_disjoint_across_splits": True,
            "all_derived_pairs_stay_with_group": True,
            "final_test_used_for_selection": False,
            "ood_used_for_selection": False,
        },
        "groups": all_groups,
        "rows": all_rows,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }
    validation = validate_dataset(payload)
    payload.update(
        {
            "group_count": validation["group_count"],
            "row_count": validation["row_count"],
            "unique_prompt_sequence_count": validation["unique_prompt_sequence_count"],
        }
    )
    return payload


def receipt_for(payload: Mapping[str, object], encoded_rows: bytes) -> dict:
    validation = validate_dataset(payload)
    return {
        "schema": "induction_selector_payload_three_source_rows_rung578_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT.parent)),
        "rows_sha256": hashlib.sha256(encoded_rows).hexdigest(),
        "preregistration_sha256": file_sha256(PREREG),
        **validation,
        "within_group_condition_reuse_declared": True,
        "tokenizer": "tiktoken:gpt2",
        "prior_r552_rows_sha256": "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460",
        "prior_r554_result_sha256": "fb6cdbb9196d118691a0655507b9d2869291f11fe8920674380188513d24dd1d",
        "r552_brittleness_resolved_by": (
            "generic source edits now act on X->E, whose source and payload are absent "
            "from the correct-vs-other target-payload margin"
        ),
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": [],
    }


def main() -> None:
    payload = build_dataset()
    encoded = (json.dumps(payload, indent=1) + "\n").encode()
    OUT.write_bytes(encoded)
    receipt = receipt_for(payload, encoded)
    RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
