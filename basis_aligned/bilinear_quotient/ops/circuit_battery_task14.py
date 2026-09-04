#!/usr/bin/env python3
"""CPU-only linked authority generator for task 14: subject–verb agreement.

The task asks whether the next copula should be `` is`` or `` are``.  For an
ordinary noun phrase the answer follows the grammatical number of the subject
head, not the nearer attractor noun.  The coordinated-subject control has two
singular conjuncts but a plural answer.  This module defines rows only; it has
no model, checkpoint, intervention, result, or queue path.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from typing import Any

import tiktoken

import circuit_battery_integration_contract as contract


TASK_ID = "subject_verb.number_agreement"
SCHEMA = "circuit_battery_subject_verb_agreement_v1"
ENCODING = tiktoken.get_encoding("gpt2")
GROUPS_PER_PHASE = 32

# Every singular and plural form is one GPT-2 token after a space.  Sixty-four
# noun pairs give four disjoint phase pools of sixteen pairs.  Collective nouns
# are deliberately absent because their agreement varies by dialect.
NOUN_PAIRS = (
    ("key", "keys"), ("door", "doors"), ("book", "books"), ("car", "cars"),
    ("star", "stars"), ("lamp", "lamps"), ("train", "trains"), ("ship", "ships"),
    ("dog", "dogs"), ("cat", "cats"), ("bird", "birds"), ("tree", "trees"),
    ("road", "roads"), ("house", "houses"), ("table", "tables"), ("garden", "gardens"),
    ("cabinet", "cabinets"), ("window", "windows"), ("bottle", "bottles"),
    ("river", "rivers"), ("chair", "chairs"), ("clock", "clocks"),
    ("bridge", "bridges"), ("phone", "phones"), ("cloud", "clouds"),
    ("stone", "stones"), ("field", "fields"), ("plant", "plants"),
    ("truck", "trucks"), ("boat", "boats"), ("gate", "gates"), ("map", "maps"),
    ("ring", "rings"), ("plate", "plates"), ("box", "boxes"),
    ("branch", "branches"), ("glass", "glasses"), ("church", "churches"),
    ("horse", "horses"), ("shoe", "shoes"), ("coat", "coats"), ("room", "rooms"),
    ("school", "schools"), ("park", "parks"), ("hill", "hills"), ("farm", "farms"),
    ("lake", "lakes"), ("island", "islands"), ("valley", "valleys"),
    ("forest", "forests"), ("market", "markets"), ("paper", "papers"),
    ("file", "files"), ("note", "notes"), ("song", "songs"), ("game", "games"),
    ("player", "players"), ("friend", "friends"), ("child", "children"),
    ("person", "people"), ("mouse", "mice"), ("woman", "women"), ("man", "men"),
    ("flower", "flowers"),
)

if len(NOUN_PAIRS) != 64:  # pragma: no cover - static tripwire
    raise RuntimeError("task14 requires exactly 64 noun pairs")
for pair in NOUN_PAIRS:  # pragma: no branch - import-time tokenizer tripwire
    if any(len(ENCODING.encode(" " + word)) != 1 for word in pair):
        raise RuntimeError(f"task14 noun form is not one GPT-2 token: {pair!r}")


TASK_SPEC = contract.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_subject_verb_agreement_panel",
    answer_role="score_jointly_tokenized_is_minus_are_margin",
    transforms=(
        contract.TransformSpec("A1", "flip_subject_head_number_in_pp", True, "toward_donor"),
        contract.TransformSpec("A2", "flip_subject_head_number_in_relative_clause", True, "toward_donor"),
        contract.TransformSpec("P", "replace_attractor_lexeme_same_number", False, "invariant"),
        contract.TransformSpec("C", "flip_attractor_number_under_coordination", False, "registered_active"),
    ),
)

_CHANGED_VARIABLE = {
    "A1": "subject_head_number_in_prepositional_phrase",
    "A2": "subject_head_number_in_relative_clause",
    "P": "nearest_attractor_lexical_identity_at_fixed_number",
    "C": "nearest_attractor_number_with_coordinated_subject_fixed",
}

_TEMPLATES = {
    "fit_pp_near": "The {head} near the {attractor}",
    "fit_relative_placed_beside": "The {head} that I placed beside the {attractor}",
    "fit_pp_behind": "The {head} behind the {attractor}",
    "fit_coord_near": "The {head} and the {second_head} near the {attractor}",
    "select_pp_beside": "The {head} beside the {attractor}",
    "select_relative_noticed_behind": "The {head} that I noticed behind the {attractor}",
    "select_pp_beyond": "The {head} beyond the {attractor}",
    "select_coord_behind": "The {head} and the {second_head} behind the {attractor}",
    "test_pp_behind": "The {head} behind the {attractor}",
    "test_relative_moved_beyond": "The {head} that I moved beyond the {attractor}",
    "test_pp_across_from": "The {head} across from the {attractor}",
    "test_coord_under": "The {head} and the {second_head} under the {attractor}",
    "ood_fronted_two_attractors": "Near the {attractor} beside the {second_attractor}, the {head}",
    "ood_relative_two_attractors": (
        "The {head} that I placed near the {attractor} behind the {second_attractor}"
    ),
    "ood_pp_two_attractors": "The {head} beyond the {attractor} near the {second_attractor}",
    "ood_coord_near": (
        "The {head} and the {second_head} near the {attractor} behind the {second_attractor}"
    ),
}

_PHASE_TEMPLATES = {
    "FIT": {
        "A1": ("fit_pp_near", "fit_pp_near"),
        "A2": ("fit_relative_placed_beside", "fit_relative_placed_beside"),
        "P": ("fit_pp_behind", "fit_pp_behind"),
        "C": ("fit_coord_near", "fit_coord_near"),
    },
    "SELECT": {
        "A1": ("select_pp_beside", "select_pp_beside"),
        "A2": ("select_relative_noticed_behind", "select_relative_noticed_behind"),
        "P": ("select_pp_beyond", "select_pp_beyond"),
        "C": ("select_coord_behind", "select_coord_behind"),
    },
    "TEST": {
        "A1": ("test_pp_behind", "test_pp_behind"),
        "A2": ("test_relative_moved_beyond", "test_relative_moved_beyond"),
        "P": ("test_pp_across_from", "test_pp_across_from"),
        "C": ("test_coord_under", "test_coord_under"),
    },
    "OOD": {
        "A1": ("ood_fronted_two_attractors", "ood_fronted_two_attractors"),
        "A2": ("ood_relative_two_attractors", "ood_relative_two_attractors"),
        "P": ("ood_pp_two_attractors", "ood_pp_two_attractors"),
        "C": ("ood_coord_near", "ood_coord_near"),
    },
}


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _phase_pool(split: str, seed: int) -> tuple[tuple[str, str], ...]:
    phase_number = contract.PHASES.index(split)
    pairs = list(NOUN_PAIRS[16 * phase_number:16 * (phase_number + 1)])
    material = f"{SCHEMA}|{seed}|{split}|noun-order".encode("utf-8")
    random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big")).shuffle(pairs)
    return tuple(pairs)


def _form(pair: tuple[str, str], plural: bool) -> str:
    return pair[1 if plural else 0]


def _answer(plural: bool) -> str:
    return " are" if plural else " is"


def _joint_encoding(prompt: str, answer: str) -> tuple[list[int], int]:
    prompt_ids = ENCODING.encode(prompt)
    complete_ids = ENCODING.encode(prompt + answer)
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not stable under joint prompt-plus-answer tokenization")
    suffix = complete_ids[len(prompt_ids):]
    if len(suffix) != 1:
        raise ValueError("task14 answer is not exactly one token at its continuation boundary")
    return prompt_ids, suffix[0]


def _render(template_id: str, *, head: str, attractor: str,
            second_head: str, second_attractor: str) -> str:
    try:
        template = _TEMPLATES[template_id]
    except KeyError as error:
        raise ValueError("unknown task14 template") from error
    return template.format(
        head=head, attractor=attractor,
        second_head=second_head, second_attractor=second_attractor,
    )


def _positions(ids: list[int], words: tuple[str, ...]) -> list[int]:
    positions = []
    for word in words:
        token = ENCODING.encode(" " + word)
        if len(token) != 1 or ids.count(token[0]) != 1:
            raise ValueError(f"semantic word is not uniquely positioned: {word!r}")
        positions.append(ids.index(token[0]))
    return positions


def _make_row(
    *, split: str, seed: int, group_number: int, group_id: str, transform_id: str,
    head_pair: tuple[str, str], attractor_pair: tuple[str, str],
    second_head_pair: tuple[str, str], second_attractor_pair: tuple[str, str],
    surface_attractor_pair: tuple[str, str],
    base_head_plural: bool, donor_head_plural: bool,
    base_attractor_plural: bool, donor_attractor_plural: bool,
    base_second_attractor_plural: bool, donor_second_attractor_plural: bool,
) -> dict[str, Any]:
    base_template, donor_template = _PHASE_TEMPLATES[split][transform_id]
    coordinated = transform_id == "C"
    if coordinated:
        base_head_plural = donor_head_plural = False
    base_head = _form(head_pair, base_head_plural)
    donor_head = _form(head_pair, donor_head_plural)
    base_attractor = _form(attractor_pair, base_attractor_plural)
    donor_attractor_pair = (
        surface_attractor_pair if transform_id == "P" and split != "OOD" else attractor_pair
    )
    donor_attractor = _form(donor_attractor_pair, donor_attractor_plural)
    second_head = _form(second_head_pair, False)
    base_second_attractor = _form(second_attractor_pair, base_second_attractor_plural)
    donor_second_attractor_pair = (
        surface_attractor_pair if transform_id == "P" and split == "OOD"
        else second_attractor_pair
    )
    donor_second_attractor = _form(
        donor_second_attractor_pair, donor_second_attractor_plural
    )
    base_text = _render(
        base_template, head=base_head, attractor=base_attractor,
        second_head=second_head, second_attractor=base_second_attractor,
    )
    donor_text = _render(
        donor_template, head=donor_head, attractor=donor_attractor,
        second_head=second_head, second_attractor=donor_second_attractor,
    )
    base_subject_plural = True if coordinated else base_head_plural
    donor_subject_plural = True if coordinated else donor_head_plural
    base_answer, donor_answer = _answer(base_subject_plural), _answer(donor_subject_plural)
    base_ids, base_answer_id = _joint_encoding(base_text, base_answer)
    donor_ids, donor_answer_id = _joint_encoding(donor_text, donor_answer)
    if len(base_ids) != len(donor_ids):
        raise ValueError("task14 base and donor prompts are not position-aligned")
    changed_positions = [
        index for index, values in enumerate(zip(base_ids, donor_ids)) if values[0] != values[1]
    ]
    if len(changed_positions) != 1:
        raise ValueError("task14 transform must change exactly one aligned prompt token")
    head_words = (base_head, second_head) if coordinated else (base_head,)
    donor_head_words = (donor_head, second_head) if coordinated else (donor_head,)
    attractor_words = (
        (base_attractor, base_second_attractor) if split == "OOD" else (base_attractor,)
    )
    donor_attractor_words = (
        (donor_attractor, donor_second_attractor) if split == "OOD" else (donor_attractor,)
    )
    base_head_positions = _positions(base_ids, head_words)
    donor_head_positions = _positions(donor_ids, donor_head_words)
    base_attractor_positions = _positions(base_ids, attractor_words)
    donor_attractor_positions = _positions(donor_ids, donor_attractor_words)
    if base_head_positions != donor_head_positions or base_attractor_positions != donor_attractor_positions:
        raise ValueError("semantic token positions moved under a task14 transformation")
    expected_position = (
        base_head_positions[0] if transform_id in ("A1", "A2")
        else base_attractor_positions[-1]
    )
    if changed_positions != [expected_position]:
        raise ValueError("task14 changed token is not the declared semantic intervention")
    transform = next(item for item in TASK_SPEC.transforms if item.transform_id == transform_id)
    answer_changes = base_answer != donor_answer
    if answer_changes != transform.answer_changes:
        raise ValueError("task14 answer-change semantics differ from transform authority")
    identity = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "split": split,
        "seed": seed,
        "group_number": group_number,
        "group_id": group_id,
        "transform_id": transform_id,
        "head_pair": list(head_pair),
        "attractor_pair": list(attractor_pair),
        "second_head_pair": list(second_head_pair),
        "second_attractor_pair": list(second_attractor_pair),
        "surface_attractor_pair": list(surface_attractor_pair),
        "base_head_plural": base_head_plural,
        "donor_head_plural": donor_head_plural,
        "base_attractor_plural": base_attractor_plural,
        "donor_attractor_plural": donor_attractor_plural,
        "base_second_attractor_plural": base_second_attractor_plural,
        "donor_second_attractor_plural": donor_second_attractor_plural,
        "base_template_id": base_template,
        "donor_template_id": donor_template,
    }
    return {
        **identity,
        "row_id": _canonical_sha(identity),
        "answer_changes": answer_changes,
        "expected_effect": transform.expected_effect,
        "changed_variable": _CHANGED_VARIABLE[transform_id],
        "base_text": base_text,
        "donor_text": donor_text,
        "base_answer": base_answer,
        "donor_answer": donor_answer,
        "base_foil": _answer(not base_subject_plural),
        "donor_foil": _answer(not donor_subject_plural),
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer_id": base_answer_id,
        "donor_answer_id": donor_answer_id,
        "base_head_positions": base_head_positions,
        "donor_head_positions": donor_head_positions,
        "base_attractor_positions": base_attractor_positions,
        "donor_attractor_positions": donor_attractor_positions,
        "intervention_token_positions": changed_positions,
        "base_prediction_position": len(base_ids) - 1,
        "donor_prediction_position": len(donor_ids) - 1,
        "base_subject_number": "plural" if base_subject_plural else "singular",
        "donor_subject_number": "plural" if donor_subject_plural else "singular",
        "control_relation": (
            "two_singular_conjuncts_require_plural_agreement" if coordinated else None
        ),
    }


def _panel(split: str, group_number: int, seed: int) -> list[dict[str, Any]]:
    pool = _phase_pool(split, seed)
    head_pair = pool[group_number % 16]
    attractor_pair = pool[(5 * group_number + 3) % 16]
    second_attractor_pair = pool[(5 * group_number + 5) % 16]
    second_head_pair = pool[(5 * group_number + 7) % 16]
    surface_attractor_pair = pool[(5 * group_number + 9) % 16]
    if len({
        head_pair, attractor_pair, second_attractor_pair,
        second_head_pair, surface_attractor_pair,
    }) != 5:
        raise RuntimeError("task14 semantic noun roles collided")
    half = group_number // 16
    base_head_plural = bool(half)
    base_attractor_plural = bool((half + group_number % 2) % 2)
    second_attractor_plural = bool((half + (group_number // 2) % 2) % 2)
    group_id = f"{split}:{_canonical_sha([SCHEMA, seed, split, group_number])[:20]}"
    common = dict(
        split=split, seed=seed, group_number=group_number, group_id=group_id,
        head_pair=head_pair, attractor_pair=attractor_pair,
        second_head_pair=second_head_pair, second_attractor_pair=second_attractor_pair,
        surface_attractor_pair=surface_attractor_pair,
    )
    return [
        _make_row(
            **common, transform_id="A1",
            base_head_plural=base_head_plural, donor_head_plural=not base_head_plural,
            base_attractor_plural=base_attractor_plural,
            donor_attractor_plural=base_attractor_plural,
            base_second_attractor_plural=second_attractor_plural,
            donor_second_attractor_plural=second_attractor_plural,
        ),
        _make_row(
            **common, transform_id="A2",
            base_head_plural=base_head_plural, donor_head_plural=not base_head_plural,
            base_attractor_plural=base_attractor_plural,
            donor_attractor_plural=base_attractor_plural,
            base_second_attractor_plural=second_attractor_plural,
            donor_second_attractor_plural=second_attractor_plural,
        ),
        _make_row(
            **common, transform_id="P",
            base_head_plural=base_head_plural, donor_head_plural=base_head_plural,
            base_attractor_plural=base_attractor_plural,
            donor_attractor_plural=base_attractor_plural,
            base_second_attractor_plural=second_attractor_plural,
            donor_second_attractor_plural=second_attractor_plural,
        ),
        _make_row(
            **common, transform_id="C",
            base_head_plural=False, donor_head_plural=False,
            base_attractor_plural=base_attractor_plural,
            donor_attractor_plural=(
                base_attractor_plural if split == "OOD" else not base_attractor_plural
            ),
            base_second_attractor_plural=second_attractor_plural,
            donor_second_attractor_plural=(
                not second_attractor_plural if split == "OOD" else second_attractor_plural
            ),
        ),
    ]


def validate_authority(rows: list[dict[str, Any]]) -> str:
    """Rebuild every row and prove exact panels, positions, balance, and isolation."""
    authority_sha = contract.validate_rows(TASK_SPEC, rows)
    seeds = {row.get("seed") for row in rows}
    if len(seeds) != 1 or any(type(seed) is not int for seed in seeds):
        raise contract.BatteryContractError("task14 authority must have one integer seed")
    seed = next(iter(seeds))
    panels: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    group_numbers: dict[str, set[int]] = {phase: set() for phase in contract.PHASES}
    prompts: dict[str, set[str]] = {phase: set() for phase in contract.PHASES}
    noun_forms: dict[str, set[str]] = {phase: set() for phase in contract.PHASES}
    for row in rows:
        split = row.get("split")
        group_number = row.get("group_number")
        transform_id = row.get("transform_id")
        if split not in contract.PHASES or type(group_number) is not int \
                or group_number not in range(GROUPS_PER_PHASE):
            raise contract.BatteryContractError("task14 split or group number is invalid")
        expected = {
            item["transform_id"]: item for item in _panel(split, group_number, seed)
        }.get(transform_id)
        if expected is None or row != expected:
            raise contract.BatteryContractError("task14 row differs from exact regenerated semantics")
        group_numbers[split].add(group_number)
        panels.setdefault((split, row["group_id"]), {})[transform_id] = row
        prompts[split].update((row["base_text"], row["donor_text"]))
        for pair_field in (
            "head_pair", "attractor_pair", "second_head_pair", "second_attractor_pair",
            "surface_attractor_pair",
        ):
            noun_forms[split].update(row[pair_field])
    for phase in contract.PHASES:
        if group_numbers[phase] != set(range(GROUPS_PER_PHASE)):
            raise contract.BatteryContractError("task14 phase lacks its exact 32 groups")
        phase_panels = [panel for (split, _), panel in panels.items() if split == phase]
        if len(phase_panels) != GROUPS_PER_PHASE:
            raise contract.BatteryContractError("task14 phase has the wrong panel count")
        role_counts = {role: Counter() for role in (
            "head_pair", "attractor_pair", "second_head_pair", "second_attractor_pair",
            "surface_attractor_pair",
        )}
        number_pairs = Counter()
        answer_counts = {
            key: Counter() for key in (
                "A1_base", "A1_donor", "A2_base", "A2_donor",
                "P_base", "P_donor", "C_base", "C_donor",
            )
        }
        for panel in phase_panels:
            a1 = panel["A1"]
            for role in role_counts:
                role_counts[role][tuple(a1[role])] += 1
            number_pairs[(a1["base_head_plural"], a1["base_attractor_plural"])] += 1
            for transform in contract.TRANSFORMS:
                row = panel[transform]
                answer_counts[f"{transform}_base"][row["base_answer"]] += 1
                answer_counts[f"{transform}_donor"][row["donor_answer"]] += 1
        if any(len(counts) != 16 or set(counts.values()) != {2}
               for counts in role_counts.values()):
            raise contract.BatteryContractError("task14 noun roles are not exactly balanced")
        if set(number_pairs.values()) != {8} or len(number_pairs) != 4:
            raise contract.BatteryContractError("task14 head/attractor number cells are unbalanced")
        for key, counts in answer_counts.items():
            expected = {" are": 32} if key.startswith("C_") else {" is": 16, " are": 16}
            if dict(counts) != expected:
                raise contract.BatteryContractError("task14 answer/foil exposure is unbalanced")
    for index, phase in enumerate(contract.PHASES):
        for other in contract.PHASES[index + 1:]:
            if prompts[phase] & prompts[other] or noun_forms[phase] & noun_forms[other]:
                raise contract.BatteryContractError("task14 prompt or noun vocabulary leaks across phases")
    template_sets = {
        phase: {template for pair in _PHASE_TEMPLATES[phase].values() for template in pair}
        for phase in contract.PHASES
    }
    for index, phase in enumerate(contract.PHASES):
        for other in contract.PHASES[index + 1:]:
            if template_sets[phase] & template_sets[other]:
                raise contract.BatteryContractError("task14 template identities leak across phases")
    return authority_sha


def build_authority(
    groups_per_phase: int = GROUPS_PER_PHASE, seed: int = 71418,
) -> tuple[list[dict[str, Any]], str]:
    if groups_per_phase != GROUPS_PER_PHASE:
        raise ValueError("balanced task14 authority requires exactly 32 groups per phase")
    rows = [
        row
        for split in contract.PHASES
        for group_number in range(groups_per_phase)
        for row in _panel(split, group_number, seed)
    ]
    return rows, validate_authority(rows)


def split_rows(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], str]:
    if split not in contract.PHASES:
        raise ValueError("task14 split is invalid")
    selected = [row for row in rows if row["split"] == split]
    contract.validate_rows(TASK_SPEC, selected, required_phases=(split,))
    return selected, _canonical_sha(selected)


if __name__ == "__main__":
    authority, digest = build_authority()
    print(json.dumps({
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "rows": len(authority),
        "groups": len(authority) // 4,
        "authority_sha256": digest,
        "split_records_sha256": {
            split: split_rows(authority, split)[1] for split in contract.PHASES
        },
        "prospective_fit_price": {
            "forward_calls": 8,
            "row_side_evaluations": 256,
            "raw_float32_evidence_bytes": 2048,
        },
    }, sort_keys=True))
