#!/usr/bin/env python3
# BQGATE: EXPERIMENT
"""One-forward DEVELOPMENT probe for natural selection of ``is`` versus ``are``.

Every prompt contains both literal words exactly once.  Within each paired
prompt, the complete key and its order are held fixed; only the selected key
changes.  This tests native copy/selection capability, not grammar.  The rows
are disposable development material and are forbidden in FIT, SELECT, TEST,
and OOD.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Sequence

import tiktoken

import circuit_experiment_spec as framework
import circuit_fast_screen_managed_runner as managed
from circuit_fast_screen_producer import Bilin18TorchBackend, ModelBatch


ROOT = Path(__file__).resolve().parent.parent
RESULT_RELATIVE = Path(
    "circuits/dev_capability/is_are_natural_copy_control_v1.json"
)
RESULT = ROOT / RESULT_RELATIVE
REQUEST_ID = "dev-is-are-natural-copy-control-v1"
SPLIT = "DEVELOPMENT"
REUSE_POLICY = "forbidden_in_FIT_SELECT_TEST_OOD"
ENCODING = tiktoken.get_encoding("gpt2")
EXPECTED_BANK_SHA256 = "f6966d43a55219ae3e1aa401ed12fd4c7a85fbe4fad9bcd1639978aa0d57362b"
MIN_FAMILY_DIRECTION_ACCURACY = 0.75
REGISTERED_PREDICTIONS = (
    (
        "pred_a_runtime_one_native_batch",
        "Runtime performs exactly one native forward batch.",
    ),
    (
        "pred_b_balance_and_pair_integrity",
        "All four families, both key orders, both instances, both selected endpoints, and all exact selection pairs retain their frozen balance.",
    ),
    (
        "pred_c_integrity_finite_aligned",
        "All returned logit pairs are finite and row-aligned.",
    ),
    (
        "pred_d_native_copy_capable",
        "Every family-by-selected-endpoint cell reaches at least 75% native accuracy.",
    ),
)


class CopyControlError(ValueError):
    """The copy bank, backend evidence, or publication is malformed."""


@dataclass(frozen=True)
class SelectionPair:
    pair_id: str
    family: str
    instance_id: str
    key_order: str
    key_is: str
    key_are: str


@dataclass(frozen=True)
class ProbeExample:
    row_id: str
    family: str
    instance_id: str
    key_order: str
    selection_pair_id: str
    selected_endpoint: str
    selected_key: str
    prompt: str
    answer: str
    foil: str


def _pair(
    family: str,
    instance_id: str,
    key_order: str,
    key_is: str,
    key_are: str,
) -> SelectionPair:
    return SelectionPair(
        pair_id=f"{family}.{instance_id}.{key_order}",
        family=family,
        instance_id=instance_id,
        key_order=key_order,
        key_is=key_is,
        key_are=key_are,
    )


SELECTION_PAIRS = (
    _pair("color_card", "red_blue", "is_first", "red", "blue"),
    _pair("color_card", "red_blue", "are_first", "red", "blue"),
    _pair("color_card", "amber_teal", "is_first", "amber", "teal"),
    _pair("color_card", "amber_teal", "are_first", "amber", "teal"),
    _pair("numbered_slot", "one_two", "is_first", "1", "2"),
    _pair("numbered_slot", "one_two", "are_first", "1", "2"),
    _pair("numbered_slot", "three_four", "is_first", "3", "4"),
    _pair("numbered_slot", "three_four", "are_first", "3", "4"),
    _pair("symbol_marker", "x_y", "is_first", "X", "Y"),
    _pair("symbol_marker", "x_y", "are_first", "X", "Y"),
    _pair("symbol_marker", "q_z", "is_first", "Q", "Z"),
    _pair("symbol_marker", "q_z", "are_first", "Q", "Z"),
    _pair("name_tag", "ada_ben", "is_first", "Ada", "Ben"),
    _pair("name_tag", "ada_ben", "are_first", "Ada", "Ben"),
    _pair("name_tag", "mira_owen", "is_first", "Mira", "Owen"),
    _pair("name_tag", "mira_owen", "are_first", "Mira", "Owen"),
)


def _key_text(pair: SelectionPair) -> str:
    is_entry = f"{pair.key_is} -> is"
    are_entry = f"{pair.key_are} -> are"
    if pair.key_order == "is_first":
        return f"Key: {is_entry} | {are_entry}"
    if pair.key_order == "are_first":
        return f"Key: {are_entry} | {is_entry}"
    raise CopyControlError("unknown key order")


def _prompt(pair: SelectionPair, selected_key: str) -> str:
    return (
        "Copy one word from the key. " + _key_text(pair)
        + f"\nSelected key: {selected_key}\nCopied word:"
    )


def _build_examples(
    pairs: Sequence[SelectionPair] = SELECTION_PAIRS,
) -> tuple[ProbeExample, ...]:
    output = []
    for pair in pairs:
        for endpoint, selected_key in (("is", pair.key_is), ("are", pair.key_are)):
            answer = f" {endpoint}"
            output.append(ProbeExample(
                row_id=f"{pair.pair_id}.select_{endpoint}",
                family=pair.family,
                instance_id=pair.instance_id,
                key_order=pair.key_order,
                selection_pair_id=pair.pair_id,
                selected_endpoint=endpoint,
                selected_key=selected_key,
                prompt=_prompt(pair, selected_key),
                answer=answer,
                foil=" are" if answer == " is" else " is",
            ))
    return tuple(output)


EXAMPLES = _build_examples()


def _example_json(example: ProbeExample) -> dict[str, object]:
    return {**asdict(example), "split": SPLIT, "reuse_policy": REUSE_POLICY}


def _bank_sha256(examples: Sequence[ProbeExample]) -> str:
    return framework.canonical_sha256({
        "schema": "circuit_fast_screen_dev_is_are_copy_bank_v1",
        "scientific_status": "excluded_development_only",
        "examples": [_example_json(example) for example in examples],
    })


def _joint_token(prompt: str, continuation: str) -> tuple[tuple[int, ...], int]:
    prompt_ids = tuple(ENCODING.encode(prompt))
    complete = tuple(ENCODING.encode(prompt + continuation))
    standalone = tuple(ENCODING.encode(continuation))
    if not prompt_ids or len(standalone) != 1 or complete != prompt_ids + standalone:
        raise CopyControlError("answer/foil is not one stable continuation token")
    return prompt_ids, standalone[0]


def _selection_frame(prompt: str, selected_key: str) -> str:
    selection = f"\nSelected key: {selected_key}\n"
    if prompt.count(selection) != 1:
        raise CopyControlError("selected key is not isolated on exactly one line")
    return prompt.replace(selection, "\nSelected key: <SELECTION>\n")


def _validate_balance(examples: Sequence[ProbeExample]) -> dict[str, object]:
    if len(examples) != 32:
        raise CopyControlError("copy bank must contain exactly 32 examples")
    row_ids = [example.row_id for example in examples]
    prompts = [example.prompt for example in examples]
    if len(set(row_ids)) != 32 or len(set(prompts)) != 32:
        raise CopyControlError("row IDs and prompts must be unique")
    pairs: dict[str, list[ProbeExample]] = defaultdict(list)
    for example in examples:
        if any(not isinstance(value, str) or not value for value in asdict(example).values()):
            raise CopyControlError("all example fields must be nonempty text")
        if len(re.findall(r"\bis\b", example.prompt.lower())) != 1 \
                or len(re.findall(r"\bare\b", example.prompt.lower())) != 1:
            raise CopyControlError("every prompt must contain is and are exactly once")
        pairs[example.selection_pair_id].append(example)
    if len(pairs) != 16 or set(map(len, pairs.values())) != {2}:
        raise CopyControlError("selection-pair coverage changed")
    for rows in pairs.values():
        if {row.selected_endpoint for row in rows} != {"is", "are"} \
                or {row.answer for row in rows} != {" is", " are"}:
            raise CopyControlError("a pair does not cover both endpoint directions")
        fixed = {
            (row.family, row.instance_id, row.key_order, row.selection_pair_id)
            for row in rows
        }
        frames = {_selection_frame(row.prompt, row.selected_key) for row in rows}
        if len(fixed) != 1 or len(frames) != 1:
            raise CopyControlError("paired prompts differ by more than the selection")

    expected = {
        "family": {
            "color_card": 8,
            "numbered_slot": 8,
            "symbol_marker": 8,
            "name_tag": 8,
        },
        "key_order": {"is_first": 16, "are_first": 16},
        "selected_endpoint": {"is": 16, "are": 16},
        "answer": {" is": 16, " are": 16},
    }
    for field, wanted in expected.items():
        if dict(Counter(getattr(row, field) for row in examples)) != wanted:
            raise CopyControlError(f"{field} balance changed")
    answers_by_order: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        answers_by_order[example.key_order].add(example.answer)
    if any(answers != {" is", " are"} for answers in answers_by_order.values()):
        raise CopyControlError("key order predicts the selected endpoint")
    return {
        "families": expected["family"],
        "key_orders": expected["key_order"],
        "selected_endpoints": expected["selected_endpoint"],
        "answers": expected["answer"],
        "selection_pairs": len(pairs),
        "rows_per_family_direction": 4,
    }


def compile_probe(
    examples: Sequence[ProbeExample] = EXAMPLES,
) -> tuple[ModelBatch, dict[str, object]]:
    balance = _validate_balance(examples)
    token_rows, answer_ids, foil_ids = [], [], []
    for example in examples:
        prompt_ids, answer_id = _joint_token(example.prompt, example.answer)
        foil_prompt_ids, foil_id = _joint_token(example.prompt, example.foil)
        if foil_prompt_ids != prompt_ids or answer_id == foil_id:
            raise CopyControlError("answer/foil tokenization is not aligned")
        token_rows.append(prompt_ids)
        answer_ids.append(answer_id)
        foil_ids.append(foil_id)
    digest = _bank_sha256(examples)
    if digest != EXPECTED_BANK_SHA256:
        raise CopyControlError("development bank differs from its frozen digest")
    batch = ModelBatch(
        row_ids=tuple(example.row_id for example in examples),
        side="base",
        token_rows=tuple(token_rows),
        answer_ids=tuple(answer_ids),
        foil_ids=tuple(foil_ids),
        semantic_positions=tuple(len(row) - 1 for row in token_rows),
    )
    return batch, {
        "schema": "circuit_fast_screen_dev_is_are_copy_dryrun_v1",
        "request_id": REQUEST_ID,
        "split": SPLIT,
        "reuse_policy": REUSE_POLICY,
        "scientific_status": "excluded_development_only",
        "bank_sha256": digest,
        "balance": balance,
        "minimum_family_direction_accuracy": MIN_FAMILY_DIRECTION_ACCURACY,
        "price": {
            "forward_calls": 1,
            "example_evaluations": len(examples),
            "backward_calls": 0,
            "model_updates": 0,
            "evidence_bytes": 8 * len(examples),
        },
        "model_loaded": False,
        "gpu_accessed": False,
        "result_path": RESULT_RELATIVE.as_posix(),
        "registered_predictions": dict(REGISTERED_PREDICTIONS),
        "causal_followup_semantics": {
            "paired_interchange": "the key, key order, prompt frame, and task family stay fixed; only the selected key and therefore copied endpoint change",
            "generic_copy_or_endpoint_prediction": "head 11.3 interchange should move the is-minus-are margin toward the donor across all four independently frozen selection families and both key orders",
            "grammar_specific_prediction": "head 11.3 should have little consistent effect because every prompt already contains both words and no grammatical-number decision is requested",
            "identification_limit": "these rows are development-only; a later intervention on them remains diagnostic and requires independent frozen replication",
        },
    }


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise CopyControlError("probe timestamp must be UTC")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _capability_summary(evidence: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in evidence:
        grouped[(str(row["family"]), str(row["selected_endpoint"]))].append(
            bool(row["correct"])
        )
    cells = []
    for (family, direction), values in sorted(grouped.items()):
        accuracy = sum(values) / len(values)
        cells.append({
            "family": family,
            "selected_endpoint": direction,
            "correct_count": sum(values),
            "example_count": len(values),
            "accuracy": accuracy,
            "minimum_accuracy": MIN_FAMILY_DIRECTION_ACCURACY,
            "passed": accuracy >= MIN_FAMILY_DIRECTION_ACCURACY,
        })
    capable = (
        len(cells) == 8
        and all(cell["example_count"] == 4 and cell["passed"] for cell in cells)
    )
    return {
        "development_copy_capable": capable,
        "global_accuracy": sum(bool(row["correct"]) for row in evidence) / len(evidence),
        "family_direction_cells": cells,
    }


def run_probe(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
    backend: object | None = None,
    force_dryrun: bool = False,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    env = os.environ if environment is None else environment
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if env.get(name) not in (None, "1"):
            raise CopyControlError(f"{name} must be absent or exactly '1'")
    batch, dryrun = compile_probe()
    if force_dryrun or env.get("BQLIB_DRYRUN") == "1" \
            or env.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True), flush=True)
        return dryrun

    result_path = (root.resolve() / RESULT_RELATIVE).resolve()
    if not result_path.is_relative_to(root.resolve()):
        raise CopyControlError("result path escapes the repository root")
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite development probe: {result_path}")
    started_utc, started = wall_clock(), timer()
    executor = backend if backend is not None else Bilin18TorchBackend.load("cuda")
    output = executor.native(batch, capture=False)
    finished, finished_utc = timer(), wall_clock()
    if len(output.answer_foil) != len(EXAMPLES) or output.captured:
        raise CopyControlError("backend output is not aligned or retained activations")
    evidence = []
    for index, (example, pair) in enumerate(zip(EXAMPLES, output.answer_foil)):
        if len(pair) != 2 or any(
            type(value) not in {int, float} or not math.isfinite(float(value))
            for value in pair
        ):
            raise CopyControlError("backend returned nonfinite logit evidence")
        answer_logit, foil_logit = float(pair[0]), float(pair[1])
        margin = answer_logit - foil_logit
        evidence.append({
            **_example_json(example),
            "answer_id": batch.answer_ids[index],
            "foil_id": batch.foil_ids[index],
            "answer_logit": answer_logit,
            "foil_logit": foil_logit,
            "answer_minus_foil_margin": margin,
            "correct": margin > 0.0,
        })
    capability = _capability_summary(evidence)
    predictions = dict(zip(
        (key for key, _text in REGISTERED_PREDICTIONS),
        (True, True, True, capability["development_copy_capable"]),
    ))
    result = {
        "schema": "circuit_fast_screen_dev_is_are_copy_result_v1",
        "request_id": REQUEST_ID,
        "split": SPLIT,
        "reuse_policy": REUSE_POLICY,
        "scientific_status": "excluded_development_only_not_screen_evidence",
        "bank_sha256": EXPECTED_BANK_SHA256,
        "started_utc": _utc_text(started_utc),
        "finished_utc": _utc_text(finished_utc),
        "runtime": {
            "serial_seconds": finished - started,
            "forward_calls": 1,
            "example_evaluations": len(EXAMPLES),
            "backward_calls": 0,
            "model_updates": 0,
            "evidence_bytes": 8 * len(EXAMPLES),
        },
        "predictions": predictions,
        "capability": capability,
        "causal_followup_semantics": dryrun["causal_followup_semantics"],
        "evidence": evidence,
    }
    payload = managed.atomic_create_json(result_path, result)
    summary = {
        "result_path": RESULT_RELATIVE.as_posix(),
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "predictions": predictions,
        "development_copy_capable": capability["development_copy_capable"],
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the excluded is/are natural-copy capability probe."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the one-forward plan without loading a model",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_probe(force_dryrun=bool(args.dry_run))


if __name__ == "__main__":
    main()
