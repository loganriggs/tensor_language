#!/usr/bin/env python3
# BQGATE: EXPERIMENT
"""One-forward DEVELOPMENT probe for an unrelated `` is``/`` are`` endpoint.

The examples use ``is`` and ``are`` as arbitrary response codes for four binary
tasks that do not ask for grammatical agreement.  Both code mappings and both
orders in which the code words are mentioned occur with both answers.  The bank
is disposable wording-development material: it is forbidden in FIT, SELECT,
TEST, and OOD, and it cannot by itself support a causal circuit claim.
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
    "circuits/dev_capability/is_are_unrelated_endpoint_control_v1.json"
)
RESULT = ROOT / RESULT_RELATIVE
REQUEST_ID = "dev-is-are-unrelated-endpoint-control-v1"
SPLIT = "DEVELOPMENT"
REUSE_POLICY = "forbidden_in_FIT_SELECT_TEST_OOD"
ENCODING = tiktoken.get_encoding("gpt2")
EXPECTED_BANK_SHA256 = "fe0a7915dc086e2f7e317b22040a13624563c45178eb16d4900f3c74e030097c"
MIN_CELL_ACCURACY = 0.75
REGISTERED_PREDICTIONS = (
    ("pred_a_runtime_one_native_batch", "Runtime performs exactly one native forward batch."),
    ("pred_b_balance_and_coverage", "All four families, mappings, label orders, truths, and answers retain their frozen balance."),
    ("pred_c_integrity_finite_aligned", "All returned logit pairs are finite and row-aligned."),
    ("pred_d_wording_capable", "Every family-by-mapping cell reaches 75% native accuracy, every family-by-mapping cell has a fully correct opposite-answer pair, and each endpoint reaches 75% accuracy."),
)


class EndpointControlError(ValueError):
    """The disposable bank, output evidence, or publication is malformed."""


@dataclass(frozen=True)
class ClaimPair:
    pair_id: str
    family: str
    mapping_id: str
    label_order: str
    true_claim: str
    false_claim: str


@dataclass(frozen=True)
class ProbeExample:
    row_id: str
    family: str
    mapping_id: str
    label_order: str
    counterfactual_pair_id: str
    remap_pair_id: str
    truth_value: str
    claim_text: str
    prompt: str
    answer: str
    foil: str


# Each pair keeps the instruction, mapping, label order, and task family fixed.
# Only the claim's truth changes, so the required endpoint changes as a result.
CLAIM_PAIRS = (
    ClaimPair("arithmetic.direct.is_first", "arithmetic", "yes_is", "is_first", "Seven plus five equals twelve.", "Seven plus five equals thirteen."),
    ClaimPair("arithmetic.direct.are_first", "arithmetic", "yes_is", "are_first", "Seven plus five equals twelve.", "Seven plus five equals thirteen."),
    ClaimPair("arithmetic.inverse.is_first", "arithmetic", "yes_are", "is_first", "Seven plus five equals twelve.", "Seven plus five equals thirteen."),
    ClaimPair("arithmetic.inverse.are_first", "arithmetic", "yes_are", "are_first", "Seven plus five equals twelve.", "Seven plus five equals thirteen."),
    ClaimPair("geography.direct.is_first", "geography", "yes_is", "is_first", "Lima belongs to Peru.", "Lima belongs to Chile."),
    ClaimPair("geography.direct.are_first", "geography", "yes_is", "are_first", "Lima belongs to Peru.", "Lima belongs to Chile."),
    ClaimPair("geography.inverse.is_first", "geography", "yes_are", "is_first", "Lima belongs to Peru.", "Lima belongs to Chile."),
    ClaimPair("geography.inverse.are_first", "geography", "yes_are", "are_first", "Lima belongs to Peru.", "Lima belongs to Chile."),
    ClaimPair("category.direct.is_first", "category", "yes_is", "is_first", "A robin belongs to the bird category.", "A robin belongs to the reptile category."),
    ClaimPair("category.direct.are_first", "category", "yes_is", "are_first", "A robin belongs to the bird category.", "A robin belongs to the reptile category."),
    ClaimPair("category.inverse.is_first", "category", "yes_are", "is_first", "A robin belongs to the bird category.", "A robin belongs to the reptile category."),
    ClaimPair("category.inverse.are_first", "category", "yes_are", "are_first", "A robin belongs to the bird category.", "A robin belongs to the reptile category."),
    ClaimPair("word.direct.is_first", "word_property", "yes_is", "is_first", "The word apple begins with A.", "The word apple begins with B."),
    ClaimPair("word.direct.are_first", "word_property", "yes_is", "are_first", "The word apple begins with A.", "The word apple begins with B."),
    ClaimPair("word.inverse.is_first", "word_property", "yes_are", "is_first", "The word apple begins with A.", "The word apple begins with B."),
    ClaimPair("word.inverse.are_first", "word_property", "yes_are", "are_first", "The word apple begins with A.", "The word apple begins with B."),
)


def _instruction(mapping_id: str, label_order: str) -> str:
    wording = {
        ("yes_is", "is_first"): "Use is for YES and are for NO.",
        ("yes_is", "are_first"): "Use are for NO and is for YES.",
        ("yes_are", "is_first"): "Use is for NO and are for YES.",
        ("yes_are", "are_first"): "Use are for YES and is for NO.",
    }
    try:
        return wording[(mapping_id, label_order)]
    except KeyError as error:
        raise EndpointControlError("unknown response mapping or label order") from error


def _endpoint(mapping_id: str, truth_value: str) -> str:
    is_answer = (mapping_id == "yes_is") == (truth_value == "yes")
    return " is" if is_answer else " are"


def _build_examples(pairs: Sequence[ClaimPair] = CLAIM_PAIRS) -> tuple[ProbeExample, ...]:
    output = []
    for pair in pairs:
        instruction = _instruction(pair.mapping_id, pair.label_order)
        for truth_value, claim in (("yes", pair.true_claim), ("no", pair.false_claim)):
            answer = _endpoint(pair.mapping_id, truth_value)
            output.append(ProbeExample(
                row_id=f"{pair.pair_id}.{truth_value}",
                family=pair.family,
                mapping_id=pair.mapping_id,
                label_order=pair.label_order,
                counterfactual_pair_id=pair.pair_id,
                remap_pair_id=f"{pair.family}.{pair.label_order}.{truth_value}.remap",
                truth_value=truth_value,
                claim_text=claim,
                prompt=(
                    "Answer with exactly one code word. " + instruction
                    + "\nClaim: " + claim + "\nCode word:"
                ),
                answer=answer,
                foil=" are" if answer == " is" else " is",
            ))
    return tuple(output)


EXAMPLES = _build_examples()


def _example_json(example: ProbeExample) -> dict[str, object]:
    return {**asdict(example), "split": SPLIT, "reuse_policy": REUSE_POLICY}


def _bank_sha256(examples: Sequence[ProbeExample]) -> str:
    return framework.canonical_sha256({
        "schema": "circuit_fast_screen_dev_is_are_endpoint_bank_v1",
        "scientific_status": "excluded_development_only",
        "examples": [_example_json(example) for example in examples],
    })


def _joint_token(prompt: str, continuation: str) -> tuple[tuple[int, ...], int]:
    prompt_ids = tuple(ENCODING.encode(prompt))
    complete = tuple(ENCODING.encode(prompt + continuation))
    standalone = tuple(ENCODING.encode(continuation))
    if not prompt_ids or len(standalone) != 1 or complete != prompt_ids + standalone:
        raise EndpointControlError("answer/foil is not one stable continuation token")
    return prompt_ids, standalone[0]


def _validate_balance(examples: Sequence[ProbeExample]) -> dict[str, object]:
    if len(examples) != 32:
        raise EndpointControlError("development bank must contain exactly 32 rows")
    row_ids = [example.row_id for example in examples]
    prompts = [example.prompt for example in examples]
    if len(set(row_ids)) != len(row_ids) or len(set(prompts)) != len(prompts):
        raise EndpointControlError("row IDs and prompts must be unique")
    pair_rows: dict[str, list[ProbeExample]] = defaultdict(list)
    remap_rows: dict[str, list[ProbeExample]] = defaultdict(list)
    for example in examples:
        if any(not isinstance(value, str) or not value for value in asdict(example).values()):
            raise EndpointControlError("example fields must be nonempty text")
        if len(re.findall(r"\bis\b", example.prompt.lower())) != 1 \
                or len(re.findall(r"\bare\b", example.prompt.lower())) != 1:
            raise EndpointControlError("each prompt must mention each endpoint exactly once")
        pair_rows[example.counterfactual_pair_id].append(example)
        remap_rows[example.remap_pair_id].append(example)
    if len(pair_rows) != 16 or set(map(len, pair_rows.values())) != {2}:
        raise EndpointControlError("counterfactual pair coverage changed")
    for rows in pair_rows.values():
        fixed = {
            (row.family, row.mapping_id, row.label_order, row.counterfactual_pair_id)
            for row in rows
        }
        if len(fixed) != 1 or {row.truth_value for row in rows} != {"yes", "no"} \
                or {row.answer for row in rows} != {" is", " are"}:
            raise EndpointControlError("pair does not isolate truth with opposite endpoints")
    if len(remap_rows) != 16 or set(map(len, remap_rows.values())) != {2}:
        raise EndpointControlError("response-remapping pair coverage changed")
    for rows in remap_rows.values():
        fixed = {
            (row.family, row.label_order, row.truth_value, row.claim_text)
            for row in rows
        }
        if len(fixed) != 1 or {row.mapping_id for row in rows} != {"yes_is", "yes_are"} \
                or {row.answer for row in rows} != {" is", " are"}:
            raise EndpointControlError("remap pair does not isolate the response code")

    expected_feature_counts = {
        "family": {"arithmetic": 8, "geography": 8, "category": 8, "word_property": 8},
        "mapping_id": {"yes_is": 16, "yes_are": 16},
        "label_order": {"is_first": 16, "are_first": 16},
        "truth_value": {"yes": 16, "no": 16},
        "answer": {" is": 16, " are": 16},
    }
    for field, expected in expected_feature_counts.items():
        observed = Counter(getattr(example, field) for example in examples)
        if dict(observed) != expected:
            raise EndpointControlError(f"{field} balance changed")
    # No registered nuisance feature alone identifies the endpoint: every value
    # of family, mapping, mention order, and truth occurs with both answers.
    for field in ("family", "mapping_id", "label_order", "truth_value", "claim_text"):
        answers_by_value: dict[str, set[str]] = defaultdict(set)
        for example in examples:
            answers_by_value[getattr(example, field)].add(example.answer)
        if any(answers != {" is", " are"} for answers in answers_by_value.values()):
            raise EndpointControlError(f"{field} perfectly predicts the endpoint")
    return {
        "families": expected_feature_counts["family"],
        "mapping_ids": expected_feature_counts["mapping_id"],
        "label_orders": expected_feature_counts["label_order"],
        "truth_values": expected_feature_counts["truth_value"],
        "answers": expected_feature_counts["answer"],
        "counterfactual_pairs": len(pair_rows),
        "response_remap_pairs": len(remap_rows),
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
            raise EndpointControlError("answer/foil tokenization is not aligned")
        token_rows.append(prompt_ids)
        answer_ids.append(answer_id)
        foil_ids.append(foil_id)
    digest = _bank_sha256(examples)
    if digest != EXPECTED_BANK_SHA256:
        raise EndpointControlError("development bank differs from its frozen digest")
    batch = ModelBatch(
        row_ids=tuple(example.row_id for example in examples),
        side="base",
        token_rows=tuple(token_rows),
        answer_ids=tuple(answer_ids),
        foil_ids=tuple(foil_ids),
        semantic_positions=tuple(len(row) - 1 for row in token_rows),
    )
    dryrun = {
        "schema": "circuit_fast_screen_dev_is_are_endpoint_dryrun_v1",
        "request_id": REQUEST_ID,
        "split": SPLIT,
        "reuse_policy": REUSE_POLICY,
        "scientific_status": "excluded_development_only",
        "bank_sha256": digest,
        "balance": balance,
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
            "truth_flip_interchange": "holds family, code mapping, label order, and instruction fixed while claim truth and required endpoint change",
            "response_remap_interchange": "holds family, exact claim and truth, and label order fixed while reversing the arbitrary YES/NO-to-is/are mapping and required endpoint",
            "generic_endpoint_prediction": "head 11.3 interchange should move the is-minus-are margin toward the donor for both independently frozen truth-flip and response-remap relations across all four families",
            "grammar_specific_prediction": "head 11.3 should have little consistent effect because no grammatical-number decision is requested",
            "identification_limit": "these rows are development-only; any later intervention on them is diagnostic and must be replicated on independently frozen text before a circuit claim",
        },
    }
    return batch, dryrun


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise EndpointControlError("probe timestamp must be UTC")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _capability_summary(evidence: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    endpoint_cells: dict[str, list[bool]] = defaultdict(list)
    pair_cells: dict[str, list[bool]] = defaultdict(list)
    pair_metadata: dict[str, tuple[str, str]] = {}
    for row in evidence:
        key = (str(row["family"]), str(row["mapping_id"]))
        cells[key].append(bool(row["correct"]))
        endpoint_cells[str(row["answer"])].append(bool(row["correct"]))
        pair_id = str(row["counterfactual_pair_id"])
        pair_cells[pair_id].append(bool(row["correct"]))
        pair_metadata[pair_id] = key
    family_mapping = []
    for (family, mapping_id), values in sorted(cells.items()):
        fully_correct_pairs = sum(
            all(outcomes) for pair_id, outcomes in pair_cells.items()
            if pair_metadata[pair_id] == (family, mapping_id)
        )
        accuracy = sum(values) / len(values)
        family_mapping.append({
            "family": family,
            "mapping_id": mapping_id,
            "correct_count": sum(values),
            "example_count": len(values),
            "accuracy": accuracy,
            "minimum_accuracy": MIN_CELL_ACCURACY,
            "fully_correct_counterfactual_pairs": fully_correct_pairs,
            "passed": accuracy >= MIN_CELL_ACCURACY and fully_correct_pairs >= 1,
        })
    endpoint_accuracy = {
        endpoint: sum(values) / len(values)
        for endpoint, values in sorted(endpoint_cells.items())
    }
    suitable = (
        len(family_mapping) == 8
        and all(cell["passed"] for cell in family_mapping)
        and set(endpoint_accuracy) == {" is", " are"}
        and all(value >= MIN_CELL_ACCURACY for value in endpoint_accuracy.values())
    )
    return {
        "development_wording_capable": suitable,
        "minimum_cell_accuracy": MIN_CELL_ACCURACY,
        "global_accuracy": sum(bool(row["correct"]) for row in evidence) / len(evidence),
        "endpoint_accuracy": endpoint_accuracy,
        "fully_correct_counterfactual_pair_count": sum(all(v) for v in pair_cells.values()),
        "family_mapping_cells": family_mapping,
    }


def run_probe(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
    backend: object | None = None,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    env = os.environ if environment is None else environment
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if env.get(name) not in (None, "1"):
            raise EndpointControlError(f"{name} must be absent or exactly '1'")
    batch, dryrun = compile_probe()
    if env.get("BQLIB_DRYRUN") == "1" or env.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True), flush=True)
        return dryrun

    result_path = (root.resolve() / RESULT_RELATIVE).resolve()
    if not result_path.is_relative_to(root.resolve()):
        raise EndpointControlError("result path escapes the repository root")
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite development probe: {result_path}")
    started_utc, started = wall_clock(), timer()
    executor = backend if backend is not None else Bilin18TorchBackend.load("cuda")
    output = executor.native(batch, capture=False)
    finished, finished_utc = timer(), wall_clock()
    if len(output.answer_foil) != len(EXAMPLES) or output.captured:
        raise EndpointControlError("backend output is not aligned or retained activations")
    evidence = []
    for index, (example, pair) in enumerate(zip(EXAMPLES, output.answer_foil)):
        if len(pair) != 2 or any(
            type(value) not in {int, float} or not math.isfinite(float(value))
            for value in pair
        ):
            raise EndpointControlError("backend returned nonfinite logit evidence")
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
        (True, True, True, capability["development_wording_capable"]),
    ))
    result = {
        "schema": "circuit_fast_screen_dev_is_are_endpoint_result_v1",
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
        "development_wording_capable": capability["development_wording_capable"],
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe unrelated binary tasks using is/are as response codes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the one-forward plan without loading the model",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        run_probe(environment={"BQLIB_DRYRUN": "1"})
    else:
        run_probe()


if __name__ == "__main__":
    main()
