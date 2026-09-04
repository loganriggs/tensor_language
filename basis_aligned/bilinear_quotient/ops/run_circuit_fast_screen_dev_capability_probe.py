#!/usr/bin/env python3
# BQGATE: EXPERIMENT
"""Queueable DEVELOPMENT-only wording probe for fast-screen candidates.

These disposable examples may diagnose native next-token capability before an
authority is frozen.  They are scientifically excluded and MUST NOT be copied,
selected, or scored as FIT, SELECT, TEST, or OOD evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import tiktoken

import circuit_experiment_spec as framework
import circuit_fast_screen_managed_runner as managed
from circuit_fast_screen_producer import Bilin18TorchBackend, ModelBatch


ROOT = Path(__file__).resolve().parent.parent
RESULT_RELATIVE = Path("circuits/dev_capability/native_wording_probe_v1.json")
RESULT = ROOT / RESULT_RELATIVE
REQUEST_ID = "dev-native-wording-probe-v1"
SPLIT = "DEVELOPMENT"
REUSE_POLICY = "forbidden_in_FIT_SELECT_TEST_OOD"
ENCODING = tiktoken.get_encoding("gpt2")
EXPECTED_BANK_SHA256 = "c6f562a4dca314fb407799d6d83d265e40618a2a5c788f35d775078c36b98773"
REGISTERED_PREDICTIONS = (
    ("pred_a_runtime_one_native_batch", "Runtime performs exactly one native forward batch."),
    ("pred_b_coverage_complete", "Every registered template and answer cell is scored."),
    ("pred_c_integrity_finite_aligned", "All returned logit pairs are finite and row-aligned."),
)


@dataclass(frozen=True)
class ProbeExample:
    row_id: str
    behavior: str
    template_id: str
    prompt: str
    answer: str
    foil: str


def _example(
    row_id: str, behavior: str, template_id: str, prompt: str,
    answer: str, foil: str,
) -> ProbeExample:
    return ProbeExample(row_id, behavior, template_id, prompt, answer, foil)


EXAMPLES = (
    # Quote endpoint controls: the quote token is explicitly an inch mark, never
    # a delimiter. Each phrasing also contains matched complete-sentence rows.
    _example("quote.explicit.00", "quote_control", "explicit_instruction",
             "Add only the final punctuation. This is a complete sentence: The inventory contains exactly 24 items", ".", '"'),
    _example("quote.explicit.01", "quote_control", "explicit_instruction",
             "Add only the unit mark. In inches, the shelf measures exactly 24", '"', "."),
    _example("quote.explicit.02", "quote_control", "explicit_instruction",
             "Add only the final punctuation. This is a complete sentence: The cabinet contains exactly 17 trays", ".", '"'),
    _example("quote.explicit.03", "quote_control", "explicit_instruction",
             "Add only the unit mark. In inches, the frame measures exactly 17", '"', "."),
    _example("quote.natural.00", "quote_control", "natural_completion",
             "The clerk finished the count and wrote: The inventory contains 31 items", ".", '"'),
    _example("quote.natural.01", "quote_control", "natural_completion",
             "The carpenter recorded the shelf's length in inches as 31", '"', "."),
    _example("quote.natural.02", "quote_control", "natural_completion",
             "The keeper finished the count and wrote: The archive contains 12 boxes", ".", '"'),
    _example("quote.natural.03", "quote_control", "natural_completion",
             "The tailor recorded the ribbon's length in inches as 12", '"', "."),
    _example("quote.label.00", "quote_control", "labeled_context",
             "Sentence ending: The warehouse holds exactly 46 crates", ".", '"'),
    _example("quote.label.01", "quote_control", "labeled_context",
             "Customary inch notation: The board measures 46", '"', "."),
    _example("quote.label.02", "quote_control", "labeled_context",
             "Sentence ending: The gallery holds exactly 28 paintings", ".", '"'),
    _example("quote.label.03", "quote_control", "labeled_context",
             "Customary inch notation: The pipe measures 28", '"', "."),
    _example("quote.report.00", "quote_control", "reporting_context",
             "The report ends with this complete statement: The room contains 19 chairs", ".", '"'),
    _example("quote.report.01", "quote_control", "reporting_context",
             "The blueprint gives this measurement in inches: 19", '"', "."),
    _example("quote.report.02", "quote_control", "reporting_context",
             "The report ends with this complete statement: The yard contains 33 trees", ".", '"'),
    _example("quote.report.03", "quote_control", "reporting_context",
             "The diagram gives this measurement in inches: 33", '"', "."),

    # Pronoun task wording: explicit genders and ordinary continuations, with
    # both answers represented inside every template.
    _example("pronoun.active.00", "pronoun_antecedent", "named_active",
             "Alice is a woman and Bob is a man. Alice carried the parcel. The correct pronoun for the person who carried it is", " she", " he"),
    _example("pronoun.active.01", "pronoun_antecedent", "named_active",
             "Alice is a woman and Bob is a man. Bob opened the cabinet. The correct pronoun for the person who opened it is", " he", " she"),
    _example("pronoun.active.02", "pronoun_antecedent", "named_active",
             "Nora is a woman and Liam is a man. Nora moved the lantern. The correct pronoun for the person who moved it is", " she", " he"),
    _example("pronoun.active.03", "pronoun_antecedent", "named_active",
             "Nora is a woman and Liam is a man. Liam repaired the bicycle. The correct pronoun for the person who repaired it is", " he", " she"),
    _example("pronoun.passive.00", "pronoun_antecedent", "named_passive",
             "Clara is a woman and David is a man. The package was delivered by Clara. The correct pronoun for the person who delivered it is", " she", " he"),
    _example("pronoun.passive.01", "pronoun_antecedent", "named_passive",
             "Clara is a woman and David is a man. The package was delivered by David. The correct pronoun for the person who delivered it is", " he", " she"),
    _example("pronoun.passive.02", "pronoun_antecedent", "named_passive",
             "Maya is a woman and Owen is a man. The gate was unlocked by Maya. The correct pronoun for the person who unlocked it is", " she", " he"),
    _example("pronoun.passive.03", "pronoun_antecedent", "named_passive",
             "Maya is a woman and Owen is a man. The gate was unlocked by Owen. The correct pronoun for the person who unlocked it is", " he", " she"),
    _example("pronoun.question.00", "pronoun_antecedent", "question_answer",
             "Eva is a woman and Noah is a man. Eva signed the form. Who signed the form? The correct pronoun for that person is", " she", " he"),
    _example("pronoun.question.01", "pronoun_antecedent", "question_answer",
             "Eva is a woman and Noah is a man. Noah signed the form. Who signed the form? The correct pronoun for that person is", " he", " she"),
    _example("pronoun.question.02", "pronoun_antecedent", "question_answer",
             "Iris is a woman and Felix is a man. Iris rang the bell. Who rang the bell? The correct pronoun for that person is", " she", " he"),
    _example("pronoun.question.03", "pronoun_antecedent", "question_answer",
             "Iris is a woman and Felix is a man. Felix rang the bell. Who rang the bell? The correct pronoun for that person is", " he", " she"),
    _example("pronoun.relative.00", "pronoun_antecedent", "relative_continuation",
             "Ruth is a woman and Simon is a man. Ruth thanked Simon after lunch. The correct pronoun for the one who gave thanks is", " she", " he"),
    _example("pronoun.relative.01", "pronoun_antecedent", "relative_continuation",
             "Ruth is a woman and Simon is a man. Simon thanked Ruth after lunch. The correct pronoun for the one who gave thanks is", " he", " she"),
    _example("pronoun.relative.02", "pronoun_antecedent", "relative_continuation",
             "Zoe is a woman and Ethan is a man. Zoe greeted Ethan at noon. The correct pronoun for the one who gave the greeting is", " she", " he"),
    _example("pronoun.relative.03", "pronoun_antecedent", "relative_continuation",
             "Zoe is a woman and Ethan is a man. Ethan greeted Zoe at noon. The correct pronoun for the one who gave the greeting is", " he", " she"),

    # Cross-domain comparison wording: two quantity domains use more/less and
    # one score domain uses higher/lower, with both directions in every template.
    _example("compare.inventory.00", "scalar_comparison", "inventory_quantity",
             "Crate A contains 17 bolts. Crate B contains 9 bolts. Compared with crate B, crate A contains", " more", " less"),
    _example("compare.inventory.01", "scalar_comparison", "inventory_quantity",
             "Crate A contains 6 gears. Crate B contains 14 gears. Compared with crate B, crate A contains", " less", " more"),
    _example("compare.inventory.02", "scalar_comparison", "inventory_quantity",
             "Bin A contains 23 washers. Bin B contains 11 washers. Compared with bin B, bin A contains", " more", " less"),
    _example("compare.inventory.03", "scalar_comparison", "inventory_quantity",
             "Bin A contains 8 valves. Bin B contains 19 valves. Compared with bin B, bin A contains", " less", " more"),
    _example("compare.distance.00", "scalar_comparison", "travel_quantity",
             "Route Cedar is 42 miles long. Route Pine is 27 miles long. Route Cedar covers", " more", " less"),
    _example("compare.distance.01", "scalar_comparison", "travel_quantity",
             "Route Cedar is 18 miles long. Route Pine is 35 miles long. Route Cedar covers", " less", " more"),
    _example("compare.distance.02", "scalar_comparison", "travel_quantity",
             "Trail Amber is 29 miles long. Trail Slate is 13 miles long. Trail Amber covers", " more", " less"),
    _example("compare.distance.03", "scalar_comparison", "travel_quantity",
             "Trail Amber is 16 miles long. Trail Slate is 38 miles long. Trail Amber covers", " less", " more"),
    _example("compare.score.00", "scalar_comparison", "assessment_score",
             "Leah scored 91 points and Mark scored 74 points. Leah's score was", " higher", " lower"),
    _example("compare.score.01", "scalar_comparison", "assessment_score",
             "Leah scored 63 points and Mark scored 82 points. Leah's score was", " lower", " higher"),
    _example("compare.score.02", "scalar_comparison", "assessment_score",
             "Team Quartz scored 88 points and Team Onyx scored 57 points. Quartz's score was", " higher", " lower"),
    _example("compare.score.03", "scalar_comparison", "assessment_score",
             "Team Quartz scored 49 points and Team Onyx scored 76 points. Quartz's score was", " lower", " higher"),
)


class DevelopmentProbeError(ValueError):
    """The disposable bank, backend evidence, or publication is malformed."""


def _example_json(example: ProbeExample) -> dict[str, object]:
    return {
        **asdict(example),
        "split": SPLIT,
        "reuse_policy": REUSE_POLICY,
    }


def _bank_sha256(examples: Sequence[ProbeExample]) -> str:
    return framework.canonical_sha256({
        "schema": "circuit_fast_screen_dev_probe_bank_v1",
        "scientific_status": "excluded_development_only",
        "examples": [_example_json(example) for example in examples],
    })


def _joint_token(prompt: str, continuation: str) -> tuple[tuple[int, ...], int]:
    prompt_ids = tuple(ENCODING.encode(prompt))
    complete = tuple(ENCODING.encode(prompt + continuation))
    standalone = tuple(ENCODING.encode(continuation))
    if not prompt_ids or len(standalone) != 1 or complete != prompt_ids + standalone:
        raise DevelopmentProbeError("answer/foil is not one stable continuation token")
    return prompt_ids, standalone[0]


def compile_probe(
    examples: Sequence[ProbeExample] = EXAMPLES,
) -> tuple[ModelBatch, dict[str, object]]:
    if not isinstance(examples, (tuple, list)) or not examples:
        raise DevelopmentProbeError("development bank must be a nonempty literal sequence")
    row_ids = [example.row_id for example in examples]
    prompts = [example.prompt for example in examples]
    if len(row_ids) != len(set(row_ids)) or len(prompts) != len(set(prompts)):
        raise DevelopmentProbeError("development row IDs and prompts must be unique")
    expected_templates = {
        "quote_control": 4, "pronoun_antecedent": 4, "scalar_comparison": 3,
    }
    templates: dict[str, set[str]] = defaultdict(set)
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    token_rows, answer_ids, foil_ids = [], [], []
    for example in examples:
        if any(not isinstance(value, str) or not value for value in asdict(example).values()):
            raise DevelopmentProbeError("development example fields must be nonempty text")
        if example.answer == example.foil:
            raise DevelopmentProbeError("development answer and foil must differ")
        prompt_ids, answer_id = _joint_token(example.prompt, example.answer)
        foil_prompt_ids, foil_id = _joint_token(example.prompt, example.foil)
        if foil_prompt_ids != prompt_ids or answer_id == foil_id:
            raise DevelopmentProbeError("answer/foil tokenization is not aligned")
        templates[example.behavior].add(example.template_id)
        cell_counts[(example.behavior, example.template_id, example.answer)] += 1
        token_rows.append(prompt_ids)
        answer_ids.append(answer_id)
        foil_ids.append(foil_id)
    if {name: len(values) for name, values in templates.items()} != expected_templates:
        raise DevelopmentProbeError("development template census changed")
    if set(cell_counts.values()) != {2} or len(cell_counts) != 2 * sum(expected_templates.values()):
        raise DevelopmentProbeError("each template must contain two examples per answer")
    digest = _bank_sha256(examples)
    if digest != EXPECTED_BANK_SHA256:
        raise DevelopmentProbeError("development bank differs from its frozen digest")
    batch = ModelBatch(
        row_ids=tuple(row_ids),
        side="base",
        token_rows=tuple(token_rows),
        answer_ids=tuple(answer_ids),
        foil_ids=tuple(foil_ids),
        semantic_positions=tuple(len(row) - 1 for row in token_rows),
    )
    dryrun = {
        "schema": "circuit_fast_screen_dev_capability_dryrun_v1",
        "request_id": REQUEST_ID,
        "split": SPLIT,
        "reuse_policy": REUSE_POLICY,
        "scientific_status": "excluded_development_only",
        "bank_sha256": digest,
        "behavior_count": len(templates),
        "template_count": sum(len(values) for values in templates.values()),
        "example_count": len(examples),
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
    }
    return batch, dryrun


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise DevelopmentProbeError("probe timestamp must be UTC")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


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
            raise DevelopmentProbeError(f"{name} must be absent or exactly '1'")
    batch, dryrun = compile_probe()
    if env.get("BQLIB_DRYRUN") == "1" or env.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True), flush=True)
        return dryrun

    result_path = (root.resolve() / RESULT_RELATIVE).resolve()
    if not result_path.is_relative_to(root.resolve()):
        raise DevelopmentProbeError("result path escapes the repository root")
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite development probe: {result_path}")
    started_utc, started = wall_clock(), timer()
    executor = backend if backend is not None else Bilin18TorchBackend.load("cuda")
    output = executor.native(batch, capture=False)
    finished, finished_utc = timer(), wall_clock()
    pairs = output.answer_foil
    if len(pairs) != len(EXAMPLES):
        raise DevelopmentProbeError("backend output count differs from development bank")
    if output.captured:
        raise DevelopmentProbeError("native development probe retained activation state")
    evidence = []
    coverage: Counter[tuple[str, str, str]] = Counter()
    summaries: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for example, pair in zip(EXAMPLES, pairs):
        if len(pair) != 2 or any(
            type(value) not in {int, float} or not math.isfinite(float(value))
            for value in pair
        ):
            raise DevelopmentProbeError("backend returned nonfinite logit evidence")
        answer_logit, foil_logit = float(pair[0]), float(pair[1])
        margin = answer_logit - foil_logit
        correct = margin > 0.0
        key = (example.behavior, example.template_id, example.answer)
        coverage[key] += 1
        summaries[key].append(correct)
        evidence.append({
            **_example_json(example),
            "answer_id": batch.answer_ids[len(evidence)],
            "foil_id": batch.foil_ids[len(evidence)],
            "answer_logit": answer_logit,
            "foil_logit": foil_logit,
            "answer_minus_foil_margin": margin,
            "correct": correct,
        })
    expected_coverage = Counter(
        (example.behavior, example.template_id, example.answer) for example in EXAMPLES
    )
    pred_a_runtime_one_native_batch = True
    pred_b_coverage_complete = coverage == expected_coverage
    pred_c_integrity_finite_aligned = len(evidence) == len(EXAMPLES)
    prediction_values = (
        pred_a_runtime_one_native_batch,
        pred_b_coverage_complete,
        pred_c_integrity_finite_aligned,
    )
    predictions = dict(zip(
        (key for key, _text in REGISTERED_PREDICTIONS), prediction_values,
    ))
    cells = [
        {
            "behavior": behavior,
            "template_id": template_id,
            "answer": answer,
            "correct_count": sum(outcomes),
            "example_count": len(outcomes),
            "accuracy": sum(outcomes) / len(outcomes),
        }
        for (behavior, template_id, answer), outcomes in sorted(summaries.items())
    ]
    result = {
        "schema": "circuit_fast_screen_dev_capability_result_v1",
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
        "template_answer_accuracy": cells,
        "evidence": evidence,
    }
    payload = managed.atomic_create_json(result_path, result)
    summary = {
        "result_path": RESULT_RELATIVE.as_posix(),
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "predictions": predictions,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    run_probe()


if __name__ == "__main__":
    main()
