#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted in the result payload below.
"""Run the first reusable circuit screen through the managed GPU lane.

This is a screen, not an identification or adoption claim.  It tests exact
donor-to-recipient state replacement at one final input position over every
residual boundary and whole attention/MLP output.  Attention heads are opened
only inside the best passing attention module.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import circuit_experiment_spec as framework
import circuit_fast_screen_candidates as candidates
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_ledger as ledger
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import circuit_prior_art


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/fast_screen_sentence_terminal_prior_art.json"
RESULT_RELATIVE = Path(
    "circuits/fast_screens/sentence_terminal_semantic_choice_v1_result.json"
)
RESULT = ROOT / RESULT_RELATIVE
LEDGER = ROOT / "circuits/fast_screen_ledger.jsonl"
REQUEST_ID = "sentence-terminal-semantic-choice-v1"
EXPERIMENT_ID = "fast-screen-sentence-terminal-semantic-choice-v1"
EXPECTED_AUTHORITY_SHA256 = (
    "d0da3cda58fa77e93f982932f9a890af8b77d9e0162f5144c2cb9288004a81ab"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_spec(rows: list[dict[str, object]]) -> screen.CircuitFastScreenSpec:
    authority_sha256 = candidates.validate_rows(rows)
    if authority_sha256 != EXPECTED_AUTHORITY_SHA256:
        raise RuntimeError("sentence-terminal authority differs from the reviewed digest")
    return screen.CircuitFastScreenSpec(
        experiment_id=EXPERIMENT_ID,
        hypothesis=screen.CandidateHypothesis(
            behavior=candidates.TASK_ID,
            answer_score=screen.ANSWER_SCORE,
            information_read=(
                "whether the unfinished sentence is declarative or interrogative"
            ),
            proposed_operation=(
                "carry that sentence-mode state across two syntactic constructions"
            ),
            proposed_write="evidence for a period or question mark",
            candidate_sites=screen.CEILING_SITE_IDS,
            alternative_explanation=(
                "a construction-specific word cue or generic punctuation-token service"
            ),
            circuit_prediction=(
                "one site transfers both target constructions while sparing both controls"
            ),
            opposing_null_prediction=(
                "native capability fails or no site transfers both constructions selectively"
            ),
        ),
        task=candidates.TASK_SPEC,
        authority_sha256=authority_sha256,
        expected_fit_rows=len(rows),
        batch_size=32,
        semantic_position=screen.SemanticPositionSpec(
            role="final input token before the predicted punctuation",
            recipient_field="base_semantic_position",
            donor_field="donor_semantic_position",
        ),
        fields=screen.AuthorityFieldSpec(),
        bars=kernel.FIXED_BARS,
        declared_max_price=screen.battery.ExactPhasePrice(
            phase="FIT",
            forward_calls=264,
            example_evaluations=8448,
            backward_calls=0,
            model_updates=0,
            evidence_bytes=67584,
        ),
    )


def atomic_create_json(path: Path, value: object) -> bytes:
    payload = framework.canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o664)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("result write made no progress")
            written += count
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def selected_controls_pass(run: producer.FastScreenRun) -> bool:
    if run.selected_site is None:
        return False
    matched = [
        result for result in run.site_results if result.site == run.selected_site
    ]
    return bool(
        len(matched) == 1
        and matched[0].p_invariance_effect is not None
        and matched[0].p_invariance_effect <= kernel.MAX_P_INVARIANCE_EFFECT
        and matched[0].c_absolute_recovery is not None
        and matched[0].c_absolute_recovery <= kernel.MAX_C_ABSOLUTE_RECOVERY
    )


def main() -> None:
    rows = candidates.build_rows(candidates.TASK_ID)
    spec = build_spec(rows)
    prior = json.loads(PRIOR_ART.read_text())
    prior_sha256 = circuit_prior_art.validate_source_files(prior, ROOT)
    dryrun = producer.compile_dryrun(spec, rows)
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({
            "dryrun": dryrun,
            "prior_art_sha256": prior_sha256,
            "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        }, sort_keys=True))
        return
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite prior screen result: {RESULT}")

    started = utc_now()
    run = producer.run_science(spec, rows)
    finished = utc_now()
    serial_seconds = (finished - started).total_seconds()
    pred_a = bool(run.capability_cells) and all(
        cell.passed for cell in run.capability_cells
    )
    pred_b = run.terminal == "screen"
    pred_c = selected_controls_pass(run)
    spec_sha256 = framework.canonical_sha256(screen.spec_json(spec))
    result = {
        "schema": "circuit_fast_screen_result_v1",
        "request_id": REQUEST_ID,
        "candidate_id": candidates.TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "screen_tier_only": True,
        "prior_art_sha256": prior_sha256,
        "spec_sha256": spec_sha256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "started_utc": utc_text(started),
        "finished_utc": utc_text(finished),
        "serial_seconds": serial_seconds,
        "dryrun": dryrun,
        "terminal": run.terminal,
        "reason": run.reason,
        "selected_site_id": (
            None if run.selected_site is None else run.selected_site.site_id
        ),
        "head_stage": run.head_stage,
        "predictions": {
            "pred_a_native_capability": pred_a,
            "pred_b_cross_construction_transfer": pred_b,
            "pred_c_controls_selective": pred_c,
        },
        "fixed_bars": asdict(kernel.FIXED_BARS),
        "run": asdict(run),
    }
    payload = atomic_create_json(RESULT, result)
    result_sha256 = hashlib.sha256(payload).hexdigest()
    max_price = dryrun["max_price"]
    active_evaluations = run.timing.example_evaluations
    ledger_entry = {
        "request_id": REQUEST_ID,
        "candidate_id": candidates.TASK_ID,
        "started_utc": utc_text(started),
        "finished_utc": utc_text(finished),
        "serial_seconds": serial_seconds,
        "prior_art_sha256": prior_sha256,
        "spec_sha256": spec_sha256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "result_path": RESULT_RELATIVE.as_posix(),
        "result_sha256": result_sha256,
        "terminal": run.terminal,
        "reasons": [] if run.terminal == "screen" else [run.reason],
        "selected_site_id": (
            None if run.selected_site is None else run.selected_site.site_id
        ),
        "active_forward_calls": run.timing.forward_calls,
        "active_example_evaluations": active_evaluations,
        "active_evidence_bytes": 8 * active_evaluations,
        "max_forward_calls": max_price["forward_calls"],
        "max_example_evaluations": max_price["example_evaluations"],
        "max_evidence_bytes": max_price["evidence_bytes"],
        "relation": prior["relation"],
        "novelty": prior["novelty_delta"],
    }
    ledger.append_entry(LEDGER, ledger_entry, result_root=ROOT)
    print(json.dumps({
        "terminal": run.terminal,
        "reason": run.reason,
        "selected_site_id": ledger_entry["selected_site_id"],
        "forward_calls": run.timing.forward_calls,
        "example_evaluations": active_evaluations,
        "serial_seconds": serial_seconds,
        "native_capability": pred_a,
        "cross_construction_transfer": pred_b,
        "controls_selective": pred_c,
        "result_sha256": result_sha256,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
