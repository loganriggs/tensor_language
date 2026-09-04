"""CPU closure tests for the first managed reusable circuit screen."""

from __future__ import annotations

import json

import circuit_fast_screen_candidates as candidates
import circuit_experiment_spec as framework
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import circuit_prior_art
import run_circuit_fast_screen_sentence_terminal as run


def test_reviewed_authority_prior_art_and_exact_padded_price() -> None:
    rows = candidates.build_rows(candidates.TASK_ID)
    spec = run.build_spec(rows)
    prior = json.loads(run.PRIOR_ART.read_text())
    assert circuit_prior_art.validate_source_files(prior, run.ROOT) == (
        "13a6d697b04e3d378f5d0a0ed293efa0983808d9ed633424a79346934f7b4f6d"
    )
    receipt = producer.compile_dryrun(spec, rows)
    assert receipt["authority_sha256"] == run.EXPECTED_AUTHORITY_SHA256
    assert receipt["active_price"] == {
        "phase": "FIT", "forward_calls": 228,
        "example_evaluations": 7296, "backward_calls": 0,
        "model_updates": 0, "evidence_bytes": 58368,
    }
    assert receipt["max_price"] == {
        "phase": "FIT", "forward_calls": 264,
        "example_evaluations": 8448, "backward_calls": 0,
        "model_updates": 0, "evidence_bytes": 67584,
    }


def test_variable_length_rows_compile_to_one_right_padded_call_per_family() -> None:
    rows = candidates.build_rows(candidates.TASK_ID)
    spec = run.build_spec(rows)
    compiled = screen.compile_screen(spec, rows)
    native = compiled["call_manifest"][:8]
    assert len(native) == 8
    assert all(call["shape_validation_mode"] ==
               "dynamic_batched_token_matrix_right_padded_v1" for call in native)
    assert all(call["logical_batch_size"] == 32 for call in native)
    assert len({len(row["base_ids"]) for row in rows}) > 1
    assert len({len(row["donor_ids"]) for row in rows}) > 1


def test_result_writer_is_create_only(tmp_path) -> None:
    target = tmp_path / "result.json"
    first = run.atomic_create_json(target, {"terminal": "null"})
    assert target.read_bytes() == first
    try:
        run.atomic_create_json(target, {"terminal": "screen"})
    except FileExistsError:
        pass
    else:  # pragma: no cover - create-only invariant
        raise AssertionError("result writer overwrote an existing receipt")
    assert target.read_bytes() == first


def test_dataclass_tuple_tree_converts_to_strict_literal_json() -> None:
    converted = run.literal_json({"outer": (("x", 1),), "empty": ()})
    assert converted == {"outer": [["x", 1]], "empty": []}
    assert framework.canonical_json_bytes(converted)
