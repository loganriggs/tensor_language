#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import hashlib
import importlib
import json

import pytest

import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer


CASES = (
    (
        "circuit_fast_screen_candidate_numeric_sequence_cross_construction",
        "run_circuit_fast_screen_numeric_sequence_cross_construction",
        "10df57201ab3382f115ffc2113cca5c94d0db91b4a8a0d81205edabc773413b1",
        "b526ed7590f6a6163aa91e9f44d83f2a555082882cdc8cfa7ba818749ec0bd2a",
        "791f65643c90217561c66b45eb2e122dd62edddcc25768bf5476b2158f21ac46",
        "48149954d75ac0375cd97545c9dd909b7cf3b56e43ba5fd2ab325fbe471d9308",
    ),
    (
        "circuit_fast_screen_candidate_sequence_control_choice",
        "run_circuit_fast_screen_sequence_control_choice",
        "65af0c92b39efa860fb4035902ac307e2ad85547259d91ce0b2756effe858716",
        "f52b4d74b9942252c0c89a44c02ee4f5199dc38c9f3064389136bb56f21549c4",
        "b3600841d8839dd88815182c8a1619952a4c4ae9e1456659dbaf7b7245d77680",
        "af3174683c2906d03fbf1c82d58f0528f81e53e880088322b55a4d33eb0ca472",
    ),
    (
        "circuit_fast_screen_candidate_p_family",
        "run_circuit_fast_screen_p_family",
        "5e6e2c8669c0d082e9b85ce2b925c6b21f8f875f4a47229ae4870de181a79489",
        "edbe4d4d0d4835aa31068630646941828b086013664c41d80f555bd1e7a68b40",
        "70af537ff79be361b579ff36d3610512f309f0e5a4d2bb47dc36bbc11c6653b6",
        "b0c7d043bb7e750973dc3b8d2561699b0e139ede3fd91ff880221f7dd8b793f4",
    ),
    (
        "circuit_fast_screen_candidate_a2_family",
        "run_circuit_fast_screen_a2_family",
        "c830386d049a3894e30f9b900cef8d26d4863d2b919371193f6a5cbcdfe39872",
        "33897b10dadb2530867c33d7b45a98ae5033cb348baaf2d631ca16d99842d1b7",
        "19fdd8c8de78ebd73fa9054d8fa3b873efdbb4dbe3217edf249508eb4591d428",
        "498a7e206976999eb9e81fe07385825bffdd3c3325c5bd258b2eec2c4760c223",
    ),
)


@pytest.mark.parametrize(
    "candidate_name,runner_name,row_sha,spec_sha,call_sha,compiled_sha", CASES,
)
def test_compatibility_modules_preserve_frozen_rows_specs_and_dryruns(
    candidate_name, runner_name, row_sha, spec_sha, call_sha, compiled_sha,
) -> None:
    candidate = importlib.import_module(candidate_name)
    runner = importlib.import_module(runner_name)
    rows = candidate.build_rows()
    observed_rows = hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert observed_rows == row_sha == candidate.authority_sha256()
    spec = runner.build_spec(rows)
    assert managed.framework.canonical_sha256(managed.screen.spec_json(spec)) == spec_sha
    dryrun = producer.compile_dryrun(spec, rows)
    assert dryrun["call_manifest_sha256"] == call_sha
    assert dryrun["compiled_sha256"] == compiled_sha
    assert dryrun["model_loaded"] is False
    assert dryrun["gpu_accessed"] is False


def _rows_by_transform(module_name: str) -> dict[str, list[dict]]:
    rows = importlib.import_module(module_name).build_rows()
    output: dict[str, list[dict]] = {}
    for row in rows:
        normalized = dict(row)
        normalized.pop("task_id")
        normalized.pop("group_id")
        output.setdefault(row["transform_id"], []).append(normalized)
    return output


def test_each_discriminator_changes_only_its_named_hypothesis_rows() -> None:
    baseline = _rows_by_transform(CASES[0][0])
    changed_c = _rows_by_transform(CASES[1][0])
    changed_p = _rows_by_transform(CASES[2][0])
    changed_a2 = _rows_by_transform(CASES[3][0])
    for unchanged in ("A1", "A2", "P"):
        assert baseline[unchanged] == changed_c[unchanged]
    for unchanged in ("A1", "A2", "C"):
        assert changed_c[unchanged] == changed_p[unchanged]
    for unchanged in ("A1", "P", "C"):
        assert changed_p[unchanged] == changed_a2[unchanged]
    assert baseline["C"] != changed_c["C"]
    assert changed_c["P"] != changed_p["P"]
    assert changed_p["A2"] != changed_a2["A2"]
