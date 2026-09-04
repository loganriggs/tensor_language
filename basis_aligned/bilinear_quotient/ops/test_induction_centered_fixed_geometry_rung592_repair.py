#!/usr/bin/env python3
# BQLANE: cpu
"""Regression tests closing all seven strict R592 implementation-review xfails."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "basis_aligned" / "bilinear_quotient" / "ops"
PRODUCER = OPS / "induction_centered_fixed_geometry_rung592.py"
RUNTIME = OPS / "induction_centered_fixed_geometry_rung592_runtime.py"
ADAPTER = OPS / "execute_induction_centered_fixed_geometry_rung592.py"
FACADE = ROOT / "basis_aligned" / "polynomial_causal" / "bilin18_observed_model_facade.py"

SPEC = importlib.util.spec_from_file_location("r592_repair", PRODUCER)
assert SPEC and SPEC.loader
r592 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r592)


def constants(path: Path) -> dict[str, object]:
    output = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                output[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return output


@pytest.mark.parametrize("path", (PRODUCER, ADAPTER))
def test_authoritative_gate_accepts_repaired_entrypoints(path: Path) -> None:
    completed = subprocess.run(
        ["python", str(OPS / "gate.py"), str(path)], cwd=ROOT,
        text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_width_matches_pinned_checkpoint_topology_and_prices() -> None:
    assert constants(PRODUCER)["VOCAB"] == constants(FACADE)["LOGIT_VOCAB"] == 50_304
    schema = r592.phase_evidence_schema("FIT")
    assert schema["logit_differences.npy"]["shape"] == [3_744, 4, 50_304]
    dry = r592.build_dryrun()
    assert dry["evidence_data_bytes"] == {
        "FIT_logit_differences": 3_013_410_816,
        "SELECT_logit_differences": 1_506_705_408,
        "maximum_logit_differences": 4_520_116_224,
        "maximum_principal_raw_payload": 5_141_200_896,
    }


def one_call(kind: str):
    tokens = np.full((1, 30), r592.PAD_TOKEN, dtype="<i8")
    call = {
        "manifest_index": 0, "call_id": f"FIT:{kind}", "phase": "FIT",
        "call_kind": kind, "chunk_index": 0,
        "machine_arm": kind, "token_record_id": "FIT:toy",
        "token_sha256": r592.sha256_bytes(tokens.tobytes()), "batch_size": 1,
        "physical_width": 30, "authority_row_ids": ["e0"],
        "direction_ids": ["d0"], "query_positions": [0],
    }
    return {"phase": "FIT", "calls": [call], "token_arrays": {"FIT:toy": tokens}}, call, tokens


def valid_arrays(call, tokens):
    return {
        name: (tokens.copy() if name == "tokens.npy" else np.zeros(shape, dtype=dtype))
        for name, (dtype, shape) in r592.mandatory_call_shapes(call).items()
    }


def test_missing_mandatory_observation_is_hard_abort(tmp_path: Path) -> None:
    bundle, call, tokens = one_call("score")
    arrays = valid_arrays(call, tokens); arrays.pop("hook_deltas.npy")
    class Missing:
        def execute(self, *_args): return {"arrays": arrays}
    context = {call["call_id"]: {
        "specs": [{}], "planned": np.zeros((1, 4, 1152), dtype="<f4"), "cached": {},
    }}
    stage = tmp_path / "stage"; stage.mkdir()
    with pytest.raises(RuntimeError, match="incomplete call.*missing"):
        r592.run_manifest_calls(Missing(), bundle, context, stage=stage, public_root=tmp_path)
    assert not any((tmp_path / path.name).exists() for path in r592.PUBLIC_NAMESPACES)


def test_invalid_receipt_binds_mask_index_and_each_mask_byte(tmp_path: Path) -> None:
    bundle, call, tokens = one_call("score")
    arrays = valid_arrays(call, tokens)
    arrays["logits.npy"][0, 0] = np.nan
    arrays["hook_deltas.npy"][0, 0, 0] = np.inf
    class Nonfinite:
        def execute(self, *_args): return {"arrays": arrays}
    context = {call["call_id"]: {
        "specs": [{}], "planned": np.zeros((1, 4, 1152), dtype="<f4"),
        "cached": {"native_logits": np.zeros((1, r592.VOCAB), dtype="<f4")},
    }}
    stage = tmp_path / "stage"; stage.mkdir()
    r592.run_manifest_calls(Nonfinite(), bundle, context, stage=stage, public_root=tmp_path)
    receipt = json.loads((tmp_path / r592.INVALID_RECEIPT.name).read_text())
    names = set(receipt["evidence_files"])
    expected = {
        "calls/0000_FIT:score/nonfinite_mask_index.json",
        "calls/0000_FIT:score/nonfinite_masks/logits.mask.npy",
        "calls/0000_FIT:score/nonfinite_masks/hook_deltas.mask.npy",
    }
    assert expected <= names
    evidence = tmp_path / r592.INVALID_EVIDENCE.name
    for name, descriptor in receipt["evidence_files"].items():
        path = evidence / name
        assert descriptor == {"byte_length": path.stat().st_size, "sha256": r592.sha256_file(path)}


def test_native_full_gate_is_reconstructed_from_distinct_raw_arrays() -> None:
    call = {"batch_size": 1, "call_kind": "endpoint"}
    contract = r592.mandatory_call_shapes(call)
    assert contract["native_full_attention_write.npy"] == (
        np.dtype("<f4"), (1, 4, 1152)
    )
    runtime = RUNTIME.read_text()
    assert "_independent_full_attention_write" in runtime
    assert (
        'arrays["independent_full_native_write.npy"][local, site_index] = '
        'term["head_output"].float().numpy()'
    ) not in runtime
    producer = PRODUCER.read_text()
    evidence_body = producer[producer.index("def write_complete_phase_evidence"):]
    assert "native_full_write_reconstruction_max_abs" in evidence_body


def test_both_closed_memmaps_are_explicitly_fsynced_and_checked() -> None:
    source = PRODUCER.read_text()
    body = source[source.index("def write_complete_phase_evidence"):source.index("def publish_normal")]
    assert "_fsync_file(hook_path)" in body
    assert "_fsync_file(logit_path)" in body
    assert "if offset != nd" in body
    assert "np.isfinite(complete_array).all()" in body
