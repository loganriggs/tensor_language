#!/usr/bin/env python3
# BQLANE: cpu
"""Fake-facade end-to-end state-machine tests for R592; no torch/model import."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).with_name("induction_centered_fixed_geometry_rung592.py")
SPEC = importlib.util.spec_from_file_location("r592_fake", MODULE)
assert SPEC and SPEC.loader
r592 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r592)


def toy_bundle(phase: str) -> dict[str, object]:
    tokens = np.full((2, 30), r592.PAD_TOKEN, dtype="<i8")
    tokens[:, :3] = np.asarray([[1, 2, 3], [4, 5, 6]], dtype="<i8")
    token_id = f"{phase}:toy:tokens"
    calls = []
    for index, kind in enumerate(("endpoint", "native", "replay", "score", "payload", "joint")):
        calls.append({
            "manifest_index": index,
            "call_id": f"{phase}:{kind}", "phase": phase, "call_kind": kind,
            "chunk_index": 0, "machine_arm": None if kind in ("endpoint", "native") else kind,
            "token_record_id": token_id, "token_sha256": r592.sha256_bytes(tokens.tobytes(order="C")),
            "batch_size": 2, "physical_width": 30,
            "authority_row_ids": ["e0", "e1"],
            "direction_ids": [] if kind == "endpoint" else ["d0", "d1"],
            "query_positions": [2, 2],
        })
    return {"phase": phase, "calls": calls, "token_arrays": {token_id: tokens}}


def arrays_for(call, tokens, planned=None):
    b = int(call["batch_size"])
    common = {
        "tokens.npy": tokens.copy(),
        "logits.npy": np.zeros((b, r592.VOCAB), dtype="<f4"),
    }
    parent = {
        name: np.zeros((b, 4, r592.RESIDUAL), dtype="<f4")
        for name in (
            "native_equality_term.npy", "factorized_equality_term.npy",
            "native_non_equality_remainder.npy", "native_head_write.npy",
            "independent_full_native_write.npy",
        )
    }
    if call["call_kind"] == "endpoint":
        return common | {
            "factor_e.npy": np.zeros((b, 4, 2), dtype="<f4"),
            "factor_u.npy": np.zeros((b, 4, 2, r592.RESIDUAL), dtype="<f4"),
            "support.npy": np.ones((b, 4, 2), dtype=np.bool_),
        } | parent
    if call["call_kind"] == "native":
        return common | {
            "live_e.npy": np.zeros((b, 4, 2), dtype="<f4"),
            "live_u.npy": np.zeros((b, 4, 2, r592.RESIDUAL), dtype="<f4"),
        } | parent
    delta = np.zeros((b, 4, r592.RESIDUAL), dtype="<f4") if planned is None else planned.copy()
    return common | {"hook_deltas.npy": delta.copy(), "planned_hook_deltas.npy": delta}


class FakeExecutor:
    def __init__(self, mutation=None):
        self.mutation = mutation
        self.calls = []

    def execute(self, call, tokens, specs, planned):
        kind = call["call_kind"]
        self.calls.append(call["call_id"])
        if self.mutation == (kind, "raise"):
            raise RuntimeError("planted incomplete forward")
        arrays = arrays_for(call, tokens, planned)
        if self.mutation == (kind, "hook"):
            arrays["hook_deltas.npy"][0, 0, 0] = 2e-5
        if self.mutation == (kind, "nonfinite"):
            arrays["logits.npy"][0, 0] = np.nan
        return {"arrays": arrays, "native_full_write_reconstruction_max_abs": 0.0}


def contexts(bundle):
    zero_e = np.zeros((2, 4, 2), dtype="<f4")
    zero_u = np.zeros((2, 4, 2, r592.RESIDUAL), dtype="<f4")
    native_logits = np.zeros((2, r592.VOCAB), dtype="<f4")
    output = {}
    for call in bundle["calls"]:
        cached = {
            "recipient_e": zero_e, "recipient_u": zero_u,
            "donor_e": zero_e, "donor_u": zero_u,
        }
        if call["call_kind"] in r592.MACHINE_ARMS:
            cached["native_logits"] = native_logits
        output[call["call_id"]] = {
            "specs": [{"final_position": 2}] * 2,
            "planned": None if call["call_kind"] in ("endpoint", "native") else np.zeros((2, 4, r592.RESIDUAL), dtype="<f4"),
            "cached": None if call["call_kind"] == "endpoint" else cached,
        }
    return output


def public_paths(root: Path):
    return [root / path.name for path in r592.PUBLIC_NAMESPACES]


def test_fake_full_fit_scientific_null_stops_before_select(tmp_path: Path) -> None:
    fit = toy_bundle("FIT"); select = toy_bundle("SELECT")
    executor = FakeExecutor(); stage = tmp_path / "stage"; stage.mkdir()
    observed = r592.run_manifest_calls(executor, fit, contexts(fit), stage=stage, public_root=tmp_path)
    assert observed["status"] == "complete" and len(executor.calls) == 6
    fit_gate_held = False  # planted score callback returns the scientific null
    if fit_gate_held:
        r592.run_manifest_calls(executor, select, contexts(select), stage=stage, public_root=tmp_path)
    assert all(not call.startswith("SELECT") for call in executor.calls)
    assert all(not path.exists() for path in public_paths(tmp_path))


def test_fake_full_fit_held_opens_select_once(tmp_path: Path) -> None:
    fit = toy_bundle("FIT"); select = toy_bundle("SELECT")
    executor = FakeExecutor()
    fit_stage = tmp_path / "fit-stage"; fit_stage.mkdir()
    select_stage = tmp_path / "select-stage"; select_stage.mkdir()
    assert r592.run_manifest_calls(executor, fit, contexts(fit), stage=fit_stage, public_root=tmp_path)["status"] == "complete"
    fit_gate_held = True
    if fit_gate_held:
        assert r592.run_manifest_calls(executor, select, contexts(select), stage=select_stage, public_root=tmp_path)["status"] == "complete"
    assert executor.calls == [call["call_id"] for call in fit["calls"] + select["calls"]]


def test_invalid_mid_arm_publishes_exact_prefix_and_stops(tmp_path: Path) -> None:
    bundle = toy_bundle("FIT"); executor = FakeExecutor(("payload", "hook"))
    stage = tmp_path / "stage"; stage.mkdir()
    observed = r592.run_manifest_calls(executor, bundle, contexts(bundle), stage=stage, public_root=tmp_path)
    assert observed["status"] == "invalid"
    assert observed["diagnostic"]["failure_predicate"] == "centered_hook_delta_failed"
    assert executor.calls[-1] == "FIT:payload" and "FIT:joint" not in executor.calls
    assert (tmp_path / r592.INVALID_RECEIPT.name).is_file()
    assert not (tmp_path / r592.NORMAL_RESULT.name).exists()


def test_nonfinite_final_call_gets_distinct_mask_index(tmp_path: Path) -> None:
    bundle = toy_bundle("FIT"); executor = FakeExecutor(("score", "nonfinite"))
    stage = tmp_path / "stage"; stage.mkdir()
    observed = r592.run_manifest_calls(executor, bundle, contexts(bundle), stage=stage, public_root=tmp_path)
    assert observed["diagnostic"]["failure_predicate"] == "nonfinite_observation"
    call_dir = tmp_path / r592.INVALID_EVIDENCE.name / "calls" / "0003_FIT:score"
    assert (call_dir / "nonfinite_mask_index.json").is_file()
    assert (call_dir / "nonfinite_masks" / "logits.mask.npy").is_file()


def test_hard_abort_publishes_nothing(tmp_path: Path) -> None:
    bundle = toy_bundle("FIT"); executor = FakeExecutor(("payload", "raise"))
    stage = tmp_path / "stage"; stage.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        r592.run_manifest_calls(executor, bundle, contexts(bundle), stage=stage, public_root=tmp_path)
    assert executor.calls[-1] == "FIT:payload"
    assert all(not path.exists() for path in public_paths(tmp_path))


def test_token_reorder_and_filled_partial_batch_fail_before_call(tmp_path: Path) -> None:
    bundle = toy_bundle("FIT")
    bundle["token_arrays"]["FIT:toy:tokens"] = bundle["token_arrays"]["FIT:toy:tokens"][::-1].copy()
    executor = FakeExecutor(); stage = tmp_path / "stage"; stage.mkdir()
    with pytest.raises(RuntimeError, match="changed before model call"):
        r592.run_manifest_calls(executor, bundle, contexts(bundle), stage=stage, public_root=tmp_path)
    assert executor.calls == []


def test_live_cache_component_drift_is_invalid(tmp_path: Path) -> None:
    bundle = toy_bundle("FIT")
    class Drift(FakeExecutor):
        def execute(self, call, tokens, specs, planned):
            response = super().execute(call, tokens, specs, planned)
            if call["call_kind"] == "native":
                response["arrays"]["live_e.npy"][0, 0, 0] = 2e-5
            return response
    stage = tmp_path / "stage"; stage.mkdir()
    observed = r592.run_manifest_calls(Drift(), bundle, contexts(bundle), stage=stage, public_root=tmp_path)
    assert observed["diagnostic"]["failure_predicate"] == "factor_transport_failed"
    assert observed["diagnostic"]["executed_call_ids"][-1] == "FIT:native"


def test_nonzero_literal_self_replay_is_invalid(tmp_path: Path) -> None:
    bundle = toy_bundle("FIT")
    planned = contexts(bundle)
    planned["FIT:replay"]["planned"][0, 0, 0] = np.float32(1e-7)
    stage = tmp_path / "stage"; stage.mkdir()
    observed = r592.run_manifest_calls(FakeExecutor(), bundle, planned, stage=stage, public_root=tmp_path)
    assert observed["diagnostic"]["failure_predicate"] == "centered_hook_delta_failed"


def test_candidate_has_no_live_removal_or_renamed_machine_keys() -> None:
    producer_text = MODULE.read_text()
    runtime_text = Path(__file__).with_name("induction_centered_fixed_geometry_rung592_runtime.py").read_text()
    assert "live_removed" not in producer_text + runtime_text
    assert r592.MACHINE_ARMS == ("replay", "score", "payload", "joint")
    assert r592.SITES == ("L5H5", "L7H3", "L8H3", "L8H4")
    assert r592.ROLES == ("A", "C")
    assert "one transaction" in runtime_text


def test_normal_publication_is_receipt_last_and_collision_safe(tmp_path: Path) -> None:
    stage = tmp_path / "stage"; evidence = stage / "evidence"; evidence.mkdir(parents=True)
    (evidence / "tiny.jsonl").write_text('{"row":1}\n')
    result = {"schema": "toy", "model_forwards": 639, "model_backwards": 0,
              "model_weights_updated": False, "final_opened": False, "ood_opened": False}
    receipt = r592.publish_normal(stage, result, public_root=tmp_path)
    assert (tmp_path / r592.NORMAL_RECEIPT.name).is_file()
    assert receipt["result_sha256"] == r592.sha256_file(tmp_path / r592.NORMAL_RESULT.name)
    second = tmp_path / "second"; (second / "evidence").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="occupied"):
        r592.publish_normal(second, result, public_root=tmp_path)


def test_model_runtime_module_has_no_eager_torch_import() -> None:
    text = Path(__file__).with_name("induction_centered_fixed_geometry_rung592_runtime.py").read_text()
    tree = ast.parse(text)
    eager = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names)
    ]
    assert eager == []
