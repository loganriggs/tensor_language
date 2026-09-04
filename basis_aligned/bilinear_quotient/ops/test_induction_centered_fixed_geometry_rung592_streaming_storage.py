#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free tests for the prospective R592 streaming-storage amendment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types

import numpy as np
import pytest


OPS = Path(__file__).resolve().parent
PRODUCER = OPS / "induction_centered_fixed_geometry_rung592.py"
ADAPTER = OPS / "execute_induction_centered_fixed_geometry_rung592.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r592 = load(PRODUCER, "r592_streaming_test")


def toy_bundle(phase: str = "FIT") -> dict[str, object]:
    tokens = np.full((2, 30), r592.PAD_TOKEN, dtype="<i8")
    tokens[:, :3] = np.asarray([[1, 2, 3], [4, 5, 6]], dtype="<i8")
    token_id = f"{phase}:toy:tokens"
    calls = []
    for index, kind in enumerate(("endpoint", "native", "replay", "score", "payload", "joint")):
        call_id = f"{phase}:endpoint:0000" if kind == "endpoint" else f"{phase}:directed:0000:{kind}"
        calls.append({
            "manifest_index": index, "call_id": call_id, "phase": phase,
            "call_kind": kind, "chunk_index": 0,
            "machine_arm": None if kind in ("endpoint", "native") else kind,
            "token_record_id": token_id, "token_sha256": r592.sha256_bytes(tokens.tobytes(order="C")),
            "batch_size": 2, "physical_width": 30, "authority_row_ids": ["e0", "e1"],
            "direction_ids": [] if kind == "endpoint" else ["d0", "d1"],
            "query_positions": [2, 2],
        })
    return {"phase": phase, "calls": calls, "token_arrays": {token_id: tokens},
            "token_records": [{"token_record_id": token_id, "token_sha256": calls[0]["token_sha256"]}]}


def arrays_for(call, tokens, planned=None):
    b = int(call["batch_size"])
    arrays = {name: (tokens.copy() if name == "tokens.npy" else np.zeros(shape, dtype=dtype))
              for name, (dtype, shape) in r592.mandatory_call_shapes(call).items()}
    if call["call_kind"] == "endpoint":
        arrays["support.npy"].fill(True)
    if call["call_kind"] in r592.MACHINE_ARMS and planned is not None:
        arrays["hook_deltas.npy"][...] = planned
        arrays["planned_hook_deltas.npy"][...] = planned
    return arrays


def contexts(bundle):
    zero_e = np.zeros((2, 4, 2), dtype="<f4")
    zero_u = np.zeros((2, 4, 2, r592.RESIDUAL), dtype="<f4")
    output = {}
    for call in bundle["calls"]:
        cached = {"recipient_e": zero_e, "recipient_u": zero_u,
                  "donor_e": zero_e, "donor_u": zero_u}
        if call["call_kind"] in r592.MACHINE_ARMS:
            cached["native_logits"] = np.zeros((2, r592.VOCAB), dtype="<f4")
        output[call["call_id"]] = {
            "specs": [{"final_position": 2}] * 2,
            "planned": None if call["call_kind"] in ("endpoint", "native")
            else np.zeros((2, 4, r592.RESIDUAL), dtype="<f4"),
            "cached": None if call["call_kind"] == "endpoint" else cached,
        }
    return output


def toy_execution() -> dict[str, object]:
    endpoints = [
        {"split": "FIT", "endpoint_id": "e0", "final_position": 2, "answer_id": 10,
         "other_answer_id": 11, "family_id": "toy"},
        {"split": "FIT", "endpoint_id": "e1", "final_position": 2, "answer_id": 12,
         "other_answer_id": 11, "family_id": "toy"},
    ]
    common = {"split": "FIT", "row_id": "row", "group_id": "group", "family": "toy",
              "variant": "base", "recipient_condition": "a", "direction": "forward",
              "control_kind": None, "answer_changes": True, "recipient_other_answer_id": 11,
              "donor_coherence_sign": None}
    directions = [
        {**common, "directed_id": "d0", "recipient_endpoint_id": "e0", "donor_endpoint_id": "e1",
         "recipient_answer_id": 10, "donor_answer_id": 12},
        {**common, "directed_id": "d1", "recipient_endpoint_id": "e1", "donor_endpoint_id": "e0",
         "recipient_answer_id": 12, "donor_answer_id": 10},
    ]
    return {"endpoints": endpoints, "directions": directions,
            "manifests": {"target_cells": [], "control_cells": [], "structural_identities": []},
            "bootstrap_cells": []}


class Executor:
    def __init__(self, mutation=None, values=False):
        self.mutation, self.values, self.calls, self.observed = mutation, values, [], {}

    def execute(self, call, tokens, specs, planned):
        kind = str(call["call_kind"])
        self.calls.append(str(call["call_id"]))
        if self.mutation == (kind, "raise"):
            raise RuntimeError("planted incomplete call")
        arrays = arrays_for(call, tokens, planned)
        if self.values and kind in ("score", "payload", "joint"):
            arrays["logits.npy"].fill({"score": 1, "payload": 2, "joint": 3}[kind])
        if self.mutation == (kind, "nonfinite"):
            arrays["logits.npy"][0, 0] = np.nan
        self.observed[str(call["call_id"])] = {name: value.copy() for name, value in arrays.items()}
        return {"arrays": arrays}


def descriptor_payload(evidence: Path, descriptor: dict[str, object]) -> np.ndarray:
    array = np.load(evidence / str(descriptor["filename"]), mmap_mode="r", allow_pickle=False)
    start, stop = descriptor["axis0"]
    return np.ascontiguousarray(array[start:stop] if "axis1" not in descriptor
                                else array[start:stop, descriptor["axis1"]])


def test_exact_registered_streaming_peak_and_dryrun() -> None:
    assert 7_798_325_760 + 41_671_168 == 7_839_996_928
    canonical = 0
    for phase in ("FIT", "SELECT"):
        for name, descriptor in r592.phase_evidence_schema(phase).items():
            if name.endswith(".npy"):
                canonical += int(np.prod(descriptor["shape"])) * np.dtype(descriptor["dtype"]).itemsize
    assert canonical == 7_798_325_760
    largest_chunk = sum(
        sum(int(np.prod(shape)) * dtype.itemsize
            for dtype, shape in r592.mandatory_call_shapes({"batch_size": 32, "call_kind": kind}).values())
        for kind in r592.DIRECTED_KINDS
    )
    assert largest_chunk == 41_671_168
    assert r592.COMPLETE_CANONICAL_DATA_BYTES == 7_798_325_760
    assert r592.LARGEST_CURRENT_CHUNK_DATA_BYTES == 41_671_168
    assert r592.MAXIMUM_STREAMING_DATA_BYTES == 7_839_996_928
    storage = r592.build_dryrun()["streaming_storage"]
    assert storage["required_free_bytes_before_model"] == 9_000_000_000
    assert storage["required_free_bytes_before_select"] == 3_801_116_160
    assert storage["remaining_select_plus_chunk_bytes"] + storage["safety_margin_bytes"] == 3_801_116_160


def test_capacity_gate_fails_before_model_and_adapter_dispatch(tmp_path: Path) -> None:
    low = lambda _path: types.SimpleNamespace(f_bavail=8_999_999_999, f_frsize=1)
    with pytest.raises(RuntimeError, match="insufficient free space"):
        r592.require_free_space(tmp_path, statvfs_function=low)
    adapter = load(ADAPTER, "r592_streaming_adapter_test")
    dispatched = []
    with pytest.raises(RuntimeError, match="insufficient free space"):
        adapter.dispatch(
            {}, exec_function=lambda *_args: dispatched.append(True),
            namespace_paths=(tmp_path / "unused",), capacity_path=tmp_path,
            statvfs_function=low,
        )
    assert dispatched == []
    assert not any((tmp_path / path.name).exists() for path in r592.PUBLIC_NAMESPACES)


def test_producer_capacity_failure_precedes_runtime_construction(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(r592, "load_authority", lambda: (object(), {}))
    monkeypatch.setattr(r592, "build_phase_manifest", lambda _execution, phase: {"phase": phase})
    monkeypatch.setattr(
        r592, "require_free_space",
        lambda _path, **_kwargs: (_ for _ in ()).throw(RuntimeError("insufficient free space")),
    )
    monkeypatch.setattr(
        r592, "_immutable_module",
        lambda *_args, **_kwargs: pytest.fail("runtime constructed before capacity gate"),
    )
    with pytest.raises(RuntimeError, match="insufficient free space"):
        r592.run_science(public_root=tmp_path)
    assert not any((tmp_path / path.name).exists() for path in r592.PUBLIC_NAMESPACES)


def test_select_capacity_gate_precedes_select_calls(tmp_path: Path) -> None:
    source = PRODUCER.read_text()
    science = source[source.index("def run_science"):source.index("def build_dryrun")]
    gate = science.index("before_select_capacity, select = run_after_capacity_gate(")
    select = science.index("lambda: run_manifest_calls(", gate)
    assert gate < select
    low = lambda _path: types.SimpleNamespace(f_bavail=1, f_frsize=1)
    called = []
    with pytest.raises(RuntimeError, match="insufficient free space"):
        r592.run_after_capacity_gate(
            tmp_path, lambda: called.append(True), boundary="SELECT", statvfs_function=low
        )
    assert called == []
    assert not any((tmp_path / path.name).exists() for path in r592.PUBLIC_NAMESPACES)


def test_phase_relative_capacity_boundaries_are_exact(tmp_path: Path) -> None:
    def stat(value):
        return lambda _path: types.SimpleNamespace(f_bavail=value, f_frsize=1)

    # A current-size 9.5 GB filesystem passes initially and after exact FIT.
    initial = 9_500_000_000
    remaining = initial - 5_198_883_840
    assert r592.require_free_space(tmp_path, statvfs_function=stat(initial))["required_free_bytes"] == 9_000_000_000
    assert remaining == 4_301_116_160
    assert r592.require_free_space(
        tmp_path, boundary="SELECT", statvfs_function=stat(remaining)
    )["required_free_bytes"] == 3_801_116_160

    # Equality passes at each inclusive boundary; one byte below fails.
    r592.require_free_space(tmp_path, statvfs_function=stat(9_000_000_000))
    r592.require_free_space(tmp_path, boundary="SELECT", statvfs_function=stat(3_801_116_160))
    with pytest.raises(RuntimeError, match="before model"):
        r592.require_free_space(tmp_path, statvfs_function=stat(8_999_999_999))
    with pytest.raises(RuntimeError, match="before SELECT"):
        r592.require_free_space(tmp_path, boundary="SELECT", statvfs_function=stat(3_801_116_159))


def test_invalid_prefix_uses_canonical_slices_plus_only_current_raw_chunk(tmp_path: Path) -> None:
    bundle = toy_bundle(); executor = Executor(("payload", "nonfinite"))
    stage = tmp_path / "stage"; stage.mkdir()
    observed = r592.run_manifest_calls(
        executor, bundle, contexts(bundle), stage=stage, public_root=tmp_path
    )
    assert observed["status"] == "invalid" and executor.calls[-1] == "FIT:directed:0000:payload"
    evidence = tmp_path / r592.INVALID_EVIDENCE.name
    prefix = [json.loads(line) for line in (evidence / "call_prefix.jsonl").read_text().splitlines()]
    assert [row["call_id"] for row in prefix] == [row["call_id"] for row in bundle["calls"][:5]]
    assert prefix[0]["storage"] == "canonical_slices"
    assert {row["storage"] for row in prefix[1:]} == {"raw_current_chunk"}
    assert sorted(path.name for path in (evidence / "calls").iterdir()) == [
        "0001_FIT:directed:0000:native", "0002_FIT:directed:0000:replay",
        "0003_FIT:directed:0000:score", "0004_FIT:directed:0000:payload",
    ]
    for descriptor in prefix[0]["canonical_slices"]:
        payload = descriptor_payload(evidence, descriptor)
        assert descriptor["sha256"] == r592.canonical_slice_sha256(
            descriptor["filename"], payload, *descriptor["axis0"],
            axis1=descriptor.get("axis1"),
        )


def test_hard_abort_then_clean_recovery_keeps_no_public_namespace(tmp_path: Path) -> None:
    bundle = toy_bundle(); failed_stage = tmp_path / "failed"; failed_stage.mkdir()
    with pytest.raises(RuntimeError, match="incomplete call"):
        r592.run_manifest_calls(
            Executor(("payload", "raise")), bundle, contexts(bundle),
            stage=failed_stage, public_root=tmp_path,
        )
    assert not any((tmp_path / path.name).exists() for path in r592.PUBLIC_NAMESPACES)
    recovered_stage = tmp_path / "recovered"; recovered_stage.mkdir()
    recovered = r592.run_manifest_calls(
        Executor(), bundle, contexts(bundle), stage=recovered_stage, public_root=tmp_path
    )
    assert recovered["status"] == "complete"
    assert not any(recovered["store"].calls_root.iterdir())


def test_raw_deletion_occurs_only_after_durable_slice_ledger(monkeypatch, tmp_path: Path) -> None:
    bundle = toy_bundle(); stage = tmp_path / "stage"; stage.mkdir()
    real_rmtree = r592.shutil.rmtree
    checked = []

    def guarded(path):
        path = Path(path)
        if path.parent.name == "calls":
            evidence = stage / "evidence"
            ledger = [json.loads(line) for line in
                      (evidence / "FIT" / "canonical_slice_ledger.jsonl").read_text().splitlines()]
            call_id = path.name.split("_", 1)[1]
            record = next(row for row in ledger if row["call_id"] == call_id)
            assert record["storage"] == "canonical_slices" and record["canonical_slices"]
            for descriptor in record["canonical_slices"]:
                payload = descriptor_payload(evidence, descriptor)
                assert descriptor["sha256"] == r592.canonical_slice_sha256(
                    descriptor["filename"], payload, *descriptor["axis0"],
                    axis1=descriptor.get("axis1"),
                )
            checked.append(call_id)
        real_rmtree(path)

    monkeypatch.setattr(r592.shutil, "rmtree", guarded)
    result = r592.run_manifest_calls(
        Executor(), bundle, contexts(bundle), stage=stage, public_root=tmp_path
    )
    assert result["status"] == "complete"
    assert checked == [row["call_id"] for row in bundle["calls"]]


def test_streamed_arrays_equal_legacy_materialization_on_synthetic_calls(tmp_path: Path) -> None:
    bundle = toy_bundle(); execution = toy_execution(); stage = tmp_path / "stage"; stage.mkdir()
    executor = Executor(values=True)
    result = r592.run_manifest_calls(
        executor, bundle, r592.make_context_factory(execution, bundle), stage=stage, public_root=tmp_path
    )
    store = result["store"]
    streamed_logits = np.asarray(store.arrays["logit_differences.npy"])
    legacy_logits = np.stack([
        np.zeros((2, r592.VOCAB), dtype="<f4"),
        np.ones((2, r592.VOCAB), dtype="<f4"),
        np.full((2, r592.VOCAB), 2, dtype="<f4"),
        np.full((2, r592.VOCAB), 3, dtype="<f4"),
    ], axis=1)
    legacy_hooks = np.zeros((2, 4, 4, r592.RESIDUAL), dtype="<f4")
    assert np.array_equal(streamed_logits, legacy_logits)
    assert np.array_equal(np.asarray(store.arrays["hook_deltas.npy"]), legacy_hooks)
    legacy_root = tmp_path / "legacy"; legacy_root.mkdir()
    outputs = {}
    for call in bundle["calls"]:
        directory = legacy_root / str(call["manifest_index"]); directory.mkdir()
        for name, value in executor.observed[str(call["call_id"])].items():
            r592._write_npy(directory / name, value)
        outputs[str(call["call_id"])] = directory
    legacy_records = r592.derive_scientific_records(execution, bundle, outputs)
    assert json.dumps(store.scientific_records, sort_keys=True, allow_nan=False) == json.dumps(
        legacy_records, sort_keys=True, allow_nan=False
    )


def test_streamed_finalization_uses_no_second_call_tree(tmp_path: Path) -> None:
    bundle = toy_bundle(); execution = toy_execution()
    factory = r592.make_context_factory(execution, bundle)
    stage = tmp_path / "stage"; (stage / "evidence").mkdir(parents=True)
    result = r592.run_manifest_calls(
        Executor(values=True), bundle, factory, stage=stage, public_root=tmp_path
    )
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"rows": [
        {"split": "FIT", "family_id": "toy", "row_id": "a"},
        {"split": "FIT", "family_id": "toy", "row_id": "b"},
    ]}))
    fake_r585 = types.SimpleNamespace(
        ROWS=rows,
        strict_load_json=lambda path: json.loads(Path(path).read_text()),
        load_manifest=lambda: types.SimpleNamespace(INCLUDED_FAMILIES={"toy"}),
    )
    descriptors = r592.finalize_streamed_phase_evidence(
        result["store"], execution, bundle, {}, {}, fake_r585
    )
    assert not result["store"].calls_root.exists()
    assert descriptors["canonical_slice_ledger.jsonl"]["records"] == 6
    directed = [json.loads(line) for line in
                (stage / "evidence" / "FIT" / "directed_records.jsonl").read_text().splitlines()]
    assert len(directed) == 2 and all(set(row["arms"]) == {"score", "payload", "joint"} for row in directed)
    assert all(np.isfinite(row[condition]["correct_ce"])
               for row in directed for condition in ("native", "replay"))
