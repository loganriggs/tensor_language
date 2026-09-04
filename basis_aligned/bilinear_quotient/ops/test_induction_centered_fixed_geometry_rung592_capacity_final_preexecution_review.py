#!/usr/bin/env python3
# BQLANE: cpu
"""Independent model-free review of immutable R592 capacity commit 7c6be867f."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest


COMMIT = "7c6be867fcca7a64b3e6dffbff4540e645a32c4e"
ROOT = Path(__file__).resolve().parents[3]
PRODUCER = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592.py"
RUNTIME = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592_runtime.py"
ADAPTER = "basis_aligned/bilinear_quotient/ops/execute_induction_centered_fixed_geometry_rung592.py"
DRYRUN = "basis_aligned/bilinear_quotient/induction_centered_fixed_geometry_rung592_dryrun.json"
CAPACITY_AMENDMENT = (
    "basis_aligned/polynomial_causal/"
    "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PHASE_RELATIVE_CAPACITY_AMENDMENT.md"
)
EXPECTED = {
    PRODUCER: "e625a94216659f4cafb91114b3f253b42844f7e54cb8531b17e0f47614dc5431",
    RUNTIME: "09309b1299b85f2c57689913547fef01f2a9e7b538b2768ac62ff3e48e0f039c",
    ADAPTER: "de8b6e2977551dc19cd00449a1de5c698dbc5978c8d9c23d1ad0d21576e025c5",
    "basis_aligned/bilinear_quotient/ops/test_execute_induction_centered_fixed_geometry_rung592.py":
        "f81b5c3df85c5c7bd8def93136ac2bbbc3d826970c2571c0626dffbad6f1a4e3",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592.py":
        "59764d300fdbe3f2024ee40b32b23fb2bcc56ccd79b48e7b1abbe5c0083eb2fc",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_fake_runtime.py":
        "52d3d22e7d1eeaaa31bed66a01d28aef296974bff94e96ab7707af6fa4219e85",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_repair.py":
        "dceb2416d20e7e795f8d3d0dd59bac18c123e3ed7705d3660fe6187abfc73844",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_streaming_storage.py":
        "c927f6828e651589089217fec7a92118563aa893cae1b529651ac7a5a7e77a9e",
    DRYRUN: "5aa8ee4ce3d4d40d00c74c64d12af7431fdbac090b74c7dabd5ae8ed4cb83e38",
    CAPACITY_AMENDMENT: "da634dd10da654739d761a6c8f8ce9c1434d8946a7477ba6d9c005c873386458",
}


def blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def load_blob_module(path: str, name: str):
    source = blob(path)
    logical = ROOT / path
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(logical))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(logical)
    exec(compile(source, str(logical), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def producer():
    return load_blob_module(PRODUCER, "r592_capacity_final_review_producer")


@pytest.fixture(scope="module")
def adapter():
    return load_blob_module(ADAPTER, "r592_capacity_final_review_adapter")


def fake_stat(available: int):
    return lambda _path: types.SimpleNamespace(f_bavail=available, f_frsize=1)


def test_exact_candidate_and_transitive_hashes() -> None:
    assert subprocess.check_output(
        ["git", "rev-parse", COMMIT], cwd=ROOT, text=True
    ).strip() == COMMIT
    assert {path: hashlib.sha256(blob(path)).hexdigest() for path in EXPECTED} == EXPECTED
    adapter_source = blob(ADAPTER).decode()
    for path, digest in EXPECTED.items():
        if path not in {ADAPTER, CAPACITY_AMENDMENT}:
            assert digest in adapter_source
    assert EXPECTED[CAPACITY_AMENDMENT] in adapter_source


@pytest.mark.parametrize(
    ("boundary", "threshold"),
    (("model", 9_000_000_000), ("SELECT", 3_801_116_160)),
)
def test_capacity_equality_passes_and_one_byte_less_fails(producer, boundary, threshold) -> None:
    observed = producer.require_free_space(
        Path("."), boundary=boundary, statvfs_function=fake_stat(threshold)
    )
    assert observed == {
        "boundary": boundary,
        "available_bytes": threshold,
        "required_free_bytes": threshold,
    }
    with pytest.raises(RuntimeError, match=f"{threshold - 1} < {threshold}"):
        producer.require_free_space(
            Path("."), boundary=boundary, statvfs_function=fake_stat(threshold - 1)
        )


def test_select_capacity_gate_never_calls_operation_below_threshold(producer) -> None:
    called: list[str] = []
    operation = lambda: called.append("called")
    with pytest.raises(RuntimeError, match="before SELECT boundary"):
        producer.run_after_capacity_gate(
            Path("."), operation, boundary="SELECT",
            statvfs_function=fake_stat(3_801_116_159),
        )
    assert called == []
    capacity, result = producer.run_after_capacity_gate(
        Path("."), operation, boundary="SELECT",
        statvfs_function=fake_stat(3_801_116_160),
    )
    assert capacity["available_bytes"] == 3_801_116_160
    assert result is None and called == ["called"]


def test_exact_storage_arithmetic_vocab_and_call_price(producer) -> None:
    schema_bytes = {}
    for phase in ("FIT", "SELECT"):
        schema_bytes[phase] = sum(
            int(np.prod(item["shape"])) * np.dtype(item["dtype"]).itemsize
            for name, item in producer.phase_evidence_schema(phase).items()
            if name.endswith(".npy")
        )
    assert schema_bytes == {"FIT": 5_198_883_840, "SELECT": 2_599_441_920}
    assert producer.LARGEST_CURRENT_CHUNK_DATA_BYTES == 41_671_168
    assert 2_599_441_920 + 41_671_168 + 1_160_003_072 == 3_801_116_160
    assert 9_000_000_000 - 5_198_883_840 == 3_801_116_160
    assert producer.MAXIMUM_STREAMING_DATA_BYTES == 7_839_996_928
    assert producer.VOCAB == 50_304
    assert producer.PHASE_COUNTS["FIT"]["calls"] == 639
    assert producer.PHASE_COUNTS["SELECT"]["calls"] == 322
    assert sum(item["calls"] for item in producer.PHASE_COUNTS.values()) == 961


def test_dryrun_and_adapter_expose_same_frozen_capacity_contract(producer, adapter) -> None:
    observed = producer.build_dryrun()
    assert observed == json.loads(blob(DRYRUN))
    storage = observed["streaming_storage"]
    assert storage["required_free_bytes_before_model"] == 9_000_000_000
    assert storage["required_free_bytes_before_select"] == 3_801_116_160
    assert storage["maximum_streaming_data_bytes"] == 7_839_996_928
    assert observed["registered_max_model_forwards"] == 961
    assert observed["model_forwards"] == observed["model_backwards"] == 0
    assert observed["model_weights_updated"] is False
    assert not any(observed[key] for key in ("select_opened", "final_opened", "ood_opened"))

    plan = adapter.preflight(
        namespace_paths=(), capacity_path=Path("."),
        statvfs_function=fake_stat(9_000_000_000),
    )
    assert plan["capacity_thresholds"] == {
        "before_model": 9_000_000_000,
        "before_select_after_fit": 3_801_116_160,
        "fit_canonical_data_bytes": 5_198_883_840,
        "remaining_select_plus_chunk_bytes": 2_641_113_088,
        "safety_margin_bytes": 1_160_003_072,
    }
    with pytest.raises(RuntimeError, match="before model boundary"):
        adapter.preflight(
            namespace_paths=(), capacity_path=Path("."),
            statvfs_function=fake_stat(8_999_999_999),
        )


def test_gate_order_precedes_model_and_select_calls_and_failures_clean_stage() -> None:
    source = blob(PRODUCER).decode()
    body = source[source.index("def run_science"):source.index("def build_dryrun")]
    assert body.index("before_model_capacity = require_free_space(public_root)") < body.index(
        "runtime = _immutable_module"
    ) < body.index("executor = runtime.R592ModelExecutor")
    assert body.index("before_select_capacity, select = run_after_capacity_gate(") < body.index(
        'evaluated.append("SELECT")'
    )
    assert "except Exception:" in body
    assert "if stage.exists(): shutil.rmtree(stage)\n        raise" in body


def test_streaming_and_publication_contract_remains_receipt_last() -> None:
    source = blob(PRODUCER).decode()
    write_slice = source[source.index("    def _write_slice("):source.index("    def _canonical_record(")]
    assert write_slice.index("destination.flush()") < write_slice.index("_fsync_file(path)")
    assert write_slice.index("_fsync_file(path)") < write_slice.index("np.array_equal(observed, payload")
    endpoint = source[source.index("    def ingest_endpoint("):source.index("    def reference(")]
    directed = source[source.index("    def ingest_directed_chunk("):source.index("    def close(")]
    assert endpoint.index("self._append_ledger(record)") < endpoint.index("shutil.rmtree(raw_directory)")
    assert directed.index("self._append_ledger(record)") < directed.index("shutil.rmtree(raw_directory)")
    publication = source[source.index("def publish_normal"):source.index("def _empty_scientific_failures")]
    assert publication.index("os.replace(evidence, targets[0])") < publication.index(
        "os.replace(result_path, targets[1])"
    ) < publication.index("os.replace(receipt_path, targets[2])")
