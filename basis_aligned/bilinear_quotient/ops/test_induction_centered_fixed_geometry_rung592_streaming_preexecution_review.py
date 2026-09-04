#!/usr/bin/env python3
# BQLANE: cpu
"""Independent model-free review of immutable R592 streaming commit 521e4c38c."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest


COMMIT = "521e4c38ca55b9ede6f51cb5408aa1fdbb4486d2"
ROOT = Path(__file__).resolve().parents[3]
PRODUCER = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592.py"
RUNTIME = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592_runtime.py"
ADAPTER = "basis_aligned/bilinear_quotient/ops/execute_induction_centered_fixed_geometry_rung592.py"
DRYRUN = "basis_aligned/bilinear_quotient/induction_centered_fixed_geometry_rung592_dryrun.json"
AMENDMENT = "basis_aligned/polynomial_causal/INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_STREAMING_STORAGE_AMENDMENT.md"
EXPECTED = {
    PRODUCER: "741d7a1481e79a726d3a2edb8bb5274a5d262ce0a93803d438c5762911809efb",
    RUNTIME: "09309b1299b85f2c57689913547fef01f2a9e7b538b2768ac62ff3e48e0f039c",
    ADAPTER: "420ab088c8b361f7645ceb77f158a3e187277c2a9c849f830fb16fcdf85a6654",
    "basis_aligned/bilinear_quotient/ops/test_execute_induction_centered_fixed_geometry_rung592.py":
        "c6c4e6bb8e9b23a63b1352064f670429fba8227c92260bb638004edadeb22478",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592.py":
        "59764d300fdbe3f2024ee40b32b23fb2bcc56ccd79b48e7b1abbe5c0083eb2fc",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_fake_runtime.py":
        "52d3d22e7d1eeaaa31bed66a01d28aef296974bff94e96ab7707af6fa4219e85",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_repair.py":
        "dceb2416d20e7e795f8d3d0dd59bac18c123e3ed7705d3660fe6187abfc73844",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_streaming_storage.py":
        "2f36d595bfe7efa8f8825e9829912e1f6c70ff0d4c0d1f69fe054aebc48a7fda",
    DRYRUN: "937d5d9682ea89ca7e4feda3e646937dee83d56f31b18b8dffd4f04b26b4a1eb",
    AMENDMENT: "2df290b9670adfb8541d675e51fc607f856f7f70c083248fdba14ab8cf90df07",
}


def blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


@pytest.fixture(scope="module")
def r592():
    source = blob(PRODUCER)
    logical = ROOT / PRODUCER
    spec = importlib.util.spec_from_loader(
        "r592_streaming_review_blob", loader=None, origin=str(logical)
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(logical)
    exec(compile(source, str(logical), "exec"), module.__dict__)
    return module


def test_exact_candidate_hashes() -> None:
    assert subprocess.check_output(
        ["git", "rev-parse", COMMIT], cwd=ROOT, text=True
    ).strip() == COMMIT
    assert {path: hashlib.sha256(blob(path)).hexdigest() for path in EXPECTED} == EXPECTED


def test_exact_schema_prices_and_dryrun_binding(r592) -> None:
    phase_bytes = {}
    for phase in ("FIT", "SELECT"):
        total = 0
        for name, descriptor in r592.phase_evidence_schema(phase).items():
            if name.endswith(".npy"):
                total += int(np.prod(descriptor["shape"])) * np.dtype(descriptor["dtype"]).itemsize
        phase_bytes[phase] = total
    assert phase_bytes == {"FIT": 5_198_883_840, "SELECT": 2_599_441_920}

    directed_chunk = sum(
        sum(int(np.prod(shape)) * dtype.itemsize for dtype, shape in
            r592.mandatory_call_shapes({"batch_size": 32, "call_kind": kind}).values())
        for kind in r592.DIRECTED_KINDS
    )
    assert directed_chunk == 41_671_168
    assert phase_bytes["FIT"] + phase_bytes["SELECT"] + directed_chunk == 7_839_996_928

    observed = r592.build_dryrun()
    frozen = json.loads(blob(DRYRUN))
    assert observed == frozen
    assert observed["phase_counts"]["FIT"]["calls"] == 639
    assert observed["phase_counts"]["SELECT"]["calls"] == 322
    assert observed["registered_max_model_forwards"] == 961
    assert observed["model_forwards"] == observed["model_backwards"] == 0


def test_manifest_rows_are_exact_authority_order_and_five_call_chunks(r592) -> None:
    _r585, execution = r592.load_authority()
    for phase, expected_calls in (("FIT", 639), ("SELECT", 322)):
        bundle = r592.build_phase_manifest(execution, phase)
        endpoint_ids = [str(row["endpoint_id"]) for row in execution["endpoints"] if row["split"] == phase]
        directed_ids = [str(row["directed_id"]) for row in execution["directions"] if row["split"] == phase]
        from_calls = [value for call in bundle["calls"] if call["call_kind"] == "endpoint"
                      for value in call["authority_row_ids"]]
        from_native = [value for call in bundle["calls"] if call["call_kind"] == "native"
                       for value in call["direction_ids"]]
        assert from_calls == endpoint_ids
        assert from_native == directed_ids
        assert len(bundle["calls"]) == expected_calls
        directed_calls = [call for call in bundle["calls"] if call["call_kind"] != "endpoint"]
        for start in range(0, len(directed_calls), 5):
            group = directed_calls[start:start + 5]
            assert [call["call_kind"] for call in group] == list(r592.DIRECTED_KINDS)
            assert len({call["token_sha256"] for call in group}) == 1
            assert len({tuple(call["direction_ids"]) for call in group}) == 1


def test_streaming_state_machine_orders_durability_before_raw_deletion() -> None:
    source = blob(PRODUCER).decode()
    write_slice = source[source.index("    def _write_slice("):source.index("    def _canonical_record(")]
    for earlier, later in zip(
        ("expected[...] = payload", "destination.flush()", "_fsync_file(path)",
         "if not np.array_equal(observed, payload", "canonical_slice_sha256"),
        ("destination.flush()", "_fsync_file(path)", "if not np.array_equal(observed, payload",
         "canonical_slice_sha256", "return descriptor"),
    ):
        assert write_slice.index(earlier) < write_slice.index(later)
    endpoint = source[source.index("    def ingest_endpoint("):source.index("    def reference(")]
    assert endpoint.index("self._append_ledger(record)") < endpoint.index("shutil.rmtree(raw_directory)")
    directed = source[source.index("    def ingest_directed_chunk("):source.index("    def close(")]
    assert directed.index("self._append_ledger(record)") < directed.index("shutil.rmtree(raw_directory)")


def test_finalization_rejects_unwritten_tail_and_scans_complete_files() -> None:
    source = blob(PRODUCER).decode()
    body = source[source.index("def finalize_streamed_phase_evidence"):source.index("def publish_normal")]
    for required in (
        "store.endpoint_offset != expected_endpoint",
        "store.directed_offset != expected_directed",
        "len(store.ledger_records) != len(bundle[\"calls\"])",
        "if not _finite_memmap(path)",
        "_fsync_file(path)",
        '"byte_length": path.stat().st_size',
        '"sha256": sha256_file(path)',
    ):
        assert required in body


def test_invalid_prefix_binds_canonical_ledger_raw_chunk_and_masks() -> None:
    source = blob(PRODUCER).decode()
    invalid = source[source.index("def publish_invalid_prefix"):source.index("def evaluate_completed_call")]
    assert '"call_prefix_sha256": sha256_file(prefix_path)' in invalid
    assert 'for path in sorted(evidence.rglob("*")) if path.is_file()' in invalid
    state_machine = source[source.index("def run_manifest_calls"):source.index("def make_context_factory")]
    assert 'record["storage"] = "raw_current_chunk"' in state_machine
    assert 'prefix[-1] = store.ingest_endpoint' in state_machine
    assert 'prefix[-len(DIRECTED_KINDS):] = canonical' in state_machine
    assert 'nonfinite_terminal=(predicate == "nonfinite_observation")' in state_machine


def test_advertised_nine_gb_preflight_cannot_reach_select_on_nine_gb_disk(r592) -> None:
    # The second frozen gate asks for 9 GB still available after the full FIT
    # canonical arrays have consumed 5,198,883,840 bytes.
    advertised_initial = 9_000_000_000
    fit_canonical = 5_198_883_840
    available_before_select = advertised_initial - fit_canonical
    assert available_before_select == 3_801_116_160
    stat = lambda _path: types.SimpleNamespace(f_bavail=available_before_select, f_frsize=1)
    with pytest.raises(RuntimeError, match="insufficient free space before SELECT"):
        r592.require_free_space(Path("."), boundary="SELECT", statvfs_function=stat)

    # Keeping both 9 GB gates therefore requires at least this much initial
    # free space, before NumPy headers and metadata—not the advertised 9 GB.
    assert fit_canonical + r592.MINIMUM_FREE_BYTES == 14_198_883_840


def test_adapter_pins_streaming_candidate_and_preserves_receipt_last_publication() -> None:
    adapter = blob(ADAPTER).decode()
    for digest in EXPECTED.values():
        if digest != EXPECTED[ADAPTER]:
            assert digest in adapter
    assert "base64.b64encode(source)" in adapter
    producer = blob(PRODUCER).decode()
    publication = producer[producer.index("def publish_normal"):producer.index("def _empty_scientific_failures")]
    assert publication.index("os.replace(evidence, targets[0])") < publication.index(
        "os.replace(result_path, targets[1])"
    ) < publication.index("os.replace(receipt_path, targets[2])")
