#!/usr/bin/env python3
# BQLANE: cpu
"""Independent exact-byte, model-free review of repaired R592 commit 3f44c224e."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


COMMIT = "3f44c224ee0144a2a58da0487ffc863bfa75e7d7"
ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "basis_aligned" / "bilinear_quotient" / "ops"
PRODUCER = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592.py"
RUNTIME = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592_runtime.py"
ADAPTER = "basis_aligned/bilinear_quotient/ops/execute_induction_centered_fixed_geometry_rung592.py"
DRYRUN = "basis_aligned/bilinear_quotient/induction_centered_fixed_geometry_rung592_dryrun.json"
EXPECTED = {
    PRODUCER: "9d75aaa291af61321cee29410b4ecfa772425e3dd2298e15440fb3a5843e799b",
    RUNTIME: "09309b1299b85f2c57689913547fef01f2a9e7b538b2768ac62ff3e48e0f039c",
    ADAPTER: "64cda676fa0ba05c80af3986b5595659aa25937b26ed587034de929de97604dd",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592.py":
        "59764d300fdbe3f2024ee40b32b23fb2bcc56ccd79b48e7b1abbe5c0083eb2fc",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_fake_runtime.py":
        "52d3d22e7d1eeaaa31bed66a01d28aef296974bff94e96ab7707af6fa4219e85",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_repair.py":
        "691eb9786f344f1851447776ce0a2f5d324c60f9efbb0c780731c489e5e3c7dd",
    DRYRUN: "152c0cc38c671e7a1b96e199a76ebed607e058427b68be9cd9a53611d83c614e",
}


def blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


@pytest.fixture(scope="module")
def r592():
    source = blob(PRODUCER)
    logical = ROOT / PRODUCER
    spec = importlib.util.spec_from_loader("r592_repair_review_blob", loader=None, origin=str(logical))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(logical)
    exec(compile(source, str(logical), "exec"), module.__dict__)
    return module


def test_exact_candidate_hashes_and_authoritative_gates() -> None:
    assert subprocess.check_output(["git", "rev-parse", COMMIT], cwd=ROOT, text=True).strip() == COMMIT
    assert {path: hashlib.sha256(blob(path)).hexdigest() for path in EXPECTED} == EXPECTED
    for path in (PRODUCER, ADAPTER):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == EXPECTED[path]
        completed = subprocess.run(
            ["python", str(OPS / "gate.py"), str(ROOT / path)],
            cwd=ROOT, text=True, capture_output=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_topology_price_and_exact_call_census() -> None:
    dry = json.loads(blob(DRYRUN))
    assert dry["evidence_schemas"]["FIT"]["logit_differences.npy"]["shape"] == [3744, 4, 50304]
    assert dry["evidence_schemas"]["SELECT"]["logit_differences.npy"]["shape"] == [1872, 4, 50304]
    assert dry["evidence_data_bytes"] == {
        "FIT_logit_differences": 3_013_410_816,
        "SELECT_logit_differences": 1_506_705_408,
        "maximum_logit_differences": 4_520_116_224,
        "maximum_principal_raw_payload": 5_141_200_896,
    }
    assert dry["phase_counts"]["FIT"]["calls"] == 639
    assert dry["phase_counts"]["SELECT"]["calls"] == 322
    assert dry["select_tail_batch_sizes"] == [16] * 5
    runtime = ast.parse(blob(RUNTIME))
    forwards = [
        node for node in ast.walk(runtime) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "forward_with_dispatch"
    ]
    assert len(forwards) == 2


def _one_call(r592, kind: str):
    tokens = np.full((1, 30), r592.PAD_TOKEN, dtype="<i8")
    call = {
        "manifest_index": 0, "call_id": f"FIT:{kind}", "phase": "FIT",
        "call_kind": kind, "chunk_index": 0,
        "machine_arm": None if kind in ("endpoint", "native") else kind,
        "token_record_id": "FIT:toy", "token_sha256": r592.sha256_bytes(tokens.tobytes()),
        "batch_size": 1, "physical_width": 30, "authority_row_ids": ["e0"],
        "direction_ids": [] if kind == "endpoint" else ["d0"], "query_positions": [0],
    }
    bundle = {"phase": "FIT", "calls": [call], "token_arrays": {"FIT:toy": tokens}}
    arrays = {
        name: (tokens.copy() if name == "tokens.npy" else np.zeros(shape, dtype=dtype))
        for name, (dtype, shape) in r592.mandatory_call_shapes(call).items()
    }
    return bundle, call, tokens, arrays


def test_missing_required_observation_hard_aborts_without_namespace(r592, tmp_path: Path) -> None:
    bundle, call, _tokens, arrays = _one_call(r592, "score")
    arrays.pop("hook_deltas.npy")

    class Missing:
        def execute(self, *_args):
            return {"arrays": arrays}

    context = {call["call_id"]: {
        "specs": [{}], "planned": np.zeros((1, 4, 1152), dtype="<f4"), "cached": {},
    }}
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(RuntimeError, match="incomplete call.*missing"):
        r592.run_manifest_calls(Missing(), bundle, context, stage=stage, public_root=tmp_path)
    assert not any((tmp_path / path.name).exists() for path in r592.PUBLIC_NAMESPACES)


def test_invalid_receipt_content_binds_every_evidence_byte(r592, tmp_path: Path) -> None:
    bundle, call, _tokens, arrays = _one_call(r592, "score")
    arrays["logits.npy"][0, 0] = np.nan
    arrays["hook_deltas.npy"][0, 0, 0] = np.inf

    class Nonfinite:
        def execute(self, *_args):
            return {"arrays": arrays}

    context = {call["call_id"]: {
        "specs": [{}], "planned": np.zeros((1, 4, 1152), dtype="<f4"),
        "cached": {"native_logits": np.zeros((1, r592.VOCAB), dtype="<f4")},
    }}
    stage = tmp_path / "stage"
    stage.mkdir()
    r592.run_manifest_calls(Nonfinite(), bundle, context, stage=stage, public_root=tmp_path)
    evidence = tmp_path / r592.INVALID_EVIDENCE.name
    receipt = json.loads((tmp_path / r592.INVALID_RECEIPT.name).read_text())
    observed = {
        str(path.relative_to(evidence)): {
            "byte_length": path.stat().st_size, "sha256": r592.sha256_file(path),
        }
        for path in sorted(evidence.rglob("*")) if path.is_file()
    }
    assert receipt["evidence_files"] == observed
    index = evidence / "calls/0000_FIT:score/nonfinite_mask_index.json"
    original_digest = receipt["evidence_files"][str(index.relative_to(evidence))]["sha256"]
    index.write_bytes(index.read_bytes() + b" ")
    assert r592.sha256_file(index) != original_digest


@pytest.mark.parametrize("kind", ("endpoint", "native"))
def test_native_full_gate_uses_distinct_raw_arrays_for_both_call_kinds(r592, kind: str) -> None:
    _bundle, call, tokens, arrays = _one_call(r592, kind)
    assert arrays["native_full_attention_write.npy"] is not arrays["independent_full_native_write.npy"]
    arrays["independent_full_native_write.npy"][0, 0, 0] = np.float32(2e-5)
    predicate, details = r592.evaluate_completed_call(call, arrays, tokens)
    assert predicate == "native_full_write_reconstruction_failed"
    assert details["native_full_write_reconstruction_max_abs"] > r592.TOLERANCE


def test_runtime_reconstructs_all_nine_heads_independently() -> None:
    text = blob(RUNTIME).decode()
    body = text[text.index("    def _independent_full_attention_write"):text.index("    def _capture")]
    assert body.count(".view(batch, length, 9, 128)") == 5
    assert 'torch.einsum("bqhd,bkhd->bhqk"' in body
    assert 'torch.einsum("bhqk,bkhd->bhqd"' in body
    assert "attention.c_proj.weight" in body
    assert "factorize_attention_event" not in body
    capture = text[text.index("    def _capture"):text.index("    def _intervene")]
    assert 'full_native[local][site] = write[local, query]' in capture
    assert 'full_reconstructed[local][site] = reconstructed[local, query]' in capture


def test_complete_evidence_has_endpoint_and_directed_gate_inputs_and_safe_memmaps(r592) -> None:
    for phase, ne, nd in (("FIT", 1728, 3744), ("SELECT", 864, 1872)):
        schema = r592.phase_evidence_schema(phase)
        for stem in ("native_full_attention_write.npy", "independent_full_native_write.npy"):
            assert schema[stem]["shape"] == [ne, 4, 1152]
            assert schema["directed_" + stem]["shape"] == [nd, 4, 1152]
        assert schema["instrument_gates.json"] == {"records": 1}
    source = blob(PRODUCER).decode()
    body = source[source.index("def write_complete_phase_evidence"):source.index("def publish_normal")]
    for required in (
        "if offset != nd", "np.isfinite(complete_array).all()",
        "_fsync_file(hook_path)", "_fsync_file(logit_path)",
        "directed_native_full_attention_write.npy", "directed_independent_full_native_write.npy",
    ):
        assert required in body


def test_centered_literal_zero_hybrid_transport_l8_and_fit_first(r592) -> None:
    rng = np.random.default_rng(592)
    ex = rng.normal(size=(2, 4, 2)).astype("<f4")
    ux = rng.normal(size=(2, 4, 2, 1152)).astype("<f4")
    ey = rng.normal(size=(2, 4, 2)).astype("<f4")
    uy = rng.normal(size=(2, 4, 2, 1152)).astype("<f4")
    delta = r592.centered_deltas(ex, ux, ey, uy)
    assert np.array_equal(delta[:, 0], np.zeros_like(delta[:, 0]))
    assert np.array_equal(delta[:, 1], r592.bilinear(ey, ux) - r592.bilinear(ex, ux))
    assert np.array_equal(delta[:, 2], r592.bilinear(ex, uy) - r592.bilinear(ex, ux))
    assert np.array_equal(delta[:, 3], r592.bilinear(ey, uy) - r592.bilinear(ex, ux))
    assert set(r592.transport_maxima(ex, ux, ey, uy, ex.copy(), ux.copy())) == {"e", "u", "xx", "yx", "xy", "yy"}
    runtime = blob(RUNTIME).decode()
    intervene = runtime[runtime.index("    def _intervene"):runtime.index("    def execute")]
    assert "total = planned_gpu[local, indices].sum(dim=0)" in intervene
    assert "modified[local, query] += total" in intervene
    assert "native_equality_term" not in intervene and "-=" not in intervene
    science = blob(PRODUCER).decode()
    science = science[science.index("def run_science"):science.index("def build_dryrun")]
    assert science.index("if not any(failure_classes.values())") < science.index("executor, select_bundle")
    assert '"final_opened": False, "ood_opened": False' in science


def test_adapter_pins_repaired_transitive_closure_and_embeds_producer() -> None:
    text = blob(ADAPTER).decode()
    for digest in EXPECTED.values():
        if digest != EXPECTED[ADAPTER]:
            assert digest in text
    for digest in (
        "15219749dd1d696e52c3129052cadce6758b7186390303eace216d98c953188e",
        "7b127fc100192d2ed0eb432ad2cfbf506d151314b1e9419d1e3fa424eb487772",
        "9b8e4ce54d1b34d650ef088f841672cf01a4482257446b611ba37e1353a457cf",
    ):
        assert digest in text
    assert "base64.b64encode(source)" in text
    assert "exec(compile(_b,_p,'exec')" in text


@pytest.mark.xfail(strict=True, reason="current layout needs ~10.68 GB data peak and adapter has no free-space gate")
def test_preflight_rejects_insufficient_peak_disk_before_model_boundary() -> None:
    # Current layout retains the whole FIT call tree while materializing complete FIT evidence.
    fit_call_tree = 5_478_515_712
    fit_complete = 5_198_883_840
    assert fit_call_tree + fit_complete == 10_677_399_552
    adapter = blob(ADAPTER).decode()
    assert "statvfs" in adapter or "disk_usage" in adapter
    assert "required_free_bytes" in adapter


def test_current_free_space_is_recorded_but_not_used_as_a_stable_oracle() -> None:
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    assert type(free) is int and free > 0
