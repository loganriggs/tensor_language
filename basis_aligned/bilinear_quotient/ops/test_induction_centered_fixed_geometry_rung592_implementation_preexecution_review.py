#!/usr/bin/env python3
# BQLANE: cpu
"""Independent, model-free attacks on immutable R592 commit 0bd259b7d."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


COMMIT = "0bd259b7d5a499a863741338f8b55dc11368f344"
ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "basis_aligned" / "bilinear_quotient" / "ops"
PRODUCER_PATH = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592.py"
RUNTIME_PATH = "basis_aligned/bilinear_quotient/ops/induction_centered_fixed_geometry_rung592_runtime.py"
ADAPTER_PATH = "basis_aligned/bilinear_quotient/ops/execute_induction_centered_fixed_geometry_rung592.py"
DRYRUN_PATH = "basis_aligned/bilinear_quotient/induction_centered_fixed_geometry_rung592_dryrun.json"
FACADE_PATH = "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py"

EXPECTED = {
    PRODUCER_PATH: "c52e1225c128de98b01d33649eb4227ff99e63177a8cbd85b9fd0556b4bf5aee",
    RUNTIME_PATH: "df2d59245dc5bd407c96af0a8a6d1c98a70ae25f1925c4540dbd47bb956254a1",
    ADAPTER_PATH: "a104a53411a68527f2702ff9999a9045925ad47cffe56ee6f9966a4eb1e65531",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592.py":
        "85f73a6b35f4e9960320bf23996ebc595d02dcd5a76f34ceaef51a6d502c7d54",
    "basis_aligned/bilinear_quotient/ops/test_induction_centered_fixed_geometry_rung592_fake_runtime.py":
        "f5ea5e005991d57f6d23b5df44d1eccb500ec59469c20f70745ce37f1f6980c0",
    "basis_aligned/bilinear_quotient/ops/test_execute_induction_centered_fixed_geometry_rung592.py":
        "203225f98635680b723f56527a8325d8d2d56e84d6b552008cd3fa3d18cf4dfd",
    DRYRUN_PATH: "a2c6e760b9b87d70b5a444a11d5bd9f76b0090330fc31e67bdee710aa31e517d",
}


def blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def constants(source: bytes) -> dict[str, object]:
    output: dict[str, object] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                output[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return output


@pytest.fixture(scope="module")
def r592():
    source = blob(PRODUCER_PATH)
    logical = OPS / Path(PRODUCER_PATH).name
    spec = importlib.util.spec_from_loader("r592_independent_exact_blob", loader=None, origin=str(logical))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(logical)
    exec(compile(source, str(logical), "exec"), module.__dict__)
    return module


def test_exact_candidate_bytes_and_model_free_manifest() -> None:
    assert subprocess.check_output(["git", "rev-parse", COMMIT], cwd=ROOT, text=True).strip() == COMMIT
    assert {path: hashlib.sha256(blob(path)).hexdigest() for path in EXPECTED} == EXPECTED
    dryrun = json.loads(blob(DRYRUN_PATH))
    assert dryrun["phase_counts"]["FIT"] == {
        "calls": 639, "directed_chunks": 117, "directions": 3744,
        "endpoint_calls": 54, "endpoints": 1728, "rows": 1872,
    }
    assert dryrun["phase_counts"]["SELECT"] == {
        "calls": 322, "directed_chunks": 59, "directions": 1872,
        "endpoint_calls": 27, "endpoints": 864, "rows": 936,
    }
    assert dryrun["select_tail_batch_sizes"] == [16] * 5
    assert dryrun["registered_max_model_forwards"] == 961
    assert dryrun["model_forwards"] == dryrun["model_backwards"] == 0
    assert dryrun["select_opened"] is dryrun["final_opened"] is dryrun["ood_opened"] is False


@pytest.mark.xfail(strict=True, reason="the managed adapter and producer are both rejected by the repository gate")
@pytest.mark.parametrize("path", [ADAPTER_PATH, PRODUCER_PATH])
def test_managed_static_gate_accepts_exact_candidate(path: str) -> None:
    completed = subprocess.run(
        ["python", str(OPS / "gate.py"), str(ROOT / path)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.xfail(strict=True, reason="R592 allocates tokenizer-vocab logits but the pinned model returns logit-vocab logits")
def test_runtime_logit_width_equals_pinned_facade_output_width() -> None:
    producer = constants(blob(PRODUCER_PATH))
    facade = constants(blob(FACADE_PATH))
    assert producer["VOCAB"] == facade["LOGIT_VOCAB"] == 50_304


def _one_call(r592, kind: str) -> tuple[dict[str, object], np.ndarray]:
    tokens = np.full((1, 30), r592.PAD_TOKEN, dtype="<i8")
    tokens[0, :3] = (1, 2, 3)
    call = {
        "manifest_index": 0, "call_id": f"FIT:{kind}", "phase": "FIT",
        "call_kind": kind, "chunk_index": 0,
        "machine_arm": None if kind in ("endpoint", "native") else kind,
        "token_record_id": "FIT:toy", "token_sha256": r592.sha256_bytes(tokens.tobytes()),
        "batch_size": 1, "physical_width": 30, "authority_row_ids": ["e0"],
        "direction_ids": [] if kind == "endpoint" else ["d0"], "query_positions": [2],
    }
    return {"phase": "FIT", "calls": [call], "token_arrays": {"FIT:toy": tokens}}, tokens


def _valid_arrays(r592, call: dict[str, object], tokens: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: (tokens.copy() if name == "tokens.npy" else np.zeros(shape, dtype=dtype))
        for name, (dtype, shape) in r592.mandatory_call_shapes(call).items()
    }


def test_every_directed_arm_can_stop_on_invalid_nonfinite_or_hard_abort(r592, tmp_path: Path) -> None:
    for kind in r592.MACHINE_ARMS:
        for failure in ("hook", "nonfinite", "raise"):
            root = tmp_path / f"{kind}-{failure}"
            root.mkdir()
            bundle, tokens = _one_call(r592, kind)
            call = bundle["calls"][0]
            planned = np.zeros((1, 4, r592.RESIDUAL), dtype="<f4")

            class Executor:
                def execute(self, *_args):
                    if failure == "raise":
                        raise RuntimeError("incomplete")
                    arrays = _valid_arrays(r592, call, tokens)
                    if failure == "hook":
                        arrays["hook_deltas.npy"][0, 0, 0] = np.float32(2e-5)
                    else:
                        arrays["logits.npy"][0, 0] = np.nan
                    return {"arrays": arrays}

            context = {call["call_id"]: {
                "specs": [{"final_position": 2}], "planned": planned,
                "cached": {"native_logits": np.zeros((1, r592.VOCAB), dtype="<f4")},
            }}
            stage = root / "stage"
            stage.mkdir()
            if failure == "raise":
                with pytest.raises(RuntimeError, match="incomplete"):
                    r592.run_manifest_calls(Executor(), bundle, context, stage=stage, public_root=root)
                assert not any((root / path.name).exists() for path in r592.PUBLIC_NAMESPACES)
            else:
                observed = r592.run_manifest_calls(Executor(), bundle, context, stage=stage, public_root=root)
                expected = "centered_hook_delta_failed" if failure == "hook" else "nonfinite_observation"
                assert observed["diagnostic"]["failure_predicate"] == expected
                assert observed["diagnostic"]["executed_call_ids"] == [call["call_id"]]


@pytest.mark.xfail(strict=True, reason="missing required arrays are misclassified as a completed token-manifest diagnostic")
def test_missing_required_observation_is_unpublishable_hard_abort(r592, tmp_path: Path) -> None:
    bundle, tokens = _one_call(r592, "score")
    call = bundle["calls"][0]
    arrays = _valid_arrays(r592, call, tokens)
    arrays.pop("hook_deltas.npy")

    class Missing:
        def execute(self, *_args):
            return {"arrays": arrays}

    context = {call["call_id"]: {
        "specs": [{"final_position": 2}],
        "planned": np.zeros((1, 4, r592.RESIDUAL), dtype="<f4"), "cached": {},
    }}
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises((RuntimeError, ValueError), match="required|missing|incomplete"):
        r592.run_manifest_calls(Missing(), bundle, context, stage=stage, public_root=tmp_path)
    assert not any((tmp_path / path.name).exists() for path in r592.PUBLIC_NAMESPACES)


@pytest.mark.xfail(strict=True, reason="invalid receipt lists mask/index paths but does not content-bind those evidence bytes")
def test_invalid_receipt_content_binds_mask_index_and_mask_bytes(r592, tmp_path: Path) -> None:
    bundle, tokens = _one_call(r592, "score")
    call = bundle["calls"][0]
    arrays = _valid_arrays(r592, call, tokens)
    arrays["logits.npy"][0, 0] = np.nan

    class Nonfinite:
        def execute(self, *_args):
            return {"arrays": arrays}

    context = {call["call_id"]: {
        "specs": [{"final_position": 2}],
        "planned": np.zeros((1, 4, r592.RESIDUAL), dtype="<f4"),
        "cached": {"native_logits": np.zeros((1, r592.VOCAB), dtype="<f4")},
    }}
    stage = tmp_path / "stage"
    stage.mkdir()
    r592.run_manifest_calls(Nonfinite(), bundle, context, stage=stage, public_root=tmp_path)
    receipt = json.loads((tmp_path / r592.INVALID_RECEIPT.name).read_text())
    assert "evidence_files" in receipt
    names = set(receipt["evidence_files"])
    assert any(name.endswith("nonfinite_mask_index.json") for name in names)
    assert any(name.endswith("nonfinite_masks/logits.mask.npy") for name in names)


@pytest.mark.xfail(strict=True, reason="the retained full-native reconstruction gate is transient metadata, not reconstructible evidence")
def test_complete_evidence_independently_reconstructs_full_native_gate() -> None:
    runtime = blob(RUNTIME_PATH).decode()
    producer = blob(PRODUCER_PATH).decode()
    duplicate = (
        'arrays["independent_full_native_write.npy"][local, site_index] = '
        'term["head_output"].float().numpy()'
    )
    assert duplicate not in runtime
    assert "native_full_write_reconstruction_max_abs" in producer[producer.index("def write_complete_phase_evidence"):]


@pytest.mark.xfail(strict=True, reason="large memmap evidence is flushed but never explicitly fsynced before publication")
def test_memmap_evidence_files_are_fsynced_before_receipt_publication() -> None:
    producer = blob(PRODUCER_PATH).decode()
    body = producer[producer.index("def write_complete_phase_evidence"):producer.index("def publish_normal")]
    assert "os.fsync(hook" in body or "_fsync_file(hook_path" in body
    assert "os.fsync(logit" in body or "_fsync_file(logit_path" in body


def test_centered_runtime_uses_frozen_additions_and_one_l8_transaction() -> None:
    runtime = blob(RUNTIME_PATH).decode()
    tree = ast.parse(runtime)
    forward_calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "forward_with_dispatch"
    ]
    assert len(forward_calls) == 2
    assert runtime.count("modified[local, query] += total") == 1
    assert "total = planned_gpu[local, indices].sum(dim=0)" in runtime
    intervention = runtime[runtime.index("    def _intervene"):runtime.index("    def execute")]
    assert "native_equality_term" not in intervention
    assert "-=" not in intervention


def test_publication_is_receipt_last_and_fit_gates_select() -> None:
    producer = blob(PRODUCER_PATH).decode()
    normal = producer[producer.index("def publish_normal"):producer.index("def _empty_scientific_failures")]
    invalid = producer[producer.index("def publish_invalid_prefix"):producer.index("def evaluate_completed_call")]
    assert normal.index("os.replace(evidence") < normal.index("os.replace(result_path") < normal.index("os.replace(receipt_path")
    assert invalid.index("os.replace(evidence") < invalid.index("os.replace(diagnostic_path") < invalid.index("os.replace(receipt_path")
    science = producer[producer.index("def run_science"):producer.index("def build_dryrun")]
    assert science.index('build_phase_manifest(execution, "FIT")') < science.index("run_manifest_calls(")
    assert science.index("if not any(failure_classes.values())") < science.index("executor, select_bundle")
    assert '"final_opened": False, "ood_opened": False' in science
