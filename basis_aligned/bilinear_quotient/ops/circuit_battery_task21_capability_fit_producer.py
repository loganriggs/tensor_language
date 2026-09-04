#!/usr/bin/env python3
"""Model-facing producer for the frozen task-21 FIT capability contract.

This module is loaded only from hash-verified bytes by the managed adapter.  Its
model-free entry compiles and attacks the exact contract without importing
Torch.  Its scientific entry makes the eight registered native calls, retains
only answer and maximum-foil float32 arrays, and publishes one create-only
atomic package.  It cannot generate later phases or perform localization.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Callable, Mapping, Sequence

import numpy as np

import circuit_artifact_package as package
import circuit_battery_task21_capability_fit as capability
import circuit_experiment_spec as framework


REPO_ROOT = Path("/workspace/tensor_language")
BQ_ROOT = REPO_ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ_ROOT / "ops"
POLY = REPO_ROOT / "basis_aligned/polynomial_causal"

NAMESPACE = "circuit_battery_task21_capability_fit_v1"
PATHS = package.PackagePaths(
    root=BQ_ROOT,
    result=BQ_ROOT / f"{NAMESPACE}_results.json",
    receipt=BQ_ROOT / f"{NAMESPACE}_receipt.json",
    evidence=BQ_ROOT / f"{NAMESPACE}_evidence",
    namespace=NAMESPACE,
)

COMPILER_COMMIT = "9ebab94615eade27b1eb63e4f2c6239337b71dc9"
CHECKPOINT_CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
CHECKPOINT_WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
CHECKPOINT_WEIGHTS_BYTES = 2_067_738_635
MODEL_REVISION = "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240"
MODEL_SOURCE_SHA256 = "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2"
FACADE_SOURCE_SHA256 = "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c"

EXPECTED_RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.12.14",
    "numpy_version": "2.5.2",
    "torch_version": "2.11.0+cu128",
    "torch_cuda_version": "12.8",
    "tiktoken_version": "0.14.0",
    "einops_version": "0.8.2",
}
CANARY1_PATH = BQ_ROOT / "bilin18_canary_results.json"
CANARY2_PATH = BQ_ROOT / "bilin18_canary2_results.json"
CANARY2_FINGERPRINT_SHA256 = "6b22b221a811382775e6a64b4198a61f2f9bcc55b826d0d12d0512d1a28be99c"
CANARY2_COMPOSITION = "v2_layer17_mlp_plus_scalar"

RAW_NUMERIC_BYTES = 1_344
FORWARD_CALLS = 8
EXAMPLE_EVALUATIONS = 168
PUBLICATION_PROTOCOL = "linux_renameat2_noreplace_receipt_last_v1"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
_FORBIDDEN_RESULT_WORDS = (
    "reader", "writer", "component", "attention_head", "mlp_site",
    "activation", "localization", "selection",
)


class ProducerError(RuntimeError):
    """The model-facing execution differs from its frozen FIT contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json_bytes(value: object) -> bytes:
    return framework.canonical_json_bytes(value) + b"\n"


def strict_json_file(path: Path, label: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ProducerError(f"{label} contains duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProducerError(f"{label} contains nonfinite JSON: {token}")
            ),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProducerError(f"{label} is missing or invalid") from error
    if type(value) is not dict:
        raise ProducerError(f"{label} is not a JSON object")
    framework.canonical_json_bytes(value)
    return value


def path_entry_exists(path: Path) -> bool:
    """Recognize every directory entry, including a dangling symlink."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def require_unused_namespaces(paths: package.PackagePaths = PATHS) -> None:
    occupied = [
        str(path) for path in (paths.result, paths.receipt, paths.evidence)
        if path_entry_exists(path)
    ]
    if occupied:
        raise ProducerError(f"task21 capability namespace is occupied: {occupied}")


def entry_identity(path: Path) -> tuple[int, int, int, int]:
    status = os.lstat(path)
    return (status.st_dev, status.st_ino, status.st_mode, status.st_size)


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically move one entry while refusing every existing destination."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = library.renameat2
    except AttributeError as error:
        raise ProducerError("renameat2 is unavailable; create-only publication cannot run") from error
    function.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number, "create-only publication destination exists", str(destination)
        )
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise ProducerError(
            "renameat2(RENAME_NOREPLACE) is unavailable; refusing weaker publication"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_task21_package(
    stage: Path,
    paths: package.PackagePaths,
    *,
    crash: Callable[[str], None] | None = None,
    before_move: Callable[[str, Path, Path], None] | None = None,
) -> None:
    """Publish evidence/result/receipt last without ever replacing a final entry."""
    package._validate_stage_tree(stage, paths)
    moves = (
        (stage / "evidence", paths.evidence, "evidence"),
        (stage / "result.json", paths.result, "result"),
        (stage / "receipt.json", paths.receipt, "receipt"),
    )
    if any(path_entry_exists(destination) for _, destination, _ in moves):
        raise ProducerError("task21 final package namespace is occupied")
    installed: list[tuple[Path, Path, str, tuple[int, int, int, int]]] = []
    try:
        for source, destination, label in moves:
            expected_identity = entry_identity(source)
            if before_move is not None:
                before_move(label, source, destination)
            rename_noreplace(source, destination)
            installed.append((source, destination, label, expected_identity))
            if path_entry_exists(source) or entry_identity(destination) != expected_identity:
                raise ProducerError(f"task21 {label} identity changed during publication")
            package._fsync_directory(destination.parent)
            if crash is not None:
                crash(f"published:{label}")
    except BaseException as publication_error:
        rollback_errors = []
        for source, destination, label, expected_identity in reversed(installed):
            try:
                if path_entry_exists(source):
                    raise ProducerError(f"task21 rollback source unexpectedly exists: {label}")
                if not path_entry_exists(destination) \
                        or entry_identity(destination) != expected_identity:
                    raise ProducerError(
                        f"task21 rollback refuses externally replaced destination: {label}"
                    )
                rename_noreplace(destination, source)
                if path_entry_exists(destination) \
                        or entry_identity(source) != expected_identity:
                    raise ProducerError(f"task21 rollback identity changed: {label}")
                package._fsync_directory(source.parent)
            except BaseException as rollback_error:
                rollback_errors.append(f"{label}:{type(rollback_error).__name__}:{rollback_error}")
        if rollback_errors:
            raise ProducerError(
                "task21 publication failed and safe rollback was incomplete: "
                + " | ".join(rollback_errors)
            ) from publication_error
        raise
    (stage / "marker.json").unlink()
    stage.rmdir()
    package._fsync_directory(paths.root)


def validate_canaries(
    canary1_path: Path = CANARY1_PATH,
    canary2_path: Path = CANARY2_PATH,
) -> dict[str, object]:
    """Require the two current model/data-path health checks to be live and passing."""
    first = strict_json_file(canary1_path, "bilin18 canary 1")
    second = strict_json_file(canary2_path, "bilin18 canary 2")
    if not all(first.get(key) is True for key in ("pa", "pb", "pc")):
        raise ProducerError("bilin18 canary 1 is not passing")
    numeric = ("score_rank", "l1_cost", "ratio_5_6", "ratio_14_15")
    if any(type(first.get(key)) not in (int, float) or isinstance(first.get(key), bool)
           or not math.isfinite(float(first[key])) for key in numeric):
        raise ProducerError("bilin18 canary 1 has malformed numeric evidence")
    fingerprint = second.get("fingerprint")
    if not all(second.get(key) is True for key in (
        "canary1", "atlases", "fingerprint_stable_vs_previous", "ALL"
    )) or type(fingerprint) is not dict:
        raise ProducerError("bilin18 canary 2 is not passing")
    if fingerprint.get("composition") != CANARY2_COMPOSITION \
            or fingerprint.get("sha") != CANARY2_FINGERPRINT_SHA256:
        raise ProducerError("bilin18 canary fingerprint changed")
    return {
        "canary1_pass": True,
        "canary2_pass": True,
        "canary2_composition": CANARY2_COMPOSITION,
        "canary2_fingerprint_sha256": CANARY2_FINGERPRINT_SHA256,
    }

def runtime_receipt() -> dict[str, object]:
    """Freeze versions before the checkpoint or CUDA context is opened."""
    import einops
    import tiktoken
    import torch

    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tiktoken_version": tiktoken.__version__,
        "einops_version": einops.__version__,
    }
    if observed != EXPECTED_RUNTIME:
        raise ProducerError(f"runtime versions changed: {observed}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ProducerError("CUDA device is unavailable before checkpoint loading")
    return observed


def compile_from_captured(
    captured: Mapping[str, bytes],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compile only the captured FIT authority through the approved compiler."""
    if "fit_authority" not in captured:
        raise ProducerError("managed closure did not capture FIT authority")
    forbidden = {
        key for key in captured
        if any(name in key.lower() for name in ("select", "test", "ood", "outcome"))
    }
    if forbidden:
        raise ProducerError(f"later-phase or outcome artifact entered closure: {sorted(forbidden)}")
    rows = capability.load_fit_authority_bytes(captured["fit_authority"])
    compiled = capability.compile_fit_invocation(rows)
    if framework.canonical_sha256(compiled) != capability.COMPILED_CONTRACT_SHA256:
        raise ProducerError("captured FIT compilation differs from frozen contract")
    return rows, compiled


def call_inputs(
    rows: Sequence[Mapping[str, object]],
    call: Mapping[str, object],
    metric: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Resolve one exact request without deduplicating repeated row-side prompts."""
    if call.get("call_id") != metric.get("call_id") \
            or call.get("row_ids") != metric.get("row_ids"):
        raise ProducerError("call and native metric manifests are not aligned")
    if call.get("logical_batch_size") != 21 \
            or call.get("padded_sequence_length") != 8:
        raise ProducerError("physical task21 call shape changed")
    side = metric.get("side")
    if side not in ("base", "donor") or call.get("arm") != f"native_{side}":
        raise ProducerError("native call side changed")
    by_id = {str(row["row_id"]): row for row in rows}
    try:
        selected = [by_id[str(row_id)] for row_id in call["row_ids"]]
    except KeyError as error:
        raise ProducerError("physical call names an unknown FIT row") from error
    sequences = np.asarray([row[f"{side}_ids"] for row in selected], dtype="<i8")
    targets = np.asarray(metric["target_token_ids"], dtype="<i8")
    foils = [[int(value) for value in values] for values in metric["foil_token_ids"]]
    if sequences.shape != (21, 8) or targets.shape != (21,) \
            or len(foils) != 21 or any(not values for values in foils):
        raise ProducerError("task21 call inputs differ from exact physical shape")
    if sequences.min() < 0 or sequences.max() >= 50_257 \
            or targets.min() < 0 or targets.max() >= 50_257 \
            or any(value < 0 or value >= 50_257 for values in foils for value in values):
        raise ProducerError("task21 call contains token outside tokenizer support")
    return sequences, targets, foils


def validate_call_arrays(
    answer: np.ndarray, maximum_foil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for label, array in (("answer_logit", answer), ("max_foil_logit", maximum_foil)):
        if type(array) is not np.ndarray or array.shape != (21,) \
                or array.dtype != np.dtype("float32") or not array.flags.c_contiguous \
                or not bool(np.isfinite(array).all()):
            raise ProducerError(f"{label} differs from the exact float32[21] contract")
        values.append(array)
    return values[0], values[1]


CallEvaluator = Callable[
    [Mapping[str, object], np.ndarray, np.ndarray, Sequence[Sequence[int]]],
    tuple[np.ndarray, np.ndarray],
]


def execute_call_manifest(
    rows: Sequence[Mapping[str, object]],
    compiled: Mapping[str, object],
    evaluator: CallEvaluator,
) -> tuple[dict[str, bytes], list[dict[str, object]], list[dict[str, object]]]:
    """Execute the literal eight-call manifest and retain only declared scalars."""
    manifest = compiled.get("call_manifest")
    metrics = compiled.get("metric_manifest")
    if type(manifest) is not list or type(metrics) is not list \
            or len(manifest) != FORWARD_CALLS or len(metrics) != FORWARD_CALLS:
        raise ProducerError("task21 physical or metric call census changed")
    evidence: dict[str, bytes] = {}
    primitives: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    for index, (call, metric) in enumerate(zip(manifest, metrics)):
        sequences, targets, foils = call_inputs(rows, call, metric)
        answer, maximum_foil = validate_call_arrays(
            *evaluator(call, sequences, targets, foils)
        )
        directory = f"calls/{package.call_directory_name(index, str(call['call_id']))}"
        evidence[f"{directory}/call.json"] = strict_json_bytes(call)
        for name, array in (
            ("answer_logit", answer), ("max_foil_logit", maximum_foil)
        ):
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            evidence[f"{directory}/{name}.npy"] = stream.getvalue()
        for local, row_id in enumerate(call["row_ids"]):
            primitives.append({
                "call_id": call["call_id"],
                "row_id": row_id,
                "side": metric["side"],
                "transform_id": metric["transform_ids"][local],
                "answer_logit": float(answer[local]),
                "max_foil_logit": float(maximum_foil[local]),
            })
        completed.append(dict(call))
    if len(primitives) != EXAMPLE_EVALUATIONS:
        raise ProducerError("task21 did not execute all 168 row-side evaluations")
    package.validate_call_prefix(
        manifest, completed,
        [package.call_directory_name(index, str(call["call_id"]))
         for index, call in enumerate(completed)],
    )
    return evidence, primitives, completed


def native_logits(model, tokens):
    """Exact native bilin18 forward for the contract's dynamic 21x8 batches."""
    import torch
    import torch.nn.functional as functional

    x = model.transformer.wte(tokens)
    x = functional.rms_norm(x, (x.size(-1),))
    x0 = x
    first_value = None
    for block in model.transformer.h:
        x, first_value = block(x, first_value, x0)
    logits = model.lm_head(functional.rms_norm(x, (x.size(-1),)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if tuple(logits.shape) != (21, 8, 50_304) or not bool(torch.isfinite(logits).all()):
        raise ProducerError("native bilin18 returned malformed logits")
    return logits


def model_evaluator(model) -> CallEvaluator:
    """Return an evaluator that copies only the two registered scalar arrays to CPU."""
    import torch

    device = next(model.parameters()).device
    if device.type != "cuda" or any(parameter.device != device for parameter in model.parameters()):
        raise ProducerError("observed model is not entirely on one CUDA device")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ProducerError("observed model parameters are not float32")

    def evaluate(call, sequences, targets, foils):
        del call
        token_tensor = torch.as_tensor(sequences, dtype=torch.long, device=device)
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
        with torch.inference_mode():
            final = native_logits(model, token_tensor)[:, -1, :]
            answer = final.gather(1, target_tensor[:, None]).squeeze(1)
            maxima = []
            for local, values in enumerate(foils):
                indices = torch.as_tensor(values, dtype=torch.long, device=device)
                maxima.append(final[local].index_select(0, indices).max())
            maximum_foil = torch.stack(maxima)
        return (
            np.ascontiguousarray(answer.cpu().numpy(), dtype="<f4"),
            np.ascontiguousarray(maximum_foil.cpu().numpy(), dtype="<f4"),
        )

    return evaluate


def evidence_numeric_bytes(evidence: Mapping[str, bytes]) -> int:
    """Count array payload bytes, excluding deterministic .npy framing metadata."""
    total = 0
    for name, payload in evidence.items():
        if not name.endswith(".npy"):
            continue
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        total += int(array.nbytes)
    return total


def validate_result_surface(result: Mapping[str, object]) -> None:
    framework.canonical_json_bytes(result)

    def reject_forbidden_keys(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(word in lowered for word in _FORBIDDEN_RESULT_WORDS):
                    raise ProducerError(
                        "capability result attempted to publish an undeclared analysis surface"
                    )
                reject_forbidden_keys(child)
        elif isinstance(value, list):
            for child in value:
                reject_forbidden_keys(child)

    reject_forbidden_keys(result)
    decision = result.get("decision")
    if type(decision) is not dict or decision.get("terminal") not in ("ok", "hard_abort"):
        raise ProducerError("task21 capability decision terminal is malformed")
    if decision["terminal"] == "hard_abort" and any(
        value is not None for value in decision.get("projection", {}).values()
    ):
        raise ProducerError("capability hard-abort did not null every projection")


def make_result(
    *, compiled: Mapping[str, object], decision: Mapping[str, object],
    completed: Sequence[Mapping[str, object]], checkpoint, runtime: Mapping[str, object],
    canaries: Mapping[str, object], evidence: Mapping[str, bytes], elapsed_seconds: float,
) -> dict[str, object]:
    result = {
        "schema": "circuit_battery_task21_capability_fit_result_v1",
        "experiment_id": capability.EXPERIMENT_ID,
        "compiler_commit": COMPILER_COMMIT,
        "compiler_contract_sha256": capability.COMPILED_CONTRACT_SHA256,
        "compiler_spec_sha256": capability.SPEC_SHA256,
        "fit_authority_sha256": capability.FIT_RECORDS_SHA256,
        "task21_authority_sha256": capability.TASK21_AUTHORITY_SHA256,
        "call_manifest_sha256": capability.CALL_MANIFEST_SHA256,
        "metric_manifest_sha256": capability.METRIC_MANIFEST_SHA256,
        "completed_call_prefix_sha256": framework.canonical_sha256(list(completed)),
        "completed_forward_calls": len(completed),
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "model_backwards": 0,
        "model_updates": 0,
        "raw_numeric_evidence_bytes": evidence_numeric_bytes(evidence),
        "publication_protocol": PUBLICATION_PROTOCOL,
        "literal_price": compiled["literal_price"],
        "decision": dict(decision),
        "checkpoint": {
            "revision": checkpoint.revision,
            "config_sha256": checkpoint.config_sha256,
            "weights_sha256": checkpoint.weights_sha256,
            "weights_bytes": checkpoint.weights_bytes,
            "tokenizer_vocab": checkpoint.tokenizer_vocab,
            "logit_vocab": checkpoint.logit_vocab,
        },
        "runtime": dict(runtime),
        "canaries": dict(canaries),
        "evaluated_phases": ["FIT"],
        "forbidden_phases_opened": [],
        "later_phase_generation": False,
        "elapsed_seconds": float(elapsed_seconds),
    }
    if len(completed) != FORWARD_CALLS \
            or result["completed_call_prefix_sha256"] != capability.CALL_MANIFEST_SHA256 \
            or result["raw_numeric_evidence_bytes"] != RAW_NUMERIC_BYTES:
        raise ProducerError("task21 literal call or evidence price changed")
    if checkpoint.revision != MODEL_REVISION \
            or checkpoint.config_sha256 != CHECKPOINT_CONFIG_SHA256 \
            or checkpoint.weights_sha256 != CHECKPOINT_WEIGHTS_SHA256 \
            or checkpoint.weights_bytes != CHECKPOINT_WEIGHTS_BYTES:
        raise ProducerError("task21 checkpoint receipt differs from frozen model")
    validate_result_surface(result)
    return result


def run_dryrun(captured: Mapping[str, bytes]) -> dict[str, object]:
    """Compile and adversarially exercise the exact plan without Torch or writes."""
    require_unused_namespaces()
    rows, compiled = compile_from_captured(captured)
    calls: list[dict[str, object]] = []

    def passing_evaluator(call, sequences, targets, foils):
        del sequences, targets, foils
        calls.append(dict(call))
        return (
            np.full(21, 1.0, dtype="<f4"),
            np.full(21, 0.0, dtype="<f4"),
        )

    evidence, primitives, completed = execute_call_manifest(
        rows, compiled, passing_evaluator
    )
    held = capability.decide_capability(compiled, primitives)
    planted = [dict(row) for row in primitives]
    for row in planted:
        if row["side"] == "base" and row["transform_id"] == "A1":
            row["answer_logit"] = -1.0
    rejected = capability.decide_capability(compiled, planted)
    if held["terminal"] != "ok" or rejected["terminal"] != "hard_abort" \
            or any(value is not None for value in rejected["projection"].values()):
        raise ProducerError("task21 dry-run passing or capability-fail fixture changed")
    return {
        "schema": "circuit_battery_task21_capability_fit_producer_dryrun_v1",
        "status": "model_free_plan_validated_execution_not_authorized",
        "compiler_commit": COMPILER_COMMIT,
        "compiled_contract_sha256": framework.canonical_sha256(compiled),
        "call_manifest_sha256": framework.canonical_sha256(completed),
        "completed_calls": len(calls),
        "example_evaluations": len(primitives),
        "raw_numeric_evidence_bytes": evidence_numeric_bytes(evidence),
        "publication_protocol": PUBLICATION_PROTOCOL,
        "final_namespace_guard_counts_dangling_symlink": True,
        "evidence_file_count": len(evidence),
        "passing_fixture_terminal": held["terminal"],
        "failing_fixture_terminal": rejected["terminal"],
        "failing_fixture_projection_all_null": all(
            value is None for value in rejected["projection"].values()
        ),
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_updates": 0,
        "queue_touched": False,
        "evaluated_phases": ["FIT"],
        "forbidden_phases_opened": [],
    }


def run_science(
    captured: Mapping[str, bytes],
    *, paths: package.PackagePaths = PATHS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Execute and atomically publish the frozen capability terminal."""
    require_unused_namespaces(paths)
    runtime = runtime_receipt()
    canaries = validate_canaries()
    rows, compiled = compile_from_captured(captured)

    import torch
    import bilin18_observed_model_facade as facade

    if facade.CONFIG_SHA256 != CHECKPOINT_CONFIG_SHA256 \
            or facade.WEIGHTS_SHA256 != CHECKPOINT_WEIGHTS_SHA256 \
            or facade.MODEL_REVISION != MODEL_REVISION:
        raise ProducerError("verified facade constants differ from frozen checkpoint")
    started = clock()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32,
        verify_weights_sha256=True,
    )
    evidence, primitives, completed = execute_call_manifest(
        rows, compiled, model_evaluator(model)
    )
    decision = capability.decide_capability(compiled, primitives)
    result = make_result(
        compiled=compiled, decision=decision, completed=completed,
        checkpoint=checkpoint, runtime=runtime, canaries=canaries,
        evidence=evidence, elapsed_seconds=clock() - started,
    )
    stage = package.stage_package(paths, evidence_files=evidence, result=result)
    publish_task21_package(stage, paths)
    published = package.validate_complete_package(paths)
    if published.get("decision") != decision:
        raise ProducerError("published capability package differs from scored terminal")
    return {
        "schema": "circuit_battery_task21_capability_fit_execution_receipt_v1",
        "terminal": decision["terminal"],
        "result": str(paths.result),
        "receipt": str(paths.receipt),
        "evidence": str(paths.evidence),
        "forward_calls": len(completed),
        "example_evaluations": len(primitives),
        "raw_numeric_evidence_bytes": evidence_numeric_bytes(evidence),
    }
