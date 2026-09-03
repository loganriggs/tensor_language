"""Atomic CPU frame archive and canonical pre-TEST manifest for rung 522.

The archive is one create-only torch file containing exactly the 103 registered
CPU float32 frames and their frozen records.  The pre-TEST manifest is a second
create-only JSON file that binds that archive to every validation/null/control/
selection decision required before TEST may open.  No model, data, or CUDA code
is imported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

import attention8_selective_shared_projector_rung522_state_guard as state_guard


ARCHIVE_SCHEMA = "rung522-frame-archive-v1"
MANIFEST_SCHEMA = "rung522-pretest-manifest-v1"
VALIDATION_EVIDENCE_SCHEMA = "rung522-validation-evidence-v1"
HAAR_SEEDS = tuple(range(52400, 52420))
SCHEDULER_NAMESPACE = "a8-r522-balanced-rows-v1"
PRETEST_INFERENCE_BY_BUCKET = {
    "native_capture": 131,
    "native_replay": 131,
    "self_donor": 2,
    "fit_d0_full_attention8": 95,
    "fit_health": 206,
    "full_attention8_comparator": 36,
    "prediction_a": 2_988,
    "recovery_only": 540,
    "haar": 720,
    "all_three_selection_and_test": 180,
}
PRETEST_INFERENCE_TOTAL = 5_029
EXPECTED_OPTIMIZER = {
    "rank": 4,
    "learning_rate": 0.03,
    "updates": 200,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "loss_epsilon": 1e-12,
    "health_window": 20,
    "orthonormality_atol": 1e-5,
    "minimum_projector_distance": 0.02,
}


class ArchiveViolation(RuntimeError):
    """Raised before an incomplete, mutable, or inconsistent archive is accepted."""


@dataclass(frozen=True)
class FrameArtifact:
    spec: state_guard.FrameSpec
    frame: torch.Tensor
    tensor_sha256: str
    fit_scheduler_payload: Mapping[str, object]
    validation_scheduler_payload: Mapping[str, object]
    fit_record_payload: Mapping[str, object]
    health_record_payload: Mapping[str, object]


@dataclass(frozen=True)
class ArchivedFrameRecord:
    spec: state_guard.FrameSpec
    tensor_sha256: str
    fit_scheduler_sha256: str
    validation_scheduler_sha256: str
    combined_scheduler_sha256: str
    fit_batch_zero_selected_row_ids: Mapping[str, int]
    validation_batch_zero_selected_row_ids: Mapping[str, int]
    fit_scheduler_payload: Mapping[str, object]
    validation_scheduler_payload: Mapping[str, object]
    fit_record_payload: Mapping[str, object]
    health_record_payload: Mapping[str, object]
    fit_record_sha256: str
    health_record_sha256: str
    healthy: bool
    health_failures: tuple[str, ...]
    record_sha256: str


@dataclass(frozen=True)
class LoadedFrameArchive:
    path: Path
    file_sha256: str
    content_sha256: str
    frames: Mapping[str, torch.Tensor]
    records: Mapping[str, ArchivedFrameRecord]


@dataclass(frozen=True)
class CallLedgerSnapshot:
    optimization_forward_events: int
    optimization_backward_events: int
    inference_forward_events: int
    inference_by_bucket: Mapping[str, int]
    removal_inference_forward_events: int = 0


@dataclass(frozen=True)
class ManifestReceipt:
    path: Path
    file_sha256: str
    pretest_freeze: state_guard.PretestFreeze
    selected_all_three_frame_id: str
    selected_all_three_seed: int
    protocol_state: state_guard.ProtocolState


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArchiveViolation("manifest object is not canonical finite JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ArchiveViolation(f"{name} is not a lowercase SHA-256 hash")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and canonical contiguous CPU bytes."""
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ArchiveViolation("only CPU tensors may enter the rung522 archive")
    value = tensor.detach().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header + b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_frame(frame: torch.Tensor, frame_id: str) -> None:
    if not isinstance(frame, torch.Tensor) or frame.device.type != "cpu":
        raise ArchiveViolation(f"{frame_id}: frame must be a CPU tensor")
    if frame.dtype != torch.float32 or tuple(frame.shape) != (1152, 4):
        raise ArchiveViolation(f"{frame_id}: frame must have shape 1152x4 and dtype float32")
    if not bool(torch.isfinite(frame).all()):
        raise ArchiveViolation(f"{frame_id}: frame contains nonfinite values")
    identity = torch.eye(4, dtype=torch.float32)
    error = float((frame.mT @ frame - identity).abs().amax())
    if error > 1e-5:
        raise ArchiveViolation(f"{frame_id}: orthonormality error {error} exceeds 1e-5")


def _canonical_json_copy(value: object, name: str) -> object:
    """Return a detached JSON value, rejecting non-string mapping keys and NaNs."""
    def check(item: object) -> None:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ArchiveViolation(f"{name} contains a non-string mapping key")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    try:
        return json.loads(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ))
    except (TypeError, ValueError) as error:
        raise ArchiveViolation(f"{name} is not canonical finite JSON") from error


def _expected_scheduler_mode(spec: state_guard.FrameSpec) -> str:
    if spec.family == "target_oracle":
        return "single_target_oracle"
    if spec.family == "all_three":
        return "all_three"
    return "two_target"


def _expected_role_names(spec: state_guard.FrameSpec) -> tuple[str, ...]:
    if spec.family == "target_oracle":
        target = spec.training_targets[0]
        return (
            f"{target}:member:0", f"{target}:member:1",
            f"{target}:control:0", f"{target}:control:1",
        )
    return tuple(
        f"{target}:{kind}:0"
        for target in spec.training_targets
        for kind in ("member", "control")
    )


def _validate_scheduler_payload(
    raw: Mapping[str, object], spec: state_guard.FrameSpec, name: str
) -> tuple[dict[str, object], str, dict[str, int]]:
    payload = _canonical_json_copy(raw, name)
    if not isinstance(payload, dict) or set(payload) != {
        "namespace", "mode", "seed", "donor_map_rule", "roles"
    }:
        raise ArchiveViolation(f"{name} schema changed")
    if payload["namespace"] != SCHEDULER_NAMESPACE:
        raise ArchiveViolation(f"{name} namespace changed")
    if payload["mode"] != _expected_scheduler_mode(spec) or payload["seed"] != spec.seed:
        raise ArchiveViolation(f"{name} mode/seed differs from frame specification")
    if payload["donor_map_rule"] != "update_mod_4":
        raise ArchiveViolation(f"{name} donor-map rule changed")
    roles = payload["roles"]
    if not isinstance(roles, list) or len(roles) != len(_expected_role_names(spec)):
        raise ArchiveViolation(f"{name} role census changed")
    expected_names = _expected_role_names(spec)
    batch_zero: dict[str, int] = {}
    for index, role in enumerate(roles):
        if not isinstance(role, dict) or set(role) != {
            "name", "target", "kind", "replica", "permutation"
        }:
            raise ArchiveViolation(f"{name} role schema changed")
        expected_name = expected_names[index]
        if role["name"] != expected_name:
            raise ArchiveViolation(f"{name} role order/name changed")
        target, kind, replica = expected_name.rsplit(":", 2)
        if role["target"] != target or role["kind"] != kind or role["replica"] != int(replica):
            raise ArchiveViolation(f"{name} role metadata is inconsistent")
        permutation = role["permutation"]
        if not isinstance(permutation, list) or not permutation or any(
            not isinstance(row, int) or isinstance(row, bool) or row < 0
            for row in permutation
        ) or len(set(permutation)) != len(permutation):
            raise ArchiveViolation(f"{name} role permutation is invalid")
        batch_zero[expected_name] = permutation[0]
    return payload, _sha256_json(payload), batch_zero


def _require_record_envelope(
    payload: Mapping[str, object], *, spec: state_guard.FrameSpec,
    frame_sha256: str, scheduler_sha256: str,
    batch_zero: Mapping[str, int], scheduler_prefix: str, name: str,
) -> None:
    expected_spec = _canonical_json_copy(asdict(spec), "registered frame specification")
    required = {
        "frame_id", "spec", "frame_sha256", f"{scheduler_prefix}_scheduler_sha256",
        f"{scheduler_prefix}_batch_zero_selected_row_ids",
    }
    if not required <= set(payload):
        raise ArchiveViolation(f"{name} is missing its frame/spec/scheduler envelope")
    if payload["frame_id"] != spec.frame_id or payload["spec"] != expected_spec:
        raise ArchiveViolation(f"{name} frame_id/spec is inconsistent")
    if payload["frame_sha256"] != frame_sha256:
        raise ArchiveViolation(f"{name} frame hash is inconsistent")
    if payload[f"{scheduler_prefix}_scheduler_sha256"] != scheduler_sha256:
        raise ArchiveViolation(f"{name} scheduler hash is inconsistent")
    if payload[f"{scheduler_prefix}_batch_zero_selected_row_ids"] != dict(batch_zero):
        raise ArchiveViolation(f"{name} batch-zero selected-row IDs are inconsistent")


def _initial_frame(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(1152, 4, generator=generator, dtype=torch.float64)
    gram = raw.mT @ raw
    gram = (gram + gram.mT) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    inverse_square_root = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.mT
    return (raw @ inverse_square_root).to(torch.float32)


def _float32_projector_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    overlap = (left.mT @ right).square().sum()
    return float((left.shape[1] + right.shape[1] - 2 * overlap).clamp_min(0).sqrt())


def _validate_fit_and_health_payloads(
    artifact: FrameArtifact,
    *,
    frame_sha256: str,
    fit_scheduler_sha256: str,
    validation_scheduler_sha256: str,
    fit_batch_zero: Mapping[str, int],
    validation_batch_zero: Mapping[str, int],
) -> tuple[dict[str, object], dict[str, object], bool, tuple[str, ...]]:
    fit = _canonical_json_copy(artifact.fit_record_payload, "fit record payload")
    health = _canonical_json_copy(artifact.health_record_payload, "health record payload")
    if not isinstance(fit, dict) or not isinstance(health, dict):
        raise ArchiveViolation("fit and health records must be JSON mappings")
    _require_record_envelope(
        fit, spec=artifact.spec, frame_sha256=frame_sha256,
        scheduler_sha256=fit_scheduler_sha256, batch_zero=fit_batch_zero,
        scheduler_prefix="fit", name="fit record",
    )
    _require_record_envelope(
        health, spec=artifact.spec, frame_sha256=frame_sha256,
        scheduler_sha256=validation_scheduler_sha256, batch_zero=validation_batch_zero,
        scheduler_prefix="validation", name="health record",
    )
    fit_required = {"coefficient", "optimizer", "loss_history", "maximizing_targets"}
    health_required = {
        "healthy", "failures", "initial_validation_objective",
        "final_validation_objective", "initial_window_mean", "final_window_mean",
        "orthonormality_error", "projector_distance_from_initialization",
    }
    if not fit_required <= set(fit) or not health_required <= set(health):
        raise ArchiveViolation("fit/health record is missing registered measurements")
    expected_coefficient = 0.0 if artifact.spec.family == "recovery_only" else 24.0
    if fit["coefficient"] != expected_coefficient:
        raise ArchiveViolation("fit record control coefficient changed")
    expected_optimizer = {**EXPECTED_OPTIMIZER, "control_coefficient": expected_coefficient}
    if fit["optimizer"] != expected_optimizer:
        raise ArchiveViolation("fit record optimizer configuration changed")
    history = fit["loss_history"]
    maximizing = fit["maximizing_targets"]
    if not isinstance(history, list) or len(history) != 200 or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in history
    ):
        raise ArchiveViolation("fit record must contain exactly 200 numeric losses")
    if not isinstance(maximizing, list) or len(maximizing) != 200 or any(
        target not in artifact.spec.training_targets for target in maximizing
    ):
        raise ArchiveViolation("fit record maximizing-target history changed")
    numeric_health_names = (
        "initial_validation_objective", "final_validation_objective",
        "initial_window_mean", "final_window_mean", "orthonormality_error",
        "projector_distance_from_initialization",
    )
    if any(
        not isinstance(health[name], (int, float)) or isinstance(health[name], bool)
        or not math.isfinite(float(health[name])) for name in numeric_health_names
    ):
        raise ArchiveViolation("health record contains a nonfinite/non-numeric measurement")
    initial_window = sum(float(value) for value in history[:20]) / 20
    final_window = sum(float(value) for value in history[-20:]) / 20
    orthonormality = float(
        (artifact.frame.mT @ artifact.frame - torch.eye(4, dtype=torch.float32)).abs().amax()
    )
    distance = _float32_projector_distance(_initial_frame(artifact.spec.seed), artifact.frame)
    derived = {
        "initial_window_mean": initial_window,
        "final_window_mean": final_window,
        "orthonormality_error": orthonormality,
        "projector_distance_from_initialization": distance,
    }
    for name, observed in derived.items():
        if not math.isclose(float(health[name]), observed, rel_tol=0.0, abs_tol=1e-12):
            raise ArchiveViolation(f"health record {name} is not derived from archived data")
    failures: list[str] = []
    if not all(math.isfinite(float(value)) for value in history):
        failures.append("nonfinite_loss")
    if final_window >= initial_window:
        failures.append("final_window_not_below_initial_window")
    if float(health["final_validation_objective"]) >= float(
        health["initial_validation_objective"]
    ):
        failures.append("validation_not_better_than_initialization")
    if orthonormality > EXPECTED_OPTIMIZER["orthonormality_atol"]:
        failures.append("orthonormality")
    if distance <= EXPECTED_OPTIMIZER["minimum_projector_distance"]:
        failures.append("projector_did_not_move")
    expected_healthy = not failures
    if health["healthy"] is not expected_healthy or health["failures"] != failures:
        raise ArchiveViolation("health state/failures do not match the archived measurements")
    return fit, health, expected_healthy, tuple(failures)


def _record_payload(record: ArchivedFrameRecord) -> dict[str, object]:
    return {
        "spec": asdict(record.spec),
        "tensor_sha256": record.tensor_sha256,
        "fit_scheduler_sha256": record.fit_scheduler_sha256,
        "validation_scheduler_sha256": record.validation_scheduler_sha256,
        "combined_scheduler_sha256": record.combined_scheduler_sha256,
        "fit_batch_zero_selected_row_ids": dict(record.fit_batch_zero_selected_row_ids),
        "validation_batch_zero_selected_row_ids": dict(
            record.validation_batch_zero_selected_row_ids
        ),
        "fit_scheduler_payload": dict(record.fit_scheduler_payload),
        "validation_scheduler_payload": dict(record.validation_scheduler_payload),
        "fit_record_payload": dict(record.fit_record_payload),
        "health_record_payload": dict(record.health_record_payload),
        "fit_record_sha256": record.fit_record_sha256,
        "health_record_sha256": record.health_record_sha256,
        "healthy": record.healthy,
        "health_failures": list(record.health_failures),
    }


def _validate_artifact(artifact: FrameArtifact) -> ArchivedFrameRecord:
    expected = state_guard.EXPECTED_FRAME_SPECS.get(artifact.spec.frame_id)
    if expected is None or artifact.spec != expected:
        raise ArchiveViolation("frame specification differs from the registered 103-frame census")
    _validate_frame(artifact.frame, artifact.spec.frame_id)
    observed_tensor_hash = tensor_sha256(artifact.frame)
    if artifact.tensor_sha256 != observed_tensor_hash:
        raise ArchiveViolation(f"{artifact.spec.frame_id}: claimed tensor hash does not match bytes")
    fit_scheduler, fit_scheduler_hash, fit_batch_zero = _validate_scheduler_payload(
        artifact.fit_scheduler_payload, artifact.spec, "FIT scheduler"
    )
    validation_scheduler, validation_scheduler_hash, validation_batch_zero = (
        _validate_scheduler_payload(
            artifact.validation_scheduler_payload, artifact.spec, "VALIDATION scheduler"
        )
    )
    fit_record, health_record, healthy, failures = _validate_fit_and_health_payloads(
        artifact,
        frame_sha256=observed_tensor_hash,
        fit_scheduler_sha256=fit_scheduler_hash,
        validation_scheduler_sha256=validation_scheduler_hash,
        fit_batch_zero=fit_batch_zero,
        validation_batch_zero=validation_batch_zero,
    )
    combined_scheduler_hash = hashlib.sha256(
        (fit_scheduler_hash + validation_scheduler_hash).encode("ascii")
    ).hexdigest()
    preliminary = ArchivedFrameRecord(
        spec=artifact.spec,
        tensor_sha256=observed_tensor_hash,
        fit_scheduler_sha256=fit_scheduler_hash,
        validation_scheduler_sha256=validation_scheduler_hash,
        combined_scheduler_sha256=combined_scheduler_hash,
        fit_batch_zero_selected_row_ids=fit_batch_zero,
        validation_batch_zero_selected_row_ids=validation_batch_zero,
        fit_scheduler_payload=fit_scheduler,
        validation_scheduler_payload=validation_scheduler,
        fit_record_payload=fit_record,
        health_record_payload=health_record,
        fit_record_sha256=_sha256_json(fit_record),
        health_record_sha256=_sha256_json(health_record),
        healthy=healthy,
        health_failures=failures,
        record_sha256="",
    )
    return replace(preliminary, record_sha256=_sha256_json(_record_payload(preliminary)))


def _record_to_payload(record: ArchivedFrameRecord) -> dict[str, object]:
    return {**_record_payload(record), "record_sha256": record.record_sha256}


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite archive: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as sink:
            torch.save(value, sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_frame_archive(
    path: Path | str, artifacts: Sequence[FrameArtifact]
) -> LoadedFrameArchive:
    """Validate and atomically create one exact 103-frame archive."""
    target = Path(path)
    items = tuple(artifacts)
    if len(items) != 103:
        raise ArchiveViolation(f"frame archive requires exactly 103 artifacts, got {len(items)}")
    identifiers = [artifact.spec.frame_id for artifact in items]
    if len(set(identifiers)) != len(identifiers):
        raise ArchiveViolation("frame archive contains duplicate frame identifiers")
    missing = sorted(set(state_guard.EXPECTED_FRAME_SPECS) - set(identifiers))
    extra = sorted(set(identifiers) - set(state_guard.EXPECTED_FRAME_SPECS))
    if missing or extra:
        raise ArchiveViolation(f"frame census differs: missing={missing[:3]}, extra={extra[:3]}")

    records = {artifact.spec.frame_id: _validate_artifact(artifact) for artifact in items}
    ordered_records = [_record_to_payload(records[name]) for name in sorted(records)]
    content_sha256 = _sha256_json(ordered_records)
    payload = {
        "schema": ARCHIVE_SCHEMA,
        "frame_count": 103,
        "content_sha256": content_sha256,
        "records": ordered_records,
        "frames": {
            artifact.spec.frame_id: artifact.frame.detach().contiguous().clone()
            for artifact in sorted(items, key=lambda item: item.spec.frame_id)
        },
    }
    _atomic_torch_save(target, payload)
    return load_frame_archive(target)


def _spec_from_payload(value: Mapping[str, object]) -> state_guard.FrameSpec:
    expected_keys = {
        "frame_id", "family", "seed", "training_targets", "health_targets",
        "omitted_target", "oracle_target",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ArchiveViolation("archived frame specification schema changed")
    try:
        return state_guard.FrameSpec(
            frame_id=str(value["frame_id"]),
            family=str(value["family"]),  # type: ignore[arg-type]
            seed=int(value["seed"]),
            training_targets=tuple(value["training_targets"]),  # type: ignore[arg-type]
            health_targets=tuple(value["health_targets"]),  # type: ignore[arg-type]
            omitted_target=value.get("omitted_target"),  # type: ignore[arg-type]
            oracle_target=value.get("oracle_target"),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArchiveViolation("archived frame specification is malformed") from error


def load_frame_archive(path: Path | str) -> LoadedFrameArchive:
    """Load and fully revalidate an archive before returning any frame."""
    source = Path(path)
    if not source.is_file():
        raise ArchiveViolation(f"frame archive does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "frame_count", "content_sha256", "records", "frames"
    }:
        raise ArchiveViolation("frame archive schema changed")
    if payload["schema"] != ARCHIVE_SCHEMA or payload["frame_count"] != 103:
        raise ArchiveViolation("frame archive schema or count changed")
    if not isinstance(payload["records"], list) or not isinstance(payload["frames"], dict):
        raise ArchiveViolation("frame archive records/frames are malformed")
    if len(payload["records"]) != 103 or len(payload["frames"]) != 103:
        raise ArchiveViolation("frame archive does not contain exactly 103 records and tensors")

    if set(payload["frames"]) != set(state_guard.EXPECTED_FRAME_SPECS):
        raise ArchiveViolation("archived frame keys differ from the registered census")
    records: dict[str, ArchivedFrameRecord] = {}
    canonical_records = []
    expected_record_keys = {
        "spec", "tensor_sha256", "fit_scheduler_sha256",
        "validation_scheduler_sha256", "combined_scheduler_sha256",
        "fit_batch_zero_selected_row_ids", "validation_batch_zero_selected_row_ids",
        "fit_scheduler_payload", "validation_scheduler_payload",
        "fit_record_payload", "health_record_payload", "fit_record_sha256",
        "health_record_sha256", "healthy", "health_failures", "record_sha256",
    }
    observed_order: list[str] = []
    for raw in payload["records"]:
        if not isinstance(raw, dict) or set(raw) != expected_record_keys:
            raise ArchiveViolation("archived frame record schema changed")
        spec = _spec_from_payload(raw["spec"])
        if spec.frame_id in records:
            raise ArchiveViolation("archived frame records contain a duplicate identifier")
        if spec.frame_id not in payload["frames"]:
            raise ArchiveViolation("archived frame record has no matching tensor")
        base = {key: raw[key] for key in raw if key != "record_sha256"}
        if raw["record_sha256"] != _sha256_json(base):
            raise ArchiveViolation(f"{spec.frame_id}: record hash does not match record")
        artifact = FrameArtifact(
            spec=spec,
            frame=payload["frames"][spec.frame_id],
            tensor_sha256=raw["tensor_sha256"],
            fit_scheduler_payload=raw["fit_scheduler_payload"],
            validation_scheduler_payload=raw["validation_scheduler_payload"],
            fit_record_payload=raw["fit_record_payload"],
            health_record_payload=raw["health_record_payload"],
        )
        record = _validate_artifact(artifact)
        if _record_to_payload(record) != raw:
            raise ArchiveViolation(f"{spec.frame_id}: derived record fields do not match")
        records[spec.frame_id] = record
        canonical_records.append(raw)
        observed_order.append(spec.frame_id)

    if observed_order != sorted(observed_order) or set(records) != set(
        state_guard.EXPECTED_FRAME_SPECS
    ):
        raise ArchiveViolation("archived frame record order/census is not canonical")
    if payload["content_sha256"] != _sha256_json(canonical_records):
        raise ArchiveViolation("archive content hash does not match its records")
    frames = {}
    for frame_id in sorted(records):
        frame = payload["frames"][frame_id]
        _validate_frame(frame, frame_id)
        if tensor_sha256(frame) != records[frame_id].tensor_sha256:
            raise ArchiveViolation(f"{frame_id}: tensor bytes do not match the frozen record")
        frames[frame_id] = frame.detach().contiguous()
    return LoadedFrameArchive(
        path=source,
        file_sha256=_sha256_file(source),
        content_sha256=payload["content_sha256"],
        frames=frames,
        records=records,
    )


def _normalize_seed_hashes(
    values: Mapping[int | str, str], expected_seeds: Sequence[int], name: str
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_seed, value in values.items():
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as error:
            raise ArchiveViolation(f"{name} seed is not an integer") from error
        if str(seed) in normalized:
            raise ArchiveViolation(f"{name} contains duplicate seed {seed}")
        _require_hash(value, f"{name}:{seed}")
        normalized[str(seed)] = value
    if set(map(int, normalized)) != set(expected_seeds):
        raise ArchiveViolation(f"{name} must contain exactly seeds {tuple(expected_seeds)}")
    return {str(seed): normalized[str(seed)] for seed in sorted(expected_seeds)}


def _projector_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    overlap = float((left.mT.double() @ right.double()).square().sum())
    squared = left.shape[1] + right.shape[1] - 2 * overlap
    return math.sqrt(max(squared, 0.0))


def geometry_only_grassmann_medoid(
    archive: LoadedFrameArchive, eligible_frame_ids: Sequence[str]
) -> tuple[str, dict[str, object]]:
    """Choose minimum summed projector distance; lower seed is exact tie-break."""
    eligible = tuple(eligible_frame_ids)
    if not eligible or len(set(eligible)) != len(eligible):
        raise ArchiveViolation("eligible all-three IDs must be nonempty and unique")
    for frame_id in eligible:
        record = archive.records.get(frame_id)
        if record is None or record.spec.family != "all_three":
            raise ArchiveViolation("medoid eligibility contains a non-all-three frame")
    pairwise: dict[str, dict[str, float]] = {}
    sums = {}
    for left_id in sorted(eligible):
        row = {}
        for right_id in sorted(eligible):
            row[right_id] = _projector_distance(
                archive.frames[left_id], archive.frames[right_id]
            )
        pairwise[left_id] = row
        sums[left_id] = sum(row.values())
    selected = min(
        eligible,
        key=lambda frame_id: (sums[frame_id], archive.records[frame_id].spec.seed),
    )
    decision = {
        "rule": "grassmann_medoid_lower_seed_tiebreak",
        "eligible_all_three_frame_ids": sorted(eligible),
        "pairwise_projector_frobenius_distances": pairwise,
        "summed_distances": sums,
        "selected_frame_id": selected,
        "selected_seed": archive.records[selected].spec.seed,
        "selection_targets": list(state_guard.FITTED_TARGETS),
    }
    decision["sha256"] = _sha256_json(decision)
    return selected, decision


def _validate_call_ledger(ledger: CallLedgerSnapshot) -> None:
    if not isinstance(ledger, CallLedgerSnapshot):
        raise ArchiveViolation("call ledger must be a CallLedgerSnapshot")
    scalar_values = (
        ledger.optimization_forward_events,
        ledger.optimization_backward_events,
        ledger.inference_forward_events,
        ledger.removal_inference_forward_events,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in scalar_values
    ):
        raise ArchiveViolation("call-ledger values must be nonnegative integers")
    if ledger.optimization_forward_events != 20_600 or (
        ledger.optimization_backward_events != 20_600
    ):
        raise ArchiveViolation("pre-TEST archive requires exactly 20,600 forward/backward events")
    buckets = dict(ledger.inference_by_bucket)
    if buckets != PRETEST_INFERENCE_BY_BUCKET or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in buckets.values()
    ):
        raise ArchiveViolation("pre-TEST inference bucket ledger changed")
    if ledger.inference_forward_events != PRETEST_INFERENCE_TOTAL or (
        ledger.inference_forward_events != sum(buckets.values())
    ):
        raise ArchiveViolation("pre-TEST inference total must be exactly 5,029")
    if ledger.removal_inference_forward_events != 0:
        raise ArchiveViolation("removal inference is illegal before TEST opens")


def write_pretest_manifest(
    path: Path | str,
    *,
    archive_path: Path | str,
    null_hashes: Mapping[int | str, str],
    validation_decisions: object,
    validation_provisional_gates_passed: bool,
    validation_evidence_path: Path | str,
    haar_hashes: Mapping[int | str, str],
    eligible_all_three_frame_ids: Sequence[str],
    fit_mu_q: torch.Tensor,
    fit_mu_q_source_split: str,
    call_ledger: CallLedgerSnapshot,
    fingerprint_definition_sha256: str,
    test_sweep_plan_sha256: str,
) -> ManifestReceipt:
    """Atomically write a canonical manifest and validate it through PretestFreeze."""
    target = Path(path)
    archive = load_frame_archive(archive_path)
    if not isinstance(validation_provisional_gates_passed, bool):
        raise ArchiveViolation("validation_provisional_gates_passed must be boolean")
    if is_dataclass(validation_decisions) and not isinstance(validation_decisions, type):
        raw_validation_decisions = asdict(validation_decisions)
    elif isinstance(validation_decisions, Mapping):
        raw_validation_decisions = validation_decisions
    else:
        raise ArchiveViolation("validation decision must be a dataclass or mapping")
    decisions = _canonical_json_copy(raw_validation_decisions, "validation decision")
    if not isinstance(decisions, dict) or not {
        "pretest_passes", "eligible_all_three_frame_ids"
    } <= set(decisions):
        raise ArchiveViolation(
            "validation decision is missing pretest_passes/eligible all-three IDs"
        )
    if type(decisions["pretest_passes"]) is not bool:
        raise ArchiveViolation("validation decision pretest_passes must be a literal bool")
    if decisions["pretest_passes"] is not validation_provisional_gates_passed:
        raise ArchiveViolation("separate pass flag disagrees with validation decision")
    if not decisions["pretest_passes"]:
        raise ArchiveViolation("VALIDATION decision has pretest_passes=False")
    evidence_path = Path(validation_evidence_path)
    if not evidence_path.is_file():
        raise ArchiveViolation("create-only VALIDATION evidence file is absent")
    try:
        evidence_bytes = evidence_path.read_bytes()
        raw_evidence = json.loads(evidence_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveViolation("VALIDATION evidence is not readable canonical JSON") from error
    evidence_file_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    evidence = _canonical_json_copy(raw_evidence, "VALIDATION evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema", "validation_outputs", "provisional_validation_decision", "call_ledger"
    }:
        raise ArchiveViolation("VALIDATION evidence schema changed")
    if evidence["schema"] != VALIDATION_EVIDENCE_SCHEMA:
        raise ArchiveViolation("VALIDATION evidence namespace changed")
    if evidence["provisional_validation_decision"] != decisions:
        raise ArchiveViolation("VALIDATION evidence decision differs from the manifest decision")
    nulls = _normalize_seed_hashes(
        null_hashes, state_guard.PERMUTATION_SEEDS, "label-null hashes"
    )
    haars = _normalize_seed_hashes(haar_hashes, HAAR_SEEDS, "Haar hashes")
    validation_hash = _sha256_json(decisions)
    _require_hash(fingerprint_definition_sha256, "fingerprint_definition_sha256")
    _require_hash(test_sweep_plan_sha256, "test_sweep_plan_sha256")
    _validate_call_ledger(call_ledger)
    if evidence["call_ledger"] != asdict(call_ledger):
        raise ArchiveViolation("VALIDATION evidence ledger differs from the manifest ledger")
    if fit_mu_q_source_split != "FIT":
        raise ArchiveViolation("mu_Q must be computed from FIT only")
    if not isinstance(fit_mu_q, torch.Tensor) or fit_mu_q.device.type != "cpu" or (
        fit_mu_q.dtype != torch.float32 or tuple(fit_mu_q.shape) != (4,)
    ) or not bool(torch.isfinite(fit_mu_q).all()):
        raise ArchiveViolation("FIT mu_Q must be a finite CPU float32 vector of length four")
    mu_hash = tensor_sha256(fit_mu_q)

    decision_eligible_raw = decisions["eligible_all_three_frame_ids"]
    if not isinstance(decision_eligible_raw, list) or not decision_eligible_raw or any(
        not isinstance(frame_id, str) for frame_id in decision_eligible_raw
    ) or len(set(decision_eligible_raw)) != len(decision_eligible_raw):
        raise ArchiveViolation(
            "validation decision eligible all-three IDs must be nonempty and unique"
        )
    canonical_eligible = tuple(decision_eligible_raw)
    if canonical_eligible != tuple(sorted(canonical_eligible)):
        raise ArchiveViolation("validation decision eligible all-three IDs are not canonical")
    if tuple(eligible_all_three_frame_ids) != canonical_eligible:
        raise ArchiveViolation("caller eligible IDs disagree with validation decision")
    for frame_id in canonical_eligible:
        record = archive.records.get(frame_id)
        if record is None or record.spec.family != "all_three":
            raise ArchiveViolation("validation decision contains a non-all-three frame")
        if not record.healthy:
            raise ArchiveViolation("validation decision marks an unhealthy all-three frame eligible")
    selected, medoid = geometry_only_grassmann_medoid(archive, canonical_eligible)
    medoid_hash = medoid["sha256"]
    state = state_guard.ProtocolState()
    for frame_id in sorted(archive.records):
        record = archive.records[frame_id]
        state.authorize_training(
            frame_id,
            split="FIT",
            training_targets=record.spec.training_targets,
            health_targets=record.spec.health_targets,
        )
        state.register_frozen_frame(state_guard.FrozenFrame(
            spec=record.spec,
            frame_sha256=record.tensor_sha256,
            scheduler_sha256=record.combined_scheduler_sha256,
        ))
    state.record_optimization_events(
        call_ledger.optimization_forward_events,
        call_ledger.optimization_backward_events,
    )
    state.record_inference_events(call_ledger.inference_forward_events)

    freeze = state_guard.PretestFreeze(
        frame_manifest_sha256=state.frame_manifest_sha256(),
        scheduler_manifest_sha256=state.scheduler_manifest_sha256(),
        validation_decisions_sha256=validation_hash,
        medoid_selection_sha256=medoid_hash,
        fingerprint_definition_sha256=fingerprint_definition_sha256,
        test_sweep_plan_sha256=test_sweep_plan_sha256,
        registered_contract_sha256=state_guard.registered_contract_sha256(),
        selected_final_frame_id=selected,
        eligible_all_three_frame_ids=canonical_eligible,
        selection_targets=state_guard.FITTED_TARGETS,
        validation_provisional_gates_passed=bool(validation_provisional_gates_passed),
        medoid_selection_rule="grassmann_medoid_lower_seed_tiebreak",
        test_sweep_plan_frozen=True,
    )
    state.freeze_pretest(freeze)
    payload: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "test_opened": False,
        "post_test_fitting_allowed": False,
        "frame_archive": {
            "path": str(archive.path.resolve()),
            "file_sha256": archive.file_sha256,
            "content_sha256": archive.content_sha256,
            "frame_count": 103,
        },
        "frame_records": [
            _record_to_payload(archive.records[frame_id])
            for frame_id in sorted(archive.records)
        ],
        "frame_health_states": {
            frame_id: {
                "healthy": archive.records[frame_id].healthy,
                "failures": list(archive.records[frame_id].health_failures),
            }
            for frame_id in sorted(archive.records)
        },
        "pretest_freeze": asdict(freeze),
        "label_null_sha256": nulls,
        "validation_decisions": decisions,
        "validation_decisions_sha256": validation_hash,
        "validation_evidence": {
            "path": str(evidence_path.resolve()),
            "file_sha256": evidence_file_sha256,
            "canonical_content_sha256": _sha256_json(evidence),
        },
        "haar_sha256": haars,
        "medoid_selection": medoid,
        "fit_mu_q": {
            "source_split": "FIT",
            "dtype": "torch.float32",
            "shape": [4],
            "values": fit_mu_q.tolist(),
            "sha256": mu_hash,
            "selected_frame_id": selected,
        },
        "call_ledger": asdict(call_ledger),
        "registered_price": dict(state_guard.REGISTERED_PRICE),
        "inference_ledger": dict(state_guard.INFERENCE_LEDGER),
        "fingerprint_definition_sha256": fingerprint_definition_sha256,
        "test_sweep_plan_sha256": test_sweep_plan_sha256,
        "registered_contract_sha256": state_guard.registered_contract_sha256(),
    }
    payload["canonical_content_sha256"] = _sha256_json(payload)
    _atomic_json(target, payload)
    return ManifestReceipt(
        path=target,
        file_sha256=_sha256_file(target),
        pretest_freeze=freeze,
        selected_all_three_frame_id=selected,
        selected_all_three_seed=archive.records[selected].spec.seed,
        protocol_state=state,
    )


__all__ = [
    "ARCHIVE_SCHEMA", "ArchiveViolation", "ArchivedFrameRecord", "CallLedgerSnapshot",
    "EXPECTED_OPTIMIZER", "FrameArtifact", "HAAR_SEEDS", "LoadedFrameArchive",
    "MANIFEST_SCHEMA", "ManifestReceipt", "PRETEST_INFERENCE_BY_BUCKET",
    "PRETEST_INFERENCE_TOTAL", "SCHEDULER_NAMESPACE",
    "VALIDATION_EVIDENCE_SCHEMA",
    "geometry_only_grassmann_medoid", "load_frame_archive", "tensor_sha256",
    "write_frame_archive", "write_pretest_manifest",
]
