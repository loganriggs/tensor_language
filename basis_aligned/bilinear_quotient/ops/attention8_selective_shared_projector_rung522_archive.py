"""Atomic CPU frame archive and canonical pre-TEST manifest for rung 522.

The archive is one create-only torch file containing exactly the 103 registered
CPU float32 frames and their frozen records.  The pre-TEST manifest is a second
create-only JSON file that binds that archive to every validation/null/control/
selection decision required before TEST may open.  No model, data, or CUDA code
is imported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
HAAR_SEEDS = tuple(range(52400, 52420))


class ArchiveViolation(RuntimeError):
    """Raised before an incomplete, mutable, or inconsistent archive is accepted."""


@dataclass(frozen=True)
class FrameArtifact:
    spec: state_guard.FrameSpec
    frame: torch.Tensor
    tensor_sha256: str
    scheduler_sha256: str
    fit_record_sha256: str
    health_record_sha256: str


@dataclass(frozen=True)
class ArchivedFrameRecord:
    spec: state_guard.FrameSpec
    tensor_sha256: str
    scheduler_sha256: str
    fit_record_sha256: str
    health_record_sha256: str
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


def _record_payload(artifact: FrameArtifact) -> dict[str, object]:
    return {
        "spec": asdict(artifact.spec),
        "tensor_sha256": artifact.tensor_sha256,
        "scheduler_sha256": artifact.scheduler_sha256,
        "fit_record_sha256": artifact.fit_record_sha256,
        "health_record_sha256": artifact.health_record_sha256,
    }


def _validate_artifact(artifact: FrameArtifact) -> ArchivedFrameRecord:
    expected = state_guard.EXPECTED_FRAME_SPECS.get(artifact.spec.frame_id)
    if expected is None or artifact.spec != expected:
        raise ArchiveViolation("frame specification differs from the registered 103-frame census")
    _validate_frame(artifact.frame, artifact.spec.frame_id)
    observed_tensor_hash = tensor_sha256(artifact.frame)
    if artifact.tensor_sha256 != observed_tensor_hash:
        raise ArchiveViolation(f"{artifact.spec.frame_id}: claimed tensor hash does not match bytes")
    for name, value in (
        ("tensor_sha256", artifact.tensor_sha256),
        ("scheduler_sha256", artifact.scheduler_sha256),
        ("fit_record_sha256", artifact.fit_record_sha256),
        ("health_record_sha256", artifact.health_record_sha256),
    ):
        _require_hash(value, f"{artifact.spec.frame_id}:{name}")
    payload = _record_payload(artifact)
    return ArchivedFrameRecord(
        spec=artifact.spec,
        tensor_sha256=artifact.tensor_sha256,
        scheduler_sha256=artifact.scheduler_sha256,
        fit_record_sha256=artifact.fit_record_sha256,
        health_record_sha256=artifact.health_record_sha256,
        record_sha256=_sha256_json(payload),
    )


def _record_to_payload(record: ArchivedFrameRecord) -> dict[str, object]:
    return {
        "spec": asdict(record.spec),
        "tensor_sha256": record.tensor_sha256,
        "scheduler_sha256": record.scheduler_sha256,
        "fit_record_sha256": record.fit_record_sha256,
        "health_record_sha256": record.health_record_sha256,
        "record_sha256": record.record_sha256,
    }


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

    records: dict[str, ArchivedFrameRecord] = {}
    canonical_records = []
    for raw in payload["records"]:
        if not isinstance(raw, dict) or set(raw) != {
            "spec", "tensor_sha256", "scheduler_sha256", "fit_record_sha256",
            "health_record_sha256", "record_sha256",
        }:
            raise ArchiveViolation("archived frame record schema changed")
        spec = _spec_from_payload(raw["spec"])
        if spec.frame_id in records:
            raise ArchiveViolation("archived frame records contain a duplicate identifier")
        expected = state_guard.EXPECTED_FRAME_SPECS.get(spec.frame_id)
        if expected is None or spec != expected:
            raise ArchiveViolation("archived frame record differs from registered specification")
        base = {key: raw[key] for key in raw if key != "record_sha256"}
        if raw["record_sha256"] != _sha256_json(base):
            raise ArchiveViolation(f"{spec.frame_id}: record hash does not match record")
        for key in (
            "tensor_sha256", "scheduler_sha256", "fit_record_sha256",
            "health_record_sha256", "record_sha256",
        ):
            _require_hash(raw[key], f"{spec.frame_id}:{key}")
        record = ArchivedFrameRecord(
            spec=spec,
            tensor_sha256=raw["tensor_sha256"],
            scheduler_sha256=raw["scheduler_sha256"],
            fit_record_sha256=raw["fit_record_sha256"],
            health_record_sha256=raw["health_record_sha256"],
            record_sha256=raw["record_sha256"],
        )
        records[spec.frame_id] = record
        canonical_records.append(raw)

    if set(records) != set(state_guard.EXPECTED_FRAME_SPECS) or set(payload["frames"]) != set(records):
        raise ArchiveViolation("archived frame keys differ from the registered census")
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
    values = asdict(ledger)
    if any(not isinstance(value, int) or value < 0 for value in values.values()):
        raise ArchiveViolation("call-ledger values must be nonnegative integers")
    if ledger.optimization_forward_events != 20_600 or (
        ledger.optimization_backward_events != 20_600
    ):
        raise ArchiveViolation("pre-TEST archive requires exactly 20,600 forward/backward events")
    if ledger.inference_forward_events > 12_000:
        raise ArchiveViolation("inference-forward ceiling exceeded")
    if ledger.removal_inference_forward_events != 0:
        raise ArchiveViolation("removal inference is illegal before TEST opens")


def write_pretest_manifest(
    path: Path | str,
    *,
    archive_path: Path | str,
    null_hashes: Mapping[int | str, str],
    validation_decisions: Mapping[str, object],
    validation_provisional_gates_passed: bool,
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
    nulls = _normalize_seed_hashes(
        null_hashes, state_guard.PERMUTATION_SEEDS, "label-null hashes"
    )
    haars = _normalize_seed_hashes(haar_hashes, HAAR_SEEDS, "Haar hashes")
    validation_hash = _sha256_json(validation_decisions)
    _require_hash(fingerprint_definition_sha256, "fingerprint_definition_sha256")
    _require_hash(test_sweep_plan_sha256, "test_sweep_plan_sha256")
    _validate_call_ledger(call_ledger)
    if fit_mu_q_source_split != "FIT":
        raise ArchiveViolation("mu_Q must be computed from FIT only")
    if not isinstance(fit_mu_q, torch.Tensor) or fit_mu_q.device.type != "cpu" or (
        fit_mu_q.dtype != torch.float32 or tuple(fit_mu_q.shape) != (4,)
    ) or not bool(torch.isfinite(fit_mu_q).all()):
        raise ArchiveViolation("FIT mu_Q must be a finite CPU float32 vector of length four")
    mu_hash = tensor_sha256(fit_mu_q)

    canonical_eligible = tuple(sorted(eligible_all_three_frame_ids))
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
            scheduler_sha256=record.scheduler_sha256,
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
        "pretest_freeze": asdict(freeze),
        "label_null_sha256": nulls,
        "validation_decisions": dict(validation_decisions),
        "validation_decisions_sha256": validation_hash,
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
    "FrameArtifact", "HAAR_SEEDS", "LoadedFrameArchive", "MANIFEST_SCHEMA",
    "ManifestReceipt", "geometry_only_grassmann_medoid", "load_frame_archive",
    "tensor_sha256", "write_frame_archive", "write_pretest_manifest",
]
