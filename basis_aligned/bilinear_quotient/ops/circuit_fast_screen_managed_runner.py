"""Small generic managed harness for one reusable circuit FIT screen.

Science is operationally queue-only.  This module does not enqueue and its
dry-run path performs no model work; the shared hash-bound queue remains the
serialization and execution authority.  Candidate wrappers supply only frozen
identities, paths, hypothesis text, and the existing candidate-bank module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Protocol, Sequence

import circuit_experiment_spec as framework
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_ledger as ledger
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import circuit_prior_art


class CandidateModule(Protocol):
    TASK_ID: str
    TASK_SPEC: object

    def build_rows(self, task_id: str) -> list[dict[str, object]]: ...
    def validate_rows(self, rows: Sequence[Mapping[str, object]]) -> str: ...


@dataclass(frozen=True)
class CandidateRunConfig:
    request_id: str
    experiment_id: str
    prior_art_relative: str
    result_relative: str
    expected_prior_art_sha256: str
    expected_authority_sha256: str
    information_read: str
    proposed_operation: str
    proposed_write: str
    alternative_explanation: str
    circuit_prediction: str
    opposing_null_prediction: str
    semantic_position_role: str
    max_price: screen.battery.ExactPhasePrice
    batch_size: int = 32
    ledger_relative: str = "circuits/fast_screen_ledger.jsonl"


class ManagedScreenError(ValueError):
    """A wrapper, prior-art, result, or execution boundary is inconsistent."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ManagedScreenError("managed screen timestamp must be UTC")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def literal_json(value: object) -> object:
    """Normalize dataclass tuple/list trees to strict literal JSON containers."""
    if isinstance(value, tuple):
        return [literal_json(item) for item in value]
    if isinstance(value, list):
        return [literal_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): literal_json(item) for key, item in value.items()}
    return value


def _hash(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ManagedScreenError(f"{label} must be a lowercase SHA-256")


def _relative(root: Path, name: str, label: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise ManagedScreenError(f"{label} must be a contained relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ManagedScreenError(f"{label} escapes the repository root")
    return resolved


def validate_config(config: CandidateRunConfig) -> None:
    for label, value in asdict(config).items():
        if label in {"max_price", "batch_size"}:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ManagedScreenError(f"{label} must be nonempty text")
    _hash(config.expected_prior_art_sha256, "prior-art digest")
    _hash(config.expected_authority_sha256, "authority digest")
    if type(config.batch_size) is not int or config.batch_size <= 0:
        raise ManagedScreenError("batch_size must be positive")
    if config.max_price.phase != "FIT" or config.max_price.backward_calls != 0 \
            or config.max_price.model_updates != 0:
        raise ManagedScreenError("maximum price must be FIT-only with no backward/update work")


def build_spec(
    config: CandidateRunConfig,
    candidate: CandidateModule | ModuleType,
    rows: Sequence[Mapping[str, object]],
) -> screen.CircuitFastScreenSpec:
    """Bind one current candidate module to the shared declarative screen."""
    validate_config(config)
    task_id = getattr(candidate, "TASK_ID", None)
    task_spec = getattr(candidate, "TASK_SPEC", None)
    validator = getattr(candidate, "validate_rows", None)
    if not isinstance(task_id, str) or not callable(validator) or task_spec is None:
        raise ManagedScreenError("candidate module lacks the current screen-bank interface")
    authority_sha256 = validator(rows)
    if authority_sha256 != config.expected_authority_sha256:
        raise ManagedScreenError("candidate authority differs from its reviewed digest")
    return screen.CircuitFastScreenSpec(
        experiment_id=config.experiment_id,
        hypothesis=screen.CandidateHypothesis(
            behavior=task_id,
            answer_score=screen.ANSWER_SCORE,
            information_read=config.information_read,
            proposed_operation=config.proposed_operation,
            proposed_write=config.proposed_write,
            candidate_sites=screen.CEILING_SITE_IDS,
            alternative_explanation=config.alternative_explanation,
            circuit_prediction=config.circuit_prediction,
            opposing_null_prediction=config.opposing_null_prediction,
        ),
        task=task_spec,
        authority_sha256=authority_sha256,
        expected_fit_rows=len(rows),
        batch_size=config.batch_size,
        semantic_position=screen.SemanticPositionSpec(
            role=config.semantic_position_role,
            recipient_field="base_semantic_position",
            donor_field="donor_semantic_position",
        ),
        fields=screen.AuthorityFieldSpec(),
        bars=kernel.FIXED_BARS,
        declared_max_price=config.max_price,
    )


def atomic_create_json(path: Path, value: object) -> bytes:
    """Create one canonical JSON result without an overwrite code path."""
    payload = framework.canonical_json_bytes(literal_json(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o664)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("result write made no progress")
            written += count
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def selected_controls_pass(run: producer.FastScreenRun) -> bool:
    if run.selected_site is None:
        return False
    matched = [result for result in run.site_results if result.site == run.selected_site]
    return bool(
        len(matched) == 1
        and matched[0].p_invariance_effect is not None
        and matched[0].p_invariance_effect <= kernel.MAX_P_INVARIANCE_EFFECT
        and matched[0].c_absolute_recovery is not None
        and matched[0].c_absolute_recovery <= kernel.MAX_C_ABSOLUTE_RECOVERY
    )


def load_prior_art(
    config: CandidateRunConfig,
    candidate: CandidateModule | ModuleType,
    root: Path,
) -> tuple[dict[str, object], str]:
    path = _relative(root, config.prior_art_relative, "prior_art_relative")
    try:
        prior = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ManagedScreenError("prior-art receipt is unreadable") from error
    if not isinstance(prior, dict) or prior.get("candidate_id") != candidate.TASK_ID:
        raise ManagedScreenError("prior-art receipt belongs to another candidate")
    digest = circuit_prior_art.validate_source_files(prior, root)
    if digest != config.expected_prior_art_sha256:
        raise ManagedScreenError("prior-art receipt differs from its reviewed digest")
    return prior, digest


def _ledger_relation(relation: object) -> str:
    return "genuinely_new" if relation == "new_question" else str(relation)


def run_managed(
    config: CandidateRunConfig,
    candidate: CandidateModule | ModuleType,
    *,
    root: Path,
    environment: Mapping[str, str] | None = None,
    science_runner: Callable[..., producer.FastScreenRun] = producer.run_science,
    clock: Callable[[], datetime] = utc_now,
) -> dict[str, object]:
    """Dry-run or execute one candidate; this function never enqueues itself."""
    validate_config(config)
    env = os.environ if environment is None else environment
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if env.get(name) not in (None, "1"):
            raise ManagedScreenError(f"{name} must be absent or exactly '1'")
    builder = getattr(candidate, "build_rows", None)
    if not callable(builder):
        raise ManagedScreenError("candidate module lacks build_rows")
    rows = builder(candidate.TASK_ID)
    spec = build_spec(config, candidate, rows)
    prior, prior_sha256 = load_prior_art(config, candidate, root)
    dryrun = producer.compile_dryrun(spec, rows)
    if env.get("BQLIB_DRYRUN") == "1" or env.get("BQLIB_NO_MODEL") == "1":
        receipt = {
            "dryrun": dryrun,
            "prior_art_sha256": prior_sha256,
            "authority_sha256": config.expected_authority_sha256,
            "execution_policy": "managed_queue_only",
        }
        print(json.dumps(receipt, sort_keys=True))
        return receipt

    result_relative = Path(config.result_relative)
    result_path = _relative(root, config.result_relative, "result_relative")
    ledger_path = _relative(root, config.ledger_relative, "ledger_relative")
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite prior screen result: {result_path}")
    started = clock()
    run = science_runner(spec, rows)
    finished = clock()
    serial_seconds = (finished - started).total_seconds()
    capability_passed = bool(run.capability_cells) and all(
        cell.passed for cell in run.capability_cells
    )
    cross_construction_transfer = run.terminal == "screen"
    controls_selective = selected_controls_pass(run)
    spec_sha256 = framework.canonical_sha256(screen.spec_json(spec))
    result = {
        "schema": "circuit_fast_screen_result_v1",
        "request_id": config.request_id,
        "candidate_id": candidate.TASK_ID,
        "experiment_id": config.experiment_id,
        "screen_tier_only": True,
        "execution_policy": "managed_queue_only",
        "prior_art_sha256": prior_sha256,
        "spec_sha256": spec_sha256,
        "authority_sha256": config.expected_authority_sha256,
        "started_utc": utc_text(started),
        "finished_utc": utc_text(finished),
        "serial_seconds": serial_seconds,
        "dryrun": dryrun,
        "terminal": run.terminal,
        "reason": run.reason,
        "selected_site_id": None if run.selected_site is None else run.selected_site.site_id,
        "head_stage": run.head_stage,
        "predictions": {
            "pred_a_native_capability": capability_passed,
            "pred_b_cross_construction_transfer": cross_construction_transfer,
            "pred_c_controls_selective": controls_selective,
        },
        "fixed_bars": asdict(kernel.FIXED_BARS),
        "run": literal_json(asdict(run)),
    }
    payload = atomic_create_json(result_path, result)
    result_sha256 = hashlib.sha256(payload).hexdigest()
    max_price = dryrun["max_price"]
    active_evaluations = run.timing.example_evaluations
    entry = {
        "request_id": config.request_id,
        "candidate_id": candidate.TASK_ID,
        "started_utc": utc_text(started),
        "finished_utc": utc_text(finished),
        "serial_seconds": serial_seconds,
        "prior_art_sha256": prior_sha256,
        "spec_sha256": spec_sha256,
        "authority_sha256": config.expected_authority_sha256,
        "result_path": result_relative.as_posix(),
        "result_sha256": result_sha256,
        "terminal": run.terminal,
        "reasons": [] if run.terminal == "screen" else [run.reason],
        "selected_site_id": None if run.selected_site is None else run.selected_site.site_id,
        "active_forward_calls": run.timing.forward_calls,
        "active_example_evaluations": active_evaluations,
        "active_evidence_bytes": 8 * active_evaluations,
        "max_forward_calls": max_price["forward_calls"],
        "max_example_evaluations": max_price["example_evaluations"],
        "max_evidence_bytes": max_price["evidence_bytes"],
        "relation": _ledger_relation(prior["relation"]),
        "novelty": prior["novelty_delta"],
    }
    ledger.append_entry(ledger_path, entry, result_root=root)
    summary = {
        "terminal": run.terminal,
        "reason": run.reason,
        "selected_site_id": entry["selected_site_id"],
        "forward_calls": run.timing.forward_calls,
        "example_evaluations": active_evaluations,
        "serial_seconds": serial_seconds,
        "native_capability": capability_passed,
        "cross_construction_transfer": cross_construction_transfer,
        "controls_selective": controls_selective,
        "result_sha256": result_sha256,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary
