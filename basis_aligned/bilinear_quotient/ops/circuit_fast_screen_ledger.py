"""Append-only CPU bookkeeping for reusable causal-screen receipts.

This ledger records only FIT screen/null/invalid outcomes.  It is deliberately
independent of the version-2 identification/adoption registry and contains no
model, GPU, queue, or outcome-loading logic beyond hashing the declared result
file supplied by the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Iterator, Mapping, Sequence


TIMING_TOLERANCE_SECONDS = 0.05
TERMINALS = {"screen", "null", "inconclusive", "invalid"}
RELATIONS = {"genuinely_new", "replication", "extension", "contradiction_test"}
ENTRY_FIELDS = {
    "request_id", "candidate_id", "started_utc", "finished_utc", "serial_seconds",
    "prior_art_sha256", "spec_sha256", "authority_sha256",
    "result_path", "result_sha256", "terminal", "reasons", "selected_site_id",
    "active_forward_calls", "active_example_evaluations", "active_evidence_bytes",
    "max_forward_calls", "max_example_evaluations", "max_evidence_bytes",
    "relation", "novelty",
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class ScreenLedgerError(ValueError):
    """A screen receipt or append-only ledger invariant failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ScreenLedgerError("ledger entry is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise ScreenLedgerError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ScreenLedgerError(f"{label} is invalid")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ScreenLedgerError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ScreenLedgerError(f"{label} is not a valid timestamp") from error
    if parsed.tzinfo != timezone.utc or parsed.isoformat().endswith("+00:00") is False:
        raise ScreenLedgerError(f"{label} must be UTC")
    return parsed


def _validate_price_fields(entry: Mapping[str, object]) -> None:
    names = (
        "active_forward_calls", "active_example_evaluations", "active_evidence_bytes",
        "max_forward_calls", "max_example_evaluations", "max_evidence_bytes",
    )
    if any(type(entry[name]) is not int or entry[name] < 0 for name in names):
        raise ScreenLedgerError("price fields must be exact nonnegative integers")
    pairs = (
        ("active_forward_calls", "max_forward_calls"),
        ("active_example_evaluations", "max_example_evaluations"),
        ("active_evidence_bytes", "max_evidence_bytes"),
    )
    if any(entry[active] > entry[maximum] for active, maximum in pairs):
        raise ScreenLedgerError("active price exceeds the declared maximum")
    if any(entry[name] <= 0 for name in (
        "max_forward_calls", "max_example_evaluations", "max_evidence_bytes"
    )):
        raise ScreenLedgerError("maximum price must be positive")
    if entry["terminal"] in {"screen", "null"} and any(entry[name] <= 0 for name in (
        "active_forward_calls", "active_example_evaluations", "active_evidence_bytes"
    )):
        raise ScreenLedgerError("scientific screen/null receipt requires positive active price")


def _contained_result_path(result_root: Path, relative_name: object) -> Path:
    if not isinstance(relative_name, str) or not relative_name or "\\" in relative_name:
        raise ScreenLedgerError("result_path must be a safe relative POSIX path")
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ScreenLedgerError("result_path escapes the contained result root")
    root = result_root.resolve(strict=True)
    if not root.is_dir():
        raise ScreenLedgerError("result root is not a directory")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ScreenLedgerError("declared result file is missing") from error
    if not resolved.is_relative_to(root) or resolved != candidate:
        raise ScreenLedgerError("result_path uses a symlink or escapes its root")
    return candidate


def _safe_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ScreenLedgerError(f"cannot safely open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ScreenLedgerError(f"{label} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )
    if identity(before) != identity(after):
        raise ScreenLedgerError(f"{label} changed while it was read")
    return b"".join(chunks)


def validate_entry(entry: Mapping[str, object], *, result_root: Path) -> dict[str, object]:
    """Validate exact schema, timing, terminal semantics, and current result bytes."""
    if not isinstance(entry, Mapping) or set(entry) != ENTRY_FIELDS:
        raise ScreenLedgerError("ledger entry has unknown or missing fields")
    value = dict(entry)
    _canonical_bytes(value)
    _validate_identifier(value["request_id"], "request_id")
    _validate_identifier(value["candidate_id"], "candidate_id")
    for name in ("prior_art_sha256", "spec_sha256", "authority_sha256", "result_sha256"):
        _validate_hash(value[name], name)
    started = _parse_utc(value["started_utc"], "started_utc")
    finished = _parse_utc(value["finished_utc"], "finished_utc")
    elapsed = (finished - started).total_seconds()
    serial = value["serial_seconds"]
    if type(serial) not in (int, float) or isinstance(serial, bool) \
            or not math.isfinite(float(serial)) or float(serial) < 0.0:
        raise ScreenLedgerError("serial_seconds must be a finite nonnegative number")
    if elapsed < 0.0:
        raise ScreenLedgerError("finished_utc precedes started_utc")
    if abs(float(serial) - elapsed) > TIMING_TOLERANCE_SECONDS:
        raise ScreenLedgerError("serial_seconds differs from UTC chronology beyond tolerance")
    terminal = value["terminal"]
    reasons = value["reasons"]
    selected = value["selected_site_id"]
    if terminal not in TERMINALS or not isinstance(reasons, list) \
            or any(not isinstance(reason, str) or not reason.strip() for reason in reasons) \
            or len(reasons) != len(set(reasons)):
        raise ScreenLedgerError("terminal or reasons are malformed")
    if terminal == "screen":
        if reasons or not isinstance(selected, str) or not selected.strip():
            raise ScreenLedgerError("screen terminal requires one site and no failure reasons")
    elif selected is not None or not reasons:
        raise ScreenLedgerError(
            "null/inconclusive/invalid terminal requires reasons and no selected site"
        )
    if value["relation"] not in RELATIONS:
        raise ScreenLedgerError("relation is outside the prior-art vocabulary")
    if not isinstance(value["novelty"], str) or not value["novelty"].strip():
        raise ScreenLedgerError("novelty must be nonempty text")
    _validate_price_fields(value)
    result = _contained_result_path(result_root, value["result_path"])
    if _sha256_bytes(_safe_regular_bytes(result, "result file")) != value["result_sha256"]:
        raise ScreenLedgerError("current result bytes differ from result_sha256")
    return value


def _execution_key(entry: Mapping[str, object]) -> tuple[object, ...]:
    """Identity of one exact scientific execution, excluding output and clock fields."""
    return (
        entry["candidate_id"], entry["prior_art_sha256"],
        entry["spec_sha256"], entry["authority_sha256"],
        entry["max_forward_calls"], entry["max_example_evaluations"],
        entry["max_evidence_bytes"],
    )


def _validate_collection(entries: Sequence[Mapping[str, object]]) -> None:
    request_ids = [entry["request_id"] for entry in entries]
    execution_keys = [_execution_key(entry) for entry in entries]
    if len(request_ids) != len(set(request_ids)):
        raise ScreenLedgerError("ledger contains a duplicate request_id")
    if len(execution_keys) != len(set(execution_keys)):
        raise ScreenLedgerError("ledger contains a duplicate identical execution")


@contextmanager
def _locked(ledger_path: Path, operation: int) -> Iterator[None]:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o664,
    )
    try:
        fcntl.flock(descriptor, operation)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _decode_lines(payload: bytes) -> list[dict[str, object]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise ScreenLedgerError("ledger ends with an incomplete JSONL record")
    entries = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        if not raw:
            raise ScreenLedgerError(f"ledger line {line_number} is empty")
        try:
            entry = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonstandard JSON constant {value}")
                ),
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise ScreenLedgerError(f"ledger line {line_number} is invalid JSON") from error
        if not isinstance(entry, dict):
            raise ScreenLedgerError(f"ledger line {line_number} is not an object")
        if raw != _canonical_bytes(entry):
            raise ScreenLedgerError(f"ledger line {line_number} is not canonical JSON")
        entries.append(entry)
    return entries


def _read_unlocked(ledger_path: Path, result_root: Path) -> tuple[dict[str, object], ...]:
    if not ledger_path.exists():
        return ()
    if ledger_path.is_symlink():
        raise ScreenLedgerError("ledger path may not be a symlink")
    entries = _decode_lines(_safe_regular_bytes(ledger_path, "screen ledger"))
    validated = tuple(validate_entry(entry, result_root=result_root) for entry in entries)
    _validate_collection(validated)
    return validated


def read_ledger(ledger_path: Path, *, result_root: Path) -> tuple[dict[str, object], ...]:
    """Read and fully revalidate the ledger and every currently bound result."""
    with _locked(ledger_path, fcntl.LOCK_SH):
        return _read_unlocked(ledger_path, result_root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_entry(
    ledger_path: Path,
    entry: Mapping[str, object],
    *,
    result_root: Path,
) -> dict[str, object]:
    """Append one canonical JSON line under an exclusive durable file lock."""
    validated = validate_entry(entry, result_root=result_root)
    with _locked(ledger_path, fcntl.LOCK_EX):
        # Recheck after lock acquisition so a result mutation while waiting can
        # never be recorded under its former digest.
        validated = validate_entry(validated, result_root=result_root)
        line = _canonical_bytes(validated) + b"\n"
        existing = _read_unlocked(ledger_path, result_root)
        if any(item["request_id"] == validated["request_id"] for item in existing):
            raise ScreenLedgerError("duplicate request_id refused")
        key = _execution_key(validated)
        if any(_execution_key(item) == key for item in existing):
            raise ScreenLedgerError("duplicate identical execution refused")
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) \
            | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(ledger_path, flags, 0o664)
        except OSError as error:
            raise ScreenLedgerError("cannot safely open screen ledger for append") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ScreenLedgerError("screen ledger is not a regular file")
            original_size = before.st_size
            try:
                written = os.write(descriptor, line)
                if written != len(line):
                    raise OSError("partial JSONL append")
                os.fsync(descriptor)
            except OSError as error:
                os.ftruncate(descriptor, original_size)
                os.fsync(descriptor)
                raise ScreenLedgerError("atomic JSONL append failed and was rolled back") from error
            except BaseException:
                os.ftruncate(descriptor, original_size)
                os.fsync(descriptor)
                raise
        finally:
            os.close(descriptor)
        _fsync_directory(ledger_path.parent)
    return validated


def query_ledger(
    ledger_path: Path,
    *,
    result_root: Path,
    candidate_id: str | None = None,
    terminal: str | None = None,
    relation: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Return append-order entries matching exact optional screen fields."""
    if candidate_id is not None:
        _validate_identifier(candidate_id, "candidate_id query")
    if terminal is not None and terminal not in TERMINALS:
        raise ScreenLedgerError("terminal query is invalid")
    if relation is not None and relation not in RELATIONS:
        raise ScreenLedgerError("relation query is invalid")
    entries = read_ledger(ledger_path, result_root=result_root)
    return tuple(entry for entry in entries if (
        (candidate_id is None or entry["candidate_id"] == candidate_id)
        and (terminal is None or entry["terminal"] == terminal)
        and (relation is None or entry["relation"] == relation)
    ))


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(entries: Sequence[Mapping[str, object]]) -> str:
    """Render a deterministic small index; this confers no identification status."""
    values = [dict(entry) for entry in entries]
    _validate_collection(values)
    ordered = sorted(values, key=lambda item: (item["started_utc"], item["request_id"]))
    lines = [
        "# Circuit fast-screen ledger", "",
        "Screen-tier bookkeeping only. A `screen` is neither circuit identification nor adoption.",
        "",
        "| started UTC | request | candidate | terminal | seconds | selected site | relation | result | reasons | novelty |",
        "|---|---|---|---|---:|---|---|---|---|---|",
    ]
    for entry in ordered:
        reasons = "; ".join(entry["reasons"]) if entry["reasons"] else "—"
        selected = entry["selected_site_id"] if entry["selected_site_id"] is not None else "—"
        result = f"`{entry['result_path']}` `{str(entry['result_sha256'])[:12]}`"
        cells = (
            entry["started_utc"], entry["request_id"], entry["candidate_id"],
            entry["terminal"], entry["serial_seconds"], selected, entry["relation"],
            result, reasons, entry["novelty"],
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
    if not ordered:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    lines.extend(["", f"Entries: {len(ordered)}.", ""])
    return "\n".join(lines)
