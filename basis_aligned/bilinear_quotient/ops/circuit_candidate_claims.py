#!/usr/bin/env python3
"""Atomic, append-only ownership for the fast circuit-screen loop.

This is deliberately smaller than the experiment registry.  The registry says
what has already been measured; this file prevents two agents from spending the
same ten-minute window authoring the same *next* screen.  A candidate must be
claimed after a prior-art receipt exists and released when its terminal receipt
lands or when it is abandoned for a stated reason.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "circuits" / "active_screen_claims.jsonl"
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
OUTCOMES = {"screen", "null", "inconclusive", "invalid", "abandoned"}


class ClaimError(ValueError):
    """A claim ledger invariant failed."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ClaimError("claim event is not canonical JSON") from error


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ClaimError(f"{label} is invalid")
    return value


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ClaimError("timestamp must be ISO-8601 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ClaimError("timestamp is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise ClaimError("timestamp must be UTC")
    return value


def validate_event(event: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(event, Mapping):
        raise ClaimError("claim event must be an object")
    kind = event.get("event")
    common = {"schema", "event", "candidate_id", "owner", "timestamp"}
    expected = common | (
        {"prior_art_sha256", "novelty"} if kind == "claim"
        else {"outcome", "receipt", "reason"} if kind == "release"
        else set()
    )
    if kind not in {"claim", "release"} or set(event) != expected:
        raise ClaimError("claim event has unknown or missing fields")
    value = dict(event)
    if value["schema"] != "circuit_candidate_claim_v1":
        raise ClaimError("claim event schema is invalid")
    _validate_identifier(value["candidate_id"], "candidate_id")
    _validate_identifier(value["owner"], "owner")
    _validate_timestamp(value["timestamp"])
    if kind == "claim":
        if not isinstance(value["prior_art_sha256"], str) or not SHA256.fullmatch(value["prior_art_sha256"]):
            raise ClaimError("prior_art_sha256 must be a lowercase SHA-256")
        if not isinstance(value["novelty"], str) or not value["novelty"].strip():
            raise ClaimError("novelty must be nonempty")
    else:
        if value["outcome"] not in OUTCOMES:
            raise ClaimError("release outcome is invalid")
        if not isinstance(value["reason"], str) or not value["reason"].strip():
            raise ClaimError("release reason must be nonempty")
        receipt = value["receipt"]
        if value["outcome"] == "abandoned":
            if receipt is not None:
                raise ClaimError("abandoned release must not name a receipt")
        elif not isinstance(receipt, str) or not receipt.strip():
            raise ClaimError("terminal release must name its receipt")
    _canonical(value)
    return value


@contextmanager
def _locked(path: Path, operation: int) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o664)
    try:
        fcntl.flock(descriptor, operation)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_unlocked(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    if path.is_symlink():
        raise ClaimError("claim ledger may not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ClaimError("claim ledger is not a regular file")
        payload = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(descriptor)
    if payload and not payload.endswith(b"\n"):
        raise ClaimError("claim ledger has an incomplete final record")
    events = []
    for number, raw in enumerate(payload.splitlines(), start=1):
        try:
            event = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClaimError(f"claim ledger line {number} is invalid JSON") from error
        if raw != _canonical(event):
            raise ClaimError(f"claim ledger line {number} is not canonical JSON")
        events.append(validate_event(event))
    _active(events)  # validates event order as part of every read
    return tuple(events)


def _active(events: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    active: dict[str, dict[str, object]] = {}
    for event in events:
        candidate = str(event["candidate_id"])
        if event["event"] == "claim":
            if candidate in active:
                raise ClaimError(f"candidate {candidate} has overlapping claims")
            active[candidate] = dict(event)
        else:
            claim = active.get(candidate)
            if claim is None:
                raise ClaimError(f"candidate {candidate} was released without an active claim")
            if claim["owner"] != event["owner"]:
                raise ClaimError(f"candidate {candidate} was released by a different owner")
            del active[candidate]
    return active


def read_claims(path: Path = DEFAULT_LEDGER) -> tuple[dict[str, object], ...]:
    with _locked(path, fcntl.LOCK_SH):
        return _read_unlocked(path)


def active_claims(path: Path = DEFAULT_LEDGER) -> dict[str, dict[str, object]]:
    return _active(read_claims(path))


def _append(path: Path, event: Mapping[str, object]) -> dict[str, object]:
    value = validate_event(event)
    line = _canonical(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o664)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ClaimError("claim ledger is not a regular file")
        original_size = os.fstat(descriptor).st_size
        try:
            if os.write(descriptor, line) != len(line):
                raise OSError("partial append")
            os.fsync(descriptor)
        except OSError as error:
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
            raise ClaimError("claim ledger append failed and was rolled back") from error
    finally:
        os.close(descriptor)
    return value


def claim(
    candidate_id: str,
    owner: str,
    prior_art_sha256: str,
    novelty: str,
    *,
    path: Path = DEFAULT_LEDGER,
    clock: Callable[[], str] = _now,
) -> dict[str, object]:
    event = validate_event({
        "schema": "circuit_candidate_claim_v1", "event": "claim",
        "candidate_id": candidate_id, "owner": owner, "timestamp": clock(),
        "prior_art_sha256": prior_art_sha256, "novelty": novelty,
    })
    with _locked(path, fcntl.LOCK_EX):
        active = _active(_read_unlocked(path))
        if candidate_id in active:
            holder = active[candidate_id]
            raise ClaimError(
                f"candidate {candidate_id} is already claimed by {holder['owner']} at {holder['timestamp']}"
            )
        return _append(path, event)


def release(
    candidate_id: str,
    owner: str,
    outcome: str,
    reason: str,
    receipt: str | None,
    *,
    path: Path = DEFAULT_LEDGER,
    clock: Callable[[], str] = _now,
) -> dict[str, object]:
    event = validate_event({
        "schema": "circuit_candidate_claim_v1", "event": "release",
        "candidate_id": candidate_id, "owner": owner, "timestamp": clock(),
        "outcome": outcome, "receipt": receipt, "reason": reason,
    })
    with _locked(path, fcntl.LOCK_EX):
        active = _active(_read_unlocked(path))
        current = active.get(candidate_id)
        if current is None:
            raise ClaimError(f"candidate {candidate_id} has no active claim")
        if current["owner"] != owner:
            raise ClaimError(f"candidate {candidate_id} is owned by {current['owner']}, not {owner}")
        return _append(path, event)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    sub = parser.add_subparsers(dest="command", required=True)
    claim_parser = sub.add_parser("claim")
    claim_parser.add_argument("candidate_id")
    claim_parser.add_argument("--owner", required=True)
    claim_parser.add_argument("--prior-art-sha256", required=True)
    claim_parser.add_argument("--novelty", required=True)
    release_parser = sub.add_parser("release")
    release_parser.add_argument("candidate_id")
    release_parser.add_argument("--owner", required=True)
    release_parser.add_argument("--outcome", choices=sorted(OUTCOMES), required=True)
    release_parser.add_argument("--receipt")
    release_parser.add_argument("--reason", required=True)
    sub.add_parser("list")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "claim":
            value = claim(
                args.candidate_id, args.owner, args.prior_art_sha256, args.novelty,
                path=args.ledger,
            )
            print(json.dumps(value, sort_keys=True))
        elif args.command == "release":
            value = release(
                args.candidate_id, args.owner, args.outcome, args.reason, args.receipt,
                path=args.ledger,
            )
            print(json.dumps(value, sort_keys=True))
        else:
            values = active_claims(args.ledger)
            print(json.dumps(values, indent=2, sort_keys=True))
    except ClaimError as error:
        print(f"REFUSED: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
