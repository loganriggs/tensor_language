from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pytest

import circuit_candidate_claims as claims


PRIOR = hashlib.sha256(b"prior art").hexdigest()


def clock(index: int):
    return lambda: f"2026-09-04T16:{index:02d}:00.000000Z"


def test_claim_release_and_reclaim_are_append_only(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    first = claims.claim("task14.reader.mlp12", "codex", PRIOR, "New downstream reader test.", path=path, clock=clock(1))
    assert claims.active_claims(path) == {"task14.reader.mlp12": first}
    claims.release(
        "task14.reader.mlp12", "codex", "screen", "Terminal receipt landed.",
        "circuits/fast_screens/task14_reader.json", path=path, clock=clock(2),
    )
    assert claims.active_claims(path) == {}
    claims.claim("task14.reader.mlp12", "claude", PRIOR, "Explicit replication.", path=path, clock=clock(3))
    assert len(claims.read_claims(path)) == 3


def test_active_duplicate_and_wrong_owner_release_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    claims.claim("task14.reader.mlp12", "codex", PRIOR, "New reader.", path=path, clock=clock(1))
    with pytest.raises(claims.ClaimError, match="already claimed by codex"):
        claims.claim("task14.reader.mlp12", "claude", PRIOR, "Same work.", path=path, clock=clock(2))
    with pytest.raises(claims.ClaimError, match="owned by codex"):
        claims.release(
            "task14.reader.mlp12", "claude", "abandoned", "Cannot proceed.", None,
            path=path, clock=clock(3),
        )


def test_terminal_receipt_and_abandonment_contract(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    claims.claim("candidate.one", "codex", PRIOR, "Novel.", path=path, clock=clock(1))
    with pytest.raises(claims.ClaimError, match="must name its receipt"):
        claims.release("candidate.one", "codex", "null", "No effect.", None, path=path, clock=clock(2))
    claims.release("candidate.one", "codex", "abandoned", "Native task failed.", None, path=path, clock=clock(3))

    claims.claim("candidate.two", "codex", PRIOR, "Novel.", path=path, clock=clock(4))
    with pytest.raises(claims.ClaimError, match="must not name"):
        claims.release(
            "candidate.two", "codex", "abandoned", "Stopped.", "some.json",
            path=path, clock=clock(5),
        )

    claims.release(
        "candidate.two", "codex", "inconclusive", "Registered gap outcome.",
        "circuits/result.json", path=path, clock=clock(6),
    )


def test_concurrent_same_candidate_has_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"

    def attempt(owner: str) -> str:
        try:
            claims.claim(
                "task14.same", owner, PRIOR, f"Attempt by {owner}.", path=path,
                clock=clock(1),
            )
            return "won"
        except claims.ClaimError:
            return "refused"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, [f"agent{i}" for i in range(8)]))
    assert outcomes.count("won") == 1
    assert outcomes.count("refused") == 7
    assert len(claims.active_claims(path)) == 1


def test_malformed_hash_event_order_and_partial_file_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    with pytest.raises(claims.ClaimError, match="lowercase SHA"):
        claims.claim("candidate.one", "codex", "bad", "Novel.", path=path, clock=clock(1))
    path.write_text('{"partial":true}')
    with pytest.raises(claims.ClaimError, match="incomplete"):
        claims.read_claims(path)

    path.write_text("")
    event = {
        "schema": "circuit_candidate_claim_v1", "event": "release",
        "candidate_id": "candidate.one", "owner": "codex",
        "timestamp": clock(2)(), "outcome": "abandoned", "receipt": None,
        "reason": "No longer useful.",
    }
    path.write_bytes(claims._canonical(event) + b"\n")
    with pytest.raises(claims.ClaimError, match="released without"):
        claims.read_claims(path)


def test_partial_append_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "claims.jsonl"
    claims.claim("candidate.one", "codex", PRIOR, "Novel.", path=path, clock=clock(1))
    before = path.read_bytes()
    real_write = claims.os.write

    def partial(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[:-1])

    monkeypatch.setattr(claims.os, "write", partial)
    with pytest.raises(claims.ClaimError, match="rolled back"):
        claims.claim("candidate.two", "codex", PRIOR, "Novel.", path=path, clock=clock(2))
    assert path.read_bytes() == before


def test_legacy_outcomes_are_readable_but_new_writes_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    claim_event = {
        "schema": "circuit_candidate_claim_v1", "event": "claim",
        "candidate_id": "candidate.legacy", "owner": "codex",
        "timestamp": clock(1)(), "prior_art_sha256": PRIOR,
        "novelty": "Historical manually appended event.",
    }
    release_event = {
        "schema": "circuit_candidate_claim_v1", "event": "release",
        "candidate_id": "candidate.legacy", "owner": "codex",
        "timestamp": clock(2)(), "outcome": "pass",
        "receipt": "circuits/result.json", "reason": "Historical label.",
    }
    path.write_text(
        "\n".join(json.dumps(event, sort_keys=True, separators=(",", ":")) for event in (claim_event, release_event))
        + "\n"
    )

    assert claims.active_claims(path) == {}
    with pytest.raises(claims.ClaimError, match="outcome is invalid"):
        claims.release(
            "candidate.legacy", "codex", "pass", "New invalid label.",
            "circuits/result.json", path=path, clock=clock(3),
        )
