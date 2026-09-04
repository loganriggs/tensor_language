from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

import pytest

import circuit_fast_screen_ledger as ledger


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def timestamp(seconds: int) -> str:
    value = datetime(2026, 9, 4, 14, 0, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def result_file(root: Path, name: str, payload: bytes) -> tuple[str, str]:
    relative = f"results/{name}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return relative, sha256(payload)


def entry(root: Path, *, index: int = 0, terminal: str = "screen") -> dict[str, object]:
    relative, digest = result_file(root, f"result-{index}", f"result {index}\n".encode())
    return {
        "request_id": f"request-{index:03d}",
        "candidate_id": f"candidate.{index:03d}",
        "started_utc": timestamp(10 * index),
        "finished_utc": timestamp(10 * index + 5),
        "serial_seconds": 5.0,
        "prior_art_sha256": sha256(f"prior {index}".encode()),
        "spec_sha256": sha256(f"spec {index}".encode()),
        "authority_sha256": sha256(f"authority {index}".encode()),
        "result_path": relative,
        "result_sha256": digest,
        "terminal": terminal,
        "reasons": [] if terminal == "screen" else [
            "no_selective_causal_site" if terminal == "null" else "evidence_invalid"
        ],
        "selected_site_id": "attn:03" if terminal == "screen" else None,
        "active_forward_calls": 228,
        "active_example_evaluations": 7_296,
        "active_evidence_bytes": 58_368,
        "max_forward_calls": 264,
        "max_example_evaluations": 8_448,
        "max_evidence_bytes": 67_584,
        "relation": "genuinely_new",
        "novelty": "Tests a causal state not covered by the matched prior records.",
    }


def test_append_read_query_and_exact_canonical_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "screens.jsonl"
    first = entry(tmp_path, index=0, terminal="screen")
    second = entry(tmp_path, index=1, terminal="null")
    ledger.append_entry(path, first, result_root=tmp_path)
    ledger.append_entry(path, second, result_root=tmp_path)
    assert ledger.read_ledger(path, result_root=tmp_path) == (first, second)
    assert ledger.query_ledger(
        path, result_root=tmp_path, terminal="null"
    ) == (second,)
    assert ledger.query_ledger(
        path, result_root=tmp_path, candidate_id="candidate.000"
    ) == (first,)
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and len(raw.splitlines()) == 2
    assert all(b'": ' not in line and b'", ' not in line for line in raw.splitlines())


def test_duplicate_request_and_identical_execution_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "screens.jsonl"
    original = entry(tmp_path)
    ledger.append_entry(path, original, result_root=tmp_path)
    with pytest.raises(ledger.ScreenLedgerError, match="duplicate request_id"):
        ledger.append_entry(path, original, result_root=tmp_path)

    duplicate_execution = entry(tmp_path, index=1)
    duplicate_execution.update(
        candidate_id=original["candidate_id"],
        prior_art_sha256=original["prior_art_sha256"],
        spec_sha256=original["spec_sha256"],
        authority_sha256=original["authority_sha256"],
    )
    with pytest.raises(ledger.ScreenLedgerError, match="duplicate identical execution"):
        ledger.append_entry(path, duplicate_execution, result_root=tmp_path)
    assert ledger.read_ledger(path, result_root=tmp_path) == (original,)


def test_result_hash_drift_and_path_escape_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "screens.jsonl"
    value = entry(tmp_path)
    wrong = deepcopy(value)
    wrong["result_sha256"] = "0" * 64
    with pytest.raises(ledger.ScreenLedgerError, match="current result bytes"):
        ledger.append_entry(path, wrong, result_root=tmp_path)
    ledger.append_entry(path, value, result_root=tmp_path)
    (tmp_path / str(value["result_path"])).write_bytes(b"mutated\n")
    with pytest.raises(ledger.ScreenLedgerError, match="current result bytes"):
        ledger.read_ledger(path, result_root=tmp_path)

    escaped = entry(tmp_path, index=2)
    escaped["result_path"] = "../outside.json"
    with pytest.raises(ledger.ScreenLedgerError, match="escapes"):
        ledger.validate_entry(escaped, result_root=tmp_path)


def test_result_symlink_is_rejected(tmp_path: Path) -> None:
    value = entry(tmp_path)
    original = tmp_path / str(value["result_path"])
    target = original.with_name("target.json")
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(ledger.ScreenLedgerError, match="symlink"):
        ledger.validate_entry(value, result_root=tmp_path)


@pytest.mark.parametrize("mutation,match", [
    ({"terminal": "screen", "selected_site_id": None}, "screen terminal"),
    ({"terminal": "screen", "reasons": ["failure"]}, "screen terminal"),
    ({"terminal": "null", "reasons": [], "selected_site_id": None}, "null/invalid"),
    ({"terminal": "invalid", "reasons": ["bad"], "selected_site_id": "attn:03"}, "null/invalid"),
    ({"terminal": "ok"}, "terminal or reasons"),
    ({"relation": "duplicate"}, "relation"),
])
def test_terminal_relation_semantics_reject_ambiguous_entries(
    tmp_path: Path, mutation: dict[str, object], match: str
) -> None:
    value = entry(tmp_path)
    value.update(mutation)
    with pytest.raises(ledger.ScreenLedgerError, match=match):
        ledger.validate_entry(value, result_root=tmp_path)


def test_exact_schema_hash_and_price_fields_are_enforced(tmp_path: Path) -> None:
    value = entry(tmp_path)
    extra = dict(value, status="identified")
    with pytest.raises(ledger.ScreenLedgerError, match="unknown or missing"):
        ledger.validate_entry(extra, result_root=tmp_path)
    bad_hash = dict(value, spec_sha256="ABC")
    with pytest.raises(ledger.ScreenLedgerError, match="lowercase SHA"):
        ledger.validate_entry(bad_hash, result_root=tmp_path)
    over_price = dict(value, active_forward_calls=265)
    with pytest.raises(ledger.ScreenLedgerError, match="exceeds"):
        ledger.validate_entry(over_price, result_root=tmp_path)
    boolean_price = dict(value, active_forward_calls=True)
    with pytest.raises(ledger.ScreenLedgerError, match="exact nonnegative"):
        ledger.validate_entry(boolean_price, result_root=tmp_path)


def test_chronology_and_serial_tolerance_are_exact(tmp_path: Path) -> None:
    value = entry(tmp_path)
    within = dict(value, serial_seconds=5.05)
    ledger.validate_entry(within, result_root=tmp_path)
    outside = dict(value, serial_seconds=5.051)
    with pytest.raises(ledger.ScreenLedgerError, match="beyond tolerance"):
        ledger.validate_entry(outside, result_root=tmp_path)
    reversed_time = dict(value, finished_utc=timestamp(-1), serial_seconds=1.0)
    with pytest.raises(ledger.ScreenLedgerError, match="precedes"):
        ledger.validate_entry(reversed_time, result_root=tmp_path)
    non_utc = dict(value, started_utc="2026-09-04T14:00:00+01:00")
    with pytest.raises(ledger.ScreenLedgerError, match="ending in Z"):
        ledger.validate_entry(non_utc, result_root=tmp_path)


def test_threaded_appends_are_serialized_by_file_lock(tmp_path: Path) -> None:
    path = tmp_path / "screens.jsonl"
    values = [entry(tmp_path, index=index) for index in range(12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(ledger.append_entry, path, value, result_root=tmp_path)
            for value in reversed(values)
        ]
        for future in futures:
            future.result()
    observed = ledger.read_ledger(path, result_root=tmp_path)
    assert len(observed) == len(values)
    assert {item["request_id"] for item in observed} == {
        item["request_id"] for item in values
    }


def test_partial_append_is_rolled_back_to_the_locked_original_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "screens.jsonl"
    first = entry(tmp_path, index=0)
    second = entry(tmp_path, index=1)
    ledger.append_entry(path, first, result_root=tmp_path)
    original = path.read_bytes()
    real_write = ledger.os.write

    def partial_write(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[:-1])

    monkeypatch.setattr(ledger.os, "write", partial_write)
    with pytest.raises(ledger.ScreenLedgerError, match="rolled back"):
        ledger.append_entry(path, second, result_root=tmp_path)
    assert path.read_bytes() == original


def test_render_is_deterministic_sorted_and_explicitly_screen_tier(tmp_path: Path) -> None:
    first = entry(tmp_path, index=0, terminal="screen")
    second = entry(tmp_path, index=1, terminal="invalid")
    forward = ledger.render_markdown((first, second))
    reverse = ledger.render_markdown((second, first))
    assert forward == reverse
    assert "neither circuit identification nor adoption" in forward
    assert forward.index("request-000") < forward.index("request-001")
    assert "`results/result-0.json`" in forward
    assert "Entries: 2." in forward


def test_noncanonical_or_partial_jsonl_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "screens.jsonl"
    value = entry(tmp_path)
    path.write_text("{}\n")
    with pytest.raises(ledger.ScreenLedgerError, match="unknown or missing"):
        ledger.read_ledger(path, result_root=tmp_path)
    path.write_text('{"request_id":"partial"}')
    with pytest.raises(ledger.ScreenLedgerError, match="incomplete"):
        ledger.read_ledger(path, result_root=tmp_path)
    path.write_text("{ }\n")
    with pytest.raises(ledger.ScreenLedgerError, match="not canonical"):
        ledger.read_ledger(path, result_root=tmp_path)
