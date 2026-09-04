"""Tests for ops/failed_log_watch.py -- capturing a failed run's log before the retry overwrites it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import failed_log_watch as W


def _wire(tmp_path, monkeypatch, completed_text, logs=()):
    runlogs = tmp_path / "runlogs"
    runlogs.mkdir()
    (runlogs / "_completed.txt").write_text(completed_text)
    for name, body in logs:
        (runlogs / f"{name}.log").write_text(body)
    monkeypatch.setattr(W, "RUNLOGS", str(runlogs))
    monkeypatch.setattr(W, "COMPLETED", str(runlogs / "_completed.txt"))
    monkeypatch.setattr(W, "FAILED_DIR", str(runlogs / "failed"))
    return runlogs


def test_first_run_starts_at_the_end_and_does_not_re_mine_history(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, "13:21 alpha exit=1\n", logs=[("alpha", "boom")])
    offset, captured = W.sweep(None)
    assert captured == [], "history must not be re-mined on first initialisation"
    assert offset > 0


def test_a_nonzero_exit_captures_that_runs_log(tmp_path, monkeypatch):
    runlogs = _wire(tmp_path, monkeypatch, "", logs=[("beta", "traceback here")])
    offset, _ = W.sweep(None)
    (runlogs / "_completed.txt").write_text("13:25 beta exit=1\n")
    offset, captured = W.sweep(offset)
    assert len(captured) == 1 and captured[0].startswith("beta."), captured
    assert captured[0].endswith(".exit1.log")
    saved = (runlogs / "failed" / captured[0]).read_text()
    assert saved == "traceback here", "the captured log must be the failing run's content"


def test_a_successful_exit_captures_nothing(tmp_path, monkeypatch):
    runlogs = _wire(tmp_path, monkeypatch, "", logs=[("gamma", "fine")])
    offset, _ = W.sweep(None)
    (runlogs / "_completed.txt").write_text("13:25 gamma exit=0\n")
    _, captured = W.sweep(offset)
    assert captured == []


def test_a_missing_log_is_skipped_without_raising(tmp_path, monkeypatch):
    runlogs = _wire(tmp_path, monkeypatch, "")
    offset, _ = W.sweep(None)
    (runlogs / "_completed.txt").write_text("13:25 nolog exit=1\n")
    _, captured = W.sweep(offset)
    assert captured == []


def test_truncation_resets_the_offset_instead_of_seeking_past_the_end(tmp_path, monkeypatch):
    runlogs = _wire(tmp_path, monkeypatch, "x" * 500, logs=[("delta", "err")])
    (runlogs / "_completed.txt").write_text("13:25 delta exit=1\n")
    _, captured = W.sweep(10_000)          # stale offset beyond the new size
    assert len(captured) == 1, captured
