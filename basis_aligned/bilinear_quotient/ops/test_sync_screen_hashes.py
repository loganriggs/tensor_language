"""Tests for ops/sync_screen_hashes.py."""
import os
import shutil
import subprocess
import sys

OPS = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(OPS, "run_circuit_fast_screen_numeric_sequence_cross_construction.py")
PY = sys.executable
TOOL = os.path.join(OPS, "sync_screen_hashes.py")
ZERO = "0" * 64


def _run(path, *args):
    return subprocess.run([PY, TOOL, path, *args], capture_output=True, text=True)


def test_live_runners_report_current():
    r = _run(RUNNER)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "both digests current" in r.stdout


def test_a_stale_authority_is_detected_and_exits_nonzero(tmp_path):
    probe = tmp_path / "probe.py"
    text = open(RUNNER).read()
    good = [l for l in text.splitlines() if "EXPECTED_AUTHORITY_SHA256" in l]
    assert good, "runner shape changed"
    import re
    text = re.sub(r'(EXPECTED_AUTHORITY_SHA256 = \(\s*\n\s*")([0-9a-f]{64})', r"\g<1>" + ZERO, text)
    probe.write_text(text)
    r = _run(str(probe))
    assert r.returncode == 1, r.stdout
    assert "STALE" in r.stdout and "authority" in r.stdout


def test_write_repairs_the_stale_digest(tmp_path):
    probe = tmp_path / "probe.py"
    import re
    text = re.sub(r'(EXPECTED_AUTHORITY_SHA256 = \(\s*\n\s*")([0-9a-f]{64})',
                  r"\g<1>" + ZERO, open(RUNNER).read())
    probe.write_text(text)
    assert _run(str(probe), "--write").returncode == 0
    r = _run(str(probe))
    assert r.returncode == 0 and "both digests current" in r.stdout
    assert ZERO not in probe.read_text()


def test_a_runner_without_the_constants_is_refused(tmp_path):
    probe = tmp_path / "nope.py"
    probe.write_text("print('hello')\n")
    r = _run(str(probe))
    assert r.returncode != 0
