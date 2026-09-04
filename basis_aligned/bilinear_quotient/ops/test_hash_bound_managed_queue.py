"""CPU tests for lane-1 hash-bound enqueue and runner execution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess


OPS = Path(__file__).resolve().parent
RUNNER = OPS / "bqrunner.sh"
ENQUEUE = OPS / "enqueue.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def captured_runner_program() -> str:
    source = RUNNER.read_text(encoding="utf-8")
    match = re.search(
        r"# BEGIN HASH_BOUND_PYTHON\n(.*?)# END HASH_BOUND_PYTHON",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def captured_enqueue_program() -> str:
    source = ENQUEUE.read_text(encoding="utf-8")
    match = re.search(
        r"# BEGIN ENQUEUE_HASH_BOUND_PYTHON\n(.*?)# END ENQUEUE_HASH_BOUND_PYTHON",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def run_captured(expected: str, target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "-", expected, str(target)],
        input=captured_runner_program(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_scripts_parse_and_runner_program_is_literal() -> None:
    for script in (RUNNER, ENQUEUE):
        subprocess.run(["bash", "-n", str(script)], check=True)
    program = captured_runner_program()
    assert "O_NOFOLLOW" in program
    assert "compile(payload" in program
    assert "Path.read_bytes" not in program
    assert "subprocess" not in program
    enqueue_program = captured_enqueue_program()
    assert "O_NOFOLLOW" in enqueue_program
    assert "compile(payload" in enqueue_program
    assert "logical_name" in enqueue_program


def test_runner_executes_exact_matching_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "job.py"
    target.write_text("print('verified-original')\n", encoding="utf-8")
    completed = run_captured(sha256(target), target)
    assert completed.returncode == 0
    assert completed.stdout.strip() == "verified-original"


def test_runner_rejects_changed_bytes_and_symlink(tmp_path: Path) -> None:
    target = tmp_path / "job.py"
    target.write_text("print('reviewed')\n", encoding="utf-8")
    reviewed = sha256(target)
    target.write_text("print('changed')\n", encoding="utf-8")
    changed = run_captured(reviewed, target)
    assert changed.returncode != 0
    assert "SHA-256 changed" in changed.stderr
    assert "changed" not in changed.stdout

    link = tmp_path / "job-link.py"
    link.symlink_to(target)
    linked = run_captured(sha256(target), link)
    assert linked.returncode != 0
    assert "cannot safely open queued script" in linked.stderr


def test_runner_does_not_reopen_script_after_capture(tmp_path: Path) -> None:
    target = tmp_path / "job.py"
    target.write_text(
        "from pathlib import Path\n"
        "Path(__file__).write_text(\"raise RuntimeError('replacement ran')\\n\")\n"
        "print('captured-bytes-ran')\n",
        encoding="utf-8",
    )
    completed = run_captured(sha256(target), target)
    assert completed.returncode == 0
    assert completed.stdout.strip() == "captured-bytes-ran"
    assert "replacement ran" in target.read_text(encoding="utf-8")


def make_enqueue_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    ops = project / "ops"
    ops.mkdir(parents=True)
    copied = ops / "enqueue.sh"
    shutil.copy2(ENQUEUE, copied)
    copied.chmod(0o755)
    (ops / "test_fast.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (ops / "gate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (ops / "preflight.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    gpu_free = ops / "gpu_free.sh"
    gpu_free.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    gpu_free.chmod(0o755)
    queue = project / "queue.txt"
    queue.write_text("", encoding="utf-8")
    job = ops / "job.py"
    job.write_text("print('dryrun-ok')\n", encoding="utf-8")
    return copied, queue, job


def test_enqueue_writes_hash_bound_lane1_record_and_enforces_reviewed_hash(
    tmp_path: Path,
) -> None:
    enqueue, queue, job = make_enqueue_fixture(tmp_path)
    reviewed = sha256(job)
    environment = dict(os.environ, EXPECTED_SHA256=reviewed, LANE="1")
    accepted = subprocess.run(
        [str(enqueue), str(job)], env=environment, text=True,
        capture_output=True, check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert queue.read_text(encoding="utf-8") == f"{reviewed}\t{job}\n"
    assert f"script sha256 {reviewed}" in accepted.stdout

    queue.write_text("", encoding="utf-8")
    marker = tmp_path / "unreviewed-marker"
    job.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    refused = subprocess.run(
        [str(enqueue), str(job)], env=environment, text=True,
        capture_output=True, check=False,
    )
    assert refused.returncode == 1
    assert "reviewed script SHA-256 changed" in refused.stderr
    assert queue.read_text(encoding="utf-8") == ""
    assert not marker.exists(), "unreviewed bytes executed before EXPECTED_SHA256 refusal"


def test_enqueue_rejects_symlink_target(tmp_path: Path) -> None:
    enqueue, queue, job = make_enqueue_fixture(tmp_path)
    link = job.with_name("linked.py")
    link.symlink_to(job)
    refused = subprocess.run(
        [str(enqueue), str(link)], env=dict(os.environ, LANE="1"),
        text=True, capture_output=True, check=False,
    )
    assert refused.returncode == 1
    assert "may not be a symlink" in refused.stderr
    assert queue.read_text(encoding="utf-8") == ""


def test_enqueue_dryrun_uses_snapshot_and_refuses_a_late_original_edit(
    tmp_path: Path,
) -> None:
    enqueue, queue, job = make_enqueue_fixture(tmp_path)
    reviewed_marker = tmp_path / "reviewed-marker"
    substituted_marker = tmp_path / "substituted-marker"
    job.write_text(
        f"from pathlib import Path\nPath({str(reviewed_marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    reviewed = sha256(job)
    gate = enqueue.parent / "gate.py"
    gate.write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['ORIGINAL_JOB']).write_text("
        "f\"from pathlib import Path\\nPath({os.environ['SUBSTITUTED_MARKER']!r}).write_text('ran')\\n\""
        ")\n",
        encoding="utf-8",
    )
    environment = dict(
        os.environ,
        EXPECTED_SHA256=reviewed,
        LANE="1",
        ORIGINAL_JOB=str(job),
        SUBSTITUTED_MARKER=str(substituted_marker),
    )
    refused = subprocess.run(
        [str(enqueue), str(job)], env=environment, text=True,
        capture_output=True, check=False,
    )
    assert refused.returncode == 1
    assert "changed after reviewed preflight" in refused.stderr
    assert reviewed_marker.read_text(encoding="utf-8") == "ran"
    assert not substituted_marker.exists()
    assert queue.read_text(encoding="utf-8") == ""


def test_lane2_keeps_legacy_path_only_record(tmp_path: Path) -> None:
    enqueue, _queue, job = make_enqueue_fixture(tmp_path)
    job.write_text("# BQLANE: cpu\nprint('lane2-dryrun')\n", encoding="utf-8")
    completed = subprocess.run(
        [str(enqueue), str(job)], env=dict(os.environ, LANE="2"),
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    queue2 = enqueue.parent.parent / "queue2.txt"
    assert queue2.read_text(encoding="utf-8") == f"{job}\n"


def test_gate_cannot_replace_snapshot_before_dryrun(tmp_path: Path) -> None:
    enqueue, queue, job = make_enqueue_fixture(tmp_path)
    marker = tmp_path / "snapshot-substitution-marker"
    job.write_text("print('reviewed dryrun')\n", encoding="utf-8")
    reviewed = sha256(job)
    gate = enqueue.parent / "gate.py"
    gate.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"Path(sys.argv[1]).write_text(\"from pathlib import Path\\nPath({str(marker)!r}).write_text('ran')\\n\")\n",
        encoding="utf-8",
    )
    refused = subprocess.run(
        [str(enqueue), str(job)],
        env=dict(os.environ, EXPECTED_SHA256=reviewed, LANE="1"),
        text=True, capture_output=True, check=False,
    )
    assert refused.returncode == 1
    assert "plan pre-flight FAILED" in refused.stderr
    assert "snapshot SHA-256 changed" in refused.stderr
    assert not marker.exists()
    assert queue.read_text(encoding="utf-8") == ""


def test_snapshot_dryrun_preserves_original_script_semantics(tmp_path: Path) -> None:
    enqueue, queue, job = make_enqueue_fixture(tmp_path)
    marker = tmp_path / "semantics-marker"
    job.write_text(
        "import os, sys\nfrom pathlib import Path\n"
        f"assert __file__ == {str(job)!r}\n"
        f"assert sys.argv == [{str(job)!r}]\n"
        f"assert sys.path[0] == {str(job.parent)!r}\n"
        f"Path({str(marker)!r}).write_text('exact')\n",
        encoding="utf-8",
    )
    reviewed = sha256(job)
    accepted = subprocess.run(
        [str(enqueue), str(job)],
        env=dict(os.environ, EXPECTED_SHA256=reviewed, LANE="1"),
        text=True, capture_output=True, check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert marker.read_text(encoding="utf-8") == "exact"
    assert queue.read_text(encoding="utf-8") == f"{reviewed}\t{job}\n"
