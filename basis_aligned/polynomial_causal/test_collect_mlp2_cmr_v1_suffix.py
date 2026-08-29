from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

import collect_mlp2_cmr_v1_suffix as collect


def test_batch_plan_pads_only_the_last_batch() -> None:
    live = tuple(range(191))
    assert collect.batch_plan(live, 0) == ((0, 1, 2, 3), 4)
    assert collect.batch_plan(live, 188) == ((188, 189, 190, 188), 3)
    with pytest.raises(ValueError, match="batch"):
        collect.batch_plan(live, 191)


def test_runner_imports_from_outside_repository() -> None:
    script = Path(collect.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, "-c", (
            "import runpy; "
            f"runpy.run_path({str(script)!r}, run_name='suffix_import_smoke')"
        )],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_protected_inputs_and_constants_are_current() -> None:
    parents = collect.protected_inputs()
    assert parents["fit_bundle"] == collect.FIT_BUNDLE_SHA256
    assert len(collect.PROBE_SEEDS) == 8
    assert collect.CALLS * len(collect.PROBE_SEEDS) == 384

