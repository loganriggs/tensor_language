"""Static/CPU checks for the rung-522 managed GPU smoke wrapper."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


PATH = Path(__file__).with_name(
    "attention8_selective_shared_projector_rung522_gpu_smoke.py"
)
REPO = Path("/workspace/tensor_language")


def test_dryrun_validates_hashes_without_importing_torch_or_model_code():
    environment = dict(os.environ)
    environment["BQLIB_DRYRUN"] = "1"
    environment["BQLIB_NO_MODEL"] = "1"
    result = subprocess.run(
        [sys.executable, str(PATH)],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "7 frozen hashes" in result.stdout
    assert "batch=6*8*2=96" in result.stdout
    assert "no science metrics retained" in result.stdout


def test_hash_and_dryrun_gate_precede_every_nontoplevel_model_import():
    source = PATH.read_text()
    assert source.index("_PREIMPORT_HASHES = _validate_frozen_hashes()") < source.index(
        'if os.environ.get("BQLIB_DRYRUN") == "1"'
    )
    assert source.index('if os.environ.get("BQLIB_DRYRUN") == "1"') < source.index(
        "import torch"
    )
    assert source.index("import torch") < source.index(
        "import bilin18_observed_model_facade as facade"
    )


def test_registered_physical_shapes_and_no_science_scoring_are_literal():
    source = PATH.read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"RECIPIENT_ROWS", "DONOR_MAPS", "DIRECTIONS", "RANK"}
    }
    assert assignments == {
        "RECIPIENT_ROWS": 6,
        "DONOR_MAPS": 8,
        "DIRECTIONS": 2,
        "RANK": 4,
    }
    assert "cross_entropy" not in source
    assert "_score_stage_a" not in source
    assert '"scientific_metrics_retained": False' in source
    assert '"model_science_opened": False' in source
