import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_source_factor_audit_rung561.py")
TEXT = SCRIPT.read_text()


def test_audit_is_independent_and_cpu_only():
    assert "import pending_opener_source_factor_interchange_rung560" not in TEXT
    assert "import torch" not in TEXT
    assert "bootstrap_lower" in TEXT and "interactions" in TEXT and "wrong_source_controls" in TEXT


def test_pre_outcome_dryrun_does_not_require_result():
    env = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(["python", str(SCRIPT)], env=env, check=True, capture_output=True, text=True)
    assert '"status": "dryrun_passed"' in result.stdout
    assert '"result_required_only_at_execution": true' in result.stdout
    assert '"model_forwards": 0' in result.stdout
