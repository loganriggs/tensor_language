import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_target_invariance_das_audit_rung559.py")
TEXT = SCRIPT.read_text()


def test_audit_is_cpu_only_and_recomputes_raw_summaries():
    assert "torch" not in TEXT
    assert "endpoint_values" in TEXT
    assert "full_vocabulary_logit_rms_values" in TEXT
    assert "bootstrap_lower" in TEXT
    assert "rank_eligible" in TEXT


def test_audit_runs_without_cuda():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(["python", str(SCRIPT)], env=env, check=True, capture_output=True, text=True)
    assert '"strong_null_recomputed": true' in result.stdout
    assert '"model_forwards": 0' in result.stdout
