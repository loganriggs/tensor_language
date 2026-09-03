import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_source_factor_interchange_rung560.py")
TEXT = SCRIPT.read_text()


def test_exact_factorization_and_no_dimension_sweep():
    assert 'ARMS = ("score", "payload", "joint")' in TEXT
    assert "score1 * score2" in TEXT
    assert "score.unsqueeze(-1) * u" in TEXT
    assert "rank" not in TEXT.lower().replace("rung", "")
    assert '"FINAL_TEST"' not in TEXT and '"OOD"' not in TEXT


def test_dryrun_uses_source_audit_without_model():
    env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(["python", str(SCRIPT)], env=env, check=True, capture_output=True, text=True)
    assert '"status": "dryrun_passed"' in result.stdout
    assert '"selected_arm": "payload"' in result.stdout
    assert '"semantic_rows": 540' in result.stdout
    assert '"model_loaded": false' in result.stdout
