import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("pending_opener_source_factor_interchange_rung560_v2.py")
TEXT = SCRIPT.read_text()


def test_v2_correction_is_only_split_envelope_selection():
    assert "raw[split], execution" in TEXT
    assert "ORIGINAL_EVALUATE" in TEXT
    assert "parent.evaluate = evaluate_selected_split" in TEXT
    assert "threshold" not in TEXT.lower()


def test_v2_dryrun_preserves_frozen_plan():
    env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(["python", str(SCRIPT)], env=env, check=True, capture_output=True, text=True)
    assert '"status": "dryrun_passed"' in result.stdout
    assert '"selected_arm": "payload"' in result.stdout
    assert '"final_or_ood_opened": false' in result.stdout
