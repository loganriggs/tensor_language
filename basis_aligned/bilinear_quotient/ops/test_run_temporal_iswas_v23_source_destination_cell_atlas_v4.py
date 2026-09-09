import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_v23_source_destination_cell_atlas_v4 as run


def test_v4_changes_only_versioned_authority_and_output_identity():
    assert run.atlas.CANDIDATE_ID.endswith("_v4")
    assert run.atlas.SCHEMA.endswith("_v4")
    assert run.atlas.OUT.name.endswith("_v4_result.json")
    assert run.atlas.PRICE["model_forwards_exact"] == 84
    assert run.atlas.BARS["closure"] == 1e-4


def test_model_free_v4_dryrun_is_authorized_and_result_safe():
    result_existed = run.atlas.OUT.exists()
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.atlas.ROOT.parents[1],
        env={**os.environ, "BQLIB_NO_MODEL": "1"}, check=True,
        capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["candidate_id"].endswith("_v4")
    assert payload["static_authority_ok"]
    assert payload["dependency"]["status"] == "valid"
    assert not payload["gpu_accessed"] and not payload["model_loaded"]
    assert run.atlas.OUT.exists() == result_existed
