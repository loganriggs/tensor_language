import json
import os
from pathlib import Path
import subprocess
import sys

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v24 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1 as run


def test_v24_bank_is_exact_aligned_balanced_and_history_disjoint():
    rows = fresh.build_rows()
    assert len(rows) == len({row["row_id"] for row in rows}) == 64
    assert fresh.authority_sha256() == run.ROWS_SHA256
    assert {family: sum(row["family"] == family for row in rows)
            for family in run.FAMILIES} == {family: 16 for family in run.FAMILIES}
    assert all(row["base_semantic_position"] == row["donor_semantic_position"]
               for row in rows)


def test_v24_prior_freezes_target_and_control_capability_without_filtering():
    prior = json.loads(run.PRIOR.read_text())
    assert prior["frozen_design"]["bars"] == {
        "all_base_donor_semantic_positions_equal": True,
        "every_target_and_control_direction_by_side_cell_accuracy_minimum": 0.75,
        "jointly_correct_rows_per_family_minimum": 12,
        "joint_bar_denominator": 16,
    }
    assert prior["frozen_design"]["price"] == run.PRICE
    assert tuple(prior["predictions"]) == run.PREDICTION_KEYS


def test_v24_runner_binds_confirmed_v23_ancestry():
    observed = {"prior": run.sha(run.PRIOR), "builder": run.sha(run.BUILDER),
                "v23_capability": run.sha(run.V23_CAPABILITY),
                "v23_confirmation": run.sha(run.V23_CONFIRMATION)}
    assert observed == run.EXPECTED
    confirmation = json.loads(run.V23_CONFIRMATION.read_text())
    assert confirmation["terminal"] == "confirmed_selective_four_head_writer_program"
    assert all(confirmation["predictions"].values())


def test_v24_dryrun_has_exact_price_and_loads_no_model():
    env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload["model_loaded"] is False and payload["gpu_accessed"] is False
    assert payload["model_forwards_exact"] == 2
    assert payload["example_evaluations_exact"] == 128
    assert payload["all_families_scored_without_filtering"] == list(run.FAMILIES)
