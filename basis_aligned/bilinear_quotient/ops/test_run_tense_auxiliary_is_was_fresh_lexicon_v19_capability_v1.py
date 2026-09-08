import json

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v19 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v19_capability_v1 as run


def test_v19_bank_is_exact_novel_balanced_inventory():
    rows = fresh.build_rows()
    assert len(rows) == len({row["row_id"] for row in rows}) == 64
    assert fresh.authority_sha256() == run.ROWS_SHA256
    assert {family: sum(row["transform_id"] == family for row in rows)
            for family in ("A1", "A2", "P", "C")} == {
                "A1": 16, "A2": 16, "P": 16, "C": 16}
    assert all(all(row["construction_checks"].values()) for row in rows)


def test_v19_prior_freezes_capability_only_and_price():
    prior = json.loads(run.PRIOR.read_text())
    assert prior["frozen_design"]["bars"]["joint_bar_denominator"] == 16
    assert prior["frozen_design"]["price"] == {"model_forwards": 2,
        "example_evaluations": 128, "interventions": 0,
        "transformer_backwards": 0, "model_updates": 0}


def test_v19_runner_binds_v18_capability_only_ancestry():
    ancestry = json.loads(run.V18_RESULT.read_text())
    assert ancestry["terminal"] == "screen" and ancestry["causal_outcomes_opened"] is False
    assert all(ancestry["predictions"].values())
