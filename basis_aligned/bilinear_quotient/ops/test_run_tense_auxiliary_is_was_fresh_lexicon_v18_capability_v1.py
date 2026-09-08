import json

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v18_capability_v1 as run


def test_v18_bank_is_exact_novel_balanced_inventory():
    rows = fresh.build_rows()
    assert len(rows) == len({row["row_id"] for row in rows}) == 64
    assert fresh.authority_sha256() == run.ROWS_SHA256
    assert {family: sum(row["transform_id"] == family for row in rows)
            for family in ("A1", "A2", "P", "C")} == {
                "A1": 16, "A2": 16, "P": 16, "C": 16,
            }
    assert all(all(row["construction_checks"].values()) for row in rows)


def test_v18_prior_registers_correct_panel_denominator_and_price():
    prior = json.loads(run.PRIOR.read_text())
    assert prior["frozen_design"]["bars"] == {
        "cell_accuracy_minimum": 0.75,
        "jointly_correct_rows_per_A_panel_minimum": 12,
        "joint_bar_denominator": 16,
    }
    assert prior["frozen_design"]["price"] == {
        "model_forwards": 2,
        "example_evaluations": 128,
        "interventions": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def test_v18_runner_is_bound_to_capability_only_ancestry():
    ancestry = json.loads(run.V17_RESULT.read_text())
    assert ancestry["terminal"] == "screen"
    assert ancestry["causal_outcomes_opened"] is False
    assert all(ancestry["predictions"].values())
    assert tuple(run.PREDICTION_KEYS) == (
        "pred_a_authority_novelty_and_exact_population",
        "pred_b_native_a_panel_capability",
        "pred_c_joint_capable_population",
        "pred_d_no_causal_outcome_access_and_exact_price",
    )
