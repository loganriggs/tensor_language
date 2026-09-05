#!/usr/bin/env python3

import run_task14_head11_3_fresh_fronted_natural_qk_number_specificity as runner


def test_plan_is_three_forward_fresh_screen():
    plan = runner.compile_plan()
    assert plan["row_count"] == 32 and plan["authority_sha256"] == runner.authority.EXPECTED_ROWS_SHA256
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 416,
                             "backwards": 0, "parameter_updates": 0}


def _fake_evidence(same=.1, qk1=.4, qk2=.3, joint=1.):
    rows = []
    for cell in ("singular_to_plural__behind_near", "singular_to_plural__beyond_under",
                 "plural_to_singular__behind_near", "plural_to_singular__beyond_under"):
        sign = runner._expected_sign(cell)
        values = {"recipient_score": 0., "same_qk1": same*qk1, "opposite_qk1": qk1,
                  "same_qk2": same*qk2, "opposite_qk2": qk2,
                  "same_joint": same*joint, "opposite_joint": joint}
        direction = cell.split("__")[0]
        for index in range(8):
            diagnostic = f"{direction}__a{(index % 4)//2}{index % 2}"
            for condition, unsigned in values.items():
                value = sign * unsigned
                rows.append({"row_id": f"{cell}-{index}", "group_id": f"g{index}",
                             "cell_id": cell, "diagnostic_cell_id": diagnostic,
                             "condition": condition, "donor_margin": value,
                             "donor_ce": 3-value})
    return rows


def test_score_applies_fresh_joint_and_branch_bars_in_both_directions():
    evidence = _fake_evidence()
    capability = {cell: {"recipient": 8, "same": 8, "opposite": 8}
                  for cell in {row["cell_id"] for row in evidence}}
    exactness = {"native_replay_max_absolute_logit_error": 0.,
                 "source_term_identity_max_absolute_error": 0.,
                 "direct_score_identity_max_absolute_error": 0.,
                 "installed_term_max_absolute_error": 0.}
    scored = runner.score(evidence, capability, exactness, runner.compile_plan()["bars"])
    assert all(scored["predictions"].values())
    assert len(scored["direction_attractor_diagnostics"]) == 8


def test_dead_or_nonspecific_interventions_cannot_pass():
    plan = runner.compile_plan()
    capability = {
        cell: {"recipient": 8, "same": 8, "opposite": 8}
        for cell in ("singular_to_plural__behind_near", "singular_to_plural__beyond_under",
                     "plural_to_singular__behind_near", "plural_to_singular__beyond_under")
    }
    exactness = {"native_replay_max_absolute_logit_error": 0.,
                 "source_term_identity_max_absolute_error": 0.,
                 "direct_score_identity_max_absolute_error": 0.,
                 "installed_term_max_absolute_error": 0.}
    dead = runner.score(_fake_evidence(qk1=.01, qk2=.01, joint=.01), capability,
                        exactness, plan["bars"])
    assert not dead["predictions"]["pred_a_instrument_live"]
    nonspecific = runner.score(_fake_evidence(same=.5), capability,
                               exactness, plan["bars"])
    assert nonspecific["predictions"]["pred_a_instrument_live"]
    assert not nonspecific["predictions"]["pred_b_joint_fresh_number_specificity"]
