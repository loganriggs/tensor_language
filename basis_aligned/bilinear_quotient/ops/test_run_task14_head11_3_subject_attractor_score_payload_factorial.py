#!/usr/bin/env python3

import attention_source_factor_primitive as primitive
import run_task14_head11_3_subject_attractor_score_payload_factorial as runner


def test_frozen_authority_positions_and_price():
    rows = runner.authority.build_rows()
    assert len(rows) == 64
    assert all(row["base_ids"][1] != row["donor_ids"][1] for row in rows)
    assert all(row["base_semantic_position"] == len(row["base_ids"]) - 1 for row in rows)
    assert all(row["donor_semantic_position"] == len(row["donor_ids"]) - 1 for row in rows)
    plan = runner.compile_plan()
    assert plan["price"]["model_forwards"] == 7
    assert plan["closed_splits"] == ["TEST", "OOD"]


def test_runner_uses_shared_generic_source_factor_primitive():
    assert runner.source_factor is primitive
    assert runner.LAYER == 11 and runner.HEAD == 3


def test_score_distinguishes_the_four_opposing_outcomes():
    def evidence(subject_score, subject_payload, attractor):
        values = {
            "subject_score": subject_score, "subject_payload": subject_payload,
            "subject_joint": max(subject_score, subject_payload),
            "attractor_score": attractor, "attractor_payload": attractor,
            "attractor_joint": attractor, "complete_head": 1.0,
        }
        return [{"condition": name, "margin_delta": value,
                 "native_donor_ce": 2.0, "donor_ce": 2.0 - value}
                for name, value in values.items() for _ in range(8)]
    payload = runner.score(evidence(.1, .7, .05), 0, 0, 1)["predictions"]
    score = runner.score(evidence(.7, .1, .05), 0, 0, 1)["predictions"]
    attractor = runner.score(evidence(.1, .1, .7), 0, 0, 1)["predictions"]
    neither = runner.score(evidence(.1, .15, .05), 0, 0, 1)["predictions"]
    assert payload["pred_b_subject_payload"]
    assert score["pred_c_subject_score"]
    assert attractor["pred_d_attractor_driven"]
    assert neither["pred_e_neither_or_other_source"]


def test_score_rejects_a_complete_head_that_does_not_improve_ce():
    evidence = [
        {"condition": name, "margin_delta": 1.0 if name == "complete_head" else .1,
         "native_donor_ce": 2.0, "donor_ce": 2.1}
        for name in runner.CONDITIONS for _ in range(8)
    ]
    scored = runner.score(evidence, 0, 0, 1)
    assert not scored["predictions"]["pred_a_instrument_live"]
