#!/usr/bin/env python3

import run_task14_head11_3_ood_fronted_natural_qk_number_specificity as runner


def test_foreign_pairing_is_cyclic_matched_and_token8_only():
    rows = runner.build_triples()
    assert len(rows) == 32
    assert len({row["foreign_group_id"] for row in rows}) == 32
    for row in rows:
        assert row["group_id"] != row["foreign_group_id"]
        assert row["base_ids"][8] != row["same_ids"][8]
        assert row["base_ids"][:8] != row["same_ids"][:8]
        assert [i for i, pair in enumerate(zip(row["same_ids"], row["opposite_ids"]))
                if pair[0] != pair[1]] == [8]


def test_plan_has_seven_conditions_and_three_forwards():
    plan = runner.compile_plan()
    assert plan["conditions"] == list(runner.CONDITIONS)
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 416,
                             "backwards": 0, "parameter_updates": 0}


def test_patch_uses_recipient_same_and_opposite_natural_branch_scalars():
    import torch
    tokens = torch.zeros(1, 9, dtype=torch.long); finals = torch.tensor([8])
    def factors(scale):
        return {"p": torch.full((1, 9), scale ** 4), "u": torch.ones(1, 9, 1),
                "head": torch.tensor([[9 * scale ** 4]]),
                "q": scale * torch.ones(1, 2), "k": scale * torch.ones(1, 9, 2),
                "q2": scale * torch.ones(1, 2), "k2": scale * torch.ones(1, 9, 2)}
    patch = runner._compile_patch_batch(tokens, finals, factors(1.), factors(2.), factors(3.),
        [{"atlas_cell_id": "cell", "group_id": "group"}], torch)
    assert [spec[1] for spec in patch["specs"]] == list(runner.CONDITIONS)
    assert patch["installed_scalars"].tolist() == [1., 4., 9., 4., 9., 16., 81.]


def _fake_evidence(same=.1, qk1=.4, qk2=.3, joint=1.):
    rows = []
    for cell, sign in runner.EXPECTED_SIGN.items():
        values = {"recipient_score": 0., "same_qk1": same*qk1, "opposite_qk1": qk1,
                  "same_qk2": same*qk2, "opposite_qk2": qk2,
                  "same_joint": same*joint, "opposite_joint": joint}
        for index in range(16):
            for condition, unsigned in values.items():
                value = sign * unsigned
                rows.append({"row_id": f"{cell}-{index}", "group_id": f"g{index:02d}",
                             "atlas_cell_id": cell, "condition": condition,
                             "donor_margin": value, "donor_ce": 3-value})
    return rows


def test_score_applies_joint_and_branch_specificity_bars_sign_aware():
    capability = {cell: {"base": 1., "donor": 1.} for cell in runner.EXPECTED_SIGN}
    exact = {"native_replay_max_absolute_logit_error": 0.,
             "source_term_identity_max_absolute_error": 0.,
             "direct_score_identity_max_absolute_error": 0.,
             "installed_term_max_absolute_error": 0.,
             "recipient_baseline_duplicate_max_absolute_error": 0.}
    predictions = runner.score(_fake_evidence(), capability, exact,
                               runner.compile_plan()["bars"])["predictions"]
    assert all(predictions.values())
    tiny = runner.score(_fake_evidence(same=.1, qk1=.004, qk2=.003, joint=.01),
                        capability, exact, runner.compile_plan()["bars"])["predictions"]
    assert not tiny["pred_a_instrument_live"]
