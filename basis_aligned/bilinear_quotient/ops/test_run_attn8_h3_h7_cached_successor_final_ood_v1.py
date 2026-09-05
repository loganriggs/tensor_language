import sys

import torch

import run_attn8_h3_h7_cached_successor_final_ood_v1 as runner

authority = runner.authority


def test_plan_binds_cached_arm_and_exact_price():
    plan = runner.compile_plan()
    assert plan["arm"] == "cached_value_only"
    assert plan["fixed_heads"] == [3, 7]
    assert plan["pair_count"] == 256
    assert plan["price"] == {"model_forwards": 6, "example_evaluations": 1280,
                             "backwards": 0, "parameter_updates": 0}


def test_cached_replacement_preserves_recipient_scores_and_current_values():
    recipient = {"score": torch.tensor([[2., 3., 5.], [7., 11., 13.]]),
                 "cached": torch.tensor([[17., 19., 23.], [29., 31., 37.]]),
                 "own": torch.tensor([[41., 43., 47.], [53., 59., 61.]])}
    recipient["value"] = recipient["cached"] + recipient["own"]
    recipient["complete"] = (recipient["score"][:, :, None] *
                             recipient["value"][:, :, None]).sum(dim=1).repeat(1, 3)
    donor = {"score": recipient["score"] + 100,
             "cached": recipient["cached"] + 1000,
             "own": recipient["own"] + 10000}
    installed = runner.exact._replace(recipient, donor, "cached")
    expected = recipient["complete"].clone()
    for slot in range(2):
        expected[slot] += recipient["score"][slot, 2] * (
            recipient["own"][slot, 2] + donor["cached"][slot, 2]) - \
            recipient["score"][slot, 2] * recipient["value"][slot, 2]
    assert torch.equal(installed, expected)


def test_no_model_environment_exits_before_loading(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_NO_MODEL", "1")
    monkeypatch.setattr(sys, "argv", ["runner"])
    monkeypatch.setattr(runner.exact.r573.facade, "load_bilin18",
                        lambda **_: (_ for _ in ()).throw(AssertionError("model loaded")))
    runner.main()
    assert '"model_forwards": 6' in capsys.readouterr().out


def test_all_decisive_cells_are_gated():
    assert authority.POSITIVE not in authority.NEGATIVE
    assert authority.SECONDARY not in authority.NEGATIVE
    bars = runner.compile_plan()["bars"]
    assert bars["minimum_positive_donorward_fraction"] == .75
    assert bars["maximum_negative_absolute_mean_recipient_ce_change"] == .10


def test_native_capability_cannot_average_away_a_weak_endpoint():
    rows = ([{"native_recipient_answer_correct": True,
              "native_donor_answer_correct": True}] * 3 +
            [{"native_recipient_answer_correct": False,
              "native_donor_answer_correct": True}])
    recipient, donor, capable = runner._native_capability(rows, .85)
    assert recipient == .75
    assert donor == 1.
    assert capable is False
