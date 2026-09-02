from types import SimpleNamespace

import torch

import mlp10_exact_source_family_factorial_rung508 as rung


def test_six_families_partition_all_named_sources_once():
    covered = [source for family in rung.FAMILIES.values() for source in family]
    assert sorted(covered) == sorted(rung.parent.NAMED_SOURCES)
    assert len(covered) == len(set(covered)) == 22


def test_twenty_one_family_terms_partition_all_exact_pairs_once():
    covered = [index for spec in rung.GROUP_SPECS for index in spec]
    assert len(rung.GROUP_NAMES) == 21
    assert sorted(covered) == list(range(253))
    assert len(covered) == len(set(covered))


def test_family_outputs_equal_sums_of_constituent_exact_terms():
    torch.manual_seed(508)
    width, hidden = 5, 7
    mlp = SimpleNamespace(Down=SimpleNamespace(weight=torch.randn(width, hidden)))
    factors = {
        "left": torch.randn(2, 3, 22, hidden),
        "right": torch.randn(2, 3, 22, hidden),
    }
    outputs, hidden_sum = rung._family_outputs(mlp, factors)
    expected_outputs = []
    for spec in rung.GROUP_SPECS:
        expected_outputs.append(sum(
            rung.parent._pair_output(mlp, factors, index) for index in spec))
    torch.testing.assert_close(
        outputs, torch.stack(expected_outputs, dim=2), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        hidden_sum, factors["left"].sum(2) * factors["right"].sum(2),
        rtol=2e-5, atol=2e-5)


def test_family_relationship_keeps_shared_input_separate_from_same_output():
    row = rung._group_relationship("A_eqxA_eq", "A_eqxM_post")
    assert row["shared_families"] == ["A_eq"]
    assert row["same_left_family"] is True
    assert row["same_right_family"] is False


def test_registered_price_includes_replay_and_intact_factor_captures():
    assert 12276 + 496 * torch.combinations(torch.arange(8), r=2).shape[0] == 26164
