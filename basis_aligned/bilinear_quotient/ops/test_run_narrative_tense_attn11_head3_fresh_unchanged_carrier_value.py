#!/usr/bin/env python3

import hashlib

import pytest
import torch

import circuit_fast_screen_candidate_narrative_tense_fresh_unchanged_carrier as authority
import run_narrative_tense_attn11_head3_fresh_unchanged_carrier_value as runner


def test_fresh_authority_and_source_partition_are_frozen():
    rows = runner.build_rows()
    assert len(rows) == 128
    assert authority.authority_sha256() == runner.AUTHORITY_SHA256
    assert len({row["row_id"] for row in rows}) == 128
    endpoints = {(row[side + "_text"], tuple(row[side + "_ids"]))
                 for row in rows for side in ("base", "donor")}
    assert len(endpoints) == 256
    for row in rows:
        base, donor = row["base_ids"], row["donor_ids"]
        assert all(base[i] == donor[i] for i in row["R_positions"])
        assert set(row["R_positions"]).isdisjoint(row["complement_positions"])
        assert set(row["R_positions"]) | set(row["complement_positions"]) \
            == set(range(len(base)))
        assert row["S_positions"] == (len(base) - 1,)


def test_plan_binds_receipt_and_uses_effective_value_language():
    plan = runner.compile_plan()
    assert hashlib.sha256(runner.PRIOR_ART.read_bytes()).hexdigest() == runner.PRIOR_ART_SHA256
    assert plan["split"] == "fresh_confirmation"
    assert plan["price"] == {"model_forwards": 12, "example_evaluations": 1536,
                             "backwards": 0, "parameter_updates": 0}
    assert "effective source value" in plan["factor_definition"]["value"]
    assert "cached" not in str(plan).lower()


def test_exact_subset_algebra_empty_full_and_factor_modes():
    base = {
        "p": torch.tensor([[.2, .8]]),
        "u": torch.tensor([[[1., 2.], [3., 4.]]]),
    }
    donor = {
        "p": torch.tensor([[.6, .4]]),
        "u": torch.tensor([[[5., 6.], [7., 8.]]]),
    }
    base["head"] = torch.einsum("bk,bkd->bd", base["p"], base["u"])
    donor["head"] = torch.einsum("bk,bkd->bd", donor["p"], donor["u"])
    assert torch.equal(runner._group_head(base, donor, (), "joint", torch), base["head"])
    assert torch.allclose(runner._group_head(base, donor, (0, 1), "joint", torch),
                          donor["head"])
    score = runner._group_head(base, donor, (0,), "score", torch)
    value = runner._group_head(base, donor, (0,), "value", torch)
    assert torch.allclose(score, base["head"] - .2 * base["u"][:, 0]
                          + .6 * base["u"][:, 0])
    assert torch.allclose(value, base["head"] - .2 * base["u"][:, 0]
                          + .2 * donor["u"][:, 0])
    with pytest.raises(runner.ScreenError):
        runner._group_head(base, donor, (0,), "bad", torch)


def _synthetic(route=0.1, value=0.05, between=0.02, control=0.0):
    evidence = []
    target_cells = ("A1/x/past_to_present", "A1/x/present_to_past",
                    "A2/x/past_to_present", "A2/x/present_to_past")
    values = {arm: .01 for arm in runner.ARMS}
    values.update(expanded_native=0.0, native_reinstall=0.0, complete_head=1.0,
                  R_joint=route, R_effective_value=value, complement_joint=.1,
                  between_changes_effective_value=between,
                  post_last_change_effective_value=.01,
                  pre_first_change_effective_value=0.0)
    for cell in target_cells:
        family = cell.split("/")[0]
        for arm, effect in values.items():
            evidence.append({"family": family, "cell_id": cell, "arm": arm,
                             "margin_delta": effect, "donor_ce_gain": effect,
                             "answer_margin_delta": effect,
                             "base_answer_CE_change": -effect})
    for family in runner.CONTROL_FAMILIES:
        cell = f"{family}/x/control"
        for arm in runner.ARMS:
            effect = control if arm == "R_joint" else 0.0
            evidence.append({"family": family, "cell_id": cell, "arm": arm,
                             "margin_delta": 0.0, "donor_ce_gain": 0.0,
                             "answer_margin_delta": effect,
                             "base_answer_CE_change": effect})
    capability = {cell: {"base": 1.0, "donor": 1.0}
                  for cell in (*target_cells, "P/x/control", "C/x/control")}
    exactness = {"source_sum_max_absolute_error": 0.0,
                 "same_batch_native_reinstall_max_absolute_error": 0.0,
                 "pre_first_change_install_max_absolute_error": 0.0}
    liveness = {"all_registered_token_differences_nonempty": True,
                "minimum_R_factor_difference_norm_by_cell": {
                    cell: 1.0 for cell in capability}}
    return evidence, capability, exactness, liveness


def test_live_instrument_can_produce_an_honest_route_null():
    scored = runner.score(*_synthetic())
    assert scored["predictions"]["pred_a_instrument_live"]
    assert scored["predictions"]["pred_h_no_unchanged_carrier_route"]
    assert not scored["predictions"]["pred_b_unchanged_carrier_route"]


def test_route_value_selectivity_and_interactions_are_separate():
    scored = runner.score(*_synthetic(route=.8, value=.7, between=.6, control=.01))
    predictions = scored["predictions"]
    assert predictions["pred_a_instrument_live"]
    assert predictions["pred_b_unchanged_carrier_route"]
    assert predictions["pred_c_unchanged_carrier_effective_value"]
    assert predictions["pred_e_between_changes_effective_value"]
    assert scored["interactions"]
    nonselective = runner.score(*_synthetic(route=.8, value=.7, between=.6, control=.3))
    assert nonselective["predictions"]["pred_a_instrument_live"]
    assert not nonselective["predictions"]["pred_b_unchanged_carrier_route"]

