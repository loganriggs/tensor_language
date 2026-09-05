#!/usr/bin/env python3

import run_task14_head11_3_source_value_role_group_factorial as runner


def test_groups_partition_all_nonzero_sources_and_price_is_three_forwards():
    rows = runner.atlas.build_rows()
    for row in rows:
        groups = runner._group_positions(row)
        positions = [position for group in runner.GROUPS for position in groups[group]]
        assert sorted(positions) == list(range(1, len(row["base_ids"])))
        assert len(positions) == len(set(positions))
    plan = runner.compile_plan()
    assert len(plan["conditions"]) == 17
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 1344,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["closed_claims"] == ["OOD", "selectivity", "completeness",
                                      "upstream_writer_identity"]


def test_patch_heads_use_native_scores_and_subset_selected_values():
    import torch
    row = dict(runner.atlas.build_rows()[0])
    row["target_family"] = "A1"
    row["base_ids"] = list(range(7))
    tokens, finals = torch.arange(7).view(1, 7), torch.tensor([6])
    base = {"p": torch.arange(1., 8.).view(1, 7),
            "u": torch.ones(1, 7, 1), "head": torch.tensor([[28.]])}
    donor = {"p": torch.full((1, 7), 99.),
             "u": (10 * torch.arange(1., 8.)).view(1, 7, 1),
             "head": torch.tensor([[999.]])}
    patch = runner._compile_patch_batch(tokens, finals, base, donor, [row], torch)
    subset_heads = {mask: float(patch["replacement_heads"][index])
                    for index, (_row, condition, mask) in enumerate(patch["specs"])
                    if condition == "subset"}
    assert subset_heads[0] == 28.
    assert subset_heads[1] == 28. - 2. + 2. * 20.
    assert subset_heads[runner.ALL_MASK] == 1. + sum(k * (10 * k) for k in range(2, 8))
    assert float(patch["replacement_heads"][-1]) == 999.


def test_mobius_and_shapley_recover_known_pair_interaction():
    values = {}
    for mask in range(16):
        values[mask] = (1.0 if mask & 1 else 0.0) + (2.0 if mask & 2 else 0.0) \
            + (4.0 if mask & 3 == 3 else 0.0)
    dividends, shapley = runner._mobius_and_shapley(values)
    assert dividends[1] == 1.0 and dividends[2] == 2.0 and dividends[3] == 4.0
    assert all(abs(value) < 1e-12 for mask, value in dividends.items()
               if mask not in (1, 2, 3))
    assert shapley == {"S": 3.0, "I": 4.0, "B": 0.0, "A": 0.0}


def test_score_centers_game_and_enforces_named_shared_SxI_interaction():
    evidence = []
    cells = ("pp_s", "pp_p", "relative_s", "relative_p")
    for cell in cells:
        for group_index in range(16):
            group_id = f"g{group_index:02d}"
            for mask in range(16):
                value = .01 + (.2 if mask & 1 else 0) + (.2 if mask & 2 else 0) \
                    + (.4 if mask & 3 == 3 else 0) + (.15 if mask & 4 else 0) \
                    + (.05 if mask & 8 else 0)
                evidence.append({"row_id": f"{cell}:{group_id}", "group_id": group_id,
                                 "atlas_cell_id": cell, "condition": "subset",
                                 "subset_mask": mask, "donor_margin": value,
                                 "donor_ce": 2 - value, "margin_delta": value,
                                 "donor_ce_gain": value})
            evidence.append({"row_id": f"{cell}:{group_id}", "group_id": group_id,
                             "atlas_cell_id": cell, "condition": "complete_head",
                             "subset_mask": None, "margin_delta": 1.2,
                             "donor_ce_gain": 1.1})
    capability = {cell: {"base": 1., "donor": 1.} for cell in cells}
    scored = runner.score(evidence, capability, 0, 0, 0, 0, 0,
                          runner.compile_plan()["bars"])
    assert scored["mobius_reconstruction_max_absolute_error"] <= 1e-8
    assert scored["shapley_efficiency_max_absolute_error"] <= 1e-8
    assert scored["predictions"]["pred_b_SI_sufficient_all_cells"]
    assert scored["predictions"]["pred_d_SxI_interaction_shared"]
