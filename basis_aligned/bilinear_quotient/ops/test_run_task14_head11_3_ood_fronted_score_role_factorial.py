#!/usr/bin/env python3

import run_task14_head11_3_ood_fronted_score_role_factorial as runner


def test_fronted_authority_groups_and_price():
    rows = runner.build_rows(); plan = runner.compile_plan()
    assert len(rows) == 32
    assert all(len(row["base_ids"]) == 9 and row["subject_position"] == 8 for row in rows)
    assert plan["groups"] == {"E": list(range(7)), "D": [7], "S": [8]}
    assert plan["price"] == {"model_forwards": 3, "example_evaluations": 1088,
                             "backwards": 0, "parameter_updates": 0}


def test_patch_contains_global_singletons_and_all_group_corners():
    import torch
    rows = [{"base_ids": list(range(9))}]
    tokens = torch.zeros(1, 9, dtype=torch.long); finals = torch.tensor([8])
    base = {"p": torch.arange(1., 10.).view(1, 9), "u": torch.ones(1, 9, 1),
            "head": torch.tensor([[45.]])}
    donor = {"p": torch.arange(11., 20.).view(1, 9), "u": 2 * torch.ones(1, 9, 1),
             "head": torch.tensor([[270.]])}
    patch = runner._compile_patch_batch(tokens, finals, base, donor, rows, torch)
    specs = patch["specs"]
    assert len(specs) == 30
    assert sum(spec[2] == "global" for spec in specs) == 4
    assert sum(spec[2] == "singleton" for spec in specs) == 18
    assert {spec[4] for spec in specs if spec[2] == "group"} == set(range(8))


def test_mobius_reconstruction_and_shapley_efficiency():
    values = {subset: float(subset * subset + 3) for subset in range(8)}
    _coefficients, _shapley, reconstruction, efficiency = runner._mobius(values)
    assert reconstruction < 1e-12
    assert efficiency < 1e-12


def _fake_evidence(earlier=.8, self_effect=.8):
    output = []
    for cell in (runner.WEAK_CELL, "fronted_plural_to_fronted_singular"):
        for index in range(16):
            globals_ = {runner.GLOBAL[0]: 0., runner.GLOBAL[1]: .1,
                        runner.GLOBAL[2]: .5, runner.GLOBAL[3]: 1.}
            for condition, value in globals_.items():
                output.append({"atlas_cell_id": cell, "condition": condition,
                               "donor_margin": 2 * value, "margin_delta": 2 * value,
                               "donor_ce": 2-value, "donor_ce_gain": value})
            for donor_value, baseline in ((False, 0.0), (True, .5)):
                for position in range(9):
                    value = baseline + .01
                    output.append({"atlas_cell_id": cell,
                                   "condition": f"singleton_p{position}_{'donor' if donor_value else 'native'}_value",
                                   "donor_margin": 2 * value, "margin_delta": 2 * value,
                                   "donor_ce": 2-value, "donor_ce_gain": value})
            for corner in range(8):
                value = .5
                if corner == 3: value += .5 * earlier
                if corner == 4: value += .5 * self_effect
                if corner == 7: value = 1.
                output.append({"atlas_cell_id": cell, "condition": f"group_corner_{corner:03b}",
                               "donor_margin": 2 * value, "margin_delta": 2 * value,
                               "donor_ce": 2-value, "donor_ce_gain": value})
    return output


def test_score_applies_both_registered_sufficiency_bars():
    capability = {cell: {"base": 1., "donor": 1.} for cell in
                  (runner.WEAK_CELL, "fronted_plural_to_fronted_singular")}
    exact = {key: 0. for key in (
        "native_replay_max_absolute_logit_error", "source_term_identity_max_absolute_error",
        "pre_subject_value_max_absolute_error", "native_corner_max_absolute_logit_error",
        "complete_head_vector_max_absolute_error", "group_endpoint_max_absolute_logit_error",
        "parent_value_only_max_absolute_reproduction_error", "global_closure_max_absolute_error")}
    bars = runner.compile_plan()["bars"]
    both = runner.score(_fake_evidence(), capability, exact, bars)["predictions"]
    scored = runner.score(_fake_evidence(), capability, exact, bars)
    assert len(scored["cells"][runner.WEAK_CELL]["score_singletons"]["native_value"]) == 9
    assert len(scored["cells"][runner.WEAK_CELL]["score_singletons"]["donor_value"]) == 9
    weak = runner.score(_fake_evidence(earlier=.4, self_effect=.4), capability, exact, bars)["predictions"]
    assert both["pred_b_earlier_score_sufficiency"] and both["pred_c_self_score_sufficiency"]
    assert both["pred_d_redundant_score_routes"]
    assert not weak["pred_b_earlier_score_sufficiency"]
    assert not weak["pred_c_self_score_sufficiency"]
