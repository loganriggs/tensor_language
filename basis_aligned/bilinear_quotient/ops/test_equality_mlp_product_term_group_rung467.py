import torch

import equality_mlp_product_term_group_rung467 as rung


def test_selection_uses_both_sources_and_both_halves_without_topk():
    numerators = torch.zeros(2, 2, 3, rung.HIDDEN, 4, dtype=torch.float64)
    counts = torch.ones(2, 2, 4, dtype=torch.float64)
    distractor = torch.tensor((1.0, 1.0, -1.0, -1.0), dtype=torch.float64)
    numerators[:] = distractor
    for mi in range(3):
        numerators[:, :, mi, :5] = rung.TARGET
    amplitude = torch.arange(3 * rung.HIDDEN, dtype=torch.float64).view(3, rung.HIDDEN)
    groups, controls, report, _, _ = rung.select_groups(numerators, counts, amplitude)
    assert all(groups[site] == list(range(5)) for site in rung.SITES)
    assert all(report[site]["half_selection_jaccard"] == 1.0 for site in rung.SITES)
    assert all(len(controls[name][site]) == 5 for name in rung.CONTROL_TYPES
               for site in rung.SITES)
    assert all(not (set(groups[site]) & set(controls[name][site]))
               for name in rung.CONTROL_TYPES for site in rung.SITES)


def test_empty_half_selections_do_not_count_as_stable():
    numerators = torch.zeros(2, 2, 3, rung.HIDDEN, 4, dtype=torch.float64)
    counts = torch.ones(2, 2, 4, dtype=torch.float64)
    amplitude = torch.ones(3, rung.HIDDEN, dtype=torch.float64)
    groups, _, report, _, _ = rung.select_groups(numerators, counts, amplitude)
    assert all(not groups[site] for site in rung.SITES)
    assert all(report[site]["half_selection_jaccard"] == 0.0 for site in rung.SITES)


def _synthetic():
    n, nc = 96, len(rung.CELLS)
    full = torch.ones(3, n, nc, dtype=torch.float64)
    proposed = torch.ones(2, 8, n, nc, dtype=torch.float64)
    complete = torch.ones_like(proposed)
    control = torch.ones(2, 2, 4, n, nc, dtype=torch.float64)
    counts = torch.ones(n, nc, dtype=torch.float64)
    cell_index = {cell: i for i, cell in enumerate(rung.CONTEXT_CELLS)}
    full[:, :, :] = 1.0
    for si, source in enumerate(rung.SOURCES):
        scale = 1.0 if source == "N" else .9
        full[rung.ALL_SOURCES.index(source)] = 1.0 - .20 * scale
        for mask in rung.SUBSETS:
            bits = mask.bit_count()
            for ci, cell in enumerate(rung.CELLS):
                value = (bits * .02 * rung.TARGET[cell_index[cell]].item() * scale
                         if cell in cell_index else bits * .0003 * scale)
                arm_effect = .20 * scale - value
                proposed[si, mask, :, ci] = 1.0 - arm_effect
                parent_value = (bits * (.08 / 3) * rung.TARGET[cell_index[cell]].item() * scale
                                if cell in cell_index else bits * .0005 * scale)
                complete[si, mask, :, ci] = 1.0 - (.20 * scale - parent_value)
        for ti, _ in enumerate(rung.CONTROL_TYPES):
            for ki, mask in enumerate(rung.CONTROL_MASKS):
                bits = mask.bit_count()
                for ci, cell in enumerate(rung.CELLS):
                    wrong = (.002 * bits * scale if cell in cell_index and ci == 0 else 0.0)
                    control[si, ti, ki, :, ci] = 1.0 - (.20 * scale - wrong)
    selection = {site: {"selected_count": 5, "half_selection_jaccard": 1.0}
                 for site in rung.SITES}
    return full, proposed, complete, control, counts, selection


def test_exact_heldout_group_and_additive_composition_pass():
    result = rung.analyze(*_synthetic())
    assert result["pred_b_stable_split"]
    assert result["pred_c_exact_heldout_correction"]
    assert result["pred_d_beats_matched_controls"]
    assert result["pred_e_cross_module_composition"]
    assert result["composition"]["regime"] == "approximately_additive"
    assert not result["strong_science_null"]


def test_control_matching_target_defeats_selection_claim():
    full, proposed, complete, control, counts, selection = _synthetic()
    for si in range(2):
        for ti in range(2):
            control[si, ti, 3] = proposed[si, 7]
    result = rung.analyze(full, proposed, complete, control, counts, selection)
    assert not result["pred_d_beats_matched_controls"]

