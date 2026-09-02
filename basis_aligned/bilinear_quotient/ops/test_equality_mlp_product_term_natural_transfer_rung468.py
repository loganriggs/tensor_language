import copy

import torch

import equality_mlp_product_term_natural_transfer_rung468 as rung
import test_equality_mlp_product_term_group_rung467 as parent_test


def synthetic_interactive():
    full, proposed, complete, control, counts, selection = parent_test._synthetic()
    interaction = torch.tensor((.008, -.003, .006, -.005), dtype=torch.float64)
    cell_index = {cell: i for i, cell in enumerate(rung.CONTEXT_CELLS)}
    for si, source in enumerate(rung.SOURCES):
        scale = 1.0 if source == "N" else .9
        for ci, cell in enumerate(rung.CELLS):
            if cell in cell_index:
                proposed[si, 7, :, ci] += interaction[cell_index[cell]] * scale
    base = rung.parent.analyze(full, proposed, complete, control, counts, selection)
    code = {"analysis": {"composition": {"source_interactions": copy.deepcopy(
        base["composition"]["source_interactions"]
    )}}}
    return full, proposed, complete, control, counts, selection, code


def test_cross_register_transfer_passes_with_stable_interaction():
    result = rung.analyze_transfer(*synthetic_interactive())
    assert result["pred_b_union_transfer"]
    assert result["pred_c_control_separation"]
    assert result["pred_d_cross_module_transfer"]
    assert result["pred_e_interaction_transfer"]
    assert not result["transfer_strong_science_null"]


def test_changed_interaction_fails_only_interaction_transfer():
    args = list(synthetic_interactive())
    args[-1]["analysis"]["composition"]["source_interactions"] = {
        source: {"vector": [1.0, 1.0, 1.0, 1.0]} for source in rung.SOURCES
    }
    result = rung.analyze_transfer(*args)
    assert result["pred_b_union_transfer"]
    assert result["pred_c_control_separation"]
    assert result["pred_d_cross_module_transfer"]
    assert not result["pred_e_interaction_transfer"]

