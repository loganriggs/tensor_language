import torch

import equality_correction_group_factorial_rung466 as rung


CORRECTION = torch.tensor([-.08, .04, .09, -.07], dtype=torch.float64)
TASK = .75 * CORRECTION
SUPPRESSOR = torch.tensor([-.03, -.025, -.015, -.03], dtype=torch.float64)
INTERACTION = torch.tensor([-.01, .008, .012, -.009], dtype=torch.float64)


def synthetic():
    base = torch.ones(rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    losses = torch.ones(len(rung.SOURCES), len(rung.SUBSETS), rung.DOCUMENTS,
                        len(rung.CELLS), dtype=torch.float64)
    direct = torch.ones(len(rung.SOURCES), rung.DOCUMENTS, len(rung.CELLS),
                        dtype=torch.float64)
    counts = torch.ones(rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    cell_index = {cell: i for i, cell in enumerate(rung.CONTEXT_CELLS)}
    for si, source in enumerate(rung.SOURCES):
        scale = 1.0 if source == "N" else .9
        for ci, cell in enumerate(rung.CELLS):
            full_effect = .12 * scale
            correction = CORRECTION[cell_index[cell]].item() * scale if cell in cell_index else .002
            direct[si, :, ci] -= full_effect - correction
            for mask in rung.SUBSETS:
                value = 0.0
                if mask & rung.TASK_MASK:
                    value += TASK[cell_index[cell]].item() * scale if cell in cell_index else .001
                if mask & rung.SUPPRESSOR_MASK:
                    value += SUPPRESSOR[cell_index[cell]].item() * scale if cell in cell_index else .001
                if (mask & rung.TASK_MASK) and (mask & rung.SUPPRESSOR_MASK):
                    value += INTERACTION[cell_index[cell]].item() * scale if cell in cell_index else 0
                losses[si, mask, :, ci] -= full_effect - value
    return base, losses, direct, counts


def test_group_roles_and_interaction_pass():
    result = rung.analyze(*synthetic())
    assert result["pred_b_task_group_context"]
    assert result["pred_c_broad_suppressor_role"]
    assert result["pred_d_cross_group_interaction"]
    assert result["pred_e_five_site_extraction"]
    assert not result["strong_science_null"]


def test_mobius_dividends_reconstruct_subset_value():
    result = rung.analyze(*synthetic())
    pooled = result["pooled"]
    source = "N"
    for cell in rung.CELLS:
        reconstructed = sum(
            row[cell] for mask, row in pooled["mobius_dividends"][source].items()
            if mask & ~rung.ALL_MASK == 0
        )
        assert abs(reconstructed - pooled["subset_values"][source][rung.ALL_MASK][cell]) < 1e-10


def test_zero_task_group_fires_null():
    base, losses, direct, counts = synthetic()
    for si in range(len(rung.SOURCES)):
        losses[si, rung.TASK_MASK] = losses[si, 0]
    result = rung.analyze(base, losses, direct, counts)
    assert not result["pred_b_task_group_context"]


def test_subset_site_mapping_is_exact():
    assert rung.subset_sites(rung.TASK_MASK) == rung.TASK_SITES
    assert rung.subset_sites(rung.SUPPRESSOR_MASK) == rung.SUPPRESSOR_SITES
    assert set(rung.subset_sites(rung.ALL_MASK)) == set(rung.SITES)
