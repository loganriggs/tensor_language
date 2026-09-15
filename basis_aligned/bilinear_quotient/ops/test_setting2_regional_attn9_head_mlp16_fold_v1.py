import torch

from run_setting2_regional_attn9_head_mlp16_fold_v1 import projected_head_writes


def test_projected_heads_sum_to_output_projection():
    torch.manual_seed(9)
    values = torch.randn(5, 9 * 128, dtype=torch.float64)
    projection = torch.randn(17, 9 * 128, dtype=torch.float64)
    coefficient = 1.7
    heads = projected_head_writes(values, projection, coefficient)
    assert len(heads) == 9
    assert torch.allclose(sum(heads), coefficient * values @ projection.T,
                          rtol=1e-11, atol=1e-11)
