import torch

from run_setting2_regional_mlp16_upstream_source_fold_v1 import propagated_coefficients


def test_propagated_coefficients_reconstruct_recurrence():
    torch.manual_seed(18)
    lambdas = torch.randn(18, 2, dtype=torch.float64)
    x0 = torch.randn(7, dtype=torch.float64)
    attention = [torch.randn(7, dtype=torch.float64) for _ in range(17)]
    mlp = [torch.randn(7, dtype=torch.float64) for _ in range(17)]
    x = x0
    for layer in range(17):
        x = lambdas[layer, 0] * x + lambdas[layer, 1] * x0 + attention[layer] + mlp[layer]
    raw17 = lambdas[17, 0] * x + lambdas[17, 1] * x0
    embed, coefficients = propagated_coefficients(lambdas)
    reconstructed = embed * x0
    for layer in range(17):
        reconstructed = reconstructed + coefficients[f"attn{layer}"] * attention[layer]
        reconstructed = reconstructed + coefficients[f"mlp{layer}"] * mlp[layer]
    assert torch.allclose(reconstructed, raw17, rtol=1e-11, atol=1e-11)
