"""Exact interaction compression for a single-source attention write."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def direct_write(routing, value_delta, output_weight):
    return F.linear(routing[..., None] * value_delta[:, None, :], output_weight)


def rankone_write(routing, value_delta, output_weight):
    writer = F.linear(value_delta, output_weight)
    return routing[..., None] * writer[:, None, :]


def price(tokens, value_width=128, residual_width=1152):
    dense_interface = tokens * residual_width
    rankone_interface = tokens + residual_width
    direct_multiplies = tokens * value_width * residual_width
    rankone_multiplies = value_width * residual_width + tokens * residual_width
    return {
        "tokens": tokens, "dense_interface_scalars": dense_interface,
        "rankone_interface_scalars": rankone_interface,
        "interface_fraction_saved": 1 - rankone_interface / dense_interface,
        "direct_projection_multiplies": direct_multiplies,
        "rankone_projection_and_scale_multiplies": rankone_multiplies,
        "multiply_fraction_saved": 1 - rankone_multiplies / direct_multiplies,
    }


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_RANKONE_EDGE_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(14091836)
    errors = []
    for tokens in (12, 20, 31):
        routing = torch.randn(3, tokens, dtype=torch.float64)
        value_delta = torch.randn(3, 128, dtype=torch.float64)
        weight = torch.randn(1152, 128, dtype=torch.float64)
        direct = direct_write(routing, value_delta, weight)
        factored = rankone_write(routing, value_delta, weight)
        errors.append(float((direct - factored).norm() / direct.norm().clamp_min(1e-30)))
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": max(errors) <= 1e-12,
        "relative_errors": errors, "prices": [price(tokens) for tokens in (18, 19, 20, 21)],
        "static_weight_scalars": 2 * 1152 * 128,
        "runtime_input_scope": "normalized recipient/donor token embeddings plus destination routing scalars",
        "scope": "Model-free distributive factorization of one attention source write. Price excludes upstream routing/embedding generation and downstream state/suffix.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
