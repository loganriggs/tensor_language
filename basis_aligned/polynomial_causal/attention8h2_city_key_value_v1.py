"""Exact two-factor expansion for one attention source channel."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def factorial_channels(r0, rd, v0, vd):
    """Return recipient, donor, routing-only, value-only, additive and mixed channels."""
    recipient = r0[..., None] * v0
    donor = rd[..., None] * vd
    routing = rd[..., None] * v0
    value = r0[..., None] * vd
    additive = routing + value - recipient
    mixed = (rd - r0)[..., None] * (vd - v0)
    return {
        "recipient": recipient,
        "donor": donor,
        "routing": routing,
        "value": value,
        "additive": additive,
        "mixed": mixed,
    }


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_CITY_KEY_VALUE_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(14091817)
    errors = []
    mixed_norms = []
    for shape in [(2, 17), (3, 23)]:
        r0, rd = torch.randn(shape, dtype=torch.float64), torch.randn(shape, dtype=torch.float64)
        v0 = torch.randn(*shape, 128, dtype=torch.float64)
        vd = torch.randn(*shape, 128, dtype=torch.float64)
        arms = factorial_channels(r0, rd, v0, vd)
        errors.append(float((arms["additive"] + arms["mixed"] - arms["donor"]).abs().max()))
        mixed_norms.append(float(arms["mixed"].norm()))
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pred_a": max(errors) <= 1e-12 and min(mixed_norms) >= 1e-8,
        "max_factorial_error": max(errors),
        "mixed_norms": mixed_norms,
        "scope": "Model-free routing/value factorial identity for a single attention source; no native model scores or causal claim.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
