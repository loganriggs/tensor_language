"""Exact current/inherited split for one mixed head8.2 value source."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def value_arms(attn, current_recipient, current_donor, first_recipient, first_donor):
    current0 = (1 - attn) * current_recipient
    currentd = (1 - attn) * current_donor
    first0 = attn * first_recipient
    firstd = attn * first_donor
    return {
        "recipient": current0 + first0,
        "donor": currentd + firstd,
        "current": currentd + first0,
        "first": current0 + firstd,
    }


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_CITY_VALUE_SOURCE_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(14091823)
    values = [torch.randn(4, 128, dtype=torch.float64) for _ in range(4)]
    arms = value_arms(torch.tensor(.37, dtype=torch.float64), *values)
    error = float(((arms["current"] - arms["recipient"]) + (arms["first"] - arms["recipient"]) - (arms["donor"] - arms["recipient"])).abs().max())
    current_norm = float((arms["current"] - arms["recipient"]).norm())
    first_norm = float((arms["first"] - arms["recipient"]).norm())
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": error <= 1e-12 and min(current_norm, first_norm) >= 1e-8,
        "max_value_sum_error": error, "current_delta_norm": current_norm,
        "first_delta_norm": first_norm,
        "scope": "Model-free linear split of an attention value mixture; no native model score or causal claim.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
