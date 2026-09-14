"""Model-free row, mask and factor control for the fresh inherited-city chain."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from attention8h2_city_key_value_v1 import factorial_channels
from attention8h2_city_value_source_v1 import value_arms
from odd_contextual_positions_v1 import contextual_masks
from odd_semantic_positions_v1 import semantic_masks


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_CHAIN_FRESH_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    rows = json.loads((root / "ODD_ATTENTION8H2_CHAIN_FRESH_V1_ROWS.json").read_text())["rows"]
    context, semantic = contextual_masks(rows), semantic_masks(rows)
    paired = all(
        torch.equal(context[i]["city"], context[i ^ 1]["city"])
        and torch.equal(semantic[i]["framing"], semantic[i ^ 1]["framing"])
        for i in range(len(rows))
    )
    torch.manual_seed(14091828)
    r0, rd = torch.randn(2, 19, 19, dtype=torch.float64), torch.randn(2, 19, 19, dtype=torch.float64)
    raw = [torch.randn(2, 1, 19, 128, dtype=torch.float64) for _ in range(4)]
    vals = value_arms(torch.tensor(.41, dtype=torch.float64), *raw)
    factors = factorial_channels(r0, rd, vals["recipient"], vals["donor"])
    factor_error = float((factors["additive"] + factors["mixed"] - factors["donor"]).abs().max())
    value_error = float(((vals["current"] - vals["recipient"]) + (vals["first"] - vals["recipient"]) - (vals["donor"] - vals["recipient"])).abs().max())
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pred_a": paired and factor_error <= 1e-12 and value_error <= 1e-12,
        "paired_city_and_framing_masks": paired,
        "city_counts": sorted(set(int(mask["city"].sum()) for mask in context)),
        "framing_counts": sorted(set(int(mask["framing"].sum()) for mask in semantic)),
        "max_factorial_error": factor_error, "max_value_sum_error": value_error,
        "rows": len(rows),
        "scope": "Model-free fresh-row mask and routing/value/current/inherited identities; no native scores or causal conclusion.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
