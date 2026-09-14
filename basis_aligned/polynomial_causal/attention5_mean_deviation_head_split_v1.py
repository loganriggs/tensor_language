"""CPU algebra control for the attention5 mean/deviation head decomposition."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    torch.manual_seed(5057)
    n, heads, hd, d = 37, 9, 7, 31
    z = torch.randn(n, heads, hd, dtype=torch.float64)
    weight = torch.randn(d, heads, hd, dtype=torch.float64)
    bias = torch.randn(d, dtype=torch.float64)
    parts = torch.einsum("nhk,dhk->nhd", z, weight)
    native = parts.sum(1) + bias
    means = parts[:19].mean(0)
    mean_full = means.sum(0) + bias
    deviations = parts - means.unsqueeze(0)
    rebuilt = mean_full + deviations.sum(1)
    subset = (5, 7)
    selected = mean_full + deviations[:, subset].sum(1)
    complement = mean_full + deviations[:, tuple(h for h in range(heads) if h not in subset)].sum(1)
    interaction_identity = selected + complement - mean_full - rebuilt
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pred_a": bool((rebuilt - native).abs().max() < 1e-12),
        "max_rebuild_error": float((rebuilt - native).abs().max()),
        "max_partition_identity_error": float(interaction_identity.abs().max()),
        "shape": [n, heads, hd, d],
        "scope": "Synthetic float64 algebra only; no model, corpus, score, head ranking, or circuit verdict.",
    }
    out = Path(__file__).with_name("ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V1_CPU_CONTROL.json")
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
