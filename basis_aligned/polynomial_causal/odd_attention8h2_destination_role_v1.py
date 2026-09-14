"""Model-free destination partition for the inherited-city head8.2 write."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from odd_framing_role_split_v1 import role_masks


def destination_parts(write, masks):
    return {name: write * mask.to(write.device)[None, :, None] for name, mask in masks.items()}


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    rows = json.loads((root / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ROWS.json").read_text())["rows"]
    masks = role_masks(rows)
    torch.manual_seed(14091832)
    errors, outside = [], []
    for index in (0, 24, 48, 72):
        write = torch.randn(1, len(rows[index]["ids"]), 1152, dtype=torch.float64)
        parts = destination_parts(write, masks[index])
        errors.append(float((parts["description"] + parts["instruction"] - parts["framing"]).abs().max()))
        outside.append(float(parts["framing"][:, ~masks[index]["framing"]].abs().max()))
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": max(errors) == 0 and max(outside) == 0,
        "partition_max_abs_errors": errors, "outside_framing_max_abs": outside,
        "representative_counts": [{key: int(value.sum()) for key, value in masks[index].items()} for index in (0, 24, 48, 72)],
        "rows": len(rows),
        "scope": "Model-free exact destination masking of head8.2 writes; no native scores or semantic conclusion.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
