"""Model-free midpoint and row checks for corpus edge removal."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def projected_writer(program, recipient, donor):
    delta = program["mixture"] * F.linear(donor - recipient, program["value_weight"])
    return F.linear(delta, program["output_weight"])


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_CORPUS_MIDPOINT_REMOVAL_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    rows = json.loads((root / "ODD_ATTENTION8H2_CORPUS_TRANSFER_V1_ROWS.json").read_text())["rows"]
    paired = all(
        rows[i]["context_id"] == rows[i + 1]["context_id"]
        and rows[i]["endpoint"] == rows[i + 1]["endpoint"]
        and rows[i]["destination_positions"] == rows[i + 1]["destination_positions"]
        and sum(a != b for a, b in zip(rows[i]["ids"], rows[i + 1]["ids"])) == 1
        for i in range(0, len(rows), 2)
    )
    program = torch.load(root / "ODD_ATTENTION8H2_ROUTING_CLOSURE_V1_PROGRAM.pt", weights_only=True)
    torch.manual_seed(14091909)
    errors = []
    for _ in range(8):
        recipient = F.rms_norm(torch.randn(2, 1152), (1152,))
        donor = F.rms_norm(torch.randn(2, 1152), (1152,))
        full = projected_writer(program, recipient, donor)
        midpoint = projected_writer(program, recipient, (recipient + donor) / 2)
        errors.append(float((midpoint - .5 * full).norm() / full.norm().clamp_min(1e-8)))
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pred_a": paired and max(errors) <= 1e-6,
        "rows": len(rows), "pairs": len(rows) // 2,
        "relative_half_write_errors": errors,
        "scope": "Model-free pair and exact midpoint-write checks on the frozen corpus panel. Native downstream necessity remains untested.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
