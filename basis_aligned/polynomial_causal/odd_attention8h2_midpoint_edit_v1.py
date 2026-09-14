"""Model-free half-strength city-value edit for the composed odd relay."""
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
    out = root / "ODD_ATTENTION8H2_MIDPOINT_EDIT_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    program = torch.load(root / "ODD_ATTENTION8H2_ROUTING_CLOSURE_V1_PROGRAM.pt", weights_only=True)
    torch.manual_seed(14091852)
    errors = []
    for _ in range(4):
        recipient = F.rms_norm(torch.randn(3, 1152), (1152,))
        donor = F.rms_norm(torch.randn(3, 1152), (1152,))
        midpoint = (recipient + donor) / 2
        full = projected_writer(program, recipient, donor)
        half = projected_writer(program, recipient, midpoint)
        errors.append(float((half - .5 * full).norm() / full.norm().clamp_min(1e-8)))
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": max(errors) <= 1e-6,
        "relative_half_write_errors": errors,
        "scope": "Model-free exact half-strength edit in normalized embedding space. Downstream RMS/O/suffix nonlinearity remains for native testing.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
