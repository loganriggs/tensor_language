"""Synthetic metric control for the attention5 response quotient."""
import json
from datetime import datetime, timezone
from pathlib import Path
import torch


def main():
    torch.manual_seed(5730)
    gains = torch.randn(9, 3, dtype=torch.float64)
    basis = torch.linalg.qr(torch.randn(41, 3, dtype=torch.float64))[0].T
    discovery = gains @ basis
    heldout = (gains * torch.tensor([1.1, .9, 1.2])) @ basis
    _, s, vh = torch.linalg.svd(discovery, full_matrices=False)
    q = vh[:3]
    energy = float((s[:3] ** 2).sum() / (s ** 2).sum())
    recon = float((heldout - (heldout @ q.T) @ q).norm() / heldout.norm())
    full = discovery.sum(0)
    composition = float((discovery.sum(0) - full).norm() / full.norm())
    out = Path(__file__).with_name("ATTENTION5_DEVIATION_RESPONSE_QUOTIENT_V1_CPU_CONTROL.json")
    if out.exists():
        raise FileExistsError(out)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "pred_a": energy > .999999 and recon < 1e-12 and composition == 0.0,
              "rank3_energy": energy, "heldout_reconstruction_error": recon, "composition_error": composition,
              "scope": "Synthetic float64 SVD/projection/composition metrics only; no model or circuit verdict."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
