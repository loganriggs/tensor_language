"""Replay four native contexts through the composed edge package."""
import hashlib
import json
from pathlib import Path

import torch
from execute import execute, load_program


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    program = load_program()
    artifact = torch.load(root.parents[1] / "ODD_ATTENTION8H2_O_COMPOSED_EDGE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)
    errors, families = [], []
    for item in artifact["representatives"]:
        predicted = execute(
            program, item["block8_current"], item["city_index"],
            item["recipient_embedding"], item["donor_embedding"],
            item["block9_residual"], item["initial"], item["first_values"],
            item["reentry"], item["framing_mask"],
        )
        target = item["delta"]
        errors.append(float((predicted - target).norm() / target.norm().clamp_min(1e-8)))
        families.append(item["family"])
    result = {
        "pred_a": sorted(families) == [0, 1, 2, 3] and max(errors) <= 1e-5,
        "relative_errors": errors, "families": families,
        "program_sha256": digest(root / "program.pt"),
        "scope": "Standalone replay of four composed head8.2-to-head9.8-O deltas. Native state generators and suffix are absent.",
    }
    (root / "CONTROL.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

