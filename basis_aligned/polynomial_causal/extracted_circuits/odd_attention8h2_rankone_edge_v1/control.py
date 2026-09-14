"""Replay four frozen native contexts through the standalone edge executor."""
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
    artifact = torch.load(root.parents[1] / "ODD_ATTENTION8H2_RANKONE_EDGE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)
    errors = []
    families = []
    for item in artifact["representatives"]:
        predicted = execute(program, item["routing"], item["recipient_embedding"], item["donor_embedding"])
        target = item["direct_write"]
        errors.append(float((predicted - target).norm() / target.norm().clamp_min(1e-8)))
        families.append(item["family"])
    result = {
        "pred_a": len(errors) == 4 and sorted(families) == [0, 1, 2, 3] and max(errors) <= 1e-5,
        "relative_errors": errors, "families": families,
        "program_sha256": digest(root / "program.pt"),
        "scope": "Standalone replay of four frozen native rank-one writes. Routing and normalized token embeddings are explicit inputs; downstream computation is absent.",
    }
    (root / "CONTROL.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

