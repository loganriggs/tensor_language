"""Replay four native contexts through the routing-closed executor."""
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
    artifact = torch.load(root.parents[1] / "ODD_ATTENTION8H2_ROUTING_CLOSURE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)
    errors, families = [], []
    for item in artifact["representatives"]:
        predicted = execute(program, item["current"], item["city_index"], item["recipient_embedding"], item["donor_embedding"])
        target = item["direct_write"]
        errors.append(float((predicted - target).norm() / target.norm().clamp_min(1e-8)))
        families.append(item["family"])
    result = {
        "pred_a": sorted(families) == [0, 1, 2, 3] and max(errors) <= 1e-5,
        "relative_errors": errors, "families": families,
        "program_sha256": digest(root / "program.pt"),
        "scope": "Standalone four-template replay from normalized block8 state and city embeddings. Upstream state generation and downstream O/suffix are absent.",
    }
    (root / "CONTROL.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

