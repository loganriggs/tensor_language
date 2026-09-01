"""RUNG 374 -- FIXED CERTIFICATE-RAY TRANSFER TO DISTRIBUTED MLP CUTS.

The Q/K-only ray from rung356 is frozen.  Project the two-layer frontier, the
third-layer certificate-cliff arm, and the prospective QK72 mid-tier without
refitting the ray or certificate thresholds.

Frozen predictions
------------------
pred_a: rung367 cosine>=.95, R2>=.80, certificate-count error<=3.
pred_b: rung372 meets the same bars and predicts its observed cliff.
pred_c: complete rung373 meets the same bars.

Null: any target cosine<.80 or certificate-count error>8.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "distributed_mlp_certificate_ray_transfer_results.json"
RAY = ROOT / "certificate_damage_axis_transfer_results.json"
TARGETS = {
    "qk64_mlp04_p768": (
        "cev_mixed64_context_qk_mlp04_context_p768.pt",
        "mixed64_context_qk_mlp04_context_p768_ood_results.json"),
    "qk64_mlp042_p768": (
        "cev_mixed64_context_qk_mlp042_context_p768.pt",
        "mixed64_context_qk_mlp042_context_p768_ood_results.json"),
    "qk72_mlp04_p768": (
        "cev_mixed72_context_qk_mlp04_context_p768.pt",
        "mixed72_context_qk_mlp04_context_p768_ood_results.json"),
}


def _project(vector: torch.Tensor, shape: torch.Tensor) -> dict:
    scale = float(vector @ shape / shape.square().sum().clamp_min(1e-30))
    prediction = scale * shape
    cosine = float(torch.nn.functional.cosine_similarity(vector[None], prediction[None]))
    denominator = (vector - vector.mean()).square().sum().clamp_min(1e-30)
    r2 = float(1.0 - (vector - prediction).square().sum() / denominator)
    actual = int((vector < 1.0).sum())
    predicted = int((prediction < 1.0).sum())
    return {"projection_scale": scale, "cosine": cosine, "vector_r2": r2,
            "actual_certificates": actual, "predicted_certificates": predicted,
            "certificate_count_error": abs(predicted - actual)}


@torch.no_grad()
def main() -> None:
    needed = [RAY, ROOT / "circuits/BATTERY.json", ROOT / "census_state_diverse.pt"]
    for cev, receipt in TARGETS.values():
        needed.extend((ROOT / cev, ROOT / receipt))
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in needed)
        assert len(TARGETS) == 3
        print("DISTRIBUTED MLP CERT RAY | dry run: fixed ray, CEVs, receipts valid")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN

    ray = json.loads(RAY.read_text())
    shape = torch.tensor(ray["qk"]["shape"], dtype=torch.float64)
    frozen_tags = ray["tags"]
    CN.use_state("census_state_diverse.pt")
    base = CN.base_ce().float().reshape(-1).cpu()
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    tags, members, thresholds = [], [], []
    for tag in sorted(battery):
        try:
            member = CN.leaf(tag)["member"].long().cpu()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        tags.append(tag)
        members.append(member)
        thresholds.append(.5 * float(battery[tag]["mean_ablation"]["top"][0]["abs_dce_members"]))
    threshold = torch.tensor(thresholds, dtype=torch.float64)
    assert tags == frozen_tags and len(tags) == shape.numel() == 62
    assert bool((threshold > 0).all())

    targets = {}
    for name, (cev_name, receipt_name) in TARGETS.items():
        cev = torch.load(ROOT / cev_name, map_location="cpu").float().reshape(-1)
        damage = cev - base
        vector = torch.tensor([float(damage[index].abs().mean()) for index in members],
                              dtype=torch.float64) / threshold
        receipt = json.loads((ROOT / receipt_name).read_text())
        reported = int(receipt["certificates_valid"])
        reproduced = int((vector < 1.0).sum())
        if reproduced != reported:
            raise RuntimeError(f"certificate mismatch {name}: {reproduced} != {reported}")
        targets[name] = _project(vector, shape)
        targets[name]["reported_certificates"] = reported
        targets[name]["cev_file"] = cev_name

    def held(name: str) -> bool:
        row = targets[name]
        return (row["cosine"] >= .95 and row["vector_r2"] >= .80
                and row["certificate_count_error"] <= 3)

    pred_a = held("qk64_mlp04_p768")
    pred_b = held("qk64_mlp042_p768")
    pred_c = held("qk72_mlp04_p768")
    null = any(row["cosine"] < .80 or row["certificate_count_error"] > 8
               for row in targets.values())
    result = {
        "status": "distributed_mlp_certificate_ray_transfer_complete",
        "rung": 374,
        "claim_level": "fixed_qk_certificate_ray_out_of_family_transfer_audit",
        "normalization": ray["normalization"],
        "certificate_rule": ray["certificate_rule"],
        "ray_source_rung": 356,
        "ray_refit": False,
        "tags_exact": tags == frozen_tags,
        "targets": targets,
        'pred_a_two_layer_frontier_stays_on_fixed_ray': bool(pred_a),
        'pred_b_third_layer_cliff_stays_on_fixed_ray': bool(pred_b),
        'pred_c_mid_tier_stays_on_fixed_ray': bool(pred_c),
        "null_distributed_mlp_rejects_fixed_ray": bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
