"""RUNG 376 -- HELD-OUT SECOND CERTIFICATE MODE.

Subtract the frozen Q/K certificate ray, fit one residual direction on three
MLP-bearing programs, and test it on a held-out QK72+MLP04 program.  Value96
is the family-specificity control.

Frozen predictions
------------------
pred_a: training residual rank-one R2>=.60 and LOO cosine>=.50.
pred_b: held MLP residual cosine>=.60 and full R2 gain>=.003, no count harm.
pred_c: absolute value-control residual cosine<=.50 and identities exact.

Null: train residual R2<.30, held cosine<=0, or held R2 gain<.0005.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp_second_certificate_mode_transfer_results.json"
RAY = ROOT / "certificate_damage_axis_transfer_results.json"
TRAIN = {
    "qk56_mlp0_p512": ("cev_mixed56_context_qk_mlp0_context_p512.pt",
                        "mixed56_context_qk_mlp0_context_p512_ood_results.json"),
    "qk64_mlp04_p768": ("cev_mixed64_context_qk_mlp04_context_p768.pt",
                         "mixed64_context_qk_mlp04_context_p768_ood_results.json"),
    "qk64_mlp042_p768": ("cev_mixed64_context_qk_mlp042_context_p768.pt",
                          "mixed64_context_qk_mlp042_context_p768_ood_results.json"),
}
HELD = ("cev_mixed72_context_qk_mlp04_context_p768.pt",
        "mixed72_context_qk_mlp04_context_p768_ood_results.json")
VALUE = ("cev_mixed80_context_qk_value96_context.pt",
         "mixed80_context_qk_value96_context_ood_results.json")


def _ray_projection(vector: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    return (vector @ shape / shape.square().sum().clamp_min(1e-30)) * shape


def _full_metrics(vector: torch.Tensor, prediction: torch.Tensor) -> dict:
    denominator = (vector - vector.mean()).square().sum().clamp_min(1e-30)
    r2 = float(1.0 - (vector - prediction).square().sum() / denominator)
    actual = int((vector < 1.0).sum())
    predicted = int((prediction < 1.0).sum())
    return {"r2": r2, "actual_certificates": actual,
            "predicted_certificates": predicted,
            "certificate_count_error": abs(actual - predicted)}


@torch.no_grad()
def main() -> None:
    needed = [RAY, ROOT / "circuits/BATTERY.json", ROOT / "census_state_diverse.pt"]
    for pair in list(TRAIN.values()) + [HELD, VALUE]:
        needed.extend(ROOT / name for name in pair)
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in needed) and len(TRAIN) == 3
        print("SECOND CERT MODE | dry run: ray, five CEVs, receipts, bars valid")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN

    ray = json.loads(RAY.read_text())
    shape = torch.tensor(ray["qk"]["shape"], dtype=torch.float64)
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
        tags.append(tag); members.append(member)
        thresholds.append(.5 * float(battery[tag]["mean_ablation"]["top"][0]["abs_dce_members"]))
    threshold = torch.tensor(thresholds, dtype=torch.float64)
    assert tags == ray["tags"] and len(tags) == shape.numel() == 62

    def load(pair) -> torch.Tensor:
        cev = torch.load(ROOT / pair[0], map_location="cpu").float().reshape(-1)
        damage = cev - base
        vector = torch.tensor([float(damage[index].abs().mean()) for index in members],
                              dtype=torch.float64) / threshold
        reported = int(json.loads((ROOT / pair[1]).read_text())["certificates_valid"])
        assert int((vector < 1.0).sum()) == reported
        return vector

    train_vectors = {name: load(pair) for name, pair in TRAIN.items()}
    residuals = torch.stack([vector - _ray_projection(vector, shape)
                             for vector in train_vectors.values()])
    _u, _s, vh = torch.linalg.svd(residuals, full_matrices=False)
    direction = vh[0]
    reconstructed = (residuals @ direction)[:, None] * direction[None]
    train_r2 = float(1.0 - (residuals - reconstructed).square().sum()
                     / residuals.square().sum().clamp_min(1e-30))
    loo_cosines = []
    for held_index in range(3):
        keep = torch.arange(3) != held_index
        _u_i, _s_i, vh_i = torch.linalg.svd(residuals[keep], full_matrices=False)
        candidate = vh_i[0]
        cosine = float(torch.nn.functional.cosine_similarity(
            residuals[held_index][None], candidate[None]))
        loo_cosines.append(abs(cosine))

    held = load(HELD)
    held_ray = _ray_projection(held, shape)
    held_residual = held - held_ray
    held_cosine = abs(float(torch.nn.functional.cosine_similarity(
        held_residual[None], direction[None])))
    held_two = held_ray + (held_residual @ direction) * direction
    held_fixed_metrics = _full_metrics(held, held_ray)
    held_two_metrics = _full_metrics(held, held_two)
    held_gain = held_two_metrics["r2"] - held_fixed_metrics["r2"]

    value = load(VALUE)
    value_residual = value - _ray_projection(value, shape)
    value_cosine = abs(float(torch.nn.functional.cosine_similarity(
        value_residual[None], direction[None])))
    pred_a = train_r2 >= .60 and min(loo_cosines) >= .50
    pred_b = (held_cosine >= .60 and held_gain >= .003
              and held_two_metrics["certificate_count_error"]
              <= held_fixed_metrics["certificate_count_error"])
    pred_c = value_cosine <= .50 and tags == ray["tags"]
    null = train_r2 < .30 or held_cosine <= 0 or held_gain < .0005
    result = {
        "status": "mlp_second_certificate_mode_transfer_complete",
        "rung": 376,
        "claim_level": "heldout_vector_valued_certificate_residual_screen",
        "mode1_source": "frozen rung356 QK-only certificate ray",
        "mode2_fit_programs": list(TRAIN),
        "mode2_refit_on_heldout": False,
        "train_residual_rank1_r2": train_r2,
        "train_leave_one_residual_cosines_abs": loo_cosines,
        "heldout_residual_cosine_abs": held_cosine,
        "heldout_fixed_ray": held_fixed_metrics,
        "heldout_two_mode": held_two_metrics,
        "heldout_full_vector_r2_gain": held_gain,
        "value_control_residual_cosine_abs": value_cosine,
        "tags_exact": tags == ray["tags"],
        'pred_a_mlp_residual_has_stable_second_mode': bool(pred_a),
        'pred_b_second_mode_predicts_heldout_mlp_vector': bool(pred_b),
        'pred_c_second_mode_is_value_family_specific': bool(pred_c),
        "null_no_useful_second_certificate_mode": bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
