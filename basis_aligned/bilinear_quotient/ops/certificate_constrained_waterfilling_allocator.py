"""RUNG 357 -- PROSPECTIVE CERTIFICATE-CONSTRAINED WATER-FILLING ALLOCATOR.

Close the projection-leakage gap in rung356: learn the scalar intensity of its
fixed QK-only certificate ray from census damage on pure-QK and mixed104/MLP0
programs, then hold out every QK+MLP construction as a family.  Combine this
with rung355's component damage curves, the already measured cross-family tax
envelope, and exact prices to enumerate only the calibrated rank grid.

Frozen predictions
------------------
pred_a_intensity_transfers_to_heldout_compositions:
    Held-out log-scale R2 >=.90, median relative scale error <=.10, and
    certificate-count MAE <=3 when measured damage is the input.
pred_b_tail_plus_tax_predicts_end_to_end_counts:
    All measured cross-family taxes are in [1.019,1.053], and counts predicted
    end-to-end from component damage, the geometric-midpoint tax, and the
    intensity model have MAE <=4.
pred_c_discrete_frontier_is_resolved:
    Exact enumeration either finds a program below 512,561,462 scalars whose
    conservative upper-scale count is >=43, or records that no such program
    exists on the calibrated grid.

Null: held-out log-scale R2 <.70 or end-to-end certificate MAE >6.
This is a CPU allocation/lower-bound screen, not a physical adoption and not a
claim outside the enumerated ranks/families.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "certificate_constrained_waterfilling_allocator_results.json"
TAX_LO = 1.019
TAX_HI = 1.053
TAX_CENTER = math.sqrt(TAX_LO * TAX_HI)
FRONTIER_SCALARS = 512_561_462
FRONTIER_CERTIFICATES = 43

PURE_QK_PRICE = {
    96: 535_089_462,
    88: 530_583_862,
    80: 526_078_262,
    72: 521_572_662,
    64: 517_067_062,
    56: 512_561_462,
    48: 508_055_862,
}
MLP0_SAVING = {None: 0, 640: 3_981_312, 512: 5_308_416,
               448: 5_971_968, 384: 6_635_520, 256: 7_962_624}


def _fit_log_power(damage: torch.Tensor, scale: torch.Tensor):
    x = torch.stack((torch.ones_like(damage), damage.log()), dim=1)
    beta = torch.linalg.lstsq(x, scale.log()[:, None]).solution[:, 0]
    return beta


def _predict_scale(beta: torch.Tensor, damage: torch.Tensor):
    return (beta[0] + beta[1] * damage.log()).exp()


def _log_r2(actual: torch.Tensor, predicted: torch.Tensor):
    residual = (actual.log() - predicted.log()).square().sum()
    denominator = (actual.log() - actual.log().mean()).square().sum().clamp_min(1e-30)
    return float(1.0 - residual / denominator)


def _count(scale: float, shape: torch.Tensor):
    return int((scale * shape < 1.0).sum())


@torch.no_grad()
def main() -> None:
    needed = [
        ROOT / "certificate_damage_axis_transfer_results.json",
        ROOT / "context_metric_tail_waterfilling_law_results.json",
        ROOT / "circuits/BATTERY.json",
        ROOT / "census_state_diverse.pt",
        ROOT / "cev_mixed104_online_cv0.pt",
        ROOT / "mixed104_online_cv0_results.json",
        ROOT / "cev_mixed104_mlp0_context_rrr_frontier.pt",
        ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json",
        ROOT / "cev_mixed104_mlp0_context_rrr_lower_ranks.pt",
        ROOT / "mixed104_mlp0_context_metric_lower_rank_ood_results.json",
        ROOT / "cev_mixed96_context_qk_mlp0_context_p512_p640.pt",
        ROOT / "mixed96_context_qk_mlp0_context_p512_p640_ood_results.json",
        ROOT / "cev_mixed96_context_qk_mlp0_context_p448.pt",
        ROOT / "mixed96_context_qk_mlp0_context_p448_ood_results.json",
        ROOT / "cev_mixed56_context_qk_mlp0_context_p512.pt",
        ROOT / "mixed56_context_qk_mlp0_context_p512_ood_results.json",
    ]
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in needed)
        assert sorted(PURE_QK_PRICE) == [48, 56, 64, 72, 80, 88, 96]
        assert sorted(rank for rank in MLP0_SAVING if rank is not None) == [256, 384, 448, 512, 640]
        print("CERTIFICATE-CONSTRAINED ALLOCATOR | dry run: files, grid, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN

    CN.use_state("census_state_diverse.pt")
    base = CN.base_ce().float().reshape(-1).cpu()
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    members, thresholds = [], []
    for tag in sorted(battery):
        try:
            member = CN.leaf(tag)["member"].long().cpu()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        members.append(member)
        thresholds.append(.5 * float(battery[tag]["mean_ablation"]["top"][0]["abs_dce_members"]))
    threshold = torch.tensor(thresholds, dtype=torch.float64)
    assert len(members) == 62 and bool((threshold > 0).all())

    ray = json.loads((ROOT / "certificate_damage_axis_transfer_results.json").read_text())
    shape = torch.tensor(ray["qk"]["shape"], dtype=torch.float64)

    def project(cev: torch.Tensor):
        damage_vector = cev.float().reshape(-1).cpu() - base
        vector = torch.tensor([float(damage_vector[index].abs().mean()) for index in members],
                              dtype=torch.float64) / threshold
        scale = float(vector @ shape / shape.square().sum().clamp_min(1e-30))
        return scale, int((vector < 1.0).sum())

    # Training family 1: the seven pure-QK points.  Their ray scales were
    # computed directly from their saved CEVs in rung356.
    qk_damage = json.loads((ROOT / "context_metric_tail_waterfilling_law_results.json").read_text())["qk"]["measured_census_damage"]
    train = []
    for rank, scale, count in zip(ray["qk"]["ranks_high_to_low"], ray["qk"]["scales"],
                                  ray["qk"]["reported_certificates"]):
        train.append({"name": f"qk{rank}", "damage": float(qk_damage[str(rank)]),
                      "scale": float(scale), "certificates": int(count)})

    # Training family 2: mixed104 alone and mixed104 plus each saved MLP0 rank.
    base_receipt = json.loads((ROOT / "mixed104_online_cv0_results.json").read_text())
    base_cev = torch.load(ROOT / "cev_mixed104_online_cv0.pt", map_location="cpu")
    scale, count = project(base_cev)
    assert count == int(base_receipt["certificates_valid"])
    train.append({"name": "mixed104", "damage": float(base_receipt["census_damage"]),
                  "scale": scale, "certificates": count})

    frontier_cevs = torch.load(ROOT / "cev_mixed104_mlp0_context_rrr_frontier.pt", map_location="cpu")
    frontier_receipt = json.loads((ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json").read_text())
    lower_cevs = torch.load(ROOT / "cev_mixed104_mlp0_context_rrr_lower_ranks.pt", map_location="cpu")
    lower_receipt = json.loads((ROOT / "mixed104_mlp0_context_metric_lower_rank_ood_results.json").read_text())
    for rank in (640, 512, 448, 384, 256):
        cevs, receipt = ((frontier_cevs, frontier_receipt) if rank >= 512 else
                         (lower_cevs, lower_receipt))
        arm = receipt["arms"][str(rank)]
        scale, count = project(cevs[f"r{rank}"])
        assert count == int(arm["certificates_valid"])
        train.append({"name": f"mixed104_mlp0_{rank}", "damage": float(arm["census_damage"]),
                      "scale": scale, "certificates": count})

    train_damage = torch.tensor([row["damage"] for row in train], dtype=torch.float64)
    train_scale = torch.tensor([row["scale"] for row in train], dtype=torch.float64)
    beta = _fit_log_power(train_damage, train_scale)

    # Entire QK+MLP construction family is held out from the intensity fit.
    target_specs = [
        ("qk96_mlp0_640", "cev_mixed96_context_qk_mlp0_context_p512_p640.pt", "r640",
         "mixed96_context_qk_mlp0_context_p512_p640_ood_results.json", "640", 96, 640),
        ("qk96_mlp0_512", "cev_mixed96_context_qk_mlp0_context_p512_p640.pt", "r512",
         "mixed96_context_qk_mlp0_context_p512_p640_ood_results.json", "512", 96, 512),
        ("qk96_mlp0_448", "cev_mixed96_context_qk_mlp0_context_p448.pt", None,
         "mixed96_context_qk_mlp0_context_p448_ood_results.json", None, 96, 448),
        ("qk56_mlp0_512", "cev_mixed56_context_qk_mlp0_context_p512.pt", None,
         "mixed56_context_qk_mlp0_context_p512_ood_results.json", None, 56, 512),
    ]
    tail = json.loads((ROOT / "context_metric_tail_waterfilling_law_results.json").read_text())
    mlp_damage = {int(rank): float(value) for rank, value in tail["mlp0"]["measured_component_damage"].items()}
    heldout = []
    for name, cev_file, cev_key, receipt_file, arm_key, q_rank, m_rank in target_specs:
        cev_object = torch.load(ROOT / cev_file, map_location="cpu")
        cev = cev_object[cev_key] if cev_key is not None else cev_object
        receipt = json.loads((ROOT / receipt_file).read_text())
        arm = receipt["arms"][arm_key] if arm_key is not None else receipt
        actual_scale, actual_count = project(cev)
        assert actual_count == int(arm["certificates_valid"])
        actual_damage = float(arm["census_damage"])
        scale_from_measured_damage = float(_predict_scale(beta, torch.tensor(actual_damage, dtype=torch.float64)))
        count_from_measured_damage = _count(scale_from_measured_damage, shape)
        component_sum = float(qk_damage[str(q_rank)]) + mlp_damage[m_rank]
        tax = actual_damage / component_sum
        end_to_end_damage = TAX_CENTER * component_sum
        end_to_end_scale = float(_predict_scale(beta, torch.tensor(end_to_end_damage, dtype=torch.float64)))
        heldout.append({
            "name": name,
            "qk_rank": q_rank,
            "mlp0_rank": m_rank,
            "actual_damage": actual_damage,
            "actual_scale": actual_scale,
            "actual_certificates": actual_count,
            "scale_from_measured_damage": scale_from_measured_damage,
            "relative_scale_error": abs(scale_from_measured_damage / actual_scale - 1.0),
            "certificates_from_measured_damage": count_from_measured_damage,
            "certificate_error_from_measured_damage": abs(count_from_measured_damage - actual_count),
            "component_sum": component_sum,
            "measured_interaction_tax": tax,
            "end_to_end_predicted_damage": end_to_end_damage,
            "end_to_end_predicted_scale": end_to_end_scale,
            "end_to_end_predicted_certificates": _count(end_to_end_scale, shape),
            "end_to_end_certificate_error": abs(_count(end_to_end_scale, shape) - actual_count),
        })

    actual_heldout_scale = torch.tensor([row["actual_scale"] for row in heldout], dtype=torch.float64)
    predicted_heldout_scale = torch.tensor([row["scale_from_measured_damage"] for row in heldout], dtype=torch.float64)
    heldout_log_r2 = _log_r2(actual_heldout_scale, predicted_heldout_scale)
    heldout_median_relative = float(torch.tensor([row["relative_scale_error"] for row in heldout]).median())
    heldout_max_relative = max(row["relative_scale_error"] for row in heldout)
    measured_count_mae = sum(row["certificate_error_from_measured_damage"] for row in heldout) / len(heldout)
    end_to_end_count_mae = sum(row["end_to_end_certificate_error"] for row in heldout) / len(heldout)
    all_taxes_covered = all(TAX_LO <= row["measured_interaction_tax"] <= TAX_HI for row in heldout)

    # Use the worst truly held-out scale miss, in the harmful direction, as a
    # finite-sample uncertainty factor.  This is conservative only for this
    # explicitly enumerated calibrated grid.
    uncertainty_factor = 1.0 + heldout_max_relative
    enumeration = []
    for q_rank in sorted(PURE_QK_PRICE, reverse=True):
        for m_rank in (None, 640, 512, 448, 384, 256):
            if m_rank is None:
                central_damage = float(qk_damage[str(q_rank)])
                upper_damage = central_damage
            else:
                component_sum = float(qk_damage[str(q_rank)]) + mlp_damage[m_rank]
                central_damage = TAX_CENTER * component_sum
                upper_damage = TAX_HI * component_sum
            central_scale = float(_predict_scale(beta, torch.tensor(central_damage, dtype=torch.float64)))
            conservative_scale = float(_predict_scale(beta, torch.tensor(upper_damage, dtype=torch.float64))) * uncertainty_factor
            row = {
                "qk_rank": q_rank,
                "mlp0_rank": "native" if m_rank is None else m_rank,
                "literal_standalone_scalars": PURE_QK_PRICE[q_rank] - MLP0_SAVING[m_rank],
                "central_predicted_damage": central_damage,
                "central_predicted_scale": central_scale,
                "central_predicted_certificates": _count(central_scale, shape),
                "conservative_upper_scale": conservative_scale,
                "conservative_certificate_lower_bound": _count(conservative_scale, shape),
            }
            row["strictly_improves_43_certificate_frontier"] = (
                row["literal_standalone_scalars"] < FRONTIER_SCALARS
                and row["conservative_certificate_lower_bound"] >= FRONTIER_CERTIFICATES
            )
            enumeration.append(row)

    improving = [row for row in enumeration if row["strictly_improves_43_certificate_frontier"]]
    cheaper = sorted((row for row in enumeration if row["literal_standalone_scalars"] < FRONTIER_SCALARS),
                     key=lambda row: (-row["conservative_certificate_lower_bound"],
                                      row["literal_standalone_scalars"]))
    tier_summary = {}
    for bar in (43, 40, 38, 34, 30):
        eligible = [row for row in enumeration if row["conservative_certificate_lower_bound"] >= bar]
        tier_summary[str(bar)] = min(eligible, key=lambda row: row["literal_standalone_scalars"]) if eligible else None

    pred_a = heldout_log_r2 >= .90 and heldout_median_relative <= .10 and measured_count_mae <= 3.0
    pred_b = all_taxes_covered and end_to_end_count_mae <= 4.0
    pred_c = bool(improving) or not any(row["strictly_improves_43_certificate_frontier"] for row in enumeration)
    null = heldout_log_r2 < .70 or end_to_end_count_mae > 6.0
    result = {
        "status": "certificate_constrained_waterfilling_allocator_complete",
        "rung": 357,
        "claim_level": "cpu_crossvalidated_discrete_grid_allocator_and_no_improvement_screen",
        "training_programs": train,
        "intensity_law": {
            "formula": "log(scale)=intercept+exponent*log(census_damage)",
            "intercept": float(beta[0]),
            "exponent": float(beta[1]),
        },
        "heldout_qk_mlp_family": heldout,
        "heldout_log_scale_r2": heldout_log_r2,
        "heldout_median_relative_scale_error": heldout_median_relative,
        "heldout_max_relative_scale_error": heldout_max_relative,
        "heldout_certificate_mae_from_measured_damage": measured_count_mae,
        "cross_family_tax_envelope": [TAX_LO, TAX_HI],
        "cross_family_tax_center": TAX_CENTER,
        "all_measured_taxes_covered": all_taxes_covered,
        "end_to_end_heldout_certificate_mae": end_to_end_count_mae,
        "conservative_scale_uncertainty_factor": uncertainty_factor,
        "frontier_constraint": {"scalars_less_than": FRONTIER_SCALARS,
                                "certificates_at_least": FRONTIER_CERTIFICATES},
        "enumerated_grid": enumeration,
        "strict_frontier_improvers": improving,
        "best_cheaper_rows_by_conservative_certificate_count": cheaper[:8],
        "minimum_price_by_conservative_certificate_bar": tier_summary,
        "discrete_no_improvement_at_43": len(improving) == 0,
        'pred_a_intensity_transfers_to_heldout_compositions': bool(pred_a),
        'pred_b_tail_plus_tax_predicts_end_to_end_counts': bool(pred_b),
        'pred_c_discrete_frontier_is_resolved': bool(pred_c),
        "null_allocator_is_not_predictive": bool(null),
        "scope_warning": "No theorem outside the enumerated calibrated ranks or QK/MLP0 families; conservative bound is empirical heldout error, not a probabilistic guarantee.",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "heldout_log_scale_r2": heldout_log_r2,
        "heldout_median_relative_scale_error": heldout_median_relative,
        "heldout_certificate_mae_from_measured_damage": measured_count_mae,
        "taxes": [row["measured_interaction_tax"] for row in heldout],
        "end_to_end_certificate_mae": end_to_end_count_mae,
        "strict_frontier_improvers": improving,
        "tier_summary": tier_summary,
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("CERTIFICATE-CONSTRAINED WATER-FILLING ALLOCATOR DONE", flush=True)


if __name__ == "__main__":
    main()
