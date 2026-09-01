"""RUNG 356 -- CERTIFICATE DAMAGE-AXIS TRANSFER FALSIFIER.

Normalize every certificate member damage by its pass threshold, fit a single
ray to the seven saved Q/K rank vectors, and test transfer to a cross-family
QK+MLP program and the context-value family.  This bridges aggregate-CE
water-filling to certificate first-passage constraints without new model work.

Frozen predictions
------------------
pred_a_qk_certificate_vectors_are_one_ray:
    Rank-one vector R2 >=.85 and leave-one-rank projected certificate MAE <=2.
pred_b_cross_family_combo_stays_on_ray:
    QK56+MLP0-p512 cosine >=.90, vector R2 >=.70, certificate error <=3.
pred_c_value_family_stays_on_ray:
    Context-value96 cosine >=.85, vector R2 >=.50, certificate error <=5.

Null: value cosine <.60 or value certificate error >10.  Exact saved-CEV
identities and reported certificate counts must reproduce.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "certificate_damage_axis_transfer_results.json"
QK = {
    96: ("cev_mixed96_context_metric_qk_split_b.pt", "mixed96_context_metric_qk_split_ood_results.json"),
    88: ("cev_mixed88_context_metric_qk.pt", "mixed88_context_metric_qk_ood_results.json"),
    80: ("cev_mixed80_context_metric_qk.pt", "mixed80_context_metric_qk_ood_results.json"),
    72: ("cev_mixed72_context_metric_qk.pt", "mixed72_context_metric_qk_ood_results.json"),
    64: ("cev_mixed64_context_metric_qk.pt", "mixed64_context_metric_qk_ood_results.json"),
    56: ("cev_mixed56_context_metric_qk.pt", "mixed56_context_metric_qk_newcorpus_ood_results.json"),
    48: ("cev_mixed48_context_metric_qk.pt", "mixed48_context_metric_qk_newcorpus_ood_results.json"),
}
TARGETS = {
    "qk56_mlp0_p512": ("cev_mixed56_context_qk_mlp0_context_p512.pt",
                       "mixed56_context_qk_mlp0_context_p512_ood_results.json"),
    "qk80_value96": ("cev_mixed80_context_qk_value96_context.pt",
                     "mixed80_context_qk_value96_context_ood_results.json"),
}


def _shape(matrix):
    _u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
    shape = vh[0]
    if float(shape.sum()) < 0:
        shape = -shape
    scale = matrix @ shape / shape.square().sum().clamp_min(1e-30)
    reconstruction = scale[:, None] * shape[None, :]
    denominator = (matrix - matrix.mean()).square().sum().clamp_min(1e-30)
    r2 = float(1.0 - (matrix - reconstruction).square().sum() / denominator)
    return shape, scale, reconstruction, r2


def _project(vector, shape):
    scale = float(vector @ shape / shape.square().sum().clamp_min(1e-30))
    prediction = scale * shape
    cosine = float(torch.nn.functional.cosine_similarity(vector[None], prediction[None]))
    denominator = (vector - vector.mean()).square().sum().clamp_min(1e-30)
    r2 = float(1.0 - (vector - prediction).square().sum() / denominator)
    actual_count = int((vector < 1.0).sum())
    predicted_count = int((prediction < 1.0).sum())
    return {
        "projection_scale": scale,
        "cosine": cosine,
        "vector_r2": r2,
        "actual_certificates": actual_count,
        "predicted_certificates": predicted_count,
        "certificate_count_error": abs(predicted_count - actual_count),
    }


@torch.no_grad()
def main() -> None:
    needed = [ROOT / "circuits/BATTERY.json", ROOT / "census_state_diverse.pt"]
    for cev, receipt in list(QK.values()) + list(TARGETS.values()):
        needed.extend((ROOT / cev, ROOT / receipt))
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in needed)
        assert sorted(QK) == [48, 56, 64, 72, 80, 88, 96]
        print("CERTIFICATE DAMAGE AXIS | dry run: CEVs, receipts, battery, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN

    CN.use_state("census_state_diverse.pt")
    base = CN.base_ce().float().reshape(-1).cpu()
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    tags = []
    members = []
    thresholds = []
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
    assert len(tags) == 62 and bool((threshold > 0).all())

    def vector(cev_name, receipt_name):
        cev = torch.load(ROOT / cev_name, map_location="cpu").float().reshape(-1)
        damage = cev - base
        value = torch.tensor([float(damage[index].abs().mean()) for index in members],
                             dtype=torch.float64) / threshold
        receipt = json.loads((ROOT / receipt_name).read_text())
        reported = int(receipt["certificates_valid"])
        reproduced = int((value < 1.0).sum())
        if reproduced != reported:
            raise RuntimeError(f"certificate count mismatch {cev_name}: {reproduced} != {reported}")
        return value, reported

    qk_ranks = sorted(QK, reverse=True)
    qk_vectors = []
    qk_counts = []
    for rank in qk_ranks:
        value, count = vector(*QK[rank])
        qk_vectors.append(value)
        qk_counts.append(count)
    matrix = torch.stack(qk_vectors)
    shape, scales, reconstruction, qk_r2 = _shape(matrix)

    leave_one = []
    for heldout, rank in enumerate(qk_ranks):
        keep = torch.arange(len(qk_ranks)) != heldout
        held_shape, _scale, _reconstruction, _r2 = _shape(matrix[keep])
        row = _project(matrix[heldout], held_shape)
        row["rank"] = rank
        leave_one.append(row)
    qk_mae = float(torch.tensor([row["certificate_count_error"] for row in leave_one],
                                dtype=torch.float64).mean())

    targets = {}
    for name, paths in TARGETS.items():
        value, reported = vector(*paths)
        targets[name] = _project(value, shape)
        targets[name]["reported_certificates"] = reported

    combo = targets["qk56_mlp0_p512"]
    value = targets["qk80_value96"]
    pred_a = qk_r2 >= .85 and qk_mae <= 2.0
    pred_b = (combo["cosine"] >= .90 and combo["vector_r2"] >= .70
              and combo["certificate_count_error"] <= 3)
    pred_c = (value["cosine"] >= .85 and value["vector_r2"] >= .50
              and value["certificate_count_error"] <= 5)
    null = value["cosine"] < .60 or value["certificate_count_error"] > 10
    result = {
        "status": "certificate_damage_axis_transfer_complete",
        "rung": 356,
        "claim_level": "saved_cev_certificate_shape_transfer_calibration",
        "normalization": "mean absolute member CEV damage divided by half native ablation",
        "certificate_rule": "normalized member damage < 1",
        "tags": tags,
        "qk": {
            "ranks_high_to_low": qk_ranks,
            "reported_certificates": qk_counts,
            "rank_one_vector_r2": qk_r2,
            "scales": [float(value) for value in scales],
            "leave_one_rank_out": leave_one,
            "leave_one_certificate_mae": qk_mae,
            "shape": [float(value) for value in shape],
        },
        "targets": targets,
        'pred_a_qk_certificate_vectors_are_one_ray': bool(pred_a),
        'pred_b_cross_family_combo_stays_on_ray': bool(pred_b),
        'pred_c_value_family_stays_on_ray': bool(pred_c),
        "null_value_family_rejects_universal_ray": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qk_r2": qk_r2, "qk_loocv_cert_mae": qk_mae,
                      "targets": targets, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("CERTIFICATE DAMAGE AXIS TRANSFER DONE", flush=True)


if __name__ == "__main__":
    main()
