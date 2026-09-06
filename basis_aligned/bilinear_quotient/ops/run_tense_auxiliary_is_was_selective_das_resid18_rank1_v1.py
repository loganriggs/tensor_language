#!/usr/bin/env python3
"""Selective rank-one is/was writer and conditional reciprocal-transfer test."""

# BQGATE: EXPERIMENT pred_a_authority_toy_head_and_rank pred_b_heldout_is_was_a pred_c_heldout_is_was_selectivity pred_d_shared_writer_geometry pred_e_reciprocal_has_had_transfer pred_f_exact_coverage
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_aspectual_different_readout_is_was_v2 as v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as v3
import circuit_candidate_aspectual_lexical_holdout_v5 as has_had
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_selective_das_resid18_rank1_v1.json"
TOY = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_toy_result.json"
V2_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
V3_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
UNCONSTRAINED = ROOT / "circuits/followups/tense_auxiliary_is_was_das_resid18_rank1_transfer_v1_result.json"
HAS_SCREEN = ROOT / "circuits/followups/aspectual_anchor_explicit_path_lexical_holdout_v2_result.json"
HAS_BASIS = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
HAS_BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
DAS_LIBRARY = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.selective_das_resid18_rank1_v1"
SITE = "resid:18"
RANK = 1
SEED = 0
STEPS_PER_BLOCK = 50
RHO_SCHEDULE = (1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0, 32.0, 32.0)
STEPS = STEPS_PER_BLOCK * len(RHO_SCHEDULE)
LR = 0.05
CONSTRAINT = 0.10
MODEL_FORWARDS_MAX = 27
EXAMPLE_EVALUATIONS_MAX = 392
EXPECTED_PRIOR_SHA256 = "11750a8d9fa69a71d2d32f3979a37c8c11335d7bd6a71fbb7a6185c83283c2b9"
EXPECTED = {
    TOY: "f9a617782656bd1ae90c94fa503987fc374e8a8d5a29f2f5cd33c8f546174e9a",
    V2_CAP: "f76fbcd6174cc9e8e3f77352ee5461815156cdae421ecc43a0e0c3576b63af7e",
    V3_CAP: "744d2fd3c8200ca00005357961df3d435a7a13dbdbd7c3a51a487daee76acec3",
    UNCONSTRAINED: "faefc825d3d481fd1a73fdbbde36c1daaa8b856dfc5590c6ee4f45540c3a9330",
    HAS_SCREEN: "fd1b4ae15e1d327001c8b172bcbecb0f15609d6da01bec8c8dddbf8de107549e",
    HAS_BASIS: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    V2_BUILDER: "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    V3_BUILDER: "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    HAS_BUILDER: "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859",
    DAS_LIBRARY: "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(rows):
    return {family: [row for row in rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    toy = json.loads(TOY.read_text())
    caps = [json.loads(V2_CAP.read_text()), json.loads(V3_CAP.read_text())]
    old = json.loads(UNCONSTRAINED.read_text())
    has_screen = json.loads(HAS_SCREEN.read_text())
    has_basis = json.loads(HAS_BASIS.read_text())
    rows2, rows3, rows_h = v2.build_rows(), v3.build_rows(), has_had.build_rows()
    config_ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["frozen_design"]["rank"] == RANK
        and prior["frozen_design"]["seed"] == SEED
        and toy.get("passed") is True
        and toy["optimizer"]["rho_schedule"] == list(RHO_SCHEDULE)
        and toy["optimizer"]["steps_per_block"] == STEPS_PER_BLOCK
        and toy["optimizer"]["lr"] == LR
        and toy["optimizer"]["constraint"] == CONSTRAINT
        and all(result.get("terminal") == "screen" for result in caps)
        and all(all(cell["passed"] for cell in result["capability_cells"]) for result in caps)
        and old.get("terminal") == "null"
        and old["predictions"]["pred_e_same_answer_selectivity"] is False
        and has_screen.get("terminal") == "screen"
        and has_basis.get("terminal") == "screen"
        and has_basis["basis"]["shape"] == [1152, 1]
        and has_basis["basis"]["sha256"] == "123c6e098fcccf68bd9b881bb81c6b95858a258baa688b79a947a3043bb61e39"
        and v2.validate_rows(rows2) == "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911"
        and v3.validate_rows(rows3) == "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2"
        and has_had.validate_rows(rows_h) == "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
    )
    if not config_ok:
        raise ExperimentError("authority, capability, toy, prior, rows, or fixed optimizer changed")
    return rows2, rows3, rows_h, has_basis


def margins(backend, x, rows):
    torch = backend.torch
    idx = torch.arange(len(rows), device=backend.device)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device, dtype=torch.long)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device, dtype=torch.long)
    logits = das.head_logits(backend, x)
    return logits[idx, answer] - logits[idx, foil]


def fit_selective(backend, base_a, donor_a, rows_a, base_p, donor_p, rows_p, base_c, donor_c, rows_c, scale):
    torch = backend.torch
    torch.manual_seed(SEED)
    raw = (0.02 * torch.randn(base_a.shape[1], 1, device=backend.device, dtype=base_a.dtype)).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)
    multipliers = torch.zeros(2, device=backend.device, dtype=base_a.dtype)
    delta_a, delta_p, delta_c = (donor_a - base_a).detach(), (donor_p - base_p).detach(), (donor_c - base_c).detach()
    with torch.no_grad():
        target_a = margins(backend, donor_a, rows_a).detach()
        base_margin_p = margins(backend, base_p, rows_p).detach()
        base_margin_c = margins(backend, base_c, rows_c).detach()
    block_receipts = []
    for rho in RHO_SCHEDULE:
        for _ in range(STEPS_PER_BLOCK):
            optimizer.zero_grad()
            q = torch.linalg.qr(raw, mode="reduced")[0]
            patched_a = base_a + (delta_a @ q) @ q.T
            patched_p = base_p + (delta_p @ q) @ q.T
            patched_c = base_c + (delta_c @ q) @ q.T
            a_loss = (((margins(backend, patched_a, rows_a) - target_a) / scale) ** 2).mean()
            effects = torch.stack(
                (
                    (margins(backend, patched_p, rows_p) - base_margin_p).abs().mean() / scale,
                    (margins(backend, patched_c, rows_c) - base_margin_c).abs().mean() / scale,
                )
            )
            violation = effects - CONSTRAINT
            shifted = torch.relu(violation + multipliers / rho)
            loss = a_loss + (0.5 * rho * shifted.square() - multipliers.square() / (2.0 * rho)).sum()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            q = torch.linalg.qr(raw, mode="reduced")[0]
            patched_p = base_p + (delta_p @ q) @ q.T
            patched_c = base_c + (delta_c @ q) @ q.T
            effects = torch.stack(
                (
                    (margins(backend, patched_p, rows_p) - base_margin_p).abs().mean() / scale,
                    (margins(backend, patched_c, rows_c) - base_margin_c).abs().mean() / scale,
                )
            )
            multipliers = torch.clamp(multipliers + rho * (effects - CONSTRAINT), min=0.0)
            block_receipts.append({"rho": rho, "p_effect": float(effects[0]), "c_effect": float(effects[1]), "lambda_p": float(multipliers[0]), "lambda_c": float(multipliers[1])})
    with torch.no_grad():
        q = torch.linalg.qr(raw, mode="reduced")[0]
    return q.detach(), block_receipts


def capture_groups(backend, groups):
    names, rows, spans = [], [], {}
    for name, group_rows in groups:
        start = len(rows)
        rows.extend(group_rows)
        spans[name] = slice(start, len(rows))
        names.append(name)
    base, donor, _ = das.capture_site(backend, rows, SITE)
    return rows, base, donor, spans


def evaluate_groups(backend, groups, q, scale):
    rows, base, donor, spans = capture_groups(backend, groups)
    report = {}
    for name, family_rows in groups:
        span = spans[name]
        if family_rows[0]["family"] in ("A1", "A2"):
            mean, absolute, count = das.subspace_recovery(
                backend, base[span], donor[span], q,
                [row["donor_answer_id"] for row in family_rows], [row["donor_foil_id"] for row in family_rows],
            )
            report[name] = {"mean_recovery": mean, "mean_absolute_recovery": absolute, "rows": count}
        else:
            effect, count = das.subspace_same_answer_effect(
                backend, base[span], donor[span], q,
                [row["donor_answer_id"] for row in family_rows], [row["donor_foil_id"] for row in family_rows], scale,
            )
            report[name] = {"same_answer_effect": effect, "rows": count}
    return report, 2, 2 * len(rows)


def main():
    rows2, rows3, rows_h, has_basis_result = validate_static()
    plan = {
        "schema": "tense_auxiliary_is_was_selective_das_resid18_rank1_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "site": SITE, "rank": RANK, "seed": SEED,
        "steps": STEPS, "steps_per_block": STEPS_PER_BLOCK, "rho_schedule": list(RHO_SCHEDULE),
        "lr": LR, "constraint": CONSTRAINT, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "transformer_backwards": 0,
        "model_updates": 0, "final_head_gradient_steps": STEPS, "fit_parameters": 1152,
        "rank_grid_points": 0, "penalty_grid_points": 0, "seed_grid_points": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    by2, by3, by_h = group(rows2), group(rows3), group(rows_h)
    fit_groups = [("fit_A1", by2["A1"][:8]), ("fit_P", by2["P"][:8]), ("fit_C", by2["C"][:8])]
    fit_rows, fit_base, fit_donor, fit_spans = capture_groups(backend, fit_groups)
    forward_calls, evaluations = 2, 2 * len(fit_rows)
    head_ok, head_error = das.verify_head(backend, by2["A1"][:8], SITE)
    forward_calls += 1
    evaluations += 8
    if not head_ok:
        raise ExperimentError(f"exact head verification failed: {head_error}")
    a_span, p_span, c_span = fit_spans["fit_A1"], fit_spans["fit_P"], fit_spans["fit_C"]
    scale = das.target_scale(
        backend, fit_base[a_span], fit_donor[a_span],
        [row["donor_answer_id"] for row in by2["A1"][:8]], [row["donor_foil_id"] for row in by2["A1"][:8]],
    )
    q, blocks = fit_selective(
        backend,
        fit_base[a_span], fit_donor[a_span], by2["A1"][:8],
        fit_base[p_span], fit_donor[p_span], by2["P"][:8],
        fit_base[c_span], fit_donor[c_span], by2["C"][:8], scale,
    )

    v2_groups = [("v2_A1_heldout", by2["A1"][8:]), ("v2_A2", by2["A2"]), ("v2_P_heldout", by2["P"][8:]), ("v2_C_heldout", by2["C"][8:])]
    v3_groups = [(f"v3_{family}", by3[family]) for family in ("A1", "A2", "P", "C")]
    report, calls, examples = evaluate_groups(backend, v2_groups, q, scale)
    forward_calls += calls
    evaluations += examples
    report3, calls, examples = evaluate_groups(backend, v3_groups, q, scale)
    report.update(report3)
    forward_calls += calls
    evaluations += examples
    report["target_scale"] = scale

    pred_b = all(report[name]["mean_absolute_recovery"] >= 0.50 for name in ("v2_A1_heldout", "v2_A2", "v3_A1", "v3_A2"))
    pred_c = all(report[name]["same_answer_effect"] <= 0.20 for name in ("v2_P_heldout", "v2_C_heldout", "v3_P", "v3_C"))
    shared = None
    reciprocal = None
    pred_d = None
    pred_e = None
    if pred_b and pred_c:
        q_has = backend.torch.as_tensor(has_basis_result["basis"]["values_column_major"], device=backend.device, dtype=q.dtype).reshape(1152, 1)
        cosine = float((q.T @ q_has).abs().item())
        has_groups = [(f"has_had_v5_{family}", by_h[family]) for family in ("A1", "A2", "P", "C")]
        reciprocal, calls, examples = evaluate_groups(backend, has_groups, q, scale)
        forward_calls += calls
        evaluations += examples
        pred_d = cosine >= 0.50
        pred_e = (
            reciprocal["has_had_v5_A1"]["mean_absolute_recovery"] >= 0.25
            and reciprocal["has_had_v5_A2"]["mean_absolute_recovery"] >= 0.25
            and reciprocal["has_had_v5_P"]["same_answer_effect"] <= 0.20
            and reciprocal["has_had_v5_C"]["same_answer_effect"] <= 0.20
        )
        shared = {"absolute_cosine_with_q_has": cosine, "q_has_basis_sha256": has_basis_result["basis"]["sha256"]}

    basis = q.detach().cpu().reshape(-1)
    values = [float(value) for value in basis]
    basis_sha256 = hashlib.sha256(basis.numpy().tobytes()).hexdigest()
    config_ok = STEPS == 600 and RHO_SCHEDULE == (1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0, 32.0, 32.0) and LR == 0.05 and CONSTRAINT == 0.10
    pred_a = head_ok and head_error <= 1.0e-3 and q.shape == (1152, 1) and config_ok
    pred_f = forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and len(values) == 1152
    predictions = {
        "pred_a_authority_toy_head_and_rank": pred_a,
        "pred_b_heldout_is_was_a": pred_b,
        "pred_c_heldout_is_was_selectivity": pred_c,
        "pred_d_shared_writer_geometry": pred_d,
        "pred_e_reciprocal_has_had_transfer": pred_e,
        "pred_f_exact_coverage": pred_f,
    }
    terminal = "screen" if pred_a and pred_b and pred_c and pred_f else ("null" if pred_a and pred_f else "invalid")
    if terminal == "screen" and pred_d and pred_e:
        reason = "selective_rank1_shared_writer_and_reciprocal_transfer"
    elif terminal == "screen":
        reason = "selective_rank1_writer_identified_shared_writer_tests_miss_or_inconclusive"
    elif terminal == "null":
        reason = "rank1_writer_fails_heldout_recovery_or_selectivity"
    else:
        reason = "authority_toy_head_rank_optimizer_or_coverage_invalid"
    result = {
        "schema": "tense_auxiliary_is_was_selective_das_resid18_rank1_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "site": SITE, "rank": RANK,
        "optimizer": {"seed": SEED, "steps": STEPS, "steps_per_block": STEPS_PER_BLOCK, "rho_schedule": list(RHO_SCHEDULE), "lr": LR, "constraint": CONSTRAINT, "blocks": blocks},
        "head_verification": {"passed": head_ok, "max_abs_difference": head_error},
        "basis": {"shape": [1152, 1], "dtype": "float32", "sha256": basis_sha256, "values_column_major": values},
        "score": {"families": report, "forward_calls": forward_calls, "example_evaluations": evaluations, "transformer_backwards": 0, "model_updates": 0, "final_head_gradient_steps": STEPS, "fit_parameters": 1152},
        "shared_writer": shared, "reciprocal_has_had": reciprocal,
        "predictions": predictions, "terminal": terminal, "reason": reason,
        "rank_policy": "Do not increase rank after a null.",
        "next_action": "interpret shared-writer geometry and reciprocal transfer" if terminal == "screen" else "retain asymmetric boundary and do not raise rank",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "families": report, "shared_writer": shared, "reciprocal_has_had": reciprocal, "basis_sha256": basis_sha256, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
