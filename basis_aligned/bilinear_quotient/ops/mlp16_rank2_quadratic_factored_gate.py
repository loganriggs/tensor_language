"""RUNG 391 -- PHYSICAL EIGHT-PROJECTION STORAGE AND SIGNED REPRODUCTION.

Post-rung390 audit found that the scored hook retained four dense 1152x1152
quadratic matrices while billing their known rank-2 factorization.  Rebuild the
identical fit-B program, explicitly store each form as two signed projection
vectors plus two eigenvalues, discard the dense fit workspace, and rerun both
the unablated census and original-native signed a16 gate.

Frozen predictions
------------------
pred_a: dense-to-two-factor form relative error <=1e-5, held-out dense/factored
    output relative error <=1e-6, stored shapes exactly [4,1152], [4,2,1152],
    [4,2], [1152], exact 14,984 tensor scalars, no dense form in the physical
    program, and a live factored hook.
pred_b: factored unablated CEV differs from saved dense by max <=5e-4 and mean
    <=2e-5, census is within 2e-4 of .0389782861, and exactly 27/62 checks hold.
pred_c: factored-vs-dense signed-effect cosine >=.999 and normalized error <=.02;
    factored-vs-original-native cosine >=.95, error <=.40, norm ratio [.75,1.30],
    collateral Spearman >=.95, and a16-own median ratio [.70,1.30].

Strong null: form error >1e-3, CEV mean difference >.002, original-native signed
cosine <.70, inert hook, or any dense form retained in the shipped program.  A
full pass restores the literal 14,984-value price and one-composition license;
failure preserves rungs389/390 only as a dense 5,314,176-value causal surrogate.
No rank, site, metric, intervention, or population tuning follows.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp16_rank2_quadratic_factored_gate_results.json"
PARENT_RESULT = ROOT / "a16_transfer_mlp16_rank2_quadratic_results.json"
DENSE_CEV = ROOT / "cev_mlp16_rank2_quadratic_clean.pt"
DENSE_KO = ROOT / "cev_a16ko_mlp16_rank2_quadratic.pt"
NATIVE_KO = ROOT / "cev_a16ko_original_native_universal_bf16_qk56_gate.pt"
FACTORED_CEV = ROOT / "cev_mlp16_rank2_quadratic_factored.pt"
FACTORED_KO = ROOT / "cev_a16ko_mlp16_rank2_quadratic_factored.pt"
FACTORED_PROGRAM = ROOT / "mlp16_rank2_quadratic_factored.pt"
FIT_CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_A = (0, 24)
FIT_B = (24, 48)
D = 1152
R = 4
K = 2
LAYER_PRICE = 14_984
DENSE_LAYER_PRICE = R * D * D + R * D + D
MODEL_PRICE = 529_991_486
EXPECTED_CENSUS = 0.03897828608751297
EXPECTED_CERTIFICATES = 27


def _spearman(left: list[float], right: list[float]) -> float:
    a = torch.tensor(left).argsort().argsort().float()
    b = torch.tensor(right).argsort().argsort().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def _factor_program(dense: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    forms = 0.5 * (dense["forms"].double() + dense["forms"].double().transpose(-1, -2))
    values, vectors = torch.linalg.eigh(forms)
    indices = values.abs().argsort(dim=1, descending=True)[:, :K]
    kept_values = torch.gather(values, 1, indices)
    kept_vectors = torch.gather(
        vectors, 2, indices[:, None, :].expand(R, D, K)).transpose(1, 2)
    return {
        "output_directions": dense["output_directions"].float().contiguous(),
        "form_vectors": kept_vectors.float().contiguous(),
        "form_values": kept_values.float().contiguous(),
        "constant": dense["constant"].float().contiguous(),
    }


def _factored_prediction(x: torch.Tensor,
                         program: dict[str, torch.Tensor]) -> torch.Tensor:
    projections = torch.einsum("...d,rkd->...rk", x.float(), program["form_vectors"])
    coefficients = (projections.square() * program["form_values"]).sum(-1)
    return coefficients @ program["output_directions"] + program["constant"]


@torch.no_grad()
def _ce_vector(model, rows: torch.Tensor, program: dict[str, torch.Tensor],
               observed: dict[str, int]) -> torch.Tensor:
    from mlp16_tucker_physical_calibration import _manual_logits

    def replacement(_module, args, output):
        observed["candidate"] = observed.get("candidate", 0) + 1
        return _factored_prediction(args[0], program).to(output.dtype)

    handle = model.transformer.h[16].mlp.register_forward_hook(replacement)
    losses = []
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index = batch[:, :-1].to(next(model.parameters()).device)
            target = batch[:, 1:].to(index.device)
            logits = _manual_logits(model, index)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                reduction="none").cpu())
    finally:
        handle.remove()
    return torch.cat(losses)


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (PARENT_RESULT, DENSE_CEV, DENSE_KO, NATIVE_KO, FIT_CACHE,
                     ROOT / "census_state_diverse.pt", ROOT / "circuits/BATTERY.json"):
            assert path.exists(), path
        parent = json.loads(PARENT_RESULT.read_text())
        assert all(parent[key] for key in (
            "pred_a_unablated_identity_price_and_hooks_hold",
            "pred_b_original_native_signed_effect_holds",
            "pred_c_circuit_profile_holds"))
        assert LAYER_PRICE == 14_984 and DENSE_LAYER_PRICE == 5_314_176
        assert MODEL_PRICE == 529_991_486
        print("L16 FACTORED QUADRATIC | dry run: parent, correction, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mlp16_rank2_quadratic_current_gate import _build_clean, _prediction
    from mlp16_tucker_physical_calibration import (
        _capture, _certificate_metrics, _manual_logits)

    CN.use_state(str(ROOT / "census_state_diverse.pt"))
    rows = CN.rows().cpu().long()[:, :257].contiguous()
    base = CN.base_ce().float().reshape(-1).cpu()
    dense_cev = torch.load(DENSE_CEV, map_location="cpu").float().reshape(-1)
    dense_ko = torch.load(DENSE_KO, map_location="cpu").float().reshape(-1)
    native_ko = torch.load(NATIVE_KO, map_location="cpu").float().reshape(-1)
    assert base.numel() == dense_cev.numel() == dense_ko.numel() == native_ko.numel() == CN.nflat()

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_a = cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    fit_b = cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    assert fit_a.shape == fit_b.shape == (24, 257)
    xb, yb = _capture(C.m, fit_b)
    dense = _build_clean(C.m, xb, yb, random_output=False)
    program = _factor_program(dense)

    reconstructed = torch.einsum(
        "rk,rki,rkj->rij", program["form_values"].double(),
        program["form_vectors"].double(), program["form_vectors"].double())
    dense_forms = dense["forms"].double()
    form_relative_error = float((reconstructed - dense_forms).norm() / dense_forms.norm())
    xa, _ya = _capture(C.m, fit_a)
    xa = xa.to(next(C.m.parameters()).device)
    dense_prediction = _prediction(xa, dense)
    factored_prediction = _factored_prediction(xa, program)
    prediction_relative_error = float(
        (factored_prediction - dense_prediction).norm() / dense_prediction.norm().clamp_min(1e-30))
    del dense, dense_forms, reconstructed, dense_prediction, factored_prediction, xa
    torch.cuda.empty_cache()

    physical_cpu = {key: value.detach().cpu() for key, value in program.items()}
    torch.save(physical_cpu, FACTORED_PROGRAM)
    shapes = {key: list(value.shape) for key, value in physical_cpu.items()}
    stored_scalars = sum(value.numel() for value in physical_cpu.values())
    no_dense_form = "forms" not in physical_cpu and max(
        value.ndim for value in physical_cpu.values()) <= 3

    capture_sum = torch.zeros(D, device=C.DEV)
    capture_n = 0

    def capture_mean(_module, _inputs, output):
        nonlocal capture_n
        values = output[0].detach().float().reshape(-1, D)
        capture_sum.add_(values.sum(0))
        capture_n += values.shape[0]

    handle = C.m.transformer.h[16].attn.register_forward_hook(capture_mean)
    for start in range(0, 128, 4):
        index = C.FW[start:start + 4, :256].to(C.DEV)
        x = F.rms_norm(C.m.transformer.wte(index), (D,))
        x0, value0 = x, None
        for block in C.m.transformer.h:
            x, value0 = block(x, value0, x0)
    handle.remove()
    mean_value = (capture_sum / capture_n).clone()

    observed: dict[str, int] = {"candidate": 0, "ablation": 0}
    factored_cev = _ce_vector(C.m, rows, program, observed)
    torch.save(factored_cev, FACTORED_CEV)

    def candidate_hook(_module, args, output):
        observed["candidate"] += 1
        return _factored_prediction(args[0], program).to(output.dtype)

    def ablation_hook(_module, _inputs, output):
        observed["ablation"] += 1
        values, value0 = output
        return mean_value.expand_as(values).to(values.dtype), value0

    candidate_handle = C.m.transformer.h[16].mlp.register_forward_hook(candidate_hook)
    ablation_handle = C.m.transformer.h[16].attn.register_forward_hook(ablation_hook)
    try:
        losses = []
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index = batch[:, :-1].to(C.DEV)
            target = batch[:, 1:].to(C.DEV)
            logits = _manual_logits(C.m, index)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                reduction="none").cpu())
        factored_ko = torch.cat(losses)
    finally:
        ablation_handle.remove()
        candidate_handle.remove()
    torch.save(factored_ko, FACTORED_KO)

    effect_factored = factored_ko.float() - factored_cev
    effect_dense = dense_ko - dense_cev
    effect_native = native_ko - base

    def compare(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, float]:
        cosine = float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))
        error = float((left - right).norm() / right.norm().clamp_min(1e-12))
        ratio = float(left.norm() / right.norm().clamp_min(1e-12))
        return cosine, error, ratio

    dense_cosine, dense_error, dense_norm_ratio = compare(effect_factored, effect_dense)
    native_cosine, native_error, native_norm_ratio = compare(effect_factored, effect_native)

    collateral_native, collateral_factored, own_ratios = [], [], []
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        native_abs = float(effect_native[member].abs().mean())
        factored_abs = float(effect_factored[member].abs().mean())
        if receipt["mean_ablation"]["top"][0]["component"] == "a16":
            own_ratios.append(factored_abs / max(native_abs, 1e-12))
        else:
            collateral_native.append(native_abs)
            collateral_factored.append(factored_abs)
    own_ratios.sort()
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = _spearman(collateral_native, collateral_factored)

    ray_receipt = json.loads((ROOT / "certificate_damage_axis_transfer_results.json").read_text())
    metrics = _certificate_metrics(CN, base, factored_cev, battery, ray_receipt)
    cev_difference = factored_cev - dense_cev
    cev_max_difference = float(cev_difference.abs().max())
    cev_mean_difference = float(cev_difference.abs().mean())
    census_damage = float((factored_cev - base).mean())
    identity = (
        shapes == {
            "output_directions": [R, D], "form_vectors": [R, K, D],
            "form_values": [R, K], "constant": [D]}
        and stored_scalars == LAYER_PRICE and no_dense_form
        and observed["candidate"] > 0 and observed["ablation"] > 0 and capture_n > 0
    )
    pred_a = (
        form_relative_error <= 1e-5 and prediction_relative_error <= 1e-6 and identity)
    pred_b = (
        cev_max_difference <= 5e-4 and cev_mean_difference <= 2e-5
        and abs(census_damage - EXPECTED_CENSUS) <= 2e-4
        and metrics["certificates"] == EXPECTED_CERTIFICATES)
    pred_c = (
        dense_cosine >= .999 and dense_error <= .02
        and native_cosine >= .95 and native_error <= .40 and .75 <= native_norm_ratio <= 1.30
        and collateral_rho >= .95 and .70 <= own_median <= 1.30)
    inert = float(effect_factored.abs().mean()) < 1e-8
    strong_null = (
        form_relative_error > 1e-3 or cev_mean_difference > .002
        or native_cosine < .70 or inert or not no_dense_form)
    composition_licensed = bool(pred_a and pred_b and pred_c and not strong_null)
    result = {
        "status": "mlp16_rank2_quadratic_factored_gate_complete",
        "rung": 391,
        "claim_level": "physical_eight_projection_storage_and_signed_reproduction",
        "price_correction": {
            "prior_executed_dense_layer_scalars": DENSE_LAYER_PRICE,
            "factored_layer_scalars": stored_scalars,
            "factored_model_scalars": MODEL_PRICE,
        },
        "stored_program": FACTORED_PROGRAM.name,
        "stored_shapes": shapes,
        "no_dense_form_in_physical_program": no_dense_form,
        "form_relative_error": form_relative_error,
        "heldout_prediction_relative_error": prediction_relative_error,
        "unablated_cev_max_abs_difference_from_dense": cev_max_difference,
        "unablated_cev_mean_abs_difference_from_dense": cev_mean_difference,
        "unablated_census_damage": census_damage,
        "unablated_certificates": metrics["certificates"],
        "factored_vs_dense_signed_cosine": dense_cosine,
        "factored_vs_dense_signed_error": dense_error,
        "factored_vs_dense_signed_norm_ratio": dense_norm_ratio,
        "factored_vs_original_native_signed_cosine": native_cosine,
        "factored_vs_original_native_signed_error": native_error,
        "factored_vs_original_native_signed_norm_ratio": native_norm_ratio,
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "hook_calls": observed,
        "saved_factored_cev": FACTORED_CEV.name,
        "saved_factored_ko_cev": FACTORED_KO.name,
        'pred_a_physical_factorization_identity_holds': bool(pred_a),
        'pred_b_unablated_dense_reproduction_holds': bool(pred_b),
        'pred_c_signed_dense_and_native_reproduction_holds': bool(pred_c),
        "null_physicalization_or_signed_transport_fails": bool(strong_null),
        "composition_licensed": composition_licensed,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print("L16 FACTORED QUADRATIC GATE DONE", flush=True)


if __name__ == "__main__":
    main()
