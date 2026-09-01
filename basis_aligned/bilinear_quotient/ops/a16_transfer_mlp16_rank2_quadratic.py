"""RUNG 390 -- ORIGINAL-NATIVE SIGNED A16 GATE FOR THE CLEAN L16 QUADRATIC PROGRAM.

Conditional on all rung389 predictions holding, rebuild the identical fit-B-only
R4/k2 MLP16 program and apply the identical fixed original-native layer-16 attention
mean ablation in native and candidate models.  The original-native KO CEV was measured
before any model replacement; the candidate effect is KO minus its own saved unablated
CEV.  This tests whether the tiny quadratic program preserves a named causal direction,
not merely ordinary prediction.

Frozen predictions
------------------
pred_a: reproduce the rung389 unablated CEV with max absolute error <=1e-5, census
    within 1e-4 of .0389782861, exactly 27/62 checks, exact fit/rank/shape/14,984 price,
    and live candidate plus ablation hooks.
pred_b: signed candidate-vs-original-native effect cosine >=.95, normalized vector
    error <=.40, and norm ratio in [.75,1.30].
pred_c: collateral circuit-effect Spearman >=.95 and the median magnitude ratio on
    a16-owned behaviors lies in [.70,1.30].

Strong null: signed cosine <.70, collateral Spearman <.75, an inert ablation, or failed
unablated identity.  All three positives with null false license exactly one physical
composition with the adopted QK64+MLP0/4 program; failure keeps rung389 as a predictive
structural result only.  No intervention, rank, metric, or population tuning follows.
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
OUT = ROOT / "a16_transfer_mlp16_rank2_quadratic_results.json"
BASE_RESULT = ROOT / "mlp16_rank2_quadratic_current_gate_results.json"
BASE_CEV = ROOT / "cev_mlp16_rank2_quadratic_clean.pt"
NATIVE_KO = ROOT / "cev_a16ko_original_native_universal_bf16_qk56_gate.pt"
COMP_KO = ROOT / "cev_a16ko_mlp16_rank2_quadratic.pt"
FIT_CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_B = (24, 48)
D = 1152
R = 4
K = 2
LAYER_PRICE = 14_984
MODEL_PRICE = 529_991_486
EXPECTED_CENSUS = 0.03897828608751297
EXPECTED_CERTIFICATES = 27


def _spearman(left: list[float], right: list[float]) -> float:
    a = torch.tensor(left).argsort().argsort().float()
    b = torch.tensor(right).argsort().argsort().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (BASE_RESULT, BASE_CEV, NATIVE_KO, FIT_CACHE,
                     ROOT / "census_state_diverse.pt", ROOT / "circuits/BATTERY.json"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert all(baseline[key] for key in (
            "pred_a_legacy_identity_corrected_price_and_clean_identity",
            "pred_b_clean_split_predictive_behavior_and_transfer",
            "pred_c_clean_directly_dominates_rung388_tucker",
            "pred_d_output_directions_beat_random_and_ray_transfers"))
        assert baseline["prices"]["corrected_layer_scalars"] == LAYER_PRICE
        assert baseline["prices"]["program_model_scalars"] == MODEL_PRICE
        print("A16 TRANSFER MLP16 RANK2 QUADRATIC | dry run: parent, identity, bars valid")
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
    nflat = CN.nflat()
    saved_unablated = torch.load(BASE_CEV, map_location="cpu").float().reshape(-1)
    native_ko = torch.load(NATIVE_KO, map_location="cpu").float().reshape(-1)
    assert base.numel() == saved_unablated.numel() == native_ko.numel() == nflat

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_b = cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    assert fit_b.shape == (24, 257)
    xb, yb = _capture(C.m, fit_b)
    program = _build_clean(C.m, xb, yb, random_output=False)

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

    observed = {"candidate": 0, "ablation": 0}

    def candidate_hook(_module, args, output):
        observed["candidate"] += 1
        return _prediction(args[0], program).to(output.dtype)

    def ablate_hook(_module, _inputs, output):
        observed["ablation"] += 1
        values, value0 = output
        return mean_value.expand_as(values).to(values.dtype), value0

    from mlp16_rank2_quadratic_current_gate import _ce_vector
    rebuilt_unablated = _ce_vector(C.m, rows, program, observed, "candidate")
    candidate_handle = C.m.transformer.h[16].mlp.register_forward_hook(candidate_hook)
    ablation_handle = C.m.transformer.h[16].attn.register_forward_hook(ablate_hook)
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
        candidate_ko = torch.cat(losses)
    finally:
        ablation_handle.remove()
        candidate_handle.remove()
    torch.save(candidate_ko, COMP_KO)

    effect_candidate = candidate_ko.float() - saved_unablated
    effect_native = native_ko - base
    cosine = float(torch.dot(effect_candidate, effect_native) /
                   (effect_candidate.norm() * effect_native.norm()).clamp_min(1e-12))
    normalized_error = float((effect_candidate - effect_native).norm() /
                             effect_native.norm().clamp_min(1e-12))
    norm_ratio = float(effect_candidate.norm() / effect_native.norm().clamp_min(1e-12))

    collateral_native, collateral_candidate, own_ratios = [], [], []
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        native_abs = float(effect_native[member].abs().mean())
        candidate_abs = float(effect_candidate[member].abs().mean())
        if receipt["mean_ablation"]["top"][0]["component"] == "a16":
            own_ratios.append(candidate_abs / max(native_abs, 1e-12))
        else:
            collateral_native.append(native_abs)
            collateral_candidate.append(candidate_abs)
    own_ratios.sort()
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = _spearman(collateral_native, collateral_candidate)

    ray_receipt = json.loads((ROOT / "certificate_damage_axis_transfer_results.json").read_text())
    rebuilt_metrics = _certificate_metrics(CN, base, rebuilt_unablated, battery, ray_receipt)
    unablated_max_error = float((rebuilt_unablated - saved_unablated).abs().max())
    unablated_damage = float((rebuilt_unablated - base).mean())
    shapes = {key: list(program[key].shape) for key in ("output_directions", "forms", "constant")}
    identity = (
        shapes == {"output_directions": [R, D], "forms": [R, D, D], "constant": [D]}
        and LAYER_PRICE == 14_984 and MODEL_PRICE == 529_991_486
        and observed["candidate"] > 0 and observed["ablation"] > 0 and capture_n > 0
    )
    pred_a = (
        unablated_max_error <= 1e-5 and abs(unablated_damage - EXPECTED_CENSUS) <= 1e-4
        and rebuilt_metrics["certificates"] == EXPECTED_CERTIFICATES and identity
    )
    pred_b = cosine >= .95 and normalized_error <= .40 and .75 <= norm_ratio <= 1.30
    pred_c = collateral_rho >= .95 and .70 <= own_median <= 1.30
    inert = float(effect_candidate.abs().mean()) < 1e-8
    strong_null = cosine < .70 or collateral_rho < .75 or inert or not pred_a
    composition_licensed = bool(pred_a and pred_b and pred_c and not strong_null)
    result = {
        "status": "a16_transfer_mlp16_rank2_quadratic_complete",
        "rung": 390,
        "claim_level": "original_native_signed_a16_gate_for_clean_l16_quadratic_program",
        "convention": "signed effect = KO CE minus unablated CE within original-native and candidate models",
        "native_ko_measured_before_candidate_installation": True,
        "fit": {"cache": FIT_CACHE.name, "fit_b": list(FIT_B)},
        "ranks": {"output_directions": R, "rank_per_form": K},
        "program_shapes": shapes,
        "literal_layer_scalars": LAYER_PRICE,
        "literal_model_scalars": MODEL_PRICE,
        "unablated_max_abs_reproduction_error": unablated_max_error,
        "unablated_census_damage": unablated_damage,
        "unablated_certificates": rebuilt_metrics["certificates"],
        "effect_cosine": cosine,
        "effect_normalized_error": normalized_error,
        "effect_norm_ratio": norm_ratio,
        "effect_candidate_mean_abs": float(effect_candidate.abs().mean()),
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "hook_calls": observed,
        "saved_candidate_ko_cev": COMP_KO.name,
        'pred_a_unablated_identity_price_and_hooks_hold': bool(pred_a),
        'pred_b_original_native_signed_effect_holds': bool(pred_b),
        'pred_c_circuit_profile_holds': bool(pred_c),
        "null_signed_transport_fails": bool(strong_null),
        "composition_licensed": composition_licensed,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print("A16 TRANSFER MLP16 RANK2 QUADRATIC DONE", flush=True)


if __name__ == "__main__":
    main()
