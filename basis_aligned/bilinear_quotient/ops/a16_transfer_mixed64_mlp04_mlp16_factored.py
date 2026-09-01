"""RUNG 393 -- ORIGINAL-NATIVE SIGNED GATE FOR THE PHYSICAL RUNG392 COMPOSITE.

Conditional on every rung392 prediction holding, apply the identical fixed
original-native layer-16 attention mean ablation to the exact fp32 structural
QK64 + split-B MLP{0,4}@p768 + shipped 14,984-value factored-MLP16 program.
The original-native knockout was measured before any replacement.  Candidate
effect is knockout CE minus rung392's own saved unablated CE.

Frozen predictions
------------------
pred_a: reproduce the rung392 baseline identity: census <=.070, >=10 checks,
    shifted full-native max <=.220, conditional fresh max <=.040, QK64/440
    fp32 maps, MLP{0,4}@p768, exact four-tensor/no-dense L16 program, live
    factor and ablation hooks, and 495,847,230 / 1,867,449,228 bills.
pred_b: signed candidate-vs-original-native effect cosine >=.95, normalized
    error <=.40, and norm ratio in [.70,1.30].
pred_c: collateral circuit-effect Spearman >=.95 and the median magnitude
    ratio on a16-owned behaviors lies in [.70,1.30].

Strong null: cosine <.70, collateral Spearman <.75, inert factor/ablation
hook, or failed baseline identity.  A full pass adopts the physical composite
as a lower-fidelity manipulable tier.  A miss leaves rung392 predictive and
composable but not causally adopted.  No rank/site/metric/precision tuning.
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
OUT = ROOT / "a16_transfer_mixed64_mlp04_mlp16_factored_results.json"
BASE_RESULT = ROOT / "mixed64_mlp04_mlp16_factored_composition_results.json"
BASE_CEV = ROOT / "cev_mixed64_mlp04_mlp16_factored.pt"
NATIVE_KO = ROOT / "cev_a16ko_original_native_universal_bf16_qk56_gate.pt"
COMP_KO = ROOT / "cev_a16ko_mixed64_mlp04_mlp16_factored.pt"
FACTOR_PROGRAM = ROOT / "mlp16_rank2_quadratic_factored.pt"
FIT_CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
QK_FIT = (72, 96)
MLP_FIT = (24, 48)
MLP_LAYERS = (0, 4)
QK_LAYERS = tuple(range(2, 18))
QK_RANK = 64
MLP_RANK = 768
SCALARS = 495_847_230
BYTES = 1_867_449_228
D = 1152
R = 4
K = 2


def _spearman(left: list[float], right: list[float]) -> float:
    a = torch.tensor(left).argsort().argsort().float()
    b = torch.tensor(right).argsort().argsort().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def _factored_prediction(x: torch.Tensor,
                         program: dict[str, torch.Tensor]) -> torch.Tensor:
    projections = torch.einsum("...d,rkd->...rk", x.float(), program["form_vectors"])
    coefficients = (projections.square() * program["form_values"]).sum(-1)
    return coefficients @ program["output_directions"] + program["constant"]


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (BASE_RESULT, BASE_CEV, NATIVE_KO, FACTOR_PROGRAM, FIT_CACHE,
                     ROOT / "census_state_diverse.pt", ROOT / "circuits/BATTERY.json"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert all(baseline[key] for key in (
            "pred_a_physical_lower_fidelity_holds",
            "pred_b_additive_composition_law_holds",
            "pred_c_shifted_and_conditional_fresh_hold",
            "pred_d_shipped_program_identity_and_bills_hold"))
        assert baseline["signed_composite_gate_licensed"]
        assert baseline["literal_standalone_scalars"] == SCALARS
        assert baseline["literal_source_format_bytes"] == BYTES
        print("A16 PHYSICAL COMPOSITE | dry run: parent, factors, fits, bills valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    import mlp_late_context_metric_shared_input_screen as M
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    CN.use_state(str(ROOT / "census_state_diverse.pt"))
    rows = CN.rows().cpu().long()[:, :257].contiguous()
    base = CN.base_ce().float().reshape(-1).cpu()
    nflat = CN.nflat()
    baseline_result = json.loads(BASE_RESULT.read_text())
    baseline = torch.load(BASE_CEV, map_location="cpu").float().reshape(-1)
    native_ko = torch.load(NATIVE_KO, map_location="cpu").float().reshape(-1)
    assert base.numel() == baseline.numel() == native_ko.numel() == nflat

    # The fixed attention-16 replacement value is measured in the original
    # native model, before any candidate hook or fitted replacement is active.
    capture = {"sum": torch.zeros(D, device=C.DEV), "n": 0}

    def capture_mean(_module, _inputs, output):
        values = output[0].detach().float().reshape(-1, D)
        capture["sum"] += values.sum(0)
        capture["n"] += values.shape[0]

    handle = C.m.transformer.h[16].attn.register_forward_hook(capture_mean)
    for start in range(0, 128, 4):
        index = C.FW[start:start + 4, :256].to(C.DEV)
        x = F.rms_norm(C.m.transformer.wte(index), (D,))
        x0, value0 = x, None
        for block in C.m.transformer.h:
            x, value0 = block(x, value0, x0)
    handle.remove()
    mean_value = (capture["sum"] / capture["n"]).clone()

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    qk_rows = cached[QK_FIT[0]:QK_FIT[1], :257].long().contiguous()
    mlp_rows = cached[MLP_FIT[0]:MLP_FIT[1], :257].long().contiguous()
    qk_covariances = _attention_input_covariances(C.m, qk_rows, _manual_logits)
    M.LAYERS = MLP_LAYERS
    mlp_covariances = M._covariances(C.m, mlp_rows, _manual_logits)
    mlp_programs = {}
    for layer in MLP_LAYERS:
        program, _basis, _diag = M._rrr_program(
            C.m.transformer.h[layer].mlp, mlp_covariances[layer], rank=MLP_RANK)
        mlp_programs[layer] = {name: value.cpu() for name, value in program.items()}
        del program, _basis
    del mlp_covariances
    torch.cuda.empty_cache()

    factor_cpu = torch.load(FACTOR_PROGRAM, map_location="cpu")
    factor = {key: value.to(C.DEV) for key, value in factor_cpu.items()}
    expected_shapes = {
        "output_directions": [R, D], "form_vectors": [R, K, D],
        "form_values": [R, K], "constant": [D]}
    factor_shapes = {key: list(value.shape) for key, value in factor_cpu.items()}
    factor_scalars = sum(value.numel() for value in factor_cpu.values())
    factor_dtypes = sorted({str(value.dtype) for value in factor_cpu.values()})
    no_dense_form = "forms" not in factor_cpu and set(factor_cpu) == set(expected_shapes)

    observed = {"factor": 0, "ablation": 0}

    def factor_hook(_module, args, output):
        observed["factor"] += 1
        return _factored_prediction(args[0], factor).to(output.dtype)

    def ablate_hook(_module, _inputs, output):
        if not C.SEL.get("abl_on"):
            return None
        observed["ablation"] += 1
        values, value0 = output
        return mean_value.expand_as(values).to(values.dtype), value0

    C.CROWS, C.CBASE, C.NFLAT = rows, base, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": QK_RANK, "qk_rmap": {},
        "qk_extra_tail": 0, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "qk_context_covariances": qk_covariances,
        "qk_factor_storage_dtype": None,
        "final_mlp_input_programs": mlp_programs,
        "ablate_on_census": True,
        "_ablh": ablate_hook,
    })
    factor_handle = C.m.transformer.h[16].mlp.register_forward_hook(factor_hook)
    try:
        print("ARM: QK64 + MLP04 + factored-MLP16 with a16 mean ablated", flush=True)
        run = C.main()
        compiled_ko = C.SEL["cev"].float().reshape(-1).cpu()
    finally:
        factor_handle.remove()
    torch.save(compiled_ko, COMP_KO)

    effect_candidate = compiled_ko - baseline
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

    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(item[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for item in factors}
    factor_maps = sum(4 * len(heads) for heads in qk.values())
    qk_metric = C.SEL.get("_QK_METRIC")
    qk_context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    mlp_observed = {int(key): int(value) for key, value in
                    C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    active = tuple(C.SEL.get("_ORDER2", ()))
    wanted_indices = tuple(range(QK_RANK))
    identity = (
        factor_shapes == expected_shapes and factor_scalars == 14_984
        and factor_dtypes == ["torch.float32"] and no_dense_form
        and observed["factor"] > 0 and observed["ablation"] > 0
        and qk_metric == "context_rrr" and qk_context_layers == QK_LAYERS
        and set(index_sets) == set(QK_LAYERS)
        and all(value == wanted_indices for value in index_sets.values())
        and widths == {QK_RANK} and factor_maps == 440
        and mlp_observed == {0: MLP_RANK, 4: MLP_RANK}
        and not any(name in active for name in ("a0", "a1v", "tailE"))
        and baseline_result["literal_standalone_scalars"] == SCALARS
        and baseline_result["literal_source_format_bytes"] == BYTES)
    pred_a = (
        baseline_result["physical_census_damage"] <= .070
        and baseline_result["physical_certificates"] >= 10
        and baseline_result["shifted_full_native_relative_max"] <= .220
        and baseline_result["conditional_fresh8_max"] <= .040
        and run["L2_F"] <= .040 and identity)
    pred_b = cosine >= .95 and normalized_error <= .40 and .70 <= norm_ratio <= 1.30
    pred_c = collateral_rho >= .95 and .70 <= own_median <= 1.30
    inert = float(effect_candidate.abs().mean()) < 1e-8
    strong_null = (
        cosine < .70 or collateral_rho < .75 or inert
        or observed["factor"] == 0 or observed["ablation"] == 0 or not identity)
    adopted = bool(pred_a and pred_b and pred_c and not strong_null)
    result = {
        "status": "a16_transfer_mixed64_mlp04_mlp16_factored_complete",
        "rung": 393,
        "claim_level": "original_native_signed_gate_for_physical_three_family_composite",
        "convention": "signed effect = KO CE minus unablated CE within original-native and candidate models",
        "native_ko_measured_before_candidate_installation": True,
        "unablated_census_damage": baseline_result["physical_census_damage"],
        "unablated_certificates_valid": baseline_result["physical_certificates"],
        "unablated_shifted_damage_max": baseline_result["shifted_full_native_relative_max"],
        "live_conditional_fresh_damage": run["L2_F"],
        "effect_cosine": cosine,
        "effect_normalized_error": normalized_error,
        "effect_norm_ratio": norm_ratio,
        "effect_candidate_mean_abs": float(effect_candidate.abs().mean()),
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "hook_calls": observed,
        "qk_metric": qk_metric,
        "qk_context_layers": list(qk_context_layers),
        "qk_rank": QK_RANK,
        "qk_factorized_maps": factor_maps,
        "qk_factor_widths": sorted(widths),
        "mlp_input_program_observed": mlp_observed,
        "active_replacements": list(active),
        "mlp16_factor_shapes": factor_shapes,
        "mlp16_factor_scalars": factor_scalars,
        "mlp16_factor_dtypes": factor_dtypes,
        "mlp16_no_dense_form": no_dense_form,
        "literal_standalone_scalars": SCALARS,
        "literal_source_format_bytes": BYTES,
        "saved_candidate_ko_cev": COMP_KO.name,
        "pred_a_baseline_physical_identity_and_hooks_hold": bool(pred_a),
        "pred_b_original_native_signed_effect_holds": bool(pred_b),
        "pred_c_circuit_profile_holds": bool(pred_c),
        "null_signed_composite_transport_fails": bool(strong_null),
        "physical_composite_adopted_as_lower_fidelity_tier": adopted,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print("A16 PHYSICAL THREE-FAMILY COMPOSITE DONE", flush=True)


if __name__ == "__main__":
    main()
