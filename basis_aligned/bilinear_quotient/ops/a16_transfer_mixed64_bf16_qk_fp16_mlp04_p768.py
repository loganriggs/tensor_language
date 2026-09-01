"""RUNG 369 -- FINAL ORIGINAL-NATIVE SIGNED GATE FOR RUNG368.

Conditional on every rung368 positive, apply the identical fixed native a16
mean ablation to the exact global-BF16, fp16-QK64, selected MLP{0,4}@p768
artifact.  Compare with the original-native KO measured before rounding.

Frozen predictions
------------------
pred_a_final_baseline_identity_and_bill_hold:
    Reproduce rung368 <=.015/43, shifted max<=.120, fresh<=.030, all exact
    source/selection/map/dtype identities, and 511,758,646/1,023,517,292.
pred_b_final_original_native_signed_effect_holds:
    Cosine >=.98, normalized error <=.30, norm ratio [.90,1.15].
pred_c_final_circuit_profile_holds:
    Collateral Spearman >=.98 and a16-own median ratio [.90,1.15].

Null: cosine <.70 or rho <.75.  A full pass formally replaces QK56 as both
semantic-scalar and byte frontier; no tuning follows.
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
OUT = ROOT / "a16_transfer_mixed64_bf16_qk_fp16_mlp04_p768_results.json"
BASE_RESULT = ROOT / "mixed64_bf16_qk_fp16_mlp04_context_p768_ood_results.json"
BASE_CEV = ROOT / "cev_mixed64_bf16_qk_fp16_mlp04_context_p768.pt"
NATIVE_KO = ROOT / "cev_a16ko_original_native_universal_bf16_qk56_gate.pt"
COMP_KO = ROOT / "cev_a16ko_mixed64_bf16_qk_fp16_mlp04_p768.pt"
FIT_CACHE = "fineweb_n192_skip11000.pt"
QK_FIT = (72, 96)
MLP_FIT = (24, 48)
MLP_LAYERS = (0, 4)
QK_LAYERS = tuple(range(2, 18))
QK_RANK = 64
MLP_RANK = 768
SCALARS = 511_758_646
BYTES = 1_023_517_292


def _spearman(left, right):
    a = torch.tensor(left).argsort().argsort().float()
    b = torch.tensor(right).argsort().argsort().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (BASE_RESULT, BASE_CEV, NATIVE_KO,
                     ROOT / "circuits/BATTERY.json", ROOT / f".rowcache/{FIT_CACHE}"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert all(baseline[key] for key in (
            "pred_a_two_byte_census_and_certificates_hold",
            "pred_b_two_byte_shifted_and_fresh_hold",
            "pred_c_two_byte_selection_identity_and_bill_hold"))
        assert baseline["literal_standalone_scalars"] == SCALARS
        assert baseline["literal_raw_tensor_bytes"] == BYTES
        print("A16 MIXED64 BF16 QK FP16 MLP04 | dry run: baseline, identities, bars valid")
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
    from tier2_model import REPOS, _fetch

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    baseline_result = json.loads(BASE_RESULT.read_text())
    baseline = torch.load(BASE_CEV, map_location="cpu").float().reshape(-1)
    native_ko = torch.load(NATIVE_KO, map_location="cpu").float().reshape(-1)
    assert baseline.numel() == native_ko.numel() == nflat

    capture = {"sum": torch.zeros(C.D, device=C.DEV), "n": 0}

    def capture_mean(_module, _inputs, output):
        values = output[0].detach().float().reshape(-1, C.D)
        capture["sum"] += values.sum(0)
        capture["n"] += values.shape[0]

    handle = C.m.transformer.h[16].attn.register_forward_hook(capture_mean)
    for start in range(0, 128, 4):
        index = C.FW[start:start + 4, :256].to(C.DEV)
        x = F.rms_norm(C.m.transformer.wte(index), (C.D,))
        x0, value0 = x, None
        for block in C.m.transformer.h:
            x, value0 = block(x, value0, x0)
    handle.remove()
    mean_value = (capture["sum"] / capture["n"]).clone()

    def ablate(_module, _inputs, output):
        if not C.SEL.get("abl_on"):
            return None
        values, value0 = output
        return mean_value.expand_as(values).to(values.dtype), value0

    checkpoint_path = _fetch(REPOS["bilin18"], "pytorch_model.bin")
    source = torch.load(checkpoint_path, map_location="meta", weights_only=True, mmap=True)
    parameters = dict(C.m.named_parameters())
    assert len(source) == 218 and set(parameters) == set(source)
    shapes_before = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
    dtype_scalars = {"torch.float32": 0, "torch.bfloat16": 0}
    source_bf16_exact = True
    for name, parameter in parameters.items():
        source_dtype = str(source[name].dtype)
        if source_dtype not in dtype_scalars:
            raise RuntimeError(f"unsupported source dtype {source_dtype} for {name}")
        dtype_scalars[source_dtype] += parameter.numel()
        rounded = parameter.data.bfloat16().float()
        if source_dtype == "torch.bfloat16":
            source_bf16_exact = source_bf16_exact and bool(torch.equal(rounded, parameter.data))
        else:
            parameter.data.copy_(rounded)
    assert dtype_scalars == {"torch.float32": 487_931_904,
                             "torch.bfloat16": 57_970_998}
    assert source_bf16_exact
    assert {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before

    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    qk_rows = cached[QK_FIT[0]:QK_FIT[1], :257].long().contiguous()
    mlp_rows = cached[MLP_FIT[0]:MLP_FIT[1], :257].long().contiguous()
    qk_covariances = _attention_input_covariances(C.m, qk_rows, _manual_logits)
    M.LAYERS = MLP_LAYERS
    mlp_covariances = M._covariances(C.m, mlp_rows, _manual_logits)
    programs = {}
    for layer in MLP_LAYERS:
        program, _basis, _diag = M._rrr_program(
            C.m.transformer.h[layer].mlp, mlp_covariances[layer], rank=MLP_RANK)
        programs[layer] = {name: value.cpu() for name, value in program.items()}
        del program, _basis
    del mlp_covariances
    torch.cuda.empty_cache()

    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": QK_RANK, "qk_rmap": {},
        "qk_extra_tail": 0, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "qk_context_covariances": qk_covariances,
        "qk_factor_storage_dtype": "float16",
        "final_mlp_input_programs": programs,
        "ablate_on_census": True,
    })
    C.SEL["_ablh"] = ablate
    print("ARM: a16 mixed64 BF16 QK fp16 MLP04 p768 vs original native", flush=True)
    run = C.main()
    compiled_ko = C.SEL["cev"].float().reshape(-1).cpu()
    torch.save(compiled_ko, COMP_KO)
    effect_comp = compiled_ko - baseline
    effect_native = native_ko - base_ce
    cosine = float(torch.dot(effect_comp, effect_native) /
                   (effect_comp.norm() * effect_native.norm()).clamp_min(1e-12))
    normalized_error = float((effect_comp - effect_native).norm()
                             / effect_native.norm().clamp_min(1e-12))
    norm_ratio = float(effect_comp.norm() / effect_native.norm().clamp_min(1e-12))

    collateral_native, collateral_comp, own_ratios = [], [], []
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        native_abs = float(effect_native[member].abs().mean())
        comp_abs = float(effect_comp[member].abs().mean())
        if receipt["mean_ablation"]["top"][0]["component"] == "a16":
            own_ratios.append(comp_abs / max(native_abs, 1e-12))
        else:
            collateral_native.append(native_abs)
            collateral_comp.append(comp_abs)
    own_ratios.sort()
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = _spearman(collateral_native, collateral_comp)

    wanted_qk = tuple(range(QK_RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    factor_pairs = [factor for heads in qk.values() for factors in heads.values()
                    for factor in factors]
    factor_dtypes = {str(tensor.dtype) for factor in factor_pairs for tensor in factor}
    widths = {int(factor[0].shape[1]) for factor in factor_pairs}
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    active = tuple(C.SEL.get("_ORDER2", ()))
    identity = (dtype_scalars == {"torch.float32": 487_931_904,
                                  "torch.bfloat16": 57_970_998}
                and source_bf16_exact and set(parameters) == set(source)
                and {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before
                and C.SEL.get("_QK_METRIC") == "context_rrr"
                and C.SEL.get("_QK_STORAGE_DTYPE") == "float16"
                and context_layers == QK_LAYERS and set(index_sets) == set(QK_LAYERS)
                and all(value == wanted_qk for value in index_sets.values())
                and widths == {QK_RANK} and factor_dtypes == {"torch.float16"}
                and len(factor_pairs) == 440 and observed == {0: MLP_RANK, 4: MLP_RANK}
                and not any(name in active for name in ("a0", "a1v", "tailE"))
                and SCALARS == 511_758_646 and BYTES == 1_023_517_292)
    pred_a = (baseline_result["census_damage"] <= .015
              and baseline_result["certificates_valid"] >= 43
              and baseline_result["shifted_damage_row_max"] <= .120
              and run["L2_F"] <= .030 and identity)
    pred_b = (cosine >= .98 and normalized_error <= .30
              and .90 <= norm_ratio <= 1.15)
    pred_c = collateral_rho >= .98 and .90 <= own_median <= 1.15
    null = cosine < .70 or collateral_rho < .75
    result = {
        "status": "a16_transfer_mixed64_bf16_qk_fp16_mlp04_p768_complete",
        "rung": 369,
        "claim_level": "final_original_native_signed_two_byte_selected_mlp_qk64_adoption_gate",
        "convention": "signed effect = KO CE minus unablated CE within original-native and compiled models",
        "native_ko_measured_before_global_rounding": True,
        "unablated_census_damage": baseline_result["census_damage"],
        "unablated_certificates_valid": baseline_result["certificates_valid"],
        "unablated_shifted_damage_row_max": baseline_result["shifted_damage_row_max"],
        "live_unablated_fresh_damage": run["L2_F"],
        "effect_cosine": cosine,
        "effect_normalized_error": normalized_error,
        "effect_norm_ratio": norm_ratio,
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "global_storage_dtype": "source-fp32_to_bfloat16; source-bfloat16_exact",
        "qk_storage_dtype": C.SEL.get("_QK_STORAGE_DTYPE"),
        "qk_rank": QK_RANK,
        "qk_factorized_maps": len(factor_pairs),
        "mlp_input_program_observed": observed,
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        'pred_a_final_baseline_identity_and_bill_hold': bool(pred_a),
        'pred_b_final_original_native_signed_effect_holds': bool(pred_b),
        'pred_c_final_circuit_profile_holds': bool(pred_c),
        "null_final_signed_transport_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print("A16 MIXED64 BF16 QK FP16 MLP04 P768 FINAL GATE DONE", flush=True)


if __name__ == "__main__":
    main()
