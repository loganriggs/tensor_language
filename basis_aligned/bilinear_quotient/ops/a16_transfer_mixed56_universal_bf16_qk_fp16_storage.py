"""RUNG 364 -- ORIGINAL-NATIVE SIGNED GATE FOR UNIVERSAL-BF16 + FP16-QK56.

Conditional on every rung363 positive, measure the native a16 mean ablation
before rounding the live model, then apply the identical ablation to the exact
global-BF16/fp32-compute plus context-QK56-fp16 artifact.  Compare signed KO
effects within each model against their respective saved unablated CEVs.

Frozen predictions
------------------
pred_a_live_combined_baseline_price_and_identity:
    Reproduce <=.021/38 certificates, shifted max <=.140, structural fresh
    <=.030, all source/global/QK identities, and 512,561,462/1,025,122,924.
pred_b_tight_original_native_signed_effect_transfers:
    Cosine >=.98, normalized error <=.30, norm ratio in [.90,1.15].
pred_c_tight_original_native_circuit_profile_transfers:
    Collateral Spearman >=.98 and a16-own median ratio in [.90,1.15].

Null: cosine <.70 or collateral Spearman <.75.  A full pass formally adopts
the 50.42%-smaller-bytes predictive/manipulable artifact; no tuning follows.
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
OUT = ROOT / "a16_transfer_mixed56_universal_bf16_qk_fp16_storage_results.json"
BASE_RESULT = ROOT / "mixed56_universal_bf16_qk_fp16_storage_ood_results.json"
BASE_CEV = ROOT / "cev_mixed56_universal_bf16_qk_fp16_storage.pt"
COMP_KO = ROOT / "cev_a16ko_mixed56_universal_bf16_qk_fp16_storage.pt"
NATIVE_KO = ROOT / "cev_a16ko_original_native_universal_bf16_qk56_gate.pt"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 56
NATIVE_SCALARS = 545_902_902
NATIVE_BYTES = 2_067_669_612
SCALARS = 512_561_462
BYTES = 1_025_122_924


def _spearman(left, right):
    a = torch.tensor(left).argsort().argsort().float()
    b = torch.tensor(right).argsort().argsort().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def _direct_native_cev(C, rows, ablation_hook):
    ces = []
    C.SEL["qk_tail_on"] = False
    C.SEL["abl_on"] = True
    handle = C.m.transformer.h[16].attn.register_forward_hook(ablation_hook)
    try:
        for start in range(0, rows.shape[0], 4):
            batch = rows[start:start + 4, :257].to(C.DEV)
            index, target = batch[:, :256], batch[:, 1:257].reshape(-1)
            x = F.rms_norm(C.m.transformer.wte(index), (C.D,))
            x0, value0 = x, None
            for block in C.m.transformer.h:
                x, value0 = block(x, value0, x0)
            logits = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
            ces.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target,
                                       reduction="none").cpu())
    finally:
        handle.remove()
        C.SEL["abl_on"] = False
    return torch.cat(ces)


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (BASE_RESULT, BASE_CEV, ROOT / "circuits/BATTERY.json",
                     ROOT / f".rowcache/{FIT_CACHE}"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert all(baseline[key] for key in (
            "pred_a_combined_census_and_certificates_hold",
            "pred_b_combined_shifted_and_fresh_transfer_hold",
            "pred_c_combined_identity_and_two_byte_bill_hold"))
        assert baseline["literal_standalone_scalars"] == SCALARS
        assert baseline["literal_raw_tensor_bytes"] == BYTES
        print("A16 UNIVERSAL-BF16 + FP16-QK56 | dry run: passing baseline and bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import REPOS, _fetch

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    baseline_result = json.loads(BASE_RESULT.read_text())
    baseline = torch.load(BASE_CEV, map_location="cpu").float().reshape(-1)
    assert baseline.numel() == nflat

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

    # This must precede any global rounding: the causal reference is the
    # original checkpoint, not a rounded-native surrogate.
    native_ko = _direct_native_cev(C, rows, ablate)
    torch.save(native_ko, NATIVE_KO)

    checkpoint_path = _fetch(REPOS["bilin18"], "pytorch_model.bin")
    source = torch.load(checkpoint_path, map_location="meta", weights_only=True, mmap=True)
    parameters = dict(C.m.named_parameters())
    assert set(parameters) == set(source) and len(source) == 218
    source_scalars = sum(tensor.numel() for tensor in source.values())
    source_bytes = sum(tensor.numel() * tensor.element_size() for tensor in source.values())
    assert source_scalars == NATIVE_SCALARS and source_bytes == NATIVE_BYTES
    shapes_before = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
    dtype_scalars = {"torch.float32": 0, "torch.bfloat16": 0}
    source_bf16_exact = True
    changed_tensors = 0
    for name, parameter in parameters.items():
        source_dtype = str(source[name].dtype)
        if source_dtype not in dtype_scalars:
            raise RuntimeError(f"unsupported source dtype {source_dtype} for {name}")
        dtype_scalars[source_dtype] += parameter.numel()
        rounded = parameter.data.bfloat16().float()
        if source_dtype == "torch.bfloat16":
            source_bf16_exact = source_bf16_exact and bool(torch.equal(rounded, parameter.data))
        else:
            changed_tensors += int(bool((rounded != parameter.data).any()))
            parameter.data.copy_(rounded)
    assert dtype_scalars == {"torch.float32": 487_931_904,
                             "torch.bfloat16": 57_970_998}
    assert source_bf16_exact and changed_tensors > 0
    assert {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before

    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    covariances = _attention_input_covariances(C.m, fit_rows, _manual_logits)

    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": RANK, "qk_rmap": {},
        "qk_extra_tail": 0, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "qk_context_covariances": covariances,
        "qk_factor_storage_dtype": "float16",
        "ablate_on_census": True,
    })
    C.SEL["_ablh"] = ablate
    print("ARM: original-native a16 vs universal-BF16 + fp16-QK56 a16", flush=True)
    run = C.main()
    compiled_ko = C.SEL["cev"].float().reshape(-1).cpu()
    torch.save(compiled_ko, COMP_KO)
    effect_comp = compiled_ko - baseline
    effect_real = native_ko - base_ce
    cosine = float(torch.dot(effect_comp, effect_real) /
                   (effect_comp.norm() * effect_real.norm()).clamp_min(1e-12))
    normalized_error = float((effect_comp - effect_real).norm()
                             / effect_real.norm().clamp_min(1e-12))
    norm_ratio = float(effect_comp.norm() / effect_real.norm().clamp_min(1e-12))

    collateral_real, collateral_comp, own_ratios = [], [], []
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        real_abs = float(effect_real[member].abs().mean())
        comp_abs = float(effect_comp[member].abs().mean())
        if receipt["mean_ablation"]["top"][0]["component"] == "a16":
            own_ratios.append(comp_abs / max(real_abs, 1e-12))
        else:
            collateral_real.append(real_abs)
            collateral_comp.append(comp_abs)
    own_ratios.sort()
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = _spearman(collateral_real, collateral_comp)

    wanted_indices = tuple(range(RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    factor_pairs = [factor for heads in qk.values() for factors in heads.values()
                    for factor in factors]
    widths = {int(factor[0].shape[1]) for factor in factor_pairs}
    factor_dtypes = {str(tensor.dtype) for factor in factor_pairs for tensor in factor}
    factor_maps = len(factor_pairs)
    factor_scalars = sum(tensor.numel() for factor in factor_pairs for tensor in factor)
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    storage_dtype = C.SEL.get("_QK_STORAGE_DTYPE")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    identity = (source_scalars == NATIVE_SCALARS and source_bytes == NATIVE_BYTES
                and dtype_scalars == {"torch.float32": 487_931_904,
                                      "torch.bfloat16": 57_970_998}
                and source_bf16_exact and set(parameters) == set(source)
                and {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before
                and metric == "context_rrr" and storage_dtype == "float16"
                and context_layers == LAYERS and set(index_sets) == set(LAYERS)
                and all(value == wanted_indices for value in index_sets.values())
                and widths == {RANK} and factor_dtypes == {"torch.float16"}
                and factor_maps == 440 and factor_scalars == 31_539_200
                and not any(name in active for name in ("a0", "a1v", "tailE"))
                and SCALARS == 512_561_462 and BYTES == 1_025_122_924)
    pred_a = (baseline_result["census_damage"] <= .021
              and baseline_result["certificates_valid"] >= 38
              and baseline_result["shifted_damage_row_max"] <= .140
              and run["L2_F"] <= .030 and identity)
    pred_b = (cosine >= .98 and normalized_error <= .30
              and .90 <= norm_ratio <= 1.15)
    pred_c = collateral_rho >= .98 and .90 <= own_median <= 1.15
    null = cosine < .70 or collateral_rho < .75
    result = {
        "status": "a16_transfer_mixed56_universal_bf16_qk_fp16_storage_complete",
        "rung": 364,
        "claim_level": "tight_original_native_signed_global_bf16_fp16_qk56_adoption_gate",
        "convention": "signed effect = KO CE minus unablated CE within original-native and compiled models",
        "native_ko_measured_before_global_rounding": True,
        "unablated_census_damage": baseline_result["census_damage"],
        "unablated_certificates_valid": baseline_result["certificates_valid"],
        "unablated_shifted_damage_mean": baseline_result["shifted_damage_mean"],
        "unablated_shifted_damage_row_max": baseline_result["shifted_damage_row_max"],
        "live_unablated_fresh_structural_damage": run["L2_F"],
        "effect_cosine": cosine,
        "effect_normalized_error": normalized_error,
        "effect_norm_ratio": norm_ratio,
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "global_storage_dtype": "source-fp32_to_bfloat16; source-bfloat16_exact",
        "source_dtype_scalars": dtype_scalars,
        "source_bfloat16_tensors_bit_exact": source_bf16_exact,
        "qk_metric": metric,
        "qk_storage_dtype": storage_dtype,
        "qk_factor_tensor_dtypes": sorted(factor_dtypes),
        "qk_context_layers": list(context_layers),
        "qk_rank": RANK,
        "qk_factorized_maps": factor_maps,
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        'pred_a_live_combined_baseline_price_and_identity': bool(pred_a),
        'pred_b_tight_original_native_signed_effect_transfers': bool(pred_b),
        'pred_c_tight_original_native_circuit_profile_transfers': bool(pred_c),
        "null_signed_global_bf16_fp16_qk56_transport_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print("ORIGINAL-NATIVE SIGNED UNIVERSAL-BF16 + FP16-QK56 GATE DONE", flush=True)


if __name__ == "__main__":
    main()
