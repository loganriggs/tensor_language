"""RUNG 365 -- UNIVERSAL-BF16 HIGH-FIDELITY CENSUS/CERT/SIGNED GATE.

Make the global two-byte storage screen a fully citable high-fidelity point.
Compare source-aware BF16/fp32-compute prediction and the identical fixed
native a16 mean ablation directly against original-native saved CEVs.

Frozen predictions
------------------
pred_a_bf16_census_and_certificates_hold:
    Absolute census mean <=.004 and >=58/62 certificates.
pred_b_bf16_original_native_signed_effect_holds:
    Cosine >=.995, normalized error <=.10, norm ratio in [.95,1.05].
pred_c_bf16_circuit_profile_identity_and_bill_hold:
    Collateral Spearman >=.995, own median ratio [.95,1.05], exact checkpoint
    identities, passing two-corpus screen, and 545,902,902/1,091,805,804.

Null: |census| >=.020, <=50 certificates, or cosine <.90.  One endpoint
confirmation only; no layer, rank, or alternate-precision tuning follows.
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
OUT = ROOT / "bilin18_universal_bf16_full_gate_results.json"
SCREEN = ROOT / "bilin18_universal_bf16_storage_screen_results.json"
NATIVE_KO = ROOT / "cev_a16ko_original_native_universal_bf16_qk56_gate.pt"
BF16_CEV = ROOT / "cev_bilin18_universal_bf16.pt"
BF16_KO = ROOT / "cev_a16ko_bilin18_universal_bf16.pt"
NATIVE_SCALARS = 545_902_902
NATIVE_BYTES = 2_067_669_612
BF16_BYTES = 1_091_805_804


def _spearman(left, right):
    a = torch.tensor(left).argsort().argsort().float()
    b = torch.tensor(right).argsort().argsort().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def _certificate_count(census_lib, battery, damage):
    valid = 0
    for tag, receipt in battery.items():
        try:
            member = census_lib.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage[member].abs().mean())
        valid += int(value < .5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])
    return valid


@torch.no_grad()
def _direct_cev(C, rows, ablation_hook=None):
    ces = []
    handle = None
    C.SEL["abl_on"] = ablation_hook is not None
    if ablation_hook is not None:
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
        if handle is not None:
            handle.remove()
        C.SEL["abl_on"] = False
    return torch.cat(ces)


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert SCREEN.exists() and NATIVE_KO.exists()
        screen = json.loads(SCREEN.read_text())
        assert all(screen[key] for key in (
            "pred_a_bf16_storage_preserves_mean_on_both_corpora",
            "pred_b_bf16_storage_preserves_tails_and_transfers",
            "pred_c_checkpoint_identity_and_two_byte_bill_hold"))
        assert BF16_BYTES == 2 * NATIVE_SCALARS
        print("UNIVERSAL BF16 FULL GATE | dry run: screen, native KO, bill, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from tier2_model import REPOS, _fetch

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    native_ko = torch.load(NATIVE_KO, map_location="cpu").float().reshape(-1)
    assert native_ko.numel() == nflat

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

    bf16_cev = _direct_cev(C, rows)
    bf16_ko = _direct_cev(C, rows, ablate)
    torch.save(bf16_cev, BF16_CEV)
    torch.save(bf16_ko, BF16_KO)
    damage = bf16_cev - base_ce
    census = float(damage.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    certificates = _certificate_count(CN, battery, damage)
    effect_bf16 = bf16_ko - bf16_cev
    effect_native = native_ko - base_ce
    cosine = float(torch.dot(effect_bf16, effect_native) /
                   (effect_bf16.norm() * effect_native.norm()).clamp_min(1e-12))
    normalized_error = float((effect_bf16 - effect_native).norm()
                             / effect_native.norm().clamp_min(1e-12))
    norm_ratio = float(effect_bf16.norm() / effect_native.norm().clamp_min(1e-12))

    collateral_native, collateral_bf16, own_ratios = [], [], []
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        native_abs = float(effect_native[member].abs().mean())
        bf16_abs = float(effect_bf16[member].abs().mean())
        if receipt["mean_ablation"]["top"][0]["component"] == "a16":
            own_ratios.append(bf16_abs / max(native_abs, 1e-12))
        else:
            collateral_native.append(native_abs)
            collateral_bf16.append(bf16_abs)
    own_ratios.sort()
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = _spearman(collateral_native, collateral_bf16)
    screen = json.loads(SCREEN.read_text())
    screen_pass = all(screen[key] for key in (
        "pred_a_bf16_storage_preserves_mean_on_both_corpora",
        "pred_b_bf16_storage_preserves_tails_and_transfers",
        "pred_c_checkpoint_identity_and_two_byte_bill_hold"))
    identity = (len(source) == 218 and source_scalars == NATIVE_SCALARS
                and source_bytes == NATIVE_BYTES and source_bf16_exact
                and dtype_scalars == {"torch.float32": 487_931_904,
                                      "torch.bfloat16": 57_970_998}
                and set(parameters) == set(source)
                and {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before
                and BF16_BYTES == 1_091_805_804 and BF16_BYTES == 2 * NATIVE_SCALARS
                and screen_pass and BF16_CEV.exists() and BF16_KO.exists())
    pred_a = abs(census) <= .004 and certificates >= 58
    pred_b = (cosine >= .995 and normalized_error <= .10
              and .95 <= norm_ratio <= 1.05)
    pred_c = (collateral_rho >= .995 and .95 <= own_median <= 1.05 and identity)
    null = abs(census) >= .020 or certificates <= 50 or cosine < .90
    result = {
        "status": "bilin18_universal_bf16_full_gate_complete",
        "rung": 365,
        "claim_level": "full_census_certificate_signed_global_bf16_high_fidelity_adoption_gate",
        "convention": "BF16 CE minus original native CE; signed effect within each model",
        "census_damage": census,
        "certificates_valid": certificates,
        "effect_cosine": cosine,
        "effect_normalized_error": normalized_error,
        "effect_norm_ratio": norm_ratio,
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "checkpoint_entries": len(source),
        "source_dtype_scalars": dtype_scalars,
        "source_bfloat16_tensors_bit_exact": source_bf16_exact,
        "rounded_fp32_tensors_changed": changed_tensors,
        "global_storage_dtype": "source-fp32_to_bfloat16; source-bfloat16_exact",
        "global_compute_dtype": "float32_explicit_dequantization",
        "literal_standalone_scalars": NATIVE_SCALARS,
        "literal_raw_tensor_bytes": BF16_BYTES,
        "byte_saving_vs_native": NATIVE_BYTES - BF16_BYTES,
        "byte_saving_fraction_vs_native": (NATIVE_BYTES - BF16_BYTES) / NATIVE_BYTES,
        "two_corpus_screen_result": SCREEN.name,
        "saved_unablated_cev_file": BF16_CEV.name,
        "saved_ablation_cev_file": BF16_KO.name,
        'pred_a_bf16_census_and_certificates_hold': bool(pred_a),
        'pred_b_bf16_original_native_signed_effect_holds': bool(pred_b),
        'pred_c_bf16_circuit_profile_identity_and_bill_hold': bool(pred_c),
        "null_universal_bf16_full_gate_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print("UNIVERSAL BF16 HIGH-FIDELITY FULL GATE DONE", flush=True)


if __name__ == "__main__":
    main()
