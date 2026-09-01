"""RUNG 338 -- SIGNED a16 ADOPTION GATE FOR CONTEXT-QK88.

Apply the identical native a16 mean ablation to the fixed split-B rank88
context-QK program and compare signed effects against its saved unablated CEV.

Frozen predictions
------------------
pred_a_live_baseline_price_and_identity:
    Baseline <=.004/55 certificates, OOD max <=.020, fresh <=.010, exact
    context-RRR rank88/440-map identity and 530,583,862-scalar bill.
pred_b_signed_effect_vector_transfers:
    Cosine >=.90 and normalized vector error <=.60.
pred_c_circuit_effect_profile_transfers:
    Collateral Spearman >=.90 and a16-own median ratio in [.60,1.40].

Null: cosine <.70 or collateral Spearman <.75. Full pass formally adopts a
middle Pareto point and licenses the rank80 physical test.
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
OUT = ROOT / "a16_transfer_mixed88_context_metric_qk_results.json"
BASE_RESULT = ROOT / "mixed88_context_metric_qk_ood_results.json"
BASE_CEV = ROOT / "cev_mixed88_context_metric_qk.pt"
COMP_KO = ROOT / "cev_a16ko_mixed88_context_metric_qk.pt"
NATIVE_KO = ROOT / "cev_a16ko_native_context_qk88_gate.pt"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 88
SCALARS = 530_583_862
BYTES = 2_006_393_452
QK_STORAGE_DTYPE = None


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
        assert baseline["census_damage"] <= .004
        assert baseline["certificates_valid"] >= 55
        assert baseline["literal_standalone_scalars"] == SCALARS
        print("A16 TRANSFER CONTEXT-QK88 | dry run: baseline, fit, bill, bars valid")
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

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    baseline_result = json.loads(BASE_RESULT.read_text())
    baseline = torch.load(BASE_CEV, map_location="cpu").float().reshape(-1)
    assert baseline.numel() == nflat

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
        "qk_factor_storage_dtype": QK_STORAGE_DTYPE,
        "ablate_on_census": True,
    })
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

    C.SEL["_ablh"] = ablate
    print("ARM: split-B context-QK88 with a16 mean ablated", flush=True)
    run = C.main()
    compiled_ko = C.SEL["cev"].float().reshape(-1).cpu()
    native_ko = _direct_native_cev(C, rows, ablate)
    torch.save(compiled_ko, COMP_KO)
    torch.save(native_ko, NATIVE_KO)
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
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    factor_dtypes = {str(tensor.dtype) for heads in qk.values()
                     for factors in heads.values() for factor in factors for tensor in factor}
    factor_maps = sum(4 * len(heads) for heads in qk.values())
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    storage_dtype = C.SEL.get("_QK_STORAGE_DTYPE")
    expected_storage_dtype = "float16" if QK_STORAGE_DTYPE == "float16" else "float32"
    expected_tensor_dtype = {"torch.float16"} if QK_STORAGE_DTYPE == "float16" else {"torch.float32"}
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    pred_a = (baseline_result["census_damage"] <= .004
              and baseline_result["certificates_valid"] >= 55
              and baseline_result["shifted_damage_row_max"] <= .020
              and run["L2_F"] <= .010 and metric == "context_rrr"
              and context_layers == LAYERS and set(index_sets) == set(LAYERS)
              and all(value == wanted_indices for value in index_sets.values())
              and widths == {RANK} and factor_maps == 440
              and storage_dtype == expected_storage_dtype
              and factor_dtypes == expected_tensor_dtype
              and not any(name in active for name in ("a0", "a1v", "tailE"))
              and SCALARS == 530_583_862 and BYTES == 2_006_393_452)
    pred_b = cosine >= .90 and normalized_error <= .60
    pred_c = collateral_rho >= .90 and .60 <= own_median <= 1.40
    null = cosine < .70 or collateral_rho < .75
    result = {
        "status": "a16_transfer_mixed88_context_metric_qk_complete",
        "rung": 338,
        "claim_level": "direct_signed_a16_context_qk88_adoption_gate",
        "convention": "signed effect = KO CE minus unablated CE within each model",
        "unablated_census_damage": baseline_result["census_damage"],
        "unablated_certificates_valid": baseline_result["certificates_valid"],
        "unablated_shifted_damage_mean": baseline_result["shifted_damage_mean"],
        "unablated_shifted_damage_row_max": baseline_result["shifted_damage_row_max"],
        "live_unablated_fresh_damage": run["L2_F"],
        "effect_cosine": cosine,
        "effect_normalized_error": normalized_error,
        "effect_norm_ratio": norm_ratio,
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "qk_metric": metric,
        "qk_storage_dtype": storage_dtype,
        "qk_factor_tensor_dtypes": sorted(factor_dtypes),
        "qk_context_layers": list(context_layers),
        "qk_rank": RANK,
        "qk_factorized_maps": factor_maps,
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        'pred_a_live_baseline_price_and_identity': bool(pred_a),
        'pred_b_signed_effect_vector_transfers': bool(pred_b),
        'pred_c_circuit_effect_profile_transfers': bool(pred_c),
        "null_signed_context_qk88_transport_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print(f"wrote {OUT}, {COMP_KO}, {NATIVE_KO}", flush=True)


if __name__ == "__main__":
    main()
