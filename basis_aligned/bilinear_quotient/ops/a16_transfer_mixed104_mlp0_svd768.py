"""RUNG 317 -- SIGNED a16 ADOPTION GATE FOR MIXED104 + MLP0 SVD768.

Rung 316 established the exact unablated p768 baseline at 536,940,854 scalars,
+.00901182 census, 50/62 certificates, non-positive fresh8, and shifted OOD.
Apply the identical native attention-16 mean ablation within compiled and native
models and compare direct signed 256,000-position causal-effect vectors:

    e_comp = CE(compiled + KO) - CE(compiled)
    e_real = CE(native + KO) - CE(native).

Frozen predictions
------------------
pred_a_live_baseline_and_identity:
    Baseline census <=.011, >=48 certificates, max fresh <=.020, and live
    p768/mixed104/price identities exact.
pred_b_signed_effect_vector:
    cosine(e_comp,e_real) >=.90 and normalized vector error <=.60.
pred_c_circuit_effect_transport:
    non-a16 collateral Spearman >=.90 and a16-own median absolute-effect ratio
    lies in [.60,1.40].

Null: effect cosine <.70 or collateral Spearman <.75.  A full pass formally
adopts the new literal Pareto point.
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
OUT = ROOT / "a16_transfer_mixed104_mlp0_svd768_results.json"
BASE_RESULT = ROOT / "mixed104_mlp0_svd768_ood_results.json"
BASE_CEV = ROOT / "cev_mixed104_mlp0_svd768.pt"
COMP_KO = ROOT / "cev_a16ko_mixed104_mlp0_svd768.pt"
NATIVE_KO = ROOT / "cev_a16ko_native_svd768_gate.pt"
SCALARS = 536_940_854


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
            ces.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target, reduction="none"
            ).cpu())
    finally:
        handle.remove()
        C.SEL["abl_on"] = False
    return torch.cat(ces)


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (BASE_RESULT, BASE_CEV, ROOT / "circuits/BATTERY.json",
                     ROOT / "census_state_diverse.pt"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert baseline["pred_a_shifted_mean_tail_and_max"]
        assert baseline["certificates_valid"] >= 48
        assert baseline["literal_standalone_scalars"] == SCALARS
        print("A16 TRANSFER MIXED104 MLP0 SVD768 | dry run: baseline, effect, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed104_mlp0_shared_input_svd_frontier import _programs

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    baseline = torch.load(BASE_CEV, map_location="cpu").float().reshape(-1)
    baseline_result = json.loads(BASE_RESULT.read_text())
    assert baseline.numel() == nflat
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    program = _programs(C.m)["r768"]
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": program,
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
    print("ARM: mixed104 + MLP0 SVD768 with a16 mean ablated only at census", flush=True)
    run = C.main()

    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    wanted = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    if (observed != {0: 768} or set(index_sets) != set(range(2, 18))
            or any(value != wanted for value in index_sets.values()) or widths != {104}
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: p768 or mixed104 identity changed")

    compiled_ko = C.SEL["cev"].float().reshape(-1).cpu()
    native_ko = _direct_native_cev(C, rows, ablate)
    assert compiled_ko.numel() == native_ko.numel() == nflat
    torch.save(compiled_ko, COMP_KO)
    torch.save(native_ko, NATIVE_KO)
    effect_comp = compiled_ko - baseline
    effect_real = native_ko - base_ce
    cosine = float(torch.dot(effect_comp, effect_real) /
                   (effect_comp.norm() * effect_real.norm()).clamp_min(1e-12))
    normalized_error = float((effect_comp - effect_real).norm()
                             / effect_real.norm().clamp_min(1e-12))
    norm_ratio = float(effect_comp.norm() / effect_real.norm().clamp_min(1e-12))

    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    collateral_real, collateral_comp, own_ratios, circuit_rows = [], [], [], []
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        real_abs = float(effect_real[member].abs().mean())
        comp_abs = float(effect_comp[member].abs().mean())
        own = receipt["mean_ablation"]["top"][0]["component"] == "a16"
        circuit_rows.append({"tag": tag, "native_abs_effect": round(real_abs, 7),
                             "compiled_abs_effect": round(comp_abs, 7), "own": own})
        if own:
            own_ratios.append(comp_abs / max(real_abs, 1e-12))
        else:
            collateral_real.append(real_abs)
            collateral_comp.append(comp_abs)
    own_ratios.sort()
    if not own_ratios:
        raise SystemExit("INSTRUMENT FAIL: no a16-own circuit leaves")
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = _spearman(collateral_real, collateral_comp)
    pred_a = (baseline_result["census_damage"] <= .011
              and baseline_result["certificates_valid"] >= 48
              and baseline_result["max_fresh_damage"] <= .020
              and run["L2_F"] <= .020 and observed == {0: 768}
              and widths == {104} and SCALARS == 536_940_854)
    pred_b = cosine >= .90 and normalized_error <= .60
    pred_c = collateral_rho >= .90 and .60 <= own_median <= 1.40
    null = cosine < .70 or collateral_rho < .75
    result = {
        "status": "a16_transfer_mixed104_mlp0_svd768_complete",
        "rung": 317,
        "claim_level": "direct_signed_a16_adoption_gate",
        "convention": "signed effect = KO CE minus unablated CE within each model",
        "unablated_census_damage": baseline_result["census_damage"],
        "unablated_certificates_valid": baseline_result["certificates_valid"],
        "unablated_max_fresh_damage": baseline_result["max_fresh_damage"],
        "live_unablated_fresh_damage": run["L2_F"],
        "compiled_ko_census_damage": float((compiled_ko - base_ce).mean()),
        "native_ko_census_damage": float(effect_real.mean()),
        "effect_cosine": cosine,
        "effect_normalized_error": normalized_error,
        "effect_norm_ratio": norm_ratio,
        "collateral_spearman": collateral_rho,
        "own_effect_median_ratio": own_median,
        "own_effect_ratios": own_ratios,
        "circuits": circuit_rows,
        "mlp_input_program_observed": observed,
        "qk_singular_indices": list(wanted),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        'pred_a_live_baseline_and_identity': bool(pred_a),
        'pred_b_signed_effect_vector': bool(pred_b),
        'pred_c_circuit_effect_transport': bool(pred_c),
        "null_signed_transport_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("circuits", "own_effect_ratios")}, indent=2), flush=True)
    print(f"wrote {OUT}, {COMP_KO}, and {NATIVE_KO}", flush=True)


if __name__ == "__main__":
    main()
