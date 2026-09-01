"""RUNG 335 -- SIGNED a16 GATE FOR CONTEXT-QK96 + CONTEXT-MLP0 p448.

Apply the identical native a16 mean ablation to the fixed dual-context program
and compare direct signed effects against rung334's saved unablated CEV.

Frozen predictions
------------------
pred_a_live_baseline_price_and_dual_identity:
    Baseline <=.012/46 certificates, OOD max <=.060, fresh <=.020, exact
    context Q/K rank96/440 maps plus context MLP0 p448 and literal bill.
pred_b_signed_effect_vector_transfers:
    Cosine >=.90 and normalized vector error <=.60.
pred_c_circuit_effect_profile_transfers:
    Collateral Spearman >=.90 and a16-own median ratio in [.60,1.40].

Null: cosine <.70 or collateral Spearman <.75.  Full pass formally adopts the
529,117,494-scalar dual-context artifact as a smaller Pareto point.
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
OUT = ROOT / "a16_transfer_mixed96_context_qk_mlp0_p448_results.json"
BASE_RESULT = ROOT / "mixed96_context_qk_mlp0_context_p448_ood_results.json"
BASE_CEV = ROOT / "cev_mixed96_context_qk_mlp0_context_p448.pt"
COMP_KO = ROOT / "cev_a16ko_mixed96_context_qk_mlp0_p448.pt"
NATIVE_KO = ROOT / "cev_a16ko_native_dual_context_p448_gate.pt"
FIT_CACHE = "fineweb_n192_skip11000.pt"
QK_FIT = (72, 96)
MLP_FIT = (0, 24)
LAYERS = tuple(range(2, 18))
QK_RANK = 96
MLP_RANK = 448
SCALARS = 529_117_494
BYTES = 2_000_527_980


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
        assert baseline["census_damage"] <= .012
        assert baseline["certificates_valid"] >= 46
        assert baseline["literal_standalone_scalars"] == SCALARS
        print("A16 TRANSFER DUAL CONTEXT p448 | dry run: baseline, fits, bill, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    baseline_result = json.loads(BASE_RESULT.read_text())
    baseline = torch.load(BASE_CEV, map_location="cpu").float().reshape(-1)
    assert baseline.numel() == nflat

    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    qk_rows = cached[QK_FIT[0]:QK_FIT[1], :257].long().contiguous()
    mlp_rows = cached[MLP_FIT[0]:MLP_FIT[1], :257].long().contiguous()
    qk_covariances = _attention_input_covariances(C.m, qk_rows, _manual_logits)
    mlp_covariance = _covariance(C.m, mlp_rows, _manual_logits)
    program0, _basis, _diagnostic = _rrr_program(C.m.transformer.h[0].mlp,
                                                 mlp_covariance, rank=MLP_RANK)
    program = {0: {name: value.cpu() for name, value in program0.items()}}
    del mlp_covariance, program0, _basis
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
    print("ARM: context-QK96 + context-MLP0 p448 with a16 mean ablated", flush=True)
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

    wanted_indices = tuple(range(QK_RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    factor_maps = sum(4 * len(heads) for heads in qk.values())
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    pred_a = (baseline_result["census_damage"] <= .012
              and baseline_result["certificates_valid"] >= 46
              and baseline_result["shifted_damage_row_max"] <= .060
              and run["L2_F"] <= .020 and metric == "context_rrr"
              and context_layers == LAYERS and set(index_sets) == set(LAYERS)
              and all(value == wanted_indices for value in index_sets.values())
              and widths == {QK_RANK} and factor_maps == 440
              and observed == {0: MLP_RANK}
              and not any(name in active for name in ("a0", "a1v", "tailE"))
              and SCALARS == 529_117_494 and BYTES == 2_000_527_980)
    pred_b = cosine >= .90 and normalized_error <= .60
    pred_c = collateral_rho >= .90 and .60 <= own_median <= 1.40
    null = cosine < .70 or collateral_rho < .75
    result = {
        "status": "a16_transfer_mixed96_context_qk_mlp0_p448_complete",
        "rung": 335,
        "claim_level": "direct_signed_a16_dual_context_p448_adoption_gate",
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
        "qk_context_layers": list(context_layers),
        "qk_rank": QK_RANK,
        "qk_factorized_maps": factor_maps,
        "mlp0_rank": observed.get(0),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        'pred_a_live_baseline_price_and_dual_identity': bool(pred_a),
        'pred_b_signed_effect_vector_transfers': bool(pred_b),
        'pred_c_circuit_effect_profile_transfers': bool(pred_c),
        "null_signed_dual_context_p448_transport_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print(f"wrote {OUT}, {COMP_KO}, {NATIVE_KO}", flush=True)


if __name__ == "__main__":
    main()
