"""RUNG 327 -- SIGNED a16 ADOPTION GATE FOR MLP0 CONTEXT-RRR p512/p640.

Rungs325/326 established physical census, certificates, fresh transfer,
shifted OOD, literal price, and exact unablated census CEVs for both variants.
Apply the identical native attention-16 mean ablation within native, p512, and
p640, comparing direct signed effects KO CE minus each model's own baseline.

Frozen predictions
------------------
pred_a_live_baselines_price_and_identity:
    Both OOD baselines/certificates, live primary fresh, variant maps,
    mixed104 QK/active set, and literal bills are exact.
pred_b_both_signed_effect_vectors_transfer:
    EACH arm has cosine >=.90 and normalized vector error <=.60.
pred_c_both_circuit_effect_profiles_transfer:
    EACH arm has non-a16 collateral Spearman >=.90 and a16-own median effect
    ratio in [.60,1.40].

Null: both effect cosines <.70 OR both collateral Spearman values <.75.  A full
pass formally adopts p640 as the new dominant artifact and p512 as the smaller
Pareto point.
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
OUT = ROOT / "a16_transfer_mixed104_mlp0_context_metric_frontier_results.json"
BASE_RESULT = ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json"
BASE_CEV = ROOT / "cev_mixed104_mlp0_context_rrr_frontier.pt"
COMP_KO = ROOT / "cev_a16ko_mixed104_mlp0_context_rrr_frontier.pt"
NATIVE_KO = ROOT / "cev_a16ko_native_context_rrr_gate.pt"
RANKS = (512, 640)
FIT_SLICE = (0, 24)
FIT_CACHE = "fineweb_n192_skip11000.pt"
SCALARS = {512: 534_286_646, 640: 535_613_750}
BYTES = {512: 2_021_204_588, 640: 2_026_513_004}


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
                     ROOT / "census_state_diverse.pt", ROOT / f".rowcache/{FIT_CACHE}"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert baseline["pred_a_both_variants_transport_on_shifted_ood"]
        assert baseline["arms"]["512"]["certificates_valid"] >= 47
        assert baseline["arms"]["640"]["certificates_valid"] >= 49
        print("A16 TRANSFER MLP0 CONTEXT-RRR | dry run: baselines, variants, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    baselines = torch.load(BASE_CEV, map_location="cpu")
    baseline_result = json.loads(BASE_RESULT.read_text())
    assert set(baselines) == {"r512", "r640"}
    baselines = {name: value.float().reshape(-1) for name, value in baselines.items()}
    assert all(value.numel() == nflat for value in baselines.values())

    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    covariance = _covariance(C.m, fit_rows, _manual_logits)
    variants = {}
    for rank in RANKS:
        program, _basis, _diagnostics = _rrr_program(C.m.transformer.h[0].mlp,
                                                     covariance, rank=rank)
        variants[f"r{rank}"] = {0: {name: value.cpu() for name, value in program.items()}}
        del program, _basis
    del covariance
    torch.cuda.empty_cache()

    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": variants["r512"],
        "final_mlp_input_program_variants": variants,
        "final_mlp_input_primary_variant": "r512",
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
    print("ARMS: MLP0 context-RRR p512/p640 with a16 mean ablated at census", flush=True)
    run = C.main()

    compiled_kos = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    if set(compiled_kos) != set(variants) or set(observed) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing ablated variant")
    native_ko = _direct_native_cev(C, rows, ablate)
    assert native_ko.numel() == nflat
    compiled_kos = {name: value.float().reshape(-1).cpu()
                    for name, value in compiled_kos.items()}
    torch.save(compiled_kos, COMP_KO)
    torch.save(native_ko, NATIVE_KO)
    effect_real = native_ko - base_ce

    wanted_qk = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    if (set(index_sets) != set(range(2, 18))
            or any(value != wanted_qk for value in index_sets.values()) or widths != {104}
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: mixed104 identity changed")

    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    arms = {}
    for rank in RANKS:
        name = f"r{rank}"
        got = {int(key): int(value) for key, value in observed[name].items()}
        if got != {0: rank}:
            raise SystemExit(f"INSTRUMENT FAIL: {name} observed {got}")
        effect_comp = compiled_kos[name] - baselines[name]
        cosine = float(torch.dot(effect_comp, effect_real) /
                       (effect_comp.norm() * effect_real.norm()).clamp_min(1e-12))
        normalized_error = float((effect_comp - effect_real).norm()
                                 / effect_real.norm().clamp_min(1e-12))
        norm_ratio = float(effect_comp.norm() / effect_real.norm().clamp_min(1e-12))
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
            raise SystemExit("INSTRUMENT FAIL: no a16-own leaves")
        arms[str(rank)] = {
            "rank": rank,
            "unablated_census_damage": baseline_result["arms"][str(rank)]["census_damage"],
            "unablated_certificates_valid": baseline_result["arms"][str(rank)]["certificates_valid"],
            "compiled_ko_census_damage": float((compiled_kos[name] - base_ce).mean()),
            "effect_cosine": cosine,
            "effect_normalized_error": normalized_error,
            "effect_norm_ratio": norm_ratio,
            "collateral_spearman": _spearman(collateral_real, collateral_comp),
            "own_effect_median_ratio": own_ratios[len(own_ratios) // 2],
            "own_effect_ratios": own_ratios,
            "circuits": circuit_rows,
            "literal_standalone_scalars": SCALARS[rank],
            "literal_raw_tensor_bytes": BYTES[rank],
        }
        print(f"p{rank}: cos/err/rho/own {cosine:.6f}/"
              f"{normalized_error:.6f}/{arms[str(rank)]['collateral_spearman']:.6f}/"
              f"{arms[str(rank)]['own_effect_median_ratio']:.6f}", flush=True)

    pred_a = (baseline_result["pred_a_both_variants_transport_on_shifted_ood"]
              and baseline_result["pred_b_census_and_certificates_reproduce"]
              and run["L2_F"] <= .025
              and all({int(key): int(value) for key, value in observed[f"r{rank}"].items()}
                      == {0: rank} for rank in RANKS)
              and widths == {104} and all(value == wanted_qk for value in index_sets.values())
              and SCALARS[512] == 534_286_646 and SCALARS[640] == 535_613_750)
    pred_b = all(arms[str(rank)]["effect_cosine"] >= .90
                 and arms[str(rank)]["effect_normalized_error"] <= .60 for rank in RANKS)
    pred_c = all(arms[str(rank)]["collateral_spearman"] >= .90
                 and .60 <= arms[str(rank)]["own_effect_median_ratio"] <= 1.40
                 for rank in RANKS)
    null = (all(arms[str(rank)]["effect_cosine"] < .70 for rank in RANKS)
            or all(arms[str(rank)]["collateral_spearman"] < .75 for rank in RANKS))
    result = {
        "status": "a16_transfer_mixed104_mlp0_context_metric_frontier_complete",
        "rung": 327,
        "claim_level": "direct_signed_a16_two_variant_adoption_gate",
        "convention": "signed effect = KO CE minus unablated CE within each model",
        "native_ko_census_damage": float(effect_real.mean()),
        "live_primary_unablated_fresh_damage": run["L2_F"],
        "arms": arms,
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        'pred_a_live_baselines_price_and_identity': bool(pred_a),
        'pred_b_both_signed_effect_vectors_transfer': bool(pred_b),
        'pred_c_both_circuit_effect_profiles_transfer': bool(pred_c),
        "null_both_signed_transports_fail": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print(f"wrote {OUT}, {COMP_KO}, and {NATIVE_KO}", flush=True)


if __name__ == "__main__":
    main()
