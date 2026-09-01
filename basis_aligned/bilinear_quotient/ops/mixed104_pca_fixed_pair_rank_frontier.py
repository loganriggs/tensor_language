"""RUNG 312 -- FIXED {8,17} PCA RANK FRONTIER INSIDE MIXED104.

Rung 311 found a smooth rank-256 pair frontier: {8,17} was certificate-first
best at +.04726489 census damage and 19/62 certificates, while saving 7,667,712
scalars.  Do not search another layer subset.  Hold {8,17} fixed and buy rank:

    rank 256: save 7,667,712 -> 531,927,350 standalone scalars
    rank 384: save 6,193,152 -> 533,401,910 standalone scalars
    rank 512: save 4,718,592 -> 534,876,470 standalone scalars

All three arms share one PCA fit, mixed104 rebuild, and census evaluation.

Frozen predictions
------------------
pred_a_capacity_crosses_useful_certificate_bars:
    rank384 has census <=.040 and >=30 certificates, OR rank512 has census
    <=.030 and >=40 certificates.
pred_b_frontier_is_monotone:
    census damage strictly decreases and certificate count does not decrease
    from rank256 -> 384 -> 512.
pred_c_certificate_recovery_beats_scalar_giveback:
    relative to rank256, rank384 gains >=11 certificates for 1,474,560 fewer
    saved scalars, OR rank512 gains >=21 for 2,949,120 fewer saved scalars.

Null: both higher-rank arms retain <=25 certificates, or rank512 census >=.05.
This is a physical census/certificate/price frontier only.  A qualifying arm
still needs fresh/OOD and causal confirmation; no rank may be selected on them.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed104_pca_fixed_pair_rank_frontier_results.json"
LAYERS = (8, 17)
RANKS = (256, 384, 512)
FIT_ROWS = 16
D = 1152
H = 4608
ADOPTED_SCALARS = 539_595_062
ADOPTED_BYTES = 2_042_438_252


def _saving_per_layer(rank: int) -> int:
    return H * D - rank * (H + D)


def _certificate_count(CN, battery: dict[str, object], damage: torch.Tensor) -> tuple[int, dict[str, float]]:
    valid = 0
    member_abs = {}
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage[member].abs().mean())
        member_abs[tag] = round(value, 7)
        valid += int(value < 0.5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])
    return valid, member_abs


@torch.no_grad()
def _fit_selected_pca(model: torch.nn.Module, rows: torch.Tensor, manual_logits):
    captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
    handles = []
    for layer in LAYERS:
        def hook(_module, _args, output, layer=layer):
            captured[layer].append(output.detach().float().cpu().reshape(-1, D))
        handles.append(model.transformer.h[layer].mlp.register_forward_hook(hook))
    try:
        for start in range(0, len(rows), 2):
            manual_logits(model, rows[start:start + 2, :-1].cuda())
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for layer in LAYERS:
        output = torch.cat(captured[layer]).cuda()
        mean = output.mean(0)
        centered = output - mean
        covariance = centered.T @ centered / len(centered)
        covariance = 0.5 * (covariance + covariance.T)
        assert bool(torch.isfinite(covariance).all())
        jitter = 1e-7 * float(torch.diagonal(covariance).mean().abs().clamp_min(1e-12))
        values, vectors = torch.linalg.eigh(covariance + jitter * torch.eye(D, device="cuda"))
        order = torch.argsort(values, descending=True)[: max(RANKS)]
        basis = vectors[:, order]
        result[layer] = (basis.cpu(), mean.cpu())
        print(f"fit layer {layer}: top256/384/512 energy "
              f"{float(values[order[:256]].sum()/values.clamp_min(0).sum()):.4f}/"
              f"{float(values[order[:384]].sum()/values.clamp_min(0).sum()):.4f}/"
              f"{float(values[order].sum()/values.clamp_min(0).sum()):.4f}", flush=True)
    return result


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / "mixed104_pca_fixed_pair_frontier_results.json").exists()
        assert (ROOT / "census_state_diverse.pt").exists()
        expected = {256: 531_927_350, 384: 533_401_910, 512: 534_876_470}
        for rank, total in expected.items():
            assert ADOPTED_SCALARS - 2 * _saving_per_layer(rank) == total
        print("MIXED104 FIXED PAIR RANK FRONTIER | dry run: variants, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    import mlp_activation_pca_four_layer_composition as pca_base
    from mlp0_signed_response_rank_screen import _manual_logits

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    fit = pca_base._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    fitted = _fit_selected_pca(C.m, fit, _manual_logits)
    variants = {
        f"r{rank}": {layer: (basis[:, :rank], mean) for layer, (basis, mean) in fitted.items()}
        for rank in RANKS
    }
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_projectors": variants["r256"],
        "final_mlp_projector_variants": variants,
        "final_mlp_primary_variant": "r256",
    })
    print("ARM FAMILY: mixed104 + PCA {8,17} ranks 256/384/512", flush=True)
    run = C.main()
    variant_cevs = C.SEL.get("_final_mlp_variant_cevs", {})
    variant_observed = C.SEL.get("_final_mlp_variant_observed", {})
    if set(variant_cevs) != set(variants) or set(variant_observed) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing rank variant output")
    for rank in RANKS:
        observed = {int(key): int(value) for key, value in variant_observed[f"r{rank}"].items()}
        if observed != {8: rank, 17: rank}:
            raise SystemExit(f"INSTRUMENT FAIL: r{rank} observed {observed}")

    wanted = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    if (set(index_sets) != set(range(2, 18)) or any(value != wanted for value in index_sets.values())
            or widths != {104} or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: mixed104 identity changed")

    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    arms = {}
    for rank in RANKS:
        cev = variant_cevs[f"r{rank}"].float().reshape(-1)
        assert cev.numel() == nflat
        damage_vector = cev - base_ce
        valid, member_abs = _certificate_count(CN, battery, damage_vector)
        damage = float(damage_vector.mean())
        saving = 2 * _saving_per_layer(rank)
        arms[str(rank)] = {
            "rank": rank,
            "layers": list(LAYERS),
            "census_damage": damage,
            "certificates_valid": valid,
            "saving_scalars": saving,
            "literal_standalone_scalars": ADOPTED_SCALARS - saving,
            "literal_raw_tensor_bytes": ADOPTED_BYTES - 4 * saving,
            "member_abs_dce": member_abs,
        }
        print(f"r{rank}: census {damage:+.6f}, certs {valid}/62, save {saving:,}", flush=True)

    a256, a384, a512 = (arms[str(rank)] for rank in RANKS)
    pred_a = ((a384["census_damage"] <= .040 and a384["certificates_valid"] >= 30)
              or (a512["census_damage"] <= .030 and a512["certificates_valid"] >= 40))
    pred_b = (a256["census_damage"] > a384["census_damage"] > a512["census_damage"]
              and a256["certificates_valid"] <= a384["certificates_valid"]
              <= a512["certificates_valid"])
    pred_c = ((a384["certificates_valid"] - a256["certificates_valid"] >= 11)
              or (a512["certificates_valid"] - a256["certificates_valid"] >= 21))
    null = ((a384["certificates_valid"] <= 25 and a512["certificates_valid"] <= 25)
            or a512["census_damage"] >= .05)
    result = {
        "status": "mixed104_pca_fixed_pair_rank_frontier_complete",
        "rung": 312,
        "claim_level": "fixed_pair_common_rebuild_rank_census_certificate_frontier_only",
        "convention": "CE added above native; lower is better",
        "arms": arms,
        "primary_r256_fresh8_diagnostic_only": [float(value) for value in run["fresh8"]],
        "mixed_identity": {"qk_indices": list(wanted), "qk_widths": sorted(widths),
                           "active_replacements": list(active)},
        'pred_a_capacity_crosses_useful_certificate_bars': bool(pred_a),
        'pred_b_frontier_is_monotone': bool(pred_b),
        'pred_c_certificate_recovery_beats_scalar_giveback': bool(pred_c),
        "null_higher_rank_does_not_recover_certificates": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MIXED104 FIXED PAIR RANK FRONTIER DONE", flush=True)


if __name__ == "__main__":
    main()
