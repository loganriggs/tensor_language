"""RUNG 311 -- TWO-LAYER PCA FRONTIER INSIDE MIXED104.

The physical {0,8,17} PCA triple composed predictively but retained only 8/62
certificates.  Evaluate exactly its three two-layer subsets in one frozen
mixed104 rebuild and one common census:

    {0,8}, {0,17}, {8,17}, all rank 256.

Every pair saves 7,667,712 scalars relative to adopted mixed104, proposing
531,927,350 scalars / 2,011,767,404 raw bytes.  Pair selection happens only
after all three census/certificate results exist; no fresh/OOD credit is earned.

Frozen predictions
------------------
pred_a_some_pair_is_predictive_and_certified:
    Some pair has census damage <=.050 and >=20/62 valid certificates.
pred_b_pair_surcharges_are_in_range:
    Every pair's surcharge over adopted +.00469195 lies in [.025,.055].
pred_c_best_pair_improves_triple_tradeoff:
    The certificate-first best pair has >8 certificates and MLP surcharge per
    saved scalar no worse than the triple (equivalently pair surcharge <=2/3
    of the triple's +.06275862 surcharge).

Null: every pair has census damage >=.08, or every pair has <=5 certificates.
The selected pair still needs fresh/OOD and exact bill confirmation.
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
OUT = ROOT / "mixed104_pca_fixed_pair_frontier_results.json"
ADOPTED_DAMAGE = 0.00469195
ADOPTED_SCALARS = 539_595_062
ADOPTED_BYTES = 2_042_438_252
SAVING_EACH = 3_833_856
TRIPLE_SURCHARGE = 0.06275861807994843
RANK = 256
FIT_ROWS = 16
PAIR_LAYERS = {"p0_8": (0, 8), "p0_17": (0, 17), "p8_17": (8, 17)}


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
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / "mixed104_online_cv0_pca_fixed_triple_results.json").exists()
        assert (ROOT / "census_state_diverse.pt").exists()
        assert ADOPTED_SCALARS - 2 * SAVING_EACH == 531_927_350
        assert ADOPTED_BYTES - 2 * SAVING_EACH * 4 == 2_011_767_404
        assert set(PAIR_LAYERS) == {"p0_8", "p0_17", "p8_17"}
        print("MIXED104 PCA FIXED PAIR FRONTIER | dry run: variants, price, census, bars valid")
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
    pca = pca_base._fit_pca(pca_base._capture_outputs(C.m, fit, _manual_logits))
    variants = {name: {layer: pca[layer] for layer in layers}
                for name, layers in PAIR_LAYERS.items()}
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_projectors": variants["p0_8"],
        "final_mlp_projector_variants": variants,
        "final_mlp_primary_variant": "p0_8",
    })
    print("ARM FAMILY: mixed104 + PCA pairs {0,8}/{0,17}/{8,17}", flush=True)
    run = C.main()
    variant_cevs = C.SEL.get("_final_mlp_variant_cevs", {})
    variant_observed = C.SEL.get("_final_mlp_variant_observed", {})
    if set(variant_cevs) != set(variants) or set(variant_observed) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing pair variant output")
    for name, layers in PAIR_LAYERS.items():
        expected = {layer: RANK for layer in layers}
        observed = {int(key): int(value) for key, value in variant_observed[name].items()}
        if observed != expected:
            raise SystemExit(f"INSTRUMENT FAIL: {name} observed {observed} != {expected}")

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
    for name, cev in variant_cevs.items():
        cev = cev.float().reshape(-1)
        assert cev.numel() == nflat
        damage_vector = cev - base_ce
        valid, member_abs = _certificate_count(CN, battery, damage_vector)
        damage = float(damage_vector.mean())
        arms[name] = {
            "layers": list(PAIR_LAYERS[name]),
            "census_damage": damage,
            "mlp_surcharge_over_adopted": damage - ADOPTED_DAMAGE,
            "certificates_valid": valid,
            "member_abs_dce": member_abs,
        }
        print(f"{name} {PAIR_LAYERS[name]}: census {damage:+.6f}, certs {valid}/62", flush=True)

    ranked = sorted(arms, key=lambda name: (-arms[name]["certificates_valid"], arms[name]["census_damage"]))
    best_name = ranked[0]
    best = arms[best_name]
    pred_a = any(row["census_damage"] <= 0.050 and row["certificates_valid"] >= 20
                 for row in arms.values())
    pred_b = all(0.025 <= row["mlp_surcharge_over_adopted"] <= 0.055 for row in arms.values())
    pred_c = bool(best["certificates_valid"] > 8
                  and best["mlp_surcharge_over_adopted"] <= (2.0 / 3.0) * TRIPLE_SURCHARGE)
    null = all(row["census_damage"] >= 0.08 for row in arms.values()) or all(
        row["certificates_valid"] <= 5 for row in arms.values())
    result = {
        "status": "mixed104_pca_fixed_pair_frontier_complete",
        "rung": 311,
        "claim_level": "common_rebuild_three_pair_census_certificate_frontier_only",
        "convention": "CE added above native; lower is better",
        "price": {"saving_scalars": 2 * SAVING_EACH,
                  "literal_standalone_scalars_each": ADOPTED_SCALARS - 2 * SAVING_EACH,
                  "literal_raw_tensor_bytes_each": ADOPTED_BYTES - 2 * SAVING_EACH * 4},
        "arms": arms,
        "certificate_first_best_pair": best_name,
        "primary_pair_fresh8_diagnostic_only": {"pair": "p0_8", "fresh8": run["fresh8"]},
        "mixed_identity": {"qk_indices": list(wanted), "qk_widths": sorted(widths),
                           "active_replacements": list(active)},
        'pred_a_some_pair_is_predictive_and_certified': bool(pred_a),
        'pred_b_pair_surcharges_are_in_range': bool(pred_b),
        'pred_c_best_pair_improves_triple_tradeoff': bool(pred_c),
        "null_no_useful_pair": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"best": best_name, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MIXED104 PCA FIXED PAIR FRONTIER DONE", flush=True)


if __name__ == "__main__":
    main()
