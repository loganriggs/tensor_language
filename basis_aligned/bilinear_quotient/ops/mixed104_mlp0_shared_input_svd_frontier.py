"""RUNG 315 -- PHYSICAL MIXED104 + MLP0 SHARED-INPUT SVD FRONTIER.

Rung 314's registered exact-token RRR hypothesis failed, but its matched control
found a simple weight-level structure: factor the concatenated MLP0 Left/Right
maps through one shared right-singular basis.  On contextual text, rank512 adds
only +.01662/+ .01012 FineWeb/WikiText while saving 5,308,416 scalars; rank768
adds +.00355/+.00278 while saving 2,654,208.

Physically compose both controls with adopted mixed104 online-c_v0 in one frozen
rebuild and one census.  The literal standalone proposals are:

    p512: 534,286,646 scalars / 2,021,204,588 bytes
    p768: 536,940,854 scalars / 2,031,821,420 bytes.

Frozen predictions
------------------
pred_a_rank_specific_census_and_certificate_bars:
    p512 census <=.030 and >=35/62 certificates, AND p768 census <=.015 and
    >=48/62 certificates.
pred_b_monotone_composition_and_surcharge:
    p768 census damage < p512; both MLP surcharges over adopted mixed104 lie
    in [0,.030].
pred_c_fresh_price_and_identity:
    Primary p512 fresh8 max <=.05; observed MLP0 ranks, shared-map shapes,
    mixed104 QK indices/width, active set, and both literal bills are exact.

Null: both arms have census >=.05 or both retain <=20 certificates.  A pass is
a physical prediction/certificate/fresh/price gate; shifted OOD and signed
interventions still gate adoption.
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
OUT = ROOT / "mixed104_mlp0_shared_input_svd_frontier_results.json"
RANKS = (512, 768)
D = 1152
H = 4608
ADOPTED_DAMAGE = 0.00469195
ADOPTED_SCALARS = 539_595_062
ADOPTED_BYTES = 2_042_438_252


def _saving(rank: int) -> int:
    return 2 * H * D - rank * (D + 2 * H)


def _certificate_count(CN, battery, damage):
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
def _programs(model):
    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    stacked = torch.cat((left, right), dim=0)
    gram = stacked.T @ stacked
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    vectors = vectors[:, torch.argsort(values, descending=True)]
    result = {}
    for rank in RANKS:
        basis = vectors[:, :rank]
        coefficient = stacked @ basis
        result[f"r{rank}"] = {0: {
            "encoder": basis.T.cpu(),
            "left": coefficient[:H].cpu(),
            "right": coefficient[H:].cpu(),
            "down": down.cpu(),
            "bias": bias.cpu(),
        }}
    return result


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / "mlp0_exact_token_shared_input_encoder_results.json").exists()
        assert (ROOT / "mixed104_online_cv0_results.json").exists()
        assert ADOPTED_SCALARS - _saving(512) == 534_286_646
        assert ADOPTED_SCALARS - _saving(768) == 536_940_854
        assert ADOPTED_BYTES - 4 * _saving(512) == 2_021_204_588
        assert ADOPTED_BYTES - 4 * _saving(768) == 2_031_821_420
        print("MIXED104 MLP0 SHARED-INPUT SVD | dry run: variants, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    variants = _programs(C.m)
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": variants["r512"],
        "final_mlp_input_program_variants": variants,
        "final_mlp_input_primary_variant": "r512",
    })
    print("ARM FAMILY: mixed104 + MLP0 shared-input SVD ranks 512/768", flush=True)
    run = C.main()
    variant_cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    if set(variant_cevs) != set(variants) or set(observed) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing MLP input-program variant output")
    for rank in RANKS:
        got = {int(key): int(value) for key, value in observed[f"r{rank}"].items()}
        if got != {0: rank}:
            raise SystemExit(f"INSTRUMENT FAIL: r{rank} observed {got}")
        program = variants[f"r{rank}"][0]
        if (program["encoder"].shape != (rank, D)
                or program["left"].shape != (H, rank)
                or program["right"].shape != (H, rank)
                or program["down"].shape != (D, H)):
            raise SystemExit(f"INSTRUMENT FAIL: r{rank} map shapes changed")

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
        saving = _saving(rank)
        arms[str(rank)] = {
            "rank": rank,
            "census_damage": damage,
            "mlp_surcharge_over_adopted": damage - ADOPTED_DAMAGE,
            "certificates_valid": valid,
            "saving_scalars": saving,
            "literal_standalone_scalars": ADOPTED_SCALARS - saving,
            "literal_raw_tensor_bytes": ADOPTED_BYTES - 4 * saving,
            "member_abs_dce": member_abs,
        }
        print(f"r{rank}: census {damage:+.6f}, certs {valid}/62, save {saving:,}", flush=True)

    a512, a768 = arms["512"], arms["768"]
    fresh = [float(value) for value in run["fresh8"]]
    pred_a = (a512["census_damage"] <= .030 and a512["certificates_valid"] >= 35
              and a768["census_damage"] <= .015 and a768["certificates_valid"] >= 48)
    pred_b = (a768["census_damage"] < a512["census_damage"]
              and all(0.0 <= arm["mlp_surcharge_over_adopted"] <= .030
                      for arm in (a512, a768)))
    pred_c = (max(fresh) <= .05
              and all(observed[f"r{rank}"] == {0: rank} for rank in RANKS)
              and widths == {104} and all(value == wanted for value in index_sets.values())
              and a512["literal_standalone_scalars"] == 534_286_646
              and a768["literal_standalone_scalars"] == 536_940_854)
    null = (all(arm["census_damage"] >= .05 for arm in (a512, a768))
            or all(arm["certificates_valid"] <= 20 for arm in (a512, a768)))
    result = {
        "status": "mixed104_mlp0_shared_input_svd_frontier_complete",
        "rung": 315,
        "claim_level": "physical_mixed104_single_mlp_shared_input_census_certificate_fresh_price_gate",
        "convention": "CE added above native; lower is better",
        "arms": arms,
        "primary_r512_fresh8": fresh,
        "mixed_identity": {"qk_indices": list(wanted), "qk_widths": sorted(widths),
                           "active_replacements": list(active)},
        'pred_a_rank_specific_census_and_certificate_bars': bool(pred_a),
        'pred_b_monotone_composition_and_surcharge': bool(pred_b),
        'pred_c_fresh_price_and_identity': bool(pred_c),
        "null_no_useful_shared_input_point": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MIXED104 MLP0 SHARED-INPUT SVD FRONTIER DONE", flush=True)


if __name__ == "__main__":
    main()
