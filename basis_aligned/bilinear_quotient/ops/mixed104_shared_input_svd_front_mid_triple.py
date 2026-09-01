"""RUNG 319 -- PHYSICAL FRONT/MIDDLE SHARED-INPUT SVD TRIPLE.

Rung318 discovered a banded law: paired Left/Right p768 compression works at
13/15 layers through MLP14 and fails at every layer 15--17.  Form a new,
explicit hypothesis from that boundary rather than choosing the best observed
individuals: one equally spaced representative from the closed front/middle
band, layers {0,7,14}, all at p768.  This set is fixed before census and never
ranked by rung318 damage.

Physically compose it with adopted mixed104 online-c_v0.  Relative to the
539,595,062-scalar parent, three shared-input encoders save 7,962,624 scalars:

    531,632,438 scalars / 2,010,587,756 raw tensor bytes.

Since MLP0 p768 is already adopted, the genuinely new step is adding layers 7
and 14 for another 5,308,416-scalar saving.

Frozen predictions
------------------
pred_a_census_and_certificate_screen:
    Combined census damage <=.035 and at least 35/62 certificates survive.
pred_b_composition_surcharge_is_controlled:
    Surcharge over the adopted MLP0-p768 point lies in [.005,.030].
pred_c_fresh_price_and_identity:
    Fresh8 max <=.060; observed MLP input map is exactly {0:768,7:768,14:768};
    mixed104 QK/active identities and the literal scalar/byte bills are exact.

Null: census damage >=.080 or <=15 certificates survive.  A pass only advances
this fixed program to shifted OOD and signed intervention gates; it is not an
adoption result by itself.
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
OUT = ROOT / "mixed104_shared_input_svd_front_mid_triple_results.json"
PARENT = ROOT / "a16_transfer_mixed104_mlp0_svd768_results.json"
LAYERS = (0, 7, 14)
RANK = 768
D = 1152
H = 4608
PARENT_DAMAGE = 0.00901182
SCALARS = 531_632_438
BYTES = 2_010_587_756


def _saving_per_layer(rank: int) -> int:
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
    result = {}
    for layer in LAYERS:
        mlp = model.transformer.h[layer].mlp
        left = mlp.Left.weight.detach().float()
        right = mlp.Right.weight.detach().float()
        down = mlp.Down.weight.detach().float()
        bias = mlp.Down_bias.detach().float()
        stacked = torch.cat((left, right), dim=0)
        gram = stacked.T @ stacked
        values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
        basis = vectors[:, torch.argsort(values, descending=True)[:RANK]]
        coefficient = stacked @ basis
        result[layer] = {
            "encoder": basis.T.cpu(),
            "left": coefficient[:H].cpu(),
            "right": coefficient[H:].cpu(),
            "down": down.cpu(),
            "bias": bias.cpu(),
        }
    return result


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists()
        assert LAYERS == (0, 7, 14) and _saving_per_layer(RANK) == 2_654_208
        assert 539_595_062 - len(LAYERS) * _saving_per_layer(RANK) == SCALARS
        assert 2_042_438_252 - 4 * len(LAYERS) * _saving_per_layer(RANK) == BYTES
        print("MIXED104 FRONT/MID INPUT-SVD TRIPLE | dry run: set, price, bars valid")
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
    programs = _programs(C.m)
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": programs,
    })
    print("ARM: mixed104 + shared-input SVD768 at fixed layers {0,7,14}", flush=True)
    run = C.main()

    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    wanted_observed = {layer: RANK for layer in LAYERS}
    wanted_qk = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    if (observed != wanted_observed or set(index_sets) != set(range(2, 18))
            or any(value != wanted_qk for value in index_sets.values()) or widths != {104}
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: front/mid triple or mixed104 identity changed")
    for program in programs.values():
        if (program["encoder"].shape != (RANK, D)
                or program["left"].shape != (H, RANK)
                or program["right"].shape != (H, RANK)
                or program["down"].shape != (D, H)):
            raise SystemExit("INSTRUMENT FAIL: shared-input program shape changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    damage_vector = cev - base_ce
    census_damage = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid, member_abs = _certificate_count(CN, battery, damage_vector)
    fresh = [float(value) for value in run["fresh8"]]
    surcharge = census_damage - PARENT_DAMAGE
    pred_a = census_damage <= .035 and valid >= 35
    pred_b = .005 <= surcharge <= .030
    pred_c = (max(fresh) <= .060 and observed == wanted_observed and widths == {104}
              and all(value == wanted_qk for value in index_sets.values())
              and SCALARS == 531_632_438 and BYTES == 2_010_587_756)
    null = census_damage >= .080 or valid <= 15
    result = {
        "status": "mixed104_shared_input_svd_front_mid_triple_complete",
        "rung": 319,
        "claim_level": "physical_fixed_band_composition_census_certificate_fresh_price_gate",
        "convention": "CE added above native; lower is better",
        "layers": list(LAYERS),
        "rank": RANK,
        "census_damage": census_damage,
        "surcharge_over_adopted_mlp0_p768": surcharge,
        "certificates_valid": valid,
        "member_abs_dce": member_abs,
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "mlp_input_program_observed": observed,
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "saving_vs_native_scalars": 545_902_902 - SCALARS,
        "additional_saving_vs_mlp0_p768": 536_940_854 - SCALARS,
        'pred_a_census_and_certificate_screen': bool(pred_a),
        'pred_b_composition_surcharge_is_controlled': bool(pred_b),
        'pred_c_fresh_price_and_identity': bool(pred_c),
        "null_front_mid_composition_is_not_useful": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("member_abs_dce", "fresh8")}, indent=2), flush=True)
    print("MIXED104 FRONT/MID INPUT-SVD TRIPLE DONE", flush=True)


if __name__ == "__main__":
    main()
