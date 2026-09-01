"""RUNG 320 -- FINAL FRONT/MIDDLE SHARED-INPUT SVD PAIR FRONTIER.

The fixed {0,7,14}@p768 triple composed to +.02604 CE and 25/62 certificates
at 531,632,438 scalars.  Aggregate/fresh/price gates held, but the frozen
35-certificate bar failed.  Permit exactly one capacity decrement, in one
common rebuild: remove either endpoint 7 or 14 while retaining already adopted
MLP0.  Compare fixed pairs {0,7} and {0,14}, both at

    534,286,646 scalars / 2,021,204,588 raw tensor bytes.

No other pair, rank, or subset follows this run.  If both qualify, selection is
frozen lexicographically: more certificates, then lower census damage.

Frozen predictions
------------------
pred_a_one_pair_crosses_the_certificate_grade_screen:
    At least one pair has census <=.020 and >=38/62 certificates.
pred_b_both_pairs_lie_on_the_monotone_composition_segment:
    Both census values lie between adopted MLP0 p768's .00901182 and the
    triple's .02604432 + .003 tolerance.
pred_c_primary_fresh_price_and_all_identities:
    Primary {0,7} fresh8 max <=.045; both observed maps, mixed104 QK/active
    identities, and the common literal bill are exact.

Null: both pairs retain <=25 certificates OR both have census >=.040.  A pass
only advances the frozen winner to shifted OOD and signed intervention gates.
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
OUT = ROOT / "mixed104_shared_input_svd_front_mid_pairs_results.json"
TRIPLE = ROOT / "mixed104_shared_input_svd_front_mid_triple_results.json"
SETS = {"pair_0_7": (0, 7), "pair_0_14": (0, 14)}
PRIMARY = "pair_0_7"
RANK = 768
D = 1152
H = 4608
MLP0_PARENT_DAMAGE = 0.00901182
TRIPLE_DAMAGE = 0.02604432
SCALARS = 534_286_646
BYTES = 2_021_204_588


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
def _layer_programs(model):
    result = {}
    for layer in sorted({layer for layers in SETS.values() for layer in layers}):
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
        assert TRIPLE.exists()
        triple = json.loads(TRIPLE.read_text())
        assert abs(triple["census_damage"] - TRIPLE_DAMAGE) <= 1e-6
        assert SETS == {"pair_0_7": (0, 7), "pair_0_14": (0, 14)}
        assert 539_595_062 - 2 * 2_654_208 == SCALARS
        assert 2_042_438_252 - 8 * 2_654_208 == BYTES
        print("MIXED104 FRONT/MID INPUT-SVD PAIRS | dry run: variants, price, bars valid")
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
    layer_programs = _layer_programs(C.m)
    variants = {name: {layer: layer_programs[layer] for layer in layers}
                for name, layers in SETS.items()}
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": variants[PRIMARY],
        "final_mlp_input_program_variants": variants,
        "final_mlp_input_primary_variant": PRIMARY,
    })
    print("ARMS: mixed104 + p768 fixed pairs {0,7} / {0,14}", flush=True)
    run = C.main()

    cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    if set(cevs) != set(SETS) or set(observed) != set(SETS):
        raise SystemExit("INSTRUMENT FAIL: missing pair variant")
    wanted_observed = {name: {layer: RANK for layer in layers}
                       for name, layers in SETS.items()}
    for name in SETS:
        got = {int(key): int(value) for key, value in observed[name].items()}
        if got != wanted_observed[name]:
            raise SystemExit(f"INSTRUMENT FAIL: {name} observed {got}")

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
    for name in SETS:
        cev = cevs[name].float().reshape(-1).cpu()
        assert cev.numel() == nflat
        damage_vector = cev - base_ce
        valid, member_abs = _certificate_count(CN, battery, damage_vector)
        arms[name] = {
            "layers": list(SETS[name]),
            "census_damage": float(damage_vector.mean()),
            "certificates_valid": valid,
            "member_abs_dce": member_abs,
            "literal_standalone_scalars": SCALARS,
            "literal_raw_tensor_bytes": BYTES,
        }
        print(f"{name}: census {arms[name]['census_damage']:+.7f}, "
              f"certs {valid}/62", flush=True)

    qualifying = [name for name, arm in arms.items()
                  if arm["census_damage"] <= .020 and arm["certificates_valid"] >= 38]
    winner = None
    if qualifying:
        winner = sorted(qualifying,
                        key=lambda name: (-arms[name]["certificates_valid"],
                                          arms[name]["census_damage"], name))[0]
    fresh = [float(value) for value in run["fresh8"]]
    pred_a = bool(qualifying)
    pred_b = all(MLP0_PARENT_DAMAGE <= arm["census_damage"] <= TRIPLE_DAMAGE + .003
                 for arm in arms.values())
    pred_c = (max(fresh) <= .045 and widths == {104}
              and all({int(key): int(value) for key, value in observed[name].items()}
                      == wanted_observed[name] for name in SETS)
              and all(value == wanted_qk for value in index_sets.values())
              and SCALARS == 534_286_646 and BYTES == 2_021_204_588)
    null = (all(arm["certificates_valid"] <= 25 for arm in arms.values())
            or all(arm["census_damage"] >= .040 for arm in arms.values()))
    result = {
        "status": "mixed104_shared_input_svd_front_mid_pairs_complete",
        "rung": 320,
        "claim_level": "physical_final_fixed_pair_frontier_census_certificate_fresh_price_gate",
        "convention": "CE added above native; lower is better",
        "arms": arms,
        "jointly_qualifying_pairs": qualifying,
        "frozen_winner": winner,
        "primary_pair_0_7_fresh8": fresh,
        "max_primary_fresh_damage": max(fresh),
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        'pred_a_one_pair_crosses_the_certificate_grade_screen': bool(pred_a),
        'pred_b_both_pairs_lie_on_the_monotone_composition_segment': bool(pred_b),
        'pred_c_primary_fresh_price_and_all_identities': bool(pred_c),
        "null_pair_capacity_is_still_not_useful": bool(null),
        "stop_rule": "no_more_pair_rank_or_subset_search_after_this_run",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualifying": qualifying, "winner": winner,
        "predicates": [pred_a, pred_b, pred_c], "null": null,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("MIXED104 FRONT/MID INPUT-SVD PAIRS DONE", flush=True)


if __name__ == "__main__":
    main()
