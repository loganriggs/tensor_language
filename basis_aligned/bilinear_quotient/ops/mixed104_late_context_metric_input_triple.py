"""RUNG 322 -- PHYSICAL MIXED104 + LATE CONTEXT-RRR INPUT TRIPLE.

Rung321 found a stable operational rank at layers15--17: p768 RRR under
contextual MLP-input covariance scores near zero on two fresh corpora while
matched Frobenius weight SVD is catastrophic.  Fit the same primary program on
frozen FineWeb skip11000 rows0:24, then physically compose the fixed late
triple {15,16,17} with adopted mixed104 online-c_v0.

The maps have exactly the same literal shapes as weight-SVD p768.  Proposed
standalone price:

    531,632,438 scalars / 2,010,587,756 raw tensor bytes.

Frozen predictions
------------------
pred_a_census_and_certificate_screen:
    Combined census damage <=.020 and >=40/62 certificates survive.
pred_b_composition_surcharge_is_controlled:
    Surcharge over adopted mixed104 .00469195 lies in [0,.025].
pred_c_fresh_fit_price_and_identity:
    Fresh8 max <=.045; fit cache/split is exact; observed maps are exactly
    {15:768,16:768,17:768}; mixed104 QK/active identities and bill are exact.

Null: census damage >=.050 or <=20 certificates survive.  A pass advances the
unchanged maps to shifted OOD and direct signed intervention gates; it is not
adoption by itself.
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
OUT = ROOT / "mixed104_late_context_metric_input_triple_results.json"
SCREEN = ROOT / "mlp_late_context_metric_shared_input_screen_results.json"
LAYERS = (15, 16, 17)
RANK = 768
FIT_SLICE = (0, 24)
FIT_CACHE = "fineweb_n192_skip11000.pt"
ADOPTED_DAMAGE = 0.00469195
SCALARS = 531_632_438
BYTES = 2_010_587_756


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
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert SCREEN.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        screen = json.loads(SCREEN.read_text())
        assert screen["fit_rows"]["fit_a"] == list(FIT_SLICE)
        assert screen["fit_rows"]["cache"] == FIT_CACHE
        assert screen["rank"] == RANK and LAYERS == (15, 16, 17)
        assert 539_595_062 - 3 * 2_654_208 == SCALARS
        assert 2_042_438_252 - 12 * 2_654_208 == BYTES
        print("MIXED104 LATE CONTEXT-RRR TRIPLE | dry run: fit, price, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mlp_late_context_metric_shared_input_screen import _covariances, _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    fit_cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    fit_rows = fit_cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    assert fit_rows.shape == (24, 257)
    covariances = _covariances(C.m, fit_rows, _manual_logits)
    programs = {}
    fit_diagnostics = {}
    for layer in LAYERS:
        program, _basis, diagnostics = _rrr_program(C.m.transformer.h[layer].mlp,
                                                    covariances[layer])
        programs[layer] = {name: value.cpu() for name, value in program.items()}
        fit_diagnostics[str(layer)] = diagnostics
        del program, _basis
    del covariances
    torch.cuda.empty_cache()

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": programs,
    })
    print("ARM: mixed104 + late context-RRR p768 at {15,16,17}", flush=True)
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
        raise SystemExit("INSTRUMENT FAIL: late context-RRR or mixed104 identity changed")
    for program in programs.values():
        if (program["encoder"].shape != (RANK, 1152)
                or program["left"].shape != (4608, RANK)
                or program["right"].shape != (4608, RANK)
                or program["down"].shape != (1152, 4608)):
            raise SystemExit("INSTRUMENT FAIL: context-RRR program shape changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    damage_vector = cev - base_ce
    census_damage = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid, member_abs = _certificate_count(CN, battery, damage_vector)
    fresh = [float(value) for value in run["fresh8"]]
    surcharge = census_damage - ADOPTED_DAMAGE
    pred_a = census_damage <= .020 and valid >= 40
    pred_b = 0.0 <= surcharge <= .025
    pred_c = (max(fresh) <= .045 and observed == wanted_observed and widths == {104}
              and all(value == wanted_qk for value in index_sets.values())
              and FIT_CACHE == "fineweb_n192_skip11000.pt" and FIT_SLICE == (0, 24)
              and SCALARS == 531_632_438 and BYTES == 2_010_587_756)
    null = census_damage >= .050 or valid <= 20
    result = {
        "status": "mixed104_late_context_metric_input_triple_complete",
        "rung": 322,
        "claim_level": "physical_fixed_late_context_metric_census_certificate_fresh_price_gate",
        "convention": "CE added above native; lower is better",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "fit_diagnostics": fit_diagnostics,
        "layers": list(LAYERS),
        "rank": RANK,
        "census_damage": census_damage,
        "surcharge_over_adopted_mixed104": surcharge,
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
        'pred_a_census_and_certificate_screen': bool(pred_a),
        'pred_b_composition_surcharge_is_controlled': bool(pred_b),
        'pred_c_fresh_fit_price_and_identity': bool(pred_c),
        "null_late_context_metric_composition_not_useful": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("member_abs_dce", "fresh8")}, indent=2), flush=True)
    print("MIXED104 LATE CONTEXT-RRR TRIPLE DONE", flush=True)


if __name__ == "__main__":
    main()
