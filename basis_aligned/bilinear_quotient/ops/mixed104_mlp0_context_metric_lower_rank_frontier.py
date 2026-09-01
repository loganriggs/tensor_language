"""RUNG 328 -- PHYSICAL MLP0 CONTEXT-RRR LOWER-RANK FRONTIER.

Context-RRR p640/p512 are fully adopted.  In one common mixed104 rebuild, map
the next three frozen ranks without adaptive iteration: p448, p384, p256.

    p448: 533,623,094 scalars / 2,018,550,380 bytes
    p384: 532,959,542 scalars / 2,015,896,172 bytes
    p256: 531,632,438 scalars / 2,010,587,756 bytes.

Frozen predictions
------------------
pred_a_p448_and_p384_cross_rank_specific_bars:
    p448 census <=.014 and >=43 certs; p384 <=.018 and >=38 certs.
pred_b_p256_is_useful_and_frontier_is_monotone:
    p256 census <=.030 and >=25 certs; damage strictly decreases with rank.
pred_c_primary_fresh_fit_price_and_identity:
    Primary p256 fresh8 max <=.040; frozen contextual fit, all maps,
    mixed104 QK/active set, and all bills are exact.

Null: all arms have census >=.040 OR all retain <=20 certificates.  Only arms
crossing their own rank-specific bar advance to shifted OOD and signed gates;
no intermediate rank is inserted after the result.
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
OUT = ROOT / "mixed104_mlp0_context_metric_lower_rank_frontier_results.json"
RANKS = (256, 384, 448)
FIT_SLICE = (0, 24)
FIT_CACHE = "fineweb_n192_skip11000.pt"
SCALARS = {256: 531_632_438, 384: 532_959_542, 448: 533_623_094}
BYTES = {256: 2_010_587_756, 384: 2_015_896_172, 448: 2_018_550_380}


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
        assert (ROOT / f".rowcache/{FIT_CACHE}").exists()
        for rank in RANKS:
            saving = 2 * 4608 * 1152 - rank * (1152 + 2 * 4608)
            assert 539_595_062 - saving == SCALARS[rank]
            assert 2_042_438_252 - 4 * saving == BYTES[rank]
        print("MLP0 CONTEXT-RRR LOWER RANKS | dry run: ranks, fit, bills, bars valid")
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

    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    covariance = _covariance(C.m, fit_rows, _manual_logits)
    variants = {}
    diagnostics = {}
    for rank in RANKS:
        program, _basis, diagnostic = _rrr_program(C.m.transformer.h[0].mlp,
                                                   covariance, rank=rank)
        variants[f"r{rank}"] = {0: {name: value.cpu() for name, value in program.items()}}
        diagnostics[str(rank)] = diagnostic
        del program, _basis
    del covariance
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
        "final_mlp_input_programs": variants["r256"],
        "final_mlp_input_program_variants": variants,
        "final_mlp_input_primary_variant": "r256",
    })
    print("ARMS: mixed104 + MLP0 context-RRR p256/p384/p448", flush=True)
    run = C.main()
    cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    if set(cevs) != set(variants) or set(observed) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing lower-rank variant")

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
        damage_vector = cevs[name].float().reshape(-1).cpu() - base_ce
        assert damage_vector.numel() == nflat
        valid, member_abs = _certificate_count(CN, battery, damage_vector)
        arms[str(rank)] = {
            "rank": rank,
            "census_damage": float(damage_vector.mean()),
            "certificates_valid": valid,
            "member_abs_dce": member_abs,
            "literal_standalone_scalars": SCALARS[rank],
            "literal_raw_tensor_bytes": BYTES[rank],
        }
        print(f"p{rank}: census {arms[str(rank)]['census_damage']:+.7f}, "
              f"certs {valid}/62", flush=True)

    pred_a = (arms["448"]["census_damage"] <= .014
              and arms["448"]["certificates_valid"] >= 43
              and arms["384"]["census_damage"] <= .018
              and arms["384"]["certificates_valid"] >= 38)
    pred_b = (arms["256"]["census_damage"] <= .030
              and arms["256"]["certificates_valid"] >= 25
              and arms["448"]["census_damage"] < arms["384"]["census_damage"]
              < arms["256"]["census_damage"])
    fresh = [float(value) for value in run["fresh8"]]
    pred_c = (max(fresh) <= .040
              and all({int(key): int(value) for key, value in observed[f"r{rank}"].items()}
                      == {0: rank} for rank in RANKS)
              and widths == {104} and all(value == wanted_qk for value in index_sets.values())
              and all(arms[str(rank)]["literal_standalone_scalars"] == SCALARS[rank]
                      and arms[str(rank)]["literal_raw_tensor_bytes"] == BYTES[rank]
                      for rank in RANKS))
    null = (all(arm["census_damage"] >= .040 for arm in arms.values())
            or all(arm["certificates_valid"] <= 20 for arm in arms.values()))
    result = {
        "status": "mixed104_mlp0_context_metric_lower_rank_frontier_complete",
        "rung": 328,
        "claim_level": "physical_lower_rank_context_metric_census_certificate_fresh_price_frontier",
        "convention": "CE added above native; lower is better",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "fit_diagnostics": diagnostics,
        "arms": arms,
        "primary_p256_fresh8": fresh,
        "max_primary_fresh_damage": max(fresh),
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        'pred_a_p448_and_p384_cross_rank_specific_bars': bool(pred_a),
        'pred_b_p256_is_useful_and_frontier_is_monotone': bool(pred_b),
        'pred_c_primary_fresh_fit_price_and_identity': bool(pred_c),
        "null_lower_rank_context_metric_is_not_useful": bool(null),
        "stop_rule": "no_intermediate_rank_inserted_after_result",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP0 CONTEXT-RRR LOWER-RANK FRONTIER DONE", flush=True)


if __name__ == "__main__":
    main()
