"""RUNG 325 -- PHYSICAL MIXED104 + MLP0 CONTEXT-RRR p512/p640 FRONTIER.

Rung324's contextual metric makes MLP0 p512/p640 substantially lower-damage
than matched weight SVD on two corpora and two fit halves.  Fit the frozen
primary covariance on FineWeb skip11000 rows0:24 and physically compose both
variants with adopted mixed104 in one rebuild.

Literal standalone proposals:
    p512: 534,286,646 scalars / 2,021,204,588 bytes
    p640: 535,613,750 scalars / 2,026,513,004 bytes.

Frozen predictions
------------------
pred_a_rank_specific_census_and_certificate_bars:
    p512 census <=.012 and >=47/62 certificates, AND p640 census <=.010 and
    >=49/62 certificates.
pred_b_monotone_composition_and_surcharge:
    p640 damage < p512; both surcharges over mixed104 .00469195 are in [0,.015].
pred_c_primary_fresh_fit_price_and_identity:
    Primary p512 fresh8 max <=.025; frozen fit split, both observed maps,
    mixed104 QK/active identities, and both literal bills are exact.

Null: both arms have census >=.030 or both retain <=30 certificates.  A pass
advances qualifying arms to shifted OOD and signed intervention gates.
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
OUT = ROOT / "mixed104_mlp0_context_metric_input_frontier_results.json"
SCREEN = ROOT / "mlp0_context_metric_shared_input_frontier_results.json"
RANKS = (512, 640)
FIT_SLICE = (0, 24)
FIT_CACHE = "fineweb_n192_skip11000.pt"
ADOPTED_DAMAGE = 0.00469195
SCALARS = {512: 534_286_646, 640: 535_613_750}
BYTES = {512: 2_021_204_588, 640: 2_026_513_004}


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
        assert 539_595_062 - 5_308_416 == SCALARS[512]
        assert 539_595_062 - 3_981_312 == SCALARS[640]
        assert 2_042_438_252 - 4 * 5_308_416 == BYTES[512]
        assert 2_042_438_252 - 4 * 3_981_312 == BYTES[640]
        print("MIXED104 MLP0 CONTEXT-RRR | dry run: fit, variants, bills, bars valid")
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
    fit_diagnostics = {}
    for rank in RANKS:
        program, _basis, diagnostics = _rrr_program(C.m.transformer.h[0].mlp,
                                                    covariance, rank=rank)
        variants[f"r{rank}"] = {0: {name: value.cpu() for name, value in program.items()}}
        fit_diagnostics[str(rank)] = diagnostics
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
        "final_mlp_input_programs": variants["r512"],
        "final_mlp_input_program_variants": variants,
        "final_mlp_input_primary_variant": "r512",
    })
    print("ARMS: mixed104 + MLP0 context-RRR p512/p640", flush=True)
    run = C.main()

    cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    if set(cevs) != set(variants) or set(observed) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing context-RRR variant")
    for rank in RANKS:
        got = {int(key): int(value) for key, value in observed[f"r{rank}"].items()}
        if got != {0: rank}:
            raise SystemExit(f"INSTRUMENT FAIL: r{rank} observed {got}")

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
        cev = cevs[f"r{rank}"].float().reshape(-1).cpu()
        assert cev.numel() == nflat
        damage_vector = cev - base_ce
        valid, member_abs = _certificate_count(CN, battery, damage_vector)
        damage = float(damage_vector.mean())
        arms[str(rank)] = {
            "rank": rank,
            "census_damage": damage,
            "surcharge_over_adopted_mixed104": damage - ADOPTED_DAMAGE,
            "certificates_valid": valid,
            "member_abs_dce": member_abs,
            "literal_standalone_scalars": SCALARS[rank],
            "literal_raw_tensor_bytes": BYTES[rank],
        }
        print(f"p{rank}: census {damage:+.7f}, certs {valid}/62", flush=True)

    p512, p640 = arms["512"], arms["640"]
    fresh = [float(value) for value in run["fresh8"]]
    pred_a = (p512["census_damage"] <= .012 and p512["certificates_valid"] >= 47
              and p640["census_damage"] <= .010 and p640["certificates_valid"] >= 49)
    pred_b = (p640["census_damage"] < p512["census_damage"]
              and all(0.0 <= arm["surcharge_over_adopted_mixed104"] <= .015
                      for arm in (p512, p640)))
    pred_c = (max(fresh) <= .025
              and all({int(key): int(value) for key, value in observed[f"r{rank}"].items()}
                      == {0: rank} for rank in RANKS)
              and widths == {104} and all(value == wanted_qk for value in index_sets.values())
              and FIT_CACHE == "fineweb_n192_skip11000.pt" and FIT_SLICE == (0, 24)
              and SCALARS[512] == 534_286_646 and SCALARS[640] == 535_613_750
              and BYTES[512] == 2_021_204_588 and BYTES[640] == 2_026_513_004)
    null = (all(arm["census_damage"] >= .030 for arm in (p512, p640))
            or all(arm["certificates_valid"] <= 30 for arm in (p512, p640)))
    result = {
        "status": "mixed104_mlp0_context_metric_input_frontier_complete",
        "rung": 325,
        "claim_level": "physical_single_site_context_metric_census_certificate_fresh_price_frontier",
        "convention": "CE added above native; lower is better",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "fit_diagnostics": fit_diagnostics,
        "arms": arms,
        "primary_p512_fresh8": fresh,
        "max_primary_fresh_damage": max(fresh),
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        'pred_a_rank_specific_census_and_certificate_bars': bool(pred_a),
        'pred_b_monotone_composition_and_surcharge': bool(pred_b),
        'pred_c_primary_fresh_fit_price_and_identity': bool(pred_c),
        "null_no_useful_context_metric_mlp0_point": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MIXED104 MLP0 CONTEXT-RRR FRONTIER DONE", flush=True)


if __name__ == "__main__":
    main()
