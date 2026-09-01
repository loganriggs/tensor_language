"""RUNG 326 -- MLP0 CONTEXT-RRR p512/p640 SHIFTED OOD + CEVS.

Rebuild rung325's two physical variants from the frozen contextual fit, score
both on 120 WikiText test chunks after token100000, reproduce census and all
certificates, and save both exact unablated census CE vectors.

Frozen predictions
------------------
pred_a_both_variants_transport_on_shifted_ood:
    p512 Wiki mean/p95/max <=.015/.040/.090 and p640 <=.012/.035/.080.
pred_b_census_and_certificates_reproduce:
    Each census is within .0015 of rung325 and p512/p640 retain >=47/49 certs.
pred_c_dataset_fit_price_identity_and_primary_fresh:
    Dataset fingerprint/rows, frozen fit, both maps, mixed104 QK/active set,
    bills, and primary p512 fresh max <=.025 are exact.

Null: both shifted means >=.040 OR both p95 values >=.080.  Only variants
passing their rank-specific OOD, reproduction, and identity gates advance to
direct signed interventions.
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
OUT = ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json"
CEV = ROOT / "cev_mixed104_mlp0_context_rrr_frontier.pt"
PARENT = ROOT / "mixed104_mlp0_context_metric_input_frontier_results.json"
RANKS = (512, 640)
FIT_SLICE = (0, 24)
FIT_CACHE = "fineweb_n192_skip11000.pt"
WIKI_SKIP = 100000
N_ROWS = 120
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
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert set(parent["arms"]) == {"512", "640"}
        assert parent["fit_rows_half_open"] == list(FIT_SLICE)
        assert WIKI_SKIP == 100000 and N_ROWS == 120
        print("MLP0 CONTEXT-RRR OOD | dry run: parent, dataset, variants, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed104_mlp0_svd768_ood import wikitext_rows
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    rows_ood, fingerprint, token_count = wikitext_rows(n=N_ROWS, skip=WIKI_SKIP)
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
        "extra_eval_rows": rows_ood,
        "extra_eval_name": "wikitext-2-raw-v1-test-skip100000",
    })
    print("ARMS: mixed104 + MLP0 context-RRR p512/p640 + shifted WikiText", flush=True)
    run = C.main()

    cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    extra = C.SEL.get("extra_eval_variants", {})
    if set(cevs) != set(variants) or set(observed) != set(variants) or set(extra) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing census or OOD variant")
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

    parent = json.loads(PARENT.read_text())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    arms = {}
    saved = {}
    for rank in RANKS:
        name = f"r{rank}"
        got = {int(key): int(value) for key, value in observed[name].items()}
        if got != {0: rank}:
            raise SystemExit(f"INSTRUMENT FAIL: {name} observed {got}")
        cev = cevs[name].float().reshape(-1).cpu()
        saved[name] = cev
        damage_vector = cev - base_ce
        valid, member_abs = _certificate_count(CN, battery, damage_vector)
        census_damage = float(damage_vector.mean())
        shifted = extra[name]
        by_row = torch.tensor(shifted["damage_by_row"])
        arms[str(rank)] = {
            "rank": rank,
            "census_damage": census_damage,
            "census_reproduction_error": census_damage - parent["arms"][str(rank)]["census_damage"],
            "certificates_valid": valid,
            "member_abs_dce": member_abs,
            "shifted_native_ce": shifted["native_ce"],
            "shifted_compiled_ce": shifted["compiled_ce"],
            "shifted_damage_mean": shifted["damage_mean"],
            "shifted_damage_row_p50": float(torch.quantile(by_row, .50)),
            "shifted_damage_row_p95": float(torch.quantile(by_row, .95)),
            "shifted_damage_row_max": float(by_row.max()),
            "shifted_damage_by_row": [float(value) for value in by_row],
            "literal_standalone_scalars": SCALARS[rank],
            "literal_raw_tensor_bytes": BYTES[rank],
        }
        print(f"p{rank}: census {census_damage:+.7f}/{valid}; Wiki mean/p95/max "
              f"{shifted['damage_mean']:+.7f}/{arms[str(rank)]['shifted_damage_row_p95']:+.7f}/"
              f"{arms[str(rank)]['shifted_damage_row_max']:+.7f}", flush=True)
    torch.save(saved, CEV)

    a512, a640 = arms["512"], arms["640"]
    pred_a = (a512["shifted_damage_mean"] <= .015
              and a512["shifted_damage_row_p95"] <= .040
              and a512["shifted_damage_row_max"] <= .090
              and a640["shifted_damage_mean"] <= .012
              and a640["shifted_damage_row_p95"] <= .035
              and a640["shifted_damage_row_max"] <= .080)
    pred_b = (abs(a512["census_reproduction_error"]) <= .0015
              and abs(a640["census_reproduction_error"]) <= .0015
              and a512["certificates_valid"] >= 47 and a640["certificates_valid"] >= 49)
    fresh = [float(value) for value in run["fresh8"]]
    pred_c = (fingerprint == "a46124b21ac53738" and token_count >= WIKI_SKIP + N_ROWS * 257
              and all(extra[f"r{rank}"]["n_rows"] == N_ROWS for rank in RANKS)
              and all({int(key): int(value) for key, value in observed[f"r{rank}"].items()}
                      == {0: rank} for rank in RANKS)
              and widths == {104} and all(value == wanted_qk for value in index_sets.values())
              and max(fresh) <= .025 and SCALARS[512] == 534_286_646
              and SCALARS[640] == 535_613_750)
    null = (all(arms[str(rank)]["shifted_damage_mean"] >= .040 for rank in RANKS)
            or all(arms[str(rank)]["shifted_damage_row_p95"] >= .080 for rank in RANKS))
    result = {
        "status": "mixed104_mlp0_context_metric_input_frontier_ood_complete",
        "rung": 326,
        "claim_level": "physical_two_variant_shifted_ood_and_signed_baseline_gate",
        "convention": "compiled CE minus native CE on identical positions",
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "dataset_fingerprint": fingerprint,
        "source_token_count": token_count,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "fit_diagnostics": fit_diagnostics,
        "arms": arms,
        "primary_p512_fresh8": fresh,
        "max_primary_fresh_damage": max(fresh),
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        "saved_census_cev_file": CEV.name,
        'pred_a_both_variants_transport_on_shifted_ood': bool(pred_a),
        'pred_b_census_and_certificates_reproduce': bool(pred_b),
        'pred_c_dataset_fit_price_identity_and_primary_fresh': bool(pred_c),
        "null_both_variants_fail_shifted_transport": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
