"""RUNG 329 -- LOWER-RANK MLP0 CONTEXT-RRR SHIFTED OOD + CEVS.

Rebuild p448/p384/p256 from rung328, evaluate all on 120 WikiText test chunks
after token120000, reproduce census/certificates, verify literal identities and
bills, and save exact unablated census CEVs.

Frozen predictions
------------------
pred_a_all_rank_specific_ood_bars_hold:
    p448 mean/p95/max <=.018/.045/.090; p384 <=.022/.050/.100;
    p256 <=.035/.065/.120.
pred_b_census_and_certificates_reproduce:
    Every census is within .0015 of rung328 and certs remain >=43/38/25.
pred_c_dataset_fit_price_identity_and_primary_fresh:
    Dataset, fit, all maps, mixed104 identity, bills, and p256 fresh max <=.040.

Null: all shifted means >=.050 OR all p95 values >=.100.  No intermediate rank
is inserted.  OOD-passing arms may receive a common signed gate, with p256
still labeled lower-fidelity unless a separately stated adoption standard is met.
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
OUT = ROOT / "mixed104_mlp0_context_metric_lower_rank_ood_results.json"
CEV = ROOT / "cev_mixed104_mlp0_context_rrr_lower_ranks.pt"
PARENT = ROOT / "mixed104_mlp0_context_metric_lower_rank_frontier_results.json"
RANKS = (256, 384, 448)
FIT_SLICE = (0, 24)
FIT_CACHE = "fineweb_n192_skip11000.pt"
WIKI_SKIP = 120000
N_ROWS = 120
SCALARS = {256: 531_632_438, 384: 532_959_542, 448: 533_623_094}
BYTES = {256: 2_010_587_756, 384: 2_015_896_172, 448: 2_018_550_380}


def _certificate_count(CN, battery, damage):
    valid = 0
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage[member].abs().mean())
        valid += int(value < 0.5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])
    return valid


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert set(parent["arms"]) == {"256", "384", "448"}
        assert WIKI_SKIP == 120000 and N_ROWS == 120
        print("LOWER-RANK CONTEXT-RRR OOD | dry run: parent, variants, dataset, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed104_online_cv0_ood import wikitext_rows
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    rows_ood, fingerprint, token_count = wikitext_rows(n=N_ROWS, skip=WIKI_SKIP)
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
        "extra_eval_rows": rows_ood,
        "extra_eval_name": "wikitext-2-raw-v1-test-skip120000",
    })
    print("ARMS: lower-rank context-RRR + shifted WikiText", flush=True)
    run = C.main()
    cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    extra = C.SEL.get("extra_eval_variants", {})
    if set(cevs) != set(variants) or set(observed) != set(variants) or set(extra) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing lower-rank census/OOD variant")

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
        census = float(damage_vector.mean())
        shifted = extra[name]
        by_row = torch.tensor(shifted["damage_by_row"])
        arms[str(rank)] = {
            "rank": rank,
            "census_damage": census,
            "census_reproduction_error": census - parent["arms"][str(rank)]["census_damage"],
            "certificates_valid": _certificate_count(CN, battery, damage_vector),
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
        print(f"p{rank}: census {census:+.7f}/{arms[str(rank)]['certificates_valid']}; "
              f"Wiki {shifted['damage_mean']:+.7f}/"
              f"{arms[str(rank)]['shifted_damage_row_p95']:+.7f}/"
              f"{arms[str(rank)]['shifted_damage_row_max']:+.7f}", flush=True)
    torch.save(saved, CEV)

    limits = {448: (.018, .045, .090), 384: (.022, .050, .100),
              256: (.035, .065, .120)}
    cert_bars = {448: 43, 384: 38, 256: 25}
    pred_a = all(arms[str(rank)]["shifted_damage_mean"] <= limits[rank][0]
                 and arms[str(rank)]["shifted_damage_row_p95"] <= limits[rank][1]
                 and arms[str(rank)]["shifted_damage_row_max"] <= limits[rank][2]
                 for rank in RANKS)
    pred_b = all(abs(arms[str(rank)]["census_reproduction_error"]) <= .0015
                 and arms[str(rank)]["certificates_valid"] >= cert_bars[rank]
                 for rank in RANKS)
    fresh = [float(value) for value in run["fresh8"]]
    pred_c = (fingerprint == "a46124b21ac53738" and token_count >= WIKI_SKIP + N_ROWS * 257
              and all(extra[f"r{rank}"]["n_rows"] == N_ROWS for rank in RANKS)
              and all({int(key): int(value) for key, value in observed[f"r{rank}"].items()}
                      == {0: rank} for rank in RANKS)
              and widths == {104} and all(value == wanted_qk for value in index_sets.values())
              and max(fresh) <= .040
              and all(SCALARS[rank] == arms[str(rank)]["literal_standalone_scalars"]
                      for rank in RANKS))
    null = (all(arms[str(rank)]["shifted_damage_mean"] >= .050 for rank in RANKS)
            or all(arms[str(rank)]["shifted_damage_row_p95"] >= .100 for rank in RANKS))
    result = {
        "status": "mixed104_mlp0_context_metric_lower_rank_ood_complete",
        "rung": 329,
        "claim_level": "physical_lower_rank_shifted_ood_and_signed_baseline_gate",
        "convention": "compiled CE minus native CE on identical positions",
        "dataset_fingerprint": fingerprint,
        "source_token_count": token_count,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "fit_diagnostics": diagnostics,
        "arms": arms,
        "primary_p256_fresh8": fresh,
        "max_primary_fresh_damage": max(fresh),
        "qk_singular_indices": list(wanted_qk),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        "saved_census_cev_file": CEV.name,
        'pred_a_all_rank_specific_ood_bars_hold': bool(pred_a),
        'pred_b_census_and_certificates_reproduce': bool(pred_b),
        'pred_c_dataset_fit_price_identity_and_primary_fresh': bool(pred_c),
        "null_all_lower_ranks_fail_shifted_transport": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
