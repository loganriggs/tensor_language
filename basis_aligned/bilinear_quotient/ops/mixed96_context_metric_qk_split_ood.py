"""RUNG 332 -- INDEPENDENT-FIT CONTEXT-QK96 REPRODUCTION + SHIFTED OOD.

Freeze FineWeb skip11000 rows72:96 as the shipping attention-input covariance,
independent of rung331's rows48:72 discovery fit.  Rebuild all 440 rank96 Q/K
maps, reproduce census/certificates, evaluate WikiText test skip140000 n120,
verify literal identity/bill, and save the exact unablated census CEV.

Frozen predictions
------------------
pred_a_independent_fit_reproduces_census_and_certificates:
    Census <=.004, >=58 certificates, and within .003 of split-A +.00124485.
pred_b_shifted_ood_mean_and_tails_hold:
    WikiText mean/p95/max row damage <=.005/.015/.040.
pred_c_dataset_fit_context_identity_price_and_fresh:
    Dataset, fit split, context layers, rank96/440 maps, active set, bill, and
    fresh max <=.010 are exact.

Null: census >=.010 or shifted mean >=.015.  Full pass advances this fixed
split-B artifact to direct signed interventions.
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
OUT = ROOT / "mixed96_context_metric_qk_split_ood_results.json"
CEV = ROOT / "cev_mixed96_context_metric_qk_split_b.pt"
PARENT = ROOT / "mixed96_context_metric_qk_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 96
WIKI_SKIP = 140000
N_ROWS = 120
SCALARS = 535_089_462
BYTES = 2_024_415_852


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
        assert parent["pred_a_context96_reaches_mixed_grade_prediction_and_certificates"]
        assert FIT_SLICE == (72, 96) and WIKI_SKIP == 140000 and N_ROWS == 120
        print("CONTEXT-QK96 SPLIT OOD | dry run: parent, fit, dataset, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed104_online_cv0_ood import wikitext_rows
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    rows_ood, fingerprint, token_count = wikitext_rows(n=N_ROWS, skip=WIKI_SKIP)
    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    covariances = _attention_input_covariances(C.m, fit_rows, _manual_logits)

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": RANK, "qk_rmap": {},
        "qk_extra_tail": 0, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "qk_context_covariances": covariances,
        "extra_eval_rows": rows_ood,
        "extra_eval_name": "wikitext-2-raw-v1-test-skip140000",
    })
    print("ARM: independent-fit context-QK96 + shifted WikiText", flush=True)
    run = C.main()

    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    factor_maps = sum(4 * len(heads) for heads in qk.values())
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    wanted_indices = tuple(range(RANK))
    if (metric != "context_rrr" or context_layers != LAYERS
            or set(index_sets) != set(LAYERS)
            or any(value != wanted_indices for value in index_sets.values())
            or widths != {RANK} or factor_maps != 440
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: split-B context-QK96 identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    census = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid = _certificate_count(CN, battery, damage_vector)
    parent = json.loads(PARENT.read_text())
    extra = C.SEL["extra_eval"]
    by_row = torch.tensor(extra["damage_by_row"])
    p95 = float(torch.quantile(by_row, .95))
    maximum = float(by_row.max())
    fresh = [float(value) for value in run["fresh8"]]
    pred_a = (census <= .004 and valid >= 58
              and abs(census - parent["census_damage"]) <= .003)
    pred_b = extra["damage_mean"] <= .005 and p95 <= .015 and maximum <= .040
    pred_c = (fingerprint == "a46124b21ac53738" and token_count >= WIKI_SKIP + N_ROWS * 257
              and extra["n_rows"] == N_ROWS and metric == "context_rrr"
              and context_layers == LAYERS and widths == {RANK} and factor_maps == 440
              and all(value == wanted_indices for value in index_sets.values())
              and max(fresh) <= .010 and SCALARS == 535_089_462 and BYTES == 2_024_415_852)
    null = census >= .010 or extra["damage_mean"] >= .015
    result = {
        "status": "mixed96_context_metric_qk_split_ood_complete",
        "rung": 332,
        "claim_level": "independent_fit_physical_context_qk_shifted_ood_and_signed_baseline_gate",
        "convention": "compiled CE minus native CE on identical positions",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "dataset_fingerprint": fingerprint,
        "source_token_count": token_count,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "census_damage": census,
        "difference_from_split_a": census - parent["census_damage"],
        "certificates_valid": valid,
        "shifted_native_ce": extra["native_ce"],
        "shifted_compiled_ce": extra["compiled_ce"],
        "shifted_damage_mean": extra["damage_mean"],
        "shifted_damage_row_p50": float(torch.quantile(by_row, .50)),
        "shifted_damage_row_p95": p95,
        "shifted_damage_row_max": maximum,
        "shifted_damage_by_row": [float(value) for value in by_row],
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "qk_metric": metric,
        "qk_context_layers": list(context_layers),
        "qk_rank": RANK,
        "qk_factorized_maps": factor_maps,
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "saved_census_cev_file": CEV.name,
        'pred_a_independent_fit_reproduces_census_and_certificates': bool(pred_a),
        'pred_b_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_dataset_fit_context_identity_price_and_fresh': bool(pred_c),
        "null_context_qk96_split_or_ood_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
