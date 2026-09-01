"""RUNG 343 -- CONTEXT-QK80 PLUS CONTEXT-VALUE96 PHYSICAL OOD SCREEN.

Replace every head slice of c_v at layers2--17 by rank96 RRR factors under
the fixed split-B attention-input covariance, on top of the context-QK80
artifact. This third map family saves 3,538,944 more scalars.

Frozen predictions
------------------
pred_a_value96_adds_bounded_census_damage_and_certificates:
    Combined census <=.012, >=45 certificates, surcharge over QK80 <=.008.
pred_b_new_shifted_ood_mean_and_tails_hold:
    WikiText terminal skip270840 n56 mean/p95/max <=.015/.040/.090.
pred_c_exact_qk80_value96_context_identity_price_and_fresh:
    QK80 at 440 maps, value96 at 144 maps, split-B context layers2--17,
    dataset, active set, 522,539,318-scalar bill, and fresh max <=.020.

Null: census >=.025, <=32 certificates, or shifted mean >=.030.  Full pass
advances to signed testing; it does not assume value is cross-family-free.
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
OUT = ROOT / "mixed80_context_qk_value96_context_ood_results.json"
CEV = ROOT / "cev_mixed80_context_qk_value96_context.pt"
PARENT = ROOT / "mixed80_context_metric_qk_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
QK_RANK = 80
VALUE_RANK = 96
WIKI_SKIP = 270840
N_ROWS = 56
SCALARS = 522_539_318
BYTES = 1_974_215_276


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
        assert parent["qk_rank"] == QK_RANK and parent["census_damage"] <= .006
        saving = 16 * 9 * (128 * 1152 - VALUE_RANK * (128 + 1152))
        assert saving == 3_538_944
        assert parent["literal_standalone_scalars"] - saving == SCALARS
        assert parent["literal_raw_tensor_bytes"] - 4 * saving == BYTES
        print("QK80 + VALUE96 CONTEXT | dry run: parent, split, maps, bill, bars valid")
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
        "ext_rows": rows, "cp_swap": 4608, "qk_r": QK_RANK, "qk_rmap": {},
        "qk_extra_tail": 0, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "qk_context_covariances": covariances,
        "value_r": VALUE_RANK,
        "value_context_covariances": covariances,
        "extra_eval_rows": rows_ood,
        "extra_eval_name": f"wikitext-2-raw-v1-test-skip{WIKI_SKIP}",
    })
    print("ARM: context-QK80 + context-value96 + new shifted WikiText", flush=True)
    run = C.main()

    qk_indices = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    qk_widths = {int(factor[0].shape[1]) for heads in qk.values()
                 for factors in heads.values() for factor in factors}
    qk_maps = sum(4 * len(heads) for heads in qk.values())
    values = C.SEL.get("_VR", {})
    value_widths = {int(factor[0].shape[1]) for heads in values.values()
                    for factor in heads.values()}
    value_maps = sum(len(heads) for heads in values.values())
    active = tuple(C.SEL.get("_ORDER2", ()))
    qk_metric = C.SEL.get("_QK_METRIC")
    qk_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    value_metric = C.SEL.get("_VALUE_METRIC")
    value_layers = tuple(C.SEL.get("_VALUE_CONTEXT_LAYERS", ()))
    wanted_qk = tuple(range(QK_RANK))
    if (qk_metric != "context_rrr" or qk_layers != LAYERS
            or value_metric != "context_rrr" or value_layers != LAYERS
            or set(qk_indices) != set(LAYERS)
            or any(index != wanted_qk for index in qk_indices.values())
            or qk_widths != {QK_RANK} or qk_maps != 440
            or value_widths != {VALUE_RANK} or value_maps != 144
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: QK80/value96 context identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage = cev - base_ce
    census = float(damage.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid = _certificate_count(CN, battery, damage)
    parent = json.loads(PARENT.read_text())
    surcharge = census - parent["census_damage"]
    extra = C.SEL["extra_eval"]
    by_row = torch.tensor(extra["damage_by_row"])
    p95 = float(torch.quantile(by_row, .95))
    maximum = float(by_row.max())
    fresh = [float(value) for value in run["fresh8"]]

    pred_a = census <= .012 and valid >= 45 and surcharge <= .008
    pred_b = extra["damage_mean"] <= .015 and p95 <= .040 and maximum <= .090
    pred_c = (fingerprint == "a46124b21ac53738"
              and token_count >= WIKI_SKIP + N_ROWS * 257
              and extra["n_rows"] == N_ROWS
              and qk_metric == "context_rrr" and qk_layers == LAYERS
              and value_metric == "context_rrr" and value_layers == LAYERS
              and qk_widths == {QK_RANK} and qk_maps == 440
              and value_widths == {VALUE_RANK} and value_maps == 144
              and max(fresh) <= .020 and SCALARS == 522_539_318
              and BYTES == 1_974_215_276)
    null = census >= .025 or valid <= 32 or extra["damage_mean"] >= .030
    result = {
        "status": "mixed80_context_qk_value96_context_ood_complete",
        "rung": 343,
        "claim_level": "physical_context_value_third_family_census_ood_price_screen",
        "convention": "compiled CE minus native CE on identical positions",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "dataset_fingerprint": fingerprint,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "census_damage": census,
        "surcharge_vs_context_qk80": surcharge,
        "certificates_valid": valid,
        "shifted_damage_mean": extra["damage_mean"],
        "shifted_damage_row_p50": float(torch.quantile(by_row, .50)),
        "shifted_damage_row_p95": p95,
        "shifted_damage_row_max": maximum,
        "shifted_damage_by_row": [float(value) for value in by_row],
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "qk_metric": qk_metric,
        "qk_context_layers": list(qk_layers),
        "qk_rank": QK_RANK,
        "qk_factorized_maps": qk_maps,
        "value_metric": value_metric,
        "value_context_layers": list(value_layers),
        "value_rank": VALUE_RANK,
        "value_factorized_maps": value_maps,
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "saved_census_cev_file": CEV.name,
        'pred_a_value96_adds_bounded_census_damage_and_certificates': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_exact_qk80_value96_context_identity_price_and_fresh': bool(pred_c),
        "null_context_value96_is_not_useful": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
