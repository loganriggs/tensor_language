"""RUNG 331 -- CONTEXT-METRIC Q/K RANK96 VERSUS MIXED104 FINE BAND.

Adopted mixed104 stores 440 per-head Q/K maps at rank104 using weight-SVD
indices {0..95,120..127}.  The contextual metric repaired MLP directions that
weight energy ranked last.  Test the same hypothesis at a distinct module:
for every replaced Q/K map W, fit rank96 under the attention-input covariance
C via the SVD of W C^(1/2), storing

    (U_r S_r) [V_r^T C^(-1/2)].

Fit C at layers2--17 on FineWeb skip11000 rows48:72.  No hand-selected fine
band remains.  Removing eight factor dimensions from 440 maps saves 4,505,600
scalars relative to mixed104:

    535,089,462 scalars / 2,024,415,852 raw tensor bytes.

Frozen predictions
------------------
pred_a_context96_reaches_mixed_grade_prediction_and_certificates:
    Census <=.0075 and >=52/62 certificates.
pred_b_context96_improves_weight_top96_at_same_rank:
    Census <=.00753845, at least .001 below physical weight-top96 .00853845,
    and surcharge over mixed104 .00469195 is <=.004.
pred_c_fresh_context_metric_identity_and_price:
    Fresh8 max <=.015; context metric uses exactly layers2--17, rank96, 440
    factorized maps; active/no-table identity and literal bill are exact.

Null: census >=.015 or <=40 certificates.  A pass advances to split-fit
reproduction, shifted OOD, and signed gates before any combination with MLP0.
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
OUT = ROOT / "mixed96_context_metric_qk_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (48, 72)
LAYERS = tuple(range(2, 18))
RANK = 96
TOP96_DAMAGE = 0.00853845
MIXED104_DAMAGE = 0.00469195
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
def _attention_input_covariances(model, rows, manual_logits):
    sums = {layer: torch.zeros(1152, 1152, device="cuda") for layer in LAYERS}
    counts = {layer: 0 for layer in LAYERS}
    handles = []
    for layer in LAYERS:
        def hook(_module, args, layer=layer):
            x = args[0].detach().reshape(-1, 1152).float()
            sums[layer].addmm_(x.T, x)
            counts[layer] += x.shape[0]
        handles.append(model.transformer.h[layer].attn.register_forward_pre_hook(hook))
    try:
        for start in range(0, len(rows), 2):
            manual_logits(model, rows[start:start + 2, :-1].to("cuda"))
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer in LAYERS:
        assert counts[layer] == len(rows) * 256
        covariance = sums[layer] / counts[layer]
        result[layer] = (0.5 * (covariance + covariance.T)).cpu()
    return result


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / f".rowcache/{FIT_CACHE}").exists()
        assert FIT_SLICE == (48, 72) and LAYERS == tuple(range(2, 18))
        assert 539_595_062 - 440 * (128 + 1152) * 8 == SCALARS
        assert 2_042_438_252 - 4 * 440 * (128 + 1152) * 8 == BYTES
        print("CONTEXT-METRIC QK96 | dry run: fit, factor count, bills, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

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
    })
    print("ARM: context-metric QK rank96 at all 440 mixed104 maps", flush=True)
    run = C.main()

    wanted_indices = tuple(range(RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    factor_maps = sum(4 * len(heads) for heads in qk.values())
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    if (metric != "context_rrr" or context_layers != LAYERS
            or set(index_sets) != set(LAYERS)
            or any(value != wanted_indices for value in index_sets.values())
            or widths != {RANK} or factor_maps != 440
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: context-QK96 identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    damage_vector = cev - base_ce
    census_damage = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid = _certificate_count(CN, battery, damage_vector)
    fresh = [float(value) for value in run["fresh8"]]
    pred_a = census_damage <= .0075 and valid >= 52
    pred_b = (census_damage <= TOP96_DAMAGE - .001
              and census_damage - MIXED104_DAMAGE <= .004)
    pred_c = (max(fresh) <= .015 and metric == "context_rrr" and context_layers == LAYERS
              and widths == {RANK} and factor_maps == 440
              and all(value == wanted_indices for value in index_sets.values())
              and SCALARS == 535_089_462 and BYTES == 2_024_415_852)
    null = census_damage >= .015 or valid <= 40
    result = {
        "status": "mixed96_context_metric_qk_complete",
        "rung": 331,
        "claim_level": "physical_context_metric_qk_census_certificate_fresh_price_screen",
        "convention": "CE added above native; lower is better",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "qk_metric": metric,
        "qk_context_layers": list(context_layers),
        "qk_rank": RANK,
        "qk_factorized_maps": factor_maps,
        "census_damage": census_damage,
        "improvement_vs_weight_top96": TOP96_DAMAGE - census_damage,
        "surcharge_vs_mixed104": census_damage - MIXED104_DAMAGE,
        "certificates_valid": valid,
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "qk_factor_widths": sorted(widths),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        'pred_a_context96_reaches_mixed_grade_prediction_and_certificates': bool(pred_a),
        'pred_b_context96_improves_weight_top96_at_same_rank': bool(pred_b),
        'pred_c_fresh_context_metric_identity_and_price': bool(pred_c),
        "null_context_qk96_is_not_useful": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("CONTEXT-METRIC QK96 DONE", flush=True)


if __name__ == "__main__":
    main()
