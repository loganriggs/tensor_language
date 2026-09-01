"""RUNG 334 -- COMPOSE CONTEXT-QK96 WITH CONTEXT-MLP0 p448.

Physically combine the independently fitted split-B context-QK96 program with
the adopted MLP0 context-RRR p448 program.  Evaluate census, certificates,
fresh rows, and a new shifted WikiText slice.  The exact composed bill is
529,117,494 scalars / 2,000,527,980 raw bytes.

Frozen predictions
------------------
pred_a_census_certificates_and_composition_residual_hold:
    Census <=.018, >=36 certificates, and absolute residual from the additive
    component prediction <=.008.
pred_b_new_shifted_ood_mean_and_tails_hold:
    WikiText skip160000 mean/p95/max <=.020/.050/.100.
pred_c_exact_dual_context_identity_price_and_fresh_hold:
    Q/K context-RRR rank96 at 440 maps, MLP0 context-RRR p448, active set,
    literal bill, dataset identity, and fresh max <=.025 are exact.

Null: census >=.030 or shifted mean >=.040.  A full pass establishes a new
smaller physical Pareto point; it still requires a direct signed gate.
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
OUT = ROOT / "mixed96_context_qk_mlp0_context_p448_ood_results.json"
CEV = ROOT / "cev_mixed96_context_qk_mlp0_context_p448.pt"
QK_PARENT = ROOT / "mixed96_context_metric_qk_split_ood_results.json"
MLP_PARENT = ROOT / "mixed104_mlp0_context_metric_lower_rank_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
QK_FIT = (72, 96)
MLP_FIT = (0, 24)
LAYERS = tuple(range(2, 18))
QK_RANK = 96
MLP_RANK = 448
WIKI_SKIP = 160000
N_ROWS = 120
MIXED104_DAMAGE = 0.00469195
SCALARS = 529_117_494
BYTES = 2_000_527_980


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
        for path in (QK_PARENT, MLP_PARENT, ROOT / f".rowcache/{FIT_CACHE}"):
            assert path.exists(), path
        qk = json.loads(QK_PARENT.read_text())
        mlp = json.loads(MLP_PARENT.read_text())["arms"]["448"]
        assert qk["pred_a_independent_fit_reproduces_census_and_certificates"]
        assert mlp["census_damage"] <= .014
        assert 535_089_462 - 5_971_968 == SCALARS
        assert 2_024_415_852 - 4 * 5_971_968 == BYTES
        print("CONTEXT-QK96 + MLP0-p448 | dry run: parents, splits, bill, bars valid")
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
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    rows_ood, fingerprint, token_count = wikitext_rows(n=N_ROWS, skip=WIKI_SKIP)
    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    qk_fit_rows = cached[QK_FIT[0]:QK_FIT[1], :257].long().contiguous()
    mlp_fit_rows = cached[MLP_FIT[0]:MLP_FIT[1], :257].long().contiguous()
    qk_covariances = _attention_input_covariances(C.m, qk_fit_rows, _manual_logits)
    mlp_covariance = _covariance(C.m, mlp_fit_rows, _manual_logits)
    program0, _basis, diagnostic = _rrr_program(C.m.transformer.h[0].mlp,
                                                mlp_covariance, rank=MLP_RANK)
    program = {0: {name: value.cpu() for name, value in program0.items()}}
    del mlp_covariance, program0, _basis
    torch.cuda.empty_cache()

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
        "qk_context_covariances": qk_covariances,
        "final_mlp_input_programs": program,
        "extra_eval_rows": rows_ood,
        "extra_eval_name": "wikitext-2-raw-v1-test-skip160000",
    })
    print("ARM: split-B context-QK96 + context-MLP0 p448 + new WikiText", flush=True)
    run = C.main()

    wanted_indices = tuple(range(QK_RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    factor_maps = sum(4 * len(heads) for heads in qk.values())
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    if (metric != "context_rrr" or context_layers != LAYERS
            or set(index_sets) != set(LAYERS)
            or any(value != wanted_indices for value in index_sets.values())
            or widths != {QK_RANK} or factor_maps != 440
            or observed != {0: MLP_RANK}
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: composed context identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    census = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    valid = _certificate_count(CN, battery, damage_vector)
    qk_parent = json.loads(QK_PARENT.read_text())
    mlp_parent = json.loads(MLP_PARENT.read_text())["arms"]["448"]
    additive = qk_parent["census_damage"] + mlp_parent["census_damage"] - MIXED104_DAMAGE
    residual = census - additive
    extra = C.SEL["extra_eval"]
    by_row = torch.tensor(extra["damage_by_row"])
    p95 = float(torch.quantile(by_row, .95))
    maximum = float(by_row.max())
    fresh = [float(value) for value in run["fresh8"]]

    pred_a = census <= .018 and valid >= 36 and abs(residual) <= .008
    pred_b = extra["damage_mean"] <= .020 and p95 <= .050 and maximum <= .100
    pred_c = (fingerprint == "a46124b21ac53738"
              and token_count >= WIKI_SKIP + N_ROWS * 257
              and extra["n_rows"] == N_ROWS and metric == "context_rrr"
              and context_layers == LAYERS and widths == {QK_RANK}
              and factor_maps == 440 and observed == {0: MLP_RANK}
              and all(value == wanted_indices for value in index_sets.values())
              and max(fresh) <= .025 and SCALARS == 529_117_494
              and BYTES == 2_000_527_980)
    null = census >= .030 or extra["damage_mean"] >= .040
    result = {
        "status": "mixed96_context_qk_mlp0_context_p448_ood_complete",
        "rung": 334,
        "claim_level": "physical_dual_context_composition_census_ood_price_screen",
        "convention": "compiled CE minus native CE on identical positions",
        "fit_cache": FIT_CACHE,
        "qk_fit_rows_half_open": list(QK_FIT),
        "mlp_fit_rows_half_open": list(MLP_FIT),
        "dataset_fingerprint": fingerprint,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "census_damage": census,
        "certificates_valid": valid,
        "additive_component_prediction": additive,
        "composition_residual": residual,
        "composition_ratio": census / max(additive, 1e-12),
        "shifted_damage_mean": extra["damage_mean"],
        "shifted_damage_row_p50": float(torch.quantile(by_row, .50)),
        "shifted_damage_row_p95": p95,
        "shifted_damage_row_max": maximum,
        "shifted_damage_by_row": [float(value) for value in by_row],
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "qk_metric": metric,
        "qk_context_layers": list(context_layers),
        "qk_rank": QK_RANK,
        "qk_factorized_maps": factor_maps,
        "mlp0_rank": observed[0],
        "mlp0_fit_diagnostic": diagnostic,
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "saved_census_cev_file": CEV.name,
        'pred_a_census_certificates_and_composition_residual_hold': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_exact_dual_context_identity_price_and_fresh_hold': bool(pred_c),
        "null_dual_context_composition_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
