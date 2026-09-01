"""RUNG 337 -- CROSS-FAMILY ADDITIVITY TEST AT MLP0 p512/p640.

Compose fixed split-B context-QK96 with context-MLP0 p512 and p640 in one
physical rebuild.  This discriminates cross-family near-additivity from the
~1.3x tax seen in within-MLP compositions.

Frozen predictions
------------------
pred_a_both_compositions_are_near_additive_and_predictive:
    Each ratio to its additive component prediction <=1.15; p512 <=.010/47
    certificates and p640 <=.007/52 certificates.
pred_b_both_new_shifted_ood_tail_bars_hold:
    WikiText skip200000 p512 mean/p95/max <=.012/.035/.080 and p640
    <=.010/.030/.070.
pred_c_dual_context_variants_identity_price_and_fresh_hold:
    Both MLP ranks, QK context96/440-map identity, dataset, exact bills, active
    set, and primary p512 fresh max <=.015 hold.

Null: either composition ratio >=1.25 or both shifted means >=.025.  A full
pass supports an intra-family-tax law and advances Pareto candidates to signed
testing; failure leaves rung334 as a single positive data point.
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
OUT = ROOT / "mixed96_context_qk_mlp0_context_p512_p640_ood_results.json"
CEV = ROOT / "cev_mixed96_context_qk_mlp0_context_p512_p640.pt"
QK_PARENT = ROOT / "mixed96_context_metric_qk_split_ood_results.json"
MLP_PARENT = ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
QK_FIT = (72, 96)
MLP_FIT = (0, 24)
LAYERS = tuple(range(2, 18))
RANKS = (512, 640)
QK_RANK = 96
WIKI_SKIP = 200000
N_ROWS = 120
MIXED104_DAMAGE = 0.00469195
SCALARS = {512: 529_781_046, 640: 531_108_150}
BYTES = {512: 2_003_182_188, 640: 2_008_490_604}


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
        mlp = json.loads(MLP_PARENT.read_text())
        assert qk["census_damage"] <= .004 and set(mlp["arms"]) == {"512", "640"}
        assert 535_089_462 - 5_308_416 == SCALARS[512]
        assert 535_089_462 - 3_981_312 == SCALARS[640]
        print("QK96 + MLP0 p512/p640 | dry run: parents, splits, bills, bars valid")
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
    qk_rows = cached[QK_FIT[0]:QK_FIT[1], :257].long().contiguous()
    mlp_rows = cached[MLP_FIT[0]:MLP_FIT[1], :257].long().contiguous()
    qk_covariances = _attention_input_covariances(C.m, qk_rows, _manual_logits)
    mlp_covariance = _covariance(C.m, mlp_rows, _manual_logits)
    variants = {}
    diagnostics = {}
    for rank in RANKS:
        program, _basis, diagnostic = _rrr_program(C.m.transformer.h[0].mlp,
                                                   mlp_covariance, rank=rank)
        variants[f"r{rank}"] = {0: {name: value.cpu() for name, value in program.items()}}
        diagnostics[str(rank)] = diagnostic
        del program, _basis
    del mlp_covariance
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
        "final_mlp_input_programs": variants["r512"],
        "final_mlp_input_program_variants": variants,
        "final_mlp_input_primary_variant": "r512",
        "extra_eval_rows": rows_ood,
        "extra_eval_name": "wikitext-2-raw-v1-test-skip200000",
    })
    print("ARMS: context-QK96 + context-MLP0 p512/p640 + new WikiText", flush=True)
    run = C.main()

    cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    extra = C.SEL.get("extra_eval_variants", {})
    if set(cevs) != set(variants) or set(observed) != set(variants) or set(extra) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing cross-family variant")
    wanted_indices = tuple(range(QK_RANK))
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
            or widths != {QK_RANK} or factor_maps != 440
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: context-QK96 identity changed")

    qk_parent = json.loads(QK_PARENT.read_text())
    mlp_parent = json.loads(MLP_PARENT.read_text())["arms"]
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    arms, saved = {}, {}
    for rank in RANKS:
        name = f"r{rank}"
        got = {int(key): int(value) for key, value in observed[name].items()}
        if got != {0: rank}:
            raise SystemExit(f"INSTRUMENT FAIL: {name} observed {got}")
        cev = cevs[name].float().reshape(-1).cpu()
        saved[name] = cev
        damage = cev - base_ce
        census = float(damage.mean())
        additive = qk_parent["census_damage"] + mlp_parent[str(rank)]["census_damage"] - MIXED104_DAMAGE
        shifted = extra[name]
        by_row = torch.tensor(shifted["damage_by_row"])
        arms[str(rank)] = {
            "rank": rank,
            "census_damage": census,
            "certificates_valid": _certificate_count(CN, battery, damage),
            "additive_component_prediction": additive,
            "composition_residual": census - additive,
            "composition_ratio": census / max(additive, 1e-12),
            "shifted_damage_mean": shifted["damage_mean"],
            "shifted_damage_row_p50": float(torch.quantile(by_row, .50)),
            "shifted_damage_row_p95": float(torch.quantile(by_row, .95)),
            "shifted_damage_row_max": float(by_row.max()),
            "shifted_damage_by_row": [float(value) for value in by_row],
            "literal_standalone_scalars": SCALARS[rank],
            "literal_raw_tensor_bytes": BYTES[rank],
        }
        print(f"p{rank}: census {census:+.7f}/{arms[str(rank)]['certificates_valid']}; "
              f"ratio {arms[str(rank)]['composition_ratio']:.4f}; Wiki "
              f"{shifted['damage_mean']:+.7f}/{arms[str(rank)]['shifted_damage_row_p95']:+.7f}/"
              f"{arms[str(rank)]['shifted_damage_row_max']:+.7f}", flush=True)
    torch.save(saved, CEV)

    p512, p640 = arms["512"], arms["640"]
    pred_a = (all(arms[str(rank)]["composition_ratio"] <= 1.15 for rank in RANKS)
              and p512["census_damage"] <= .010 and p512["certificates_valid"] >= 47
              and p640["census_damage"] <= .007 and p640["certificates_valid"] >= 52)
    pred_b = (p512["shifted_damage_mean"] <= .012
              and p512["shifted_damage_row_p95"] <= .035
              and p512["shifted_damage_row_max"] <= .080
              and p640["shifted_damage_mean"] <= .010
              and p640["shifted_damage_row_p95"] <= .030
              and p640["shifted_damage_row_max"] <= .070)
    fresh = [float(value) for value in run["fresh8"]]
    pred_c = (fingerprint == "a46124b21ac53738"
              and token_count >= WIKI_SKIP + N_ROWS * 257
              and all(extra[f"r{rank}"]["n_rows"] == N_ROWS for rank in RANKS)
              and all({int(key): int(value) for key, value in observed[f"r{rank}"].items()}
                      == {0: rank} for rank in RANKS)
              and widths == {QK_RANK} and factor_maps == 440
              and all(value == wanted_indices for value in index_sets.values())
              and max(fresh) <= .015
              and SCALARS == {512: 529_781_046, 640: 531_108_150})
    null = (any(arms[str(rank)]["composition_ratio"] >= 1.25 for rank in RANKS)
            or all(arms[str(rank)]["shifted_damage_mean"] >= .025 for rank in RANKS))
    result = {
        "status": "mixed96_context_qk_mlp0_context_p512_p640_ood_complete",
        "rung": 337,
        "claim_level": "cross_family_additivity_physical_ood_price_test",
        "convention": "compiled CE minus native CE on identical positions",
        "fit_cache": FIT_CACHE,
        "qk_fit_rows_half_open": list(QK_FIT),
        "mlp_fit_rows_half_open": list(MLP_FIT),
        "dataset_fingerprint": fingerprint,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "arms": arms,
        "fit_diagnostics": diagnostics,
        "primary_p512_fresh8": fresh,
        "max_primary_fresh_damage": max(fresh),
        "qk_metric": metric,
        "qk_context_layers": list(context_layers),
        "qk_rank": QK_RANK,
        "qk_factorized_maps": factor_maps,
        "active_replacements": list(active),
        "saved_census_cev_file": CEV.name,
        'pred_a_both_compositions_are_near_additive_and_predictive': bool(pred_a),
        'pred_b_both_new_shifted_ood_tail_bars_hold': bool(pred_b),
        'pred_c_dual_context_variants_identity_price_and_fresh_hold': bool(pred_c),
        "null_cross_family_near_additivity_fails": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print(f"wrote {OUT} and {CEV}", flush=True)


if __name__ == "__main__":
    main()
