"""RUNG 367 -- CONTEXT-QK64 + SCREEN-SELECTED MLP{0,4}@P768.

Apply the frozen rung366 selection rule, build the selected programs from the
independent fit-B covariance, and compose them with fully gated context-QK64.

Frozen predictions
------------------
pred_a_census_and_certificate_frontier_improves:
    Census <=.014 and >=43 certificates at 511,758,646 scalars.
pred_b_shifted_ood_and_fresh_hold:
    WT103 mean/p95/max <=.025/.060/.120 and fresh max <=.030.
pred_c_selection_program_identity_and_price_hold:
    Rule selects layers4,0; exact fit-B p768 programs, context-QK64/440 maps,
    active set, saved CEV, and 511,758,646/1,931,092,588 bill.

Null: census >=.025 or <=35 certificates.  Pass advances one original-native
signed gate and then global BF16 composition; no subset/rank tuning follows.
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
OUT = ROOT / "mixed64_context_qk_mlp04_context_p768_ood_results.json"
CEV = ROOT / "cev_mixed64_context_qk_mlp04_context_p768.pt"
SCREEN = ROOT / "mlp_all_layer_context_metric_shared_input_screen_results.json"
QK_FIT_CACHE = "fineweb_n192_skip11000.pt"
QK_FIT_SLICE = (72, 96)
MLP_FIT_SLICE = (24, 48)
LAYERS = (0, 4)
SELECT_COUNT = 2
EXPECTED_SELECTED = (4, 0)
RANK = 768
QK_RANK = 64
QK_LAYERS = tuple(range(2, 18))
N_ROWS = 120
WIKI_SKIP = 209_712
WIKI_STOP = WIKI_SKIP + N_ROWS * 257
SCALARS = 511_758_646
BYTES = 1_931_092_588
QK_STORAGE_DTYPE = None
EXPECTED_QK_FACTOR_DTYPE = "torch.float32"
CENSUS_MAX = .014
CERTIFICATE_MIN = 43
OOD_MEAN_MAX = .025
OOD_P95_MAX = .060
OOD_MAX = .120
FRESH_MAX = .030
NULL_CENSUS = .025
NULL_CERTIFICATES = 35


def _certificate_count(census_lib, battery, damage):
    valid = 0
    member_abs = {}
    for tag, receipt in battery.items():
        try:
            member = census_lib.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage[member].abs().mean())
        member_abs[tag] = value
        valid += int(value < .5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])
    return valid, member_abs


def _selected_layers(screen):
    eligible = []
    for layer in range(18):
        arm_a = screen["arms"][str(layer)]["context_rrr_fit_a_p768"]
        arm_b = screen["arms"][str(layer)]["context_rrr_fit_b_p768"]
        overlap = screen["diagnostics"][str(layer)]["768"]["whitened_subspace_overlap"]
        if (overlap >= .80
                and max(arm_b["fineweb_damage"], arm_b["wikitext_damage"]) <= .010):
            primary_max = max(arm_a["fineweb_damage"], arm_a["wikitext_damage"])
            eligible.append((primary_max, layer))
    eligible.sort()
    return tuple(layer for _score, layer in eligible[:SELECT_COUNT]), eligible


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert SCREEN.exists() and (ROOT / f".rowcache/{QK_FIT_CACHE}").exists()
        screen = json.loads(SCREEN.read_text())
        selected, _eligible = _selected_layers(screen)
        assert selected == EXPECTED_SELECTED and set(selected) == set(LAYERS)
        assert 517_067_062 - SELECT_COUNT * 2_654_208 == SCALARS
        if QK_STORAGE_DTYPE is None:
            assert 1_952_326_252 - SELECT_COUNT * 4 * 2_654_208 == BYTES
        else:
            assert QK_STORAGE_DTYPE == "float16" and BYTES == 2 * SCALARS
        assert WIKI_STOP == WIKI_SKIP + N_ROWS * 257
        print("QK64 + MLP{0,4}P768 | dry run: selection, fits, price, population, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    import mlp_late_context_metric_shared_input_screen as M
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    screen = json.loads(SCREEN.read_text())
    selected, eligible = _selected_layers(screen)
    assert selected == EXPECTED_SELECTED and set(selected) == set(LAYERS)
    cached = torch.load(ROOT / f".rowcache/{QK_FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    qk_fit_rows = cached[QK_FIT_SLICE[0]:QK_FIT_SLICE[1], :257].long().contiguous()
    mlp_fit_rows = cached[MLP_FIT_SLICE[0]:MLP_FIT_SLICE[1], :257].long().contiguous()
    qk_covariances = _attention_input_covariances(C.m, qk_fit_rows, _manual_logits)
    M.LAYERS = LAYERS
    mlp_covariances = M._covariances(C.m, mlp_fit_rows, _manual_logits)
    programs, fit_diagnostics = {}, {}
    for layer in LAYERS:
        program, _basis, diagnostics = M._rrr_program(
            C.m.transformer.h[layer].mlp, mlp_covariances[layer], rank=RANK)
        programs[layer] = {name: value.cpu() for name, value in program.items()}
        fit_diagnostics[str(layer)] = diagnostics
        del program, _basis
    del mlp_covariances
    torch.cuda.empty_cache()

    rows_ood, fingerprint, token_count = _wikitext103_train_rows(
        n=N_ROWS, width=257, skip=WIKI_SKIP)
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
        "qk_factor_storage_dtype": QK_STORAGE_DTYPE,
        "final_mlp_input_programs": programs,
        "extra_eval_rows": rows_ood,
        "extra_eval_name": f"wikitext-103-raw-v1-train-skip{WIKI_SKIP}",
    })
    print("ARM: context-QK64 + split-B context-MLP{0,4}@p768", flush=True)
    run = C.main()

    observed = {int(key): int(value) for key, value in
                C.SEL.get("_final_mlp_input_programs_observed", {}).items()}
    wanted_observed = {layer: RANK for layer in LAYERS}
    wanted_qk = tuple(range(QK_RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    factor_pairs = [factor for heads in qk.values() for factors in heads.values()
                    for factor in factors]
    widths = {int(factor[0].shape[1]) for factor in factor_pairs}
    factor_dtypes = {str(tensor.dtype) for factor in factor_pairs for tensor in factor}
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    if (observed != wanted_observed or metric != "context_rrr"
            or context_layers != QK_LAYERS or set(index_sets) != set(QK_LAYERS)
            or any(value != wanted_qk for value in index_sets.values())
            or widths != {QK_RANK} or factor_dtypes != {EXPECTED_QK_FACTOR_DTYPE}
            or len(factor_pairs) != 440
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: selected MLP or context-QK64 identity changed")
    for program in programs.values():
        assert program["encoder"].shape == (RANK, 1152)
        assert program["left"].shape == program["right"].shape == (4608, RANK)
        assert program["down"].shape == (1152, 4608)

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    census = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    certificates, member_abs = _certificate_count(CN, battery, damage_vector)
    extra = C.SEL["extra_eval"]
    by_row = torch.tensor(extra["damage_by_row"], dtype=torch.float64)
    p95 = float(torch.quantile(by_row, .95))
    maximum = float(by_row.max())
    fresh = [float(value) for value in run["fresh8"]]

    pred_a = census <= CENSUS_MAX and certificates >= CERTIFICATE_MIN
    pred_b = (extra["damage_mean"] <= OOD_MEAN_MAX and p95 <= OOD_P95_MAX
              and maximum <= OOD_MAX and max(fresh) <= FRESH_MAX)
    pred_c = (selected == EXPECTED_SELECTED and observed == wanted_observed
              and fingerprint == "7dabb830ac9ebb0d" and token_count == 675_457
              and QK_FIT_SLICE == (72, 96) and MLP_FIT_SLICE == (24, 48)
              and metric == "context_rrr" and context_layers == QK_LAYERS
              and widths == {QK_RANK} and factor_dtypes == {EXPECTED_QK_FACTOR_DTYPE}
              and len(factor_pairs) == 440 and all(value == wanted_qk for value in index_sets.values())
              and SCALARS == 517_067_062 - SELECT_COUNT * 2_654_208
              and BYTES > 0 and CEV.exists())
    null = census >= NULL_CENSUS or certificates <= NULL_CERTIFICATES
    result = {
        "status": "mixed64_context_qk_mlp04_context_p768_ood_complete",
        "rung": 367,
        "claim_level": "physical_selected_two_mlp_context_qk64_census_certificate_ood_screen",
        "convention": "compiled CE minus original native CE; lower is better",
        "selection_rule": "two smallest fit-A max among overlap>=.80 and fit-B max<=.010",
        "selection_ranking": [[score, layer] for score, layer in eligible],
        "selected_layers_ordered": list(selected),
        "mlp_fit_cache": QK_FIT_CACHE,
        "mlp_fit_rows_half_open": list(MLP_FIT_SLICE),
        "mlp_fit_diagnostics": fit_diagnostics,
        "mlp_rank": RANK,
        "mlp_input_program_observed": observed,
        "qk_fit_rows_half_open": list(QK_FIT_SLICE),
        "qk_metric": metric,
        "qk_storage_dtype": C.SEL.get("_QK_STORAGE_DTYPE"),
        "qk_context_layers": list(context_layers),
        "qk_rank": QK_RANK,
        "qk_factorized_maps": len(factor_pairs),
        "qk_factor_tensor_dtypes": sorted(factor_dtypes),
        "dataset_fingerprint": fingerprint,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "census_damage": census,
        "certificates_valid": certificates,
        "member_abs_dce": member_abs,
        "shifted_damage_mean": float(extra["damage_mean"]),
        "shifted_damage_row_p95": p95,
        "shifted_damage_row_max": maximum,
        "shifted_damage_by_row": [float(value) for value in by_row],
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "active_replacements": list(active),
        "saved_census_cev_file": CEV.name,
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "saving_vs_native_scalars": 545_902_902 - SCALARS,
        'pred_a_census_and_certificate_frontier_improves': bool(pred_a),
        'pred_b_shifted_ood_and_fresh_hold': bool(pred_b),
        'pred_c_selection_program_identity_and_price_hold': bool(pred_c),
        "null_selected_two_mlp_context_qk64_not_useful": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("member_abs_dce", "shifted_damage_by_row", "fresh8")},
                     indent=2), flush=True)
    print("CONTEXT-QK64 + SELECTED MLP{0,4}@P768 DONE", flush=True)


if __name__ == "__main__":
    main()
