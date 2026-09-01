"""RUNG 363 -- UNIVERSAL-BF16 + FP16 CONTEXT-QK56 COMPOSITION.

Round every source-fp32 checkpoint tensor through bfloat16, dequantize to
fp32 for computation, and then build the exact split-B context-QK56 program
on that rounded model.  Store the replacement factors in fp16.  Evaluation
remains relative to the original native model.

Frozen predictions
------------------
pred_a_combined_census_and_certificates_hold:
    Original-native census damage <=.021 and >=38 certificates.
pred_b_combined_shifted_and_fresh_transfer_hold:
    New WT103 total mean/p95/max <=.030/.070/.140 and structural fresh
    max <=.030 (the separately bounded broad-rounding max is .0015).
pred_c_combined_identity_and_two_byte_bill_hold:
    Exact source-aware BF16 identity, split-B context rank56/440 fp16 maps,
    original-native baseline, saved CEV, and 512,561,462 scalars /
    1,025,122,924 bytes.

Null: census >=.040 or <=28 certificates.  A full pass advances one frozen
signed-a16 gate; failure ends broad precision composition without tuning.
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
OUT = ROOT / "mixed56_universal_bf16_qk_fp16_storage_ood_results.json"
CEV = ROOT / "cev_mixed56_universal_bf16_qk_fp16_storage.pt"
BF16_SCREEN = ROOT / "bilin18_universal_bf16_storage_screen_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 56
N_ROWS = 120
WIKI_SKIP = 174_760
WIKI_STOP = WIKI_SKIP + N_ROWS * 257
FACTOR_MAPS = 440
FACTOR_SCALARS = FACTOR_MAPS * (128 + 1152) * RANK
NATIVE_SCALARS = 545_902_902
NATIVE_BYTES = 2_067_669_612
SCALARS = 512_561_462
BYTES = 2 * SCALARS


def _certificate_count(census_lib, battery, damage):
    valid = 0
    for tag, receipt in battery.items():
        try:
            member = census_lib.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage[member].abs().mean())
        valid += int(value < .5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])
    return valid


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BF16_SCREEN.exists()
        screen = json.loads(BF16_SCREEN.read_text())
        assert all(screen[key] for key in (
            "pred_a_bf16_storage_preserves_mean_on_both_corpora",
            "pred_b_bf16_storage_preserves_tails_and_transfers",
            "pred_c_checkpoint_identity_and_two_byte_bill_hold"))
        assert (ROOT / f".rowcache/{FIT_CACHE}").exists()
        assert FACTOR_SCALARS == 31_539_200
        assert SCALARS == 512_561_462 and BYTES == 1_025_122_924
        assert WIKI_SKIP == 174_760 and WIKI_STOP == 205_600
        print("UNIVERSAL-BF16 + FP16-QK56 | dry run: license, price, population, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp0_tail_robust_context_metric_screen import _score_rows
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import REPOS, _fetch

    rows_ood, fingerprint, token_count = _wikitext103_train_rows(
        n=N_ROWS, width=257, skip=WIKI_SKIP)
    original_ood = _score_rows(C.m, rows_ood, _manual_logits).double()

    checkpoint_path = _fetch(REPOS["bilin18"], "pytorch_model.bin")
    source = torch.load(checkpoint_path, map_location="meta", weights_only=True, mmap=True)
    parameters = dict(C.m.named_parameters())
    assert set(parameters) == set(source) and len(source) == 218
    source_scalars = sum(tensor.numel() for tensor in source.values())
    source_bytes = sum(tensor.numel() * tensor.element_size() for tensor in source.values())
    assert source_scalars == NATIVE_SCALARS and source_bytes == NATIVE_BYTES
    shapes_before = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
    dtype_scalars = {"torch.float32": 0, "torch.bfloat16": 0}
    source_bf16_exact = True
    changed_tensors = 0
    changed_scalars = 0
    for name, parameter in parameters.items():
        source_dtype = str(source[name].dtype)
        if source_dtype not in dtype_scalars:
            raise RuntimeError(f"unsupported source dtype {source_dtype} for {name}")
        dtype_scalars[source_dtype] += parameter.numel()
        rounded = parameter.data.bfloat16().float()
        difference = rounded != parameter.data
        if source_dtype == "torch.bfloat16":
            source_bf16_exact = source_bf16_exact and bool(torch.equal(rounded, parameter.data))
        else:
            changed_tensors += int(bool(difference.any()))
            changed_scalars += int(difference.sum())
            parameter.data.copy_(rounded)
    assert sum(dtype_scalars.values()) == NATIVE_SCALARS
    assert changed_tensors > 0 and changed_scalars > 0 and source_bf16_exact
    assert {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before
    rounded_ood = _score_rows(C.m, rows_ood, _manual_logits).double()
    broad_rounding_by_row = rounded_ood - original_ood

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
        "qk_factor_storage_dtype": "float16",
        "extra_eval_rows": rows_ood,
        "extra_eval_name": f"wikitext-103-raw-v1-train-skip{WIKI_SKIP}",
    })
    print("ARM: universal-BF16/fp32-compute + split-B context-QK56 fp16 factors", flush=True)
    run = C.main()

    wanted_indices = tuple(range(RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    factor_pairs = [factor for heads in qk.values() for factors in heads.values()
                    for factor in factors]
    widths = {int(factor[0].shape[1]) for factor in factor_pairs}
    dtypes = {str(tensor.dtype) for factor in factor_pairs for tensor in factor}
    stored_factor_scalars = sum(tensor.numel() for factor in factor_pairs for tensor in factor)
    factor_maps = len(factor_pairs)
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    storage_dtype = C.SEL.get("_QK_STORAGE_DTYPE")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    if (metric != "context_rrr" or storage_dtype != "float16"
            or context_layers != LAYERS or set(index_sets) != set(LAYERS)
            or any(value != wanted_indices for value in index_sets.values())
            or widths != {RANK} or dtypes != {"torch.float16"}
            or factor_maps != FACTOR_MAPS or stored_factor_scalars != FACTOR_SCALARS
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: combined context-QK56 identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    census = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    certificates = _certificate_count(CN, battery, damage_vector)

    extra = C.SEL["extra_eval"]
    structural_by_row = torch.tensor(extra["damage_by_row"], dtype=torch.float64)
    assert structural_by_row.shape == broad_rounding_by_row.shape
    assert abs(float(rounded_ood.mean()) - float(extra["native_ce"])) <= 2e-5
    total_by_row = broad_rounding_by_row + structural_by_row
    shifted_mean = float(total_by_row.mean())
    shifted_p95 = float(torch.quantile(total_by_row, .95))
    shifted_max = float(total_by_row.max())
    fresh = [float(value) for value in run["fresh8"]]

    identity = (fingerprint == "7dabb830ac9ebb0d" and token_count == 675_457
                and source_scalars == NATIVE_SCALARS and source_bytes == NATIVE_BYTES
                and dtype_scalars == {"torch.float32": 487_931_904,
                                      "torch.bfloat16": 57_970_998}
                and source_bf16_exact and set(parameters) == set(source)
                and {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before
                and metric == "context_rrr" and storage_dtype == "float16"
                and context_layers == LAYERS and widths == {RANK}
                and dtypes == {"torch.float16"} and factor_maps == FACTOR_MAPS
                and stored_factor_scalars == FACTOR_SCALARS
                and all(value == wanted_indices for value in index_sets.values())
                and SCALARS == 512_561_462 and BYTES == 1_025_122_924 and CEV.exists())
    pred_a = census <= .021 and certificates >= 38
    pred_b = (shifted_mean <= .030 and shifted_p95 <= .070 and shifted_max <= .140
              and max(fresh) <= .030)
    pred_c = identity
    null = census >= .040 or certificates <= 28
    result = {
        "status": "mixed56_universal_bf16_qk_fp16_storage_ood_complete",
        "rung": 363,
        "claim_level": "physical_global_bf16_plus_fp16_qk56_census_certificate_ood_screen",
        "convention": "compiled CE minus original native CE on identical positions",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "dataset_fingerprint": fingerprint,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "census_damage": census,
        "certificates_valid": certificates,
        "shifted_damage_mean": shifted_mean,
        "shifted_damage_row_p95": shifted_p95,
        "shifted_damage_row_max": shifted_max,
        "shifted_broad_rounding_mean": float(broad_rounding_by_row.mean()),
        "shifted_structural_mean": float(structural_by_row.mean()),
        "shifted_total_damage_by_row": [float(value) for value in total_by_row],
        "fresh8_structural_vs_rounded_native": fresh,
        "max_fresh_structural_damage": max(fresh),
        "global_storage_dtype": "source-fp32_to_bfloat16; source-bfloat16_exact",
        "global_compute_dtype": "float32_explicit_dequantization",
        "source_dtype_scalars": dtype_scalars,
        "source_bfloat16_tensors_bit_exact": source_bf16_exact,
        "rounded_fp32_tensors_changed": changed_tensors,
        "rounded_fp32_scalars_changed": changed_scalars,
        "qk_metric": metric,
        "qk_storage_dtype": storage_dtype,
        "qk_compute_dtype": "float32_explicit_dequantization",
        "qk_context_layers": list(context_layers),
        "qk_rank": RANK,
        "qk_factorized_maps": factor_maps,
        "qk_factor_scalars": stored_factor_scalars,
        "qk_factor_tensor_dtypes": sorted(dtypes),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "byte_saving_vs_native": NATIVE_BYTES - BYTES,
        "byte_saving_fraction_vs_native": (NATIVE_BYTES - BYTES) / NATIVE_BYTES,
        "saved_census_cev_file": CEV.name,
        'pred_a_combined_census_and_certificates_hold': bool(pred_a),
        'pred_b_combined_shifted_and_fresh_transfer_hold': bool(pred_b),
        'pred_c_combined_identity_and_two_byte_bill_hold': bool(pred_c),
        "null_combined_precision_structure_breaks_prediction": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_total_damage_by_row", "fresh8_structural_vs_rounded_native")},
                     indent=2), flush=True)
    print("UNIVERSAL-BF16 + FP16-QK56 DONE", flush=True)


if __name__ == "__main__":
    main()
