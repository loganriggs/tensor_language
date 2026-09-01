"""RUNG 368 -- TWO-BYTE QK64 + SELECTED MLP{0,4}@P768.

Conditional on rung367's full pass, rebuild the identical selected program
after source-aware global BF16 rounding and store all QK factors in fp16.
Compute in fp32 and compare all final scores with the original native model.

Frozen predictions
------------------
pred_a_two_byte_census_and_certificates_hold:
    Census <=.015 and >=43 certificates.
pred_b_two_byte_shifted_and_fresh_hold:
    WT103 total mean/p95/max <=.025/.060/.120 and fresh max <=.030.
pred_c_two_byte_selection_identity_and_bill_hold:
    Exact source/selection/fits/maps/dtypes/CEV identities and
    511,758,646 scalars / 1,023,517,292 bytes.

Null: census >=.025 or <=35 certificates.  A pass advances only the already
frozen original-native signed gate; no precision, layer, or rank tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed64_bf16_qk_fp16_mlp04_context_p768_ood_results.json"
CEV = ROOT / "cev_mixed64_bf16_qk_fp16_mlp04_context_p768.pt"
PARENT = ROOT / "mixed64_context_qk_mlp04_context_p768_ood_results.json"
WIKI_SKIP = 240_552
N_ROWS = 120
SCALARS = 511_758_646
BYTES = 1_023_517_292
QK_RANK = 64
RUNG = 368
STATUS = "mixed64_bf16_qk_fp16_mlp04_context_p768_ood_complete"
CLAIM_LEVEL = "physical_two_byte_selected_mlp_qk64_census_certificate_ood_screen"
CENSUS_MAX = .015
CERTIFICATE_MIN = 43
OOD_MEAN_MAX = .025
OOD_P95_MAX = .060
OOD_MAX = .120
FRESH_MAX = .030
NULL_CENSUS = .025
NULL_CERTIFICATES = 35


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists()
        parent = json.loads(PARENT.read_text())
        assert all(parent[key] for key in (
            "pred_a_census_and_certificate_frontier_improves",
            "pred_b_shifted_ood_and_fresh_hold",
            "pred_c_selection_program_identity_and_price_hold"))
        assert parent["selected_layers_ordered"] == [4, 0]
        assert parent["qk_rank"] == QK_RANK
        assert BYTES == 2 * SCALARS and WIKI_SKIP + N_ROWS * 257 > WIKI_SKIP
        print("MIXED64 BF16 QK FP16 MLP04 | dry run: parent, population, bill, bars valid")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import cevdump_ct96 as C
    import mixed64_context_qk_mlp04_context_p768_ood as harness
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp0_tail_robust_context_metric_screen import _score_rows
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import REPOS, _fetch

    rows_ood, fingerprint, token_count = _wikitext103_train_rows(
        n=N_ROWS, width=257, skip=WIKI_SKIP)
    original_ood = _score_rows(C.m, rows_ood, _manual_logits).double()
    checkpoint_path = _fetch(REPOS["bilin18"], "pytorch_model.bin")
    source = torch.load(checkpoint_path, map_location="meta", weights_only=True, mmap=True)
    parameters = dict(C.m.named_parameters())
    assert len(source) == 218 and set(parameters) == set(source)
    source_scalars = sum(tensor.numel() for tensor in source.values())
    source_bytes = sum(tensor.numel() * tensor.element_size() for tensor in source.values())
    assert source_scalars == 545_902_902 and source_bytes == 2_067_669_612
    shapes_before = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
    dtype_scalars = {"torch.float32": 0, "torch.bfloat16": 0}
    source_bf16_exact = True
    changed_tensors = 0
    for name, parameter in parameters.items():
        source_dtype = str(source[name].dtype)
        if source_dtype not in dtype_scalars:
            raise RuntimeError(f"unsupported source dtype {source_dtype} for {name}")
        dtype_scalars[source_dtype] += parameter.numel()
        rounded = parameter.data.bfloat16().float()
        if source_dtype == "torch.bfloat16":
            source_bf16_exact = source_bf16_exact and bool(torch.equal(rounded, parameter.data))
        else:
            changed_tensors += int(bool((rounded != parameter.data).any()))
            parameter.data.copy_(rounded)
    assert dtype_scalars == {"torch.float32": 487_931_904,
                             "torch.bfloat16": 57_970_998}
    assert source_bf16_exact and changed_tensors > 0
    assert {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before
    rounded_ood = _score_rows(C.m, rows_ood, _manual_logits).double()
    broad_by_row = rounded_ood - original_ood

    harness.OUT = OUT
    harness.CEV = CEV
    harness.WIKI_SKIP = WIKI_SKIP
    harness.WIKI_STOP = WIKI_SKIP + N_ROWS * 257
    harness.N_ROWS = N_ROWS
    harness.QK_RANK = QK_RANK
    harness.SCALARS = SCALARS
    harness.BYTES = BYTES
    harness.QK_STORAGE_DTYPE = "float16"
    harness.EXPECTED_QK_FACTOR_DTYPE = "torch.float16"
    harness.main()

    result = json.loads(OUT.read_text())
    structural_by_row = torch.tensor(result["shifted_damage_by_row"], dtype=torch.float64)
    assert structural_by_row.shape == broad_by_row.shape
    total_by_row = broad_by_row + structural_by_row
    shifted_mean = float(total_by_row.mean())
    shifted_p95 = float(torch.quantile(total_by_row, .95))
    shifted_max = float(total_by_row.max())
    pred_a = (result["census_damage"] <= CENSUS_MAX
              and result["certificates_valid"] >= CERTIFICATE_MIN)
    pred_b = (shifted_mean <= OOD_MEAN_MAX and shifted_p95 <= OOD_P95_MAX
              and shifted_max <= OOD_MAX and result["max_fresh_damage"] <= FRESH_MAX)
    identity = (result["selected_layers_ordered"] == [4, 0]
                and result["mlp_fit_rows_half_open"] == [24, 48]
                and result["qk_fit_rows_half_open"] == [72, 96]
                and result["mlp_input_program_observed"] == {"0": 768, "4": 768}
                and result["qk_metric"] == "context_rrr"
                and result["qk_storage_dtype"] == "float16"
                and result["qk_rank"] == QK_RANK and result["qk_factorized_maps"] == 440
                and result["qk_factor_tensor_dtypes"] == ["torch.float16"]
                and result["saved_census_cev_file"] == CEV.name
                and fingerprint == "7dabb830ac9ebb0d" and token_count == 675_457
                and len(source) == 218 and source_scalars == 545_902_902
                and source_bytes == 2_067_669_612 and source_bf16_exact
                and dtype_scalars == {"torch.float32": 487_931_904,
                                      "torch.bfloat16": 57_970_998}
                and set(parameters) == set(source)
                and {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before
                and BYTES == 2 * SCALARS
                and result["literal_standalone_scalars"] == SCALARS
                and result["literal_raw_tensor_bytes"] == BYTES and CEV.exists())
    pred_c = identity
    null = (result["census_damage"] >= NULL_CENSUS
            or result["certificates_valid"] <= NULL_CERTIFICATES)
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": STATUS,
        "rung": RUNG,
        "claim_level": CLAIM_LEVEL,
        "convention": "compiled CE minus original native CE; lower is better",
        "global_storage_dtype": "source-fp32_to_bfloat16; source-bfloat16_exact",
        "global_compute_dtype": "float32_explicit_dequantization",
        "source_dtype_scalars": dtype_scalars,
        "source_bfloat16_tensors_bit_exact": source_bf16_exact,
        "rounded_fp32_tensors_changed": changed_tensors,
        "dataset_fingerprint": fingerprint,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "shifted_broad_rounding_mean": float(broad_by_row.mean()),
        "shifted_structural_mean": float(structural_by_row.mean()),
        "shifted_damage_mean": shifted_mean,
        "shifted_damage_row_p95": shifted_p95,
        "shifted_damage_row_max": shifted_max,
        "shifted_damage_by_row": [float(value) for value in total_by_row],
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "byte_saving_vs_native": 2_067_669_612 - BYTES,
        "byte_saving_fraction_vs_native": (2_067_669_612 - BYTES) / 2_067_669_612,
        'pred_a_two_byte_census_and_certificates_hold': bool(pred_a),
        'pred_b_two_byte_shifted_and_fresh_hold': bool(pred_b),
        'pred_c_two_byte_selection_identity_and_bill_hold': bool(pred_c),
        "null_two_byte_selected_mlp_qk64_breaks_prediction": bool(null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("member_abs_dce", "shifted_damage_by_row", "fresh8")},
                     indent=2), flush=True)
    print("MIXED64 BF16 QK FP16 MLP04 CONTEXT P768 DONE", flush=True)


if __name__ == "__main__":
    main()
