"""RUNG 412 -- PHYSICAL BF16 DERIVED-MLP REPAIR OF THE 43-CERTIFICATE BYTE TIER.

Rungs 368/369 rounded source-native tensors to BF16 and stored QK factors in
fp16, but their generated MLP0/4 programs remained float32. Rebuild the exact
program while casting every shipped MLP program tensor to BF16 before the
runtime hook. The hook explicitly dequantizes to fp32 for computation.

Frozen predictions
------------------
pred_a_physical_program_dtypes_and_bill_hold:
    Exact source/QK/selection/fit identities; every generated MLP tensor BF16;
    exact 26,544,384 generated values and 1,023,517,292-byte whole bill.
pred_b_census_and_certificates_hold:
    Census <=.015 and >=43 certificates.
pred_c_shifted_and_fresh_hold:
    Untouched WT103 [409144,439984) mean/p95/max <=.025/.060/.120 and
    fresh max <=.030.
pred_d_parent_precision_delta_small:
    Versus saved mixed-dtype rung368 CEV, mean/max absolute change <=.003/.050.

Strong null: physical identity failure, census >=.025, or <=35 certificates.
A full pass licenses only an original-native signed gate. Storage encoding is
not interpretability or scalar compression; no dtype/rank/layer/bar tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed64_physical_bf16_mlp04_byte_repair_results.json"
CEV = ROOT / "cev_mixed64_physical_bf16_mlp04_byte_repair.pt"
PARENT_RESULT = ROOT / "mixed64_bf16_qk_fp16_mlp04_context_p768_ood_results.json"
PARENT_CEV = ROOT / "cev_mixed64_bf16_qk_fp16_mlp04_context_p768.pt"
WIKI_SKIP = 409_144
N_ROWS = 120
WIKI_STOP = WIKI_SKIP + N_ROWS * 257
SCALARS = 511_758_646
BYTES = 1_023_517_292
PROGRAM_SCALARS = 26_544_384
PROGRAM_BYTES = 53_088_768
OLD_MIXED_BYTES = 1_076_606_060
RUNG = 412
CENSUS_MAX = .015
CERTIFICATE_MIN = 43
OOD_MEAN_MAX = .025
OOD_P95_MAX = .060
OOD_MAX = .120
FRESH_MAX = .030
PARENT_DELTA_MEAN_MAX = .003
PARENT_DELTA_MAX = .050
NULL_CENSUS = .025
NULL_CERTIFICATES = 35


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT_RESULT.exists() and PARENT_CEV.exists()
        parent_result = json.loads(PARENT_RESULT.read_text())
        assert parent_result["literal_standalone_scalars"] == SCALARS
        assert parent_result["literal_raw_tensor_bytes"] == BYTES
        assert WIKI_STOP == 439_984 and WIKI_STOP < 675_457
        assert BYTES == 2 * SCALARS and PROGRAM_BYTES == 2 * PROGRAM_SCALARS
        assert OLD_MIXED_BYTES == 2 * (SCALARS - PROGRAM_SCALARS) + 4 * PROGRAM_SCALARS
        print("R412 PHYSICAL BF16 DERIVED MLP | dry run: authority, population, bills valid")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import cevdump_ct96 as C
    import mlp_late_context_metric_shared_input_screen as M
    import mixed64_bf16_qk_fp16_mlp04_context_p768_ood as parent

    original_rrr = M._rrr_program

    def physical_bf16_rrr(*args, **kwargs):
        program, basis, diagnostics = original_rrr(*args, **kwargs)
        program = {name: value.to(torch.bfloat16).contiguous()
                   for name, value in program.items()}
        return program, basis, diagnostics

    M._rrr_program = physical_bf16_rrr
    parent.OUT = OUT
    parent.CEV = CEV
    parent.WIKI_SKIP = WIKI_SKIP
    parent.N_ROWS = N_ROWS
    parent.RUNG = RUNG
    parent.STATUS = "mixed64_physical_bf16_mlp04_byte_repair_complete"
    parent.CLAIM_LEVEL = "physical_all_two_byte_selected_mlp_qk64_repair_screen"
    try:
        parent.main()
    finally:
        M._rrr_program = original_rrr

    result = json.loads(OUT.read_text())
    programs = C.SEL["final_mlp_input_programs"]
    assert set(programs) == {0, 4}
    expected_shapes = {
        "encoder": (768, 1152),
        "left": (4608, 768),
        "right": (4608, 768),
        "down": (1152, 4608),
        "bias": (1152,),
    }
    program_dtypes = {}
    program_shapes = {}
    scalar_count = 0
    byte_count = 0
    for layer, program in sorted(programs.items()):
        assert set(program) == set(expected_shapes)
        program_dtypes[str(layer)] = {}
        program_shapes[str(layer)] = {}
        for name, tensor in sorted(program.items()):
            program_dtypes[str(layer)][name] = str(tensor.dtype)
            program_shapes[str(layer)][name] = list(tensor.shape)
            assert tuple(tensor.shape) == expected_shapes[name]
            assert tensor.dtype == torch.bfloat16 and tensor.device.type == "cpu"
            scalar_count += tensor.numel()
            byte_count += tensor.numel() * tensor.element_size()

    new_cev = torch.load(CEV, map_location="cpu").float().reshape(-1)
    parent_cev = torch.load(PARENT_CEV, map_location="cpu").float().reshape(-1)
    assert new_cev.shape == parent_cev.shape
    precision_delta = (new_cev - parent_cev).abs()
    precision_delta_mean = float(precision_delta.mean())
    precision_delta_max = float(precision_delta.max())

    physical_identity = (
        scalar_count == PROGRAM_SCALARS
        and byte_count == PROGRAM_BYTES
        and result["literal_standalone_scalars"] == SCALARS
        and result["literal_raw_tensor_bytes"] == BYTES
        and result["selected_layers_ordered"] == [4, 0]
        and result["mlp_fit_rows_half_open"] == [24, 48]
        and result["qk_fit_rows_half_open"] == [72, 96]
        and result["mlp_input_program_observed"] == {"0": 768, "4": 768}
        and result["qk_factor_tensor_dtypes"] == ["torch.float16"]
        and result["qk_factorized_maps"] == 440
        and result["qk_rank"] == 64
        and result["global_storage_dtype"]
            == "source-fp32_to_bfloat16; source-bfloat16_exact"
        and result["global_compute_dtype"] == "float32_explicit_dequantization"
        and all(dtype == "torch.bfloat16"
                for layer in program_dtypes.values() for dtype in layer.values())
        and BYTES == 2 * SCALARS
        and OLD_MIXED_BYTES
            == 2 * (SCALARS - PROGRAM_SCALARS) + 4 * PROGRAM_SCALARS
    )
    pred_a = physical_identity
    pred_b = (result["census_damage"] <= CENSUS_MAX
              and result["certificates_valid"] >= CERTIFICATE_MIN)
    pred_c = (result["shifted_damage_mean"] <= OOD_MEAN_MAX
              and result["shifted_damage_row_p95"] <= OOD_P95_MAX
              and result["shifted_damage_row_max"] <= OOD_MAX
              and result["max_fresh_damage"] <= FRESH_MAX)
    pred_d = (precision_delta_mean <= PARENT_DELTA_MEAN_MAX
              and precision_delta_max <= PARENT_DELTA_MAX)
    null = (not physical_identity
            or result["census_damage"] >= NULL_CENSUS
            or result["certificates_valid"] <= NULL_CERTIFICATES)

    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed64_physical_bf16_mlp04_byte_repair_complete",
        "rung": RUNG,
        "claim_level": "physical_all_two_byte_selected_mlp_qk64_repair_screen",
        "generated_mlp_program_dtypes": program_dtypes,
        "generated_mlp_program_shapes": program_shapes,
        "generated_mlp_program_scalars": scalar_count,
        "generated_mlp_program_bytes": byte_count,
        "verified_old_mixed_dtype_bytes": OLD_MIXED_BYTES,
        "old_published_all_two_byte_bill_withdrawn_pending_this_repair": True,
        "parent_mixed_dtype_cev": PARENT_CEV.name,
        "parent_precision_delta_mean_abs": precision_delta_mean,
        "parent_precision_delta_max_abs": precision_delta_max,
        "row_construction": {
            "skip_tokens": WIKI_SKIP,
            "stop_tokens": WIKI_STOP,
            "n_rows": N_ROWS,
            "tokens_per_row": 257,
        },
        'pred_a_physical_program_dtypes_and_bill_hold': bool(pred_a),
        'pred_b_census_and_certificates_hold': bool(pred_b),
        'pred_c_shifted_and_fresh_hold': bool(pred_c),
        'pred_d_parent_precision_delta_small': bool(pred_d),
        "null_physical_two_byte_repair_breaks_frontier": bool(null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "census_damage": result["census_damage"],
        "certificates_valid": result["certificates_valid"],
        "shifted": [result["shifted_damage_mean"],
                    result["shifted_damage_row_p95"],
                    result["shifted_damage_row_max"]],
        "fresh_max": result["max_fresh_damage"],
        "program_scalars_bytes": [scalar_count, byte_count],
        "parent_delta_mean_max": [precision_delta_mean, precision_delta_max],
        "predicates": [pred_a, pred_b, pred_c, pred_d],
        "null": null,
    }, indent=2), flush=True)
    print("R412 PHYSICAL BF16 DERIVED MLP BYTE REPAIR DONE", flush=True)


if __name__ == "__main__":
    main()
