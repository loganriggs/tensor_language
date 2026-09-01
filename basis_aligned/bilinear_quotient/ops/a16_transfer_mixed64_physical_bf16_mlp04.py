"""RUNG 413 -- SIGNED A16 GATE FOR THE PHYSICAL BF16 DERIVED-MLP REPAIR.

Conditioned on rung412, apply the unchanged original-native attention16 mean
knockout to the exact artifact whose generated MLP0/4 tensors are physically
stored in BF16. The existing hook dequantizes them to fp32 for computation.

Frozen predictions
------------------
pred_a_physical_baseline_identity_and_bill_hold:
    R412 <=.015/43, shifted max<=.120, fresh<=.030; exact source BF16,
    QK fp16, generated-MLP BF16 shapes/counts, and 1,023,517,292-byte bill.
pred_b_original_native_signed_effect_holds:
    Cosine >=.98, normalized error <=.30, norm ratio [.90,1.15].
pred_c_circuit_profile_holds:
    Collateral Spearman >=.98 and a16-own median ratio [.90,1.15].

Strong null: cosine <.70, rho <.75, or physical identity failure. Full pass
restores formal adoption of the literal all-two-byte tier; no tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "a16_transfer_mixed64_physical_bf16_mlp04_results.json"
BASE_RESULT = ROOT / "mixed64_physical_bf16_mlp04_byte_repair_results.json"
BASE_CEV = ROOT / "cev_mixed64_physical_bf16_mlp04_byte_repair.pt"
COMP_KO = ROOT / "cev_a16ko_mixed64_physical_bf16_mlp04.pt"
SCALARS = 511_758_646
BYTES = 1_023_517_292
PROGRAM_SCALARS = 26_544_384
PROGRAM_BYTES = 53_088_768
RUNG = 413


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BASE_RESULT.exists() and BASE_CEV.exists()
        baseline = json.loads(BASE_RESULT.read_text())
        assert all(baseline[key] for key in (
            "pred_a_physical_program_dtypes_and_bill_hold",
            "pred_b_census_and_certificates_hold",
            "pred_c_shifted_and_fresh_hold",
            "pred_d_parent_precision_delta_small"))
        assert baseline["literal_standalone_scalars"] == SCALARS
        assert baseline["literal_raw_tensor_bytes"] == BYTES
        assert baseline["generated_mlp_program_scalars"] == PROGRAM_SCALARS
        assert baseline["generated_mlp_program_bytes"] == PROGRAM_BYTES
        print("R413 SIGNED PHYSICAL BF16 MLP | dry run: parent and bills valid")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import a16_transfer_mixed64_bf16_qk_fp16_mlp04_p768 as parent
    import cevdump_ct96 as C
    import mlp_late_context_metric_shared_input_screen as M

    original_rrr = M._rrr_program

    def physical_bf16_rrr(*args, **kwargs):
        program, basis, diagnostics = original_rrr(*args, **kwargs)
        program = {name: value.to(torch.bfloat16).contiguous()
                   for name, value in program.items()}
        return program, basis, diagnostics

    M._rrr_program = physical_bf16_rrr
    parent.OUT = OUT
    parent.BASE_RESULT = BASE_RESULT
    parent.BASE_CEV = BASE_CEV
    parent.COMP_KO = COMP_KO
    parent.RUNG = RUNG
    parent.STATUS = "a16_transfer_mixed64_physical_bf16_mlp04_complete"
    parent.CLAIM_LEVEL = "final_original_native_signed_physical_all_two_byte_adoption_gate"
    try:
        parent.main()
    finally:
        M._rrr_program = original_rrr

    result = json.loads(OUT.read_text())
    programs = C.SEL["final_mlp_input_programs"]
    expected_shapes = {
        "encoder": (768, 1152),
        "left": (4608, 768),
        "right": (4608, 768),
        "down": (1152, 4608),
        "bias": (1152,),
    }
    program_dtypes = {}
    scalar_count = 0
    byte_count = 0
    physical_identity = set(programs) == {0, 4}
    for layer, program in sorted(programs.items()):
        physical_identity = physical_identity and set(program) == set(expected_shapes)
        program_dtypes[str(layer)] = {}
        for name, tensor in sorted(program.items()):
            program_dtypes[str(layer)][name] = str(tensor.dtype)
            physical_identity = (
                physical_identity
                and tuple(tensor.shape) == expected_shapes[name]
                and tensor.dtype == torch.bfloat16
                and tensor.device.type == "cpu"
            )
            scalar_count += tensor.numel()
            byte_count += tensor.numel() * tensor.element_size()
    physical_identity = (
        physical_identity
        and scalar_count == PROGRAM_SCALARS
        and byte_count == PROGRAM_BYTES
        and result["literal_standalone_scalars"] == SCALARS
        and result["literal_raw_tensor_bytes"] == BYTES
        and result["qk_storage_dtype"] == "float16"
        and result["qk_factorized_maps"] == 440
        and result["mlp_input_program_observed"] == {"0": 768, "4": 768}
        and result["global_storage_dtype"]
            == "source-fp32_to_bfloat16; source-bfloat16_exact"
        and BYTES == 2 * SCALARS
    )
    parent_a = bool(result["pred_a_final_baseline_identity_and_bill_hold"])
    parent_b = bool(result["pred_b_final_original_native_signed_effect_holds"])
    parent_c = bool(result["pred_c_final_circuit_profile_holds"])
    pred_a = parent_a and physical_identity
    pred_b = parent_b
    pred_c = parent_c
    null = (not physical_identity
            or result["effect_cosine"] < .70
            or result["collateral_spearman"] < .75)

    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "a16_transfer_mixed64_physical_bf16_mlp04_complete",
        "rung": RUNG,
        "claim_level": "final_original_native_signed_physical_all_two_byte_adoption_gate",
        "generated_mlp_program_dtypes": program_dtypes,
        "generated_mlp_program_scalars": scalar_count,
        "generated_mlp_program_bytes": byte_count,
        'pred_a_physical_baseline_identity_and_bill_hold': bool(pred_a),
        'pred_b_original_native_signed_effect_holds': bool(pred_b),
        'pred_c_circuit_profile_holds': bool(pred_c),
        "null_physical_all_two_byte_signed_transport_fails": bool(null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "baseline": [result["unablated_census_damage"],
                     result["unablated_certificates_valid"]],
        "effect": [result["effect_cosine"],
                   result["effect_normalized_error"],
                   result["effect_norm_ratio"]],
        "circuit": [result["collateral_spearman"],
                    result["own_effect_median_ratio"]],
        "program_scalars_bytes": [scalar_count, byte_count],
        "predicates": [pred_a, pred_b, pred_c],
        "null": null,
    }, indent=2), flush=True)
    print("R413 SIGNED PHYSICAL BF16 MLP ADOPTION GATE DONE", flush=True)


if __name__ == "__main__":
    main()
