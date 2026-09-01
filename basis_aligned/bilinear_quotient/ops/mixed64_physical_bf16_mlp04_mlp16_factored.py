"""RUNG 414 -- PHYSICAL ALL-TWO-BYTE SUB-500M COMPOSITE.

Rebuild the adopted QK64 + MLP0/4-p768 + factored-MLP16 composite with
source-native BF16, QK factors fp16, both generated MLP programs BF16, and the
14,984-value L16 program BF16. Runtime hooks explicitly dequantize to fp32.

Frozen predictions
------------------
pred_a_physical_dtypes_shapes_hooks_and_bill_hold:
    Exact source/QK/MLP0/4/MLP16 identities, live hooks, no dense form, and
    495,847,230 values / 991,694,460 bytes.
pred_b_census_certificates_and_parent_delta_hold:
    Census <=.070, >=10 certificates, and mean/max absolute CEV change from
    saved rung392 <=.010/.100.
pred_c_additive_composition_law_holds:
    Tax ratio [.90,1.35], normalized-vector cosine >=.95, certificate
    difference <=7.
pred_d_shifted_and_fresh_hold:
    Untouched WT103 [439984,470824) full original-native mean/p95/max
    <=.075/.140/.220 and conditional fresh max <=.040.

Strong null: identity failure, census >=.10, <=5 certificates, shifted mean
>=.10, or inert L16 hook. Pass licenses one signed gate; no tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed64_physical_bf16_mlp04_mlp16_factored_results.json"
CEV = ROOT / "cev_mixed64_physical_bf16_mlp04_mlp16_factored.pt"
PARENT_RESULT = ROOT / "mixed64_mlp04_mlp16_factored_composition_results.json"
PARENT_CEV = ROOT / "cev_mixed64_mlp04_mlp16_factored.pt"
SIGNED_PARENT = ROOT / "a16_transfer_mixed64_mlp04_mlp16_factored_results.json"
SOURCE_FACTOR = ROOT / "mlp16_rank2_quadratic_factored.pt"
PHYSICAL_FACTOR = ROOT / "mlp16_rank2_quadratic_factored_bf16.pt"
WIKI_SKIP = 439_984
N_ROWS = 120
WIKI_STOP = WIKI_SKIP + N_ROWS * 257
SCALARS = 495_847_230
BYTES = 991_694_460
MLP_PROGRAM_SCALARS = 26_544_384
L16_SCALARS = 14_984
L16_BYTES = 29_968
RUNG = 414


def _bf16_factored_prediction(x: torch.Tensor,
                               program: dict[str, torch.Tensor]) -> torch.Tensor:
    form_vectors = program["form_vectors"].float()
    form_values = program["form_values"].float()
    output_directions = program["output_directions"].float()
    constant = program["constant"].float()
    projections = torch.einsum("...d,rkd->...rk", x.float(), form_vectors)
    coefficients = (projections.square() * form_values).sum(-1)
    return coefficients @ output_directions + constant


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (PARENT_RESULT, PARENT_CEV, SIGNED_PARENT, SOURCE_FACTOR):
            assert path.exists(), path
        parent = json.loads(PARENT_RESULT.read_text())
        assert all(parent[key] for key in (
            "pred_a_physical_lower_fidelity_holds",
            "pred_b_additive_composition_law_holds",
            "pred_c_shifted_and_conditional_fresh_hold",
            "pred_d_shipped_program_identity_and_bills_hold"))
        signed = json.loads(SIGNED_PARENT.read_text())
        assert all(signed[key] for key in (
            "pred_a_baseline_physical_identity_and_hooks_hold",
            "pred_b_original_native_signed_effect_holds",
            "pred_c_circuit_profile_holds"))
        assert WIKI_STOP == 470_824 and WIKI_STOP < 675_457
        assert BYTES == 2 * SCALARS and L16_BYTES == 2 * L16_SCALARS
        print("R414 PHYSICAL SUB500 TIER | dry run: parents, population, bills valid")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import cevdump_ct96 as C
    import mixed64_context_qk_mlp04_context_p768_ood as harness
    import mixed64_mlp04_mlp16_factored_composition as parent
    import mlp_late_context_metric_shared_input_screen as M
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
    dtype_scalars = {"torch.float32": 0, "torch.bfloat16": 0}
    source_bf16_exact = True
    changed_tensors = 0
    shapes_before = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
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

    factor_source = torch.load(SOURCE_FACTOR, map_location="cpu")
    factor_bf16 = {name: value.to(torch.bfloat16).contiguous()
                   for name, value in factor_source.items()}
    torch.save(factor_bf16, PHYSICAL_FACTOR)

    original_rrr = M._rrr_program
    original_harness_main = harness.main
    original_factor_prediction = parent._factored_prediction

    def physical_bf16_rrr(*args, **kwargs):
        program, basis, diagnostics = original_rrr(*args, **kwargs)
        program = {name: value.to(torch.bfloat16).contiguous()
                   for name, value in program.items()}
        return program, basis, diagnostics

    def physical_harness_main():
        harness.QK_STORAGE_DTYPE = "float16"
        harness.EXPECTED_QK_FACTOR_DTYPE = "torch.float16"
        return original_harness_main()

    M._rrr_program = physical_bf16_rrr
    harness.main = physical_harness_main
    parent._factored_prediction = _bf16_factored_prediction
    parent.OUT = OUT
    parent.CEV = CEV
    parent.FACTOR_PROGRAM = PHYSICAL_FACTOR
    parent.WIKI_SKIP = WIKI_SKIP
    parent.N_ROWS = N_ROWS
    parent.SCALARS = SCALARS
    parent.BYTES = BYTES
    parent.FACTORED_L16_BYTES = L16_BYTES
    try:
        parent.main()
    finally:
        M._rrr_program = original_rrr
        harness.main = original_harness_main
        parent._factored_prediction = original_factor_prediction

    result = json.loads(OUT.read_text())
    factor_stored = torch.load(PHYSICAL_FACTOR, map_location="cpu")
    expected_factor_shapes = {
        "output_directions": (4, 1152),
        "form_vectors": (4, 2, 1152),
        "form_values": (4, 2),
        "constant": (1152,),
    }
    factor_shapes = {name: tuple(value.shape) for name, value in factor_stored.items()}
    factor_scalars = sum(value.numel() for value in factor_stored.values())
    factor_bytes = sum(value.numel() * value.element_size()
                       for value in factor_stored.values())
    factor_dtypes = sorted({str(value.dtype) for value in factor_stored.values()})
    factor_no_dense = "forms" not in factor_stored and set(factor_stored) == set(expected_factor_shapes)

    programs = C.SEL["final_mlp_input_programs"]
    expected_program_shapes = {
        "encoder": (768, 1152),
        "left": (4608, 768),
        "right": (4608, 768),
        "down": (1152, 4608),
        "bias": (1152,),
    }
    mlp_program_dtypes = {}
    mlp_program_scalars = 0
    mlp_program_bytes = 0
    mlp_program_identity = set(programs) == {0, 4}
    for layer, program in sorted(programs.items()):
        mlp_program_identity = (
            mlp_program_identity and set(program) == set(expected_program_shapes))
        mlp_program_dtypes[str(layer)] = {}
        for name, tensor in sorted(program.items()):
            mlp_program_dtypes[str(layer)][name] = str(tensor.dtype)
            mlp_program_identity = (
                mlp_program_identity
                and tuple(tensor.shape) == expected_program_shapes[name]
                and tensor.dtype == torch.bfloat16
                and tensor.device.type == "cpu")
            mlp_program_scalars += tensor.numel()
            mlp_program_bytes += tensor.numel() * tensor.element_size()

    stored_total_by_row = torch.tensor(
        result["shifted_full_damage_by_row"], dtype=torch.float64)
    assert stored_total_by_row.shape == broad_by_row.shape == (N_ROWS,)
    full_original_by_row = stored_total_by_row + broad_by_row
    shifted_mean = float(full_original_by_row.mean())
    shifted_p95 = float(torch.quantile(full_original_by_row, .95))
    shifted_max = float(full_original_by_row.max())

    physical_cev = torch.load(CEV, map_location="cpu").float().reshape(-1)
    parent_cev = torch.load(PARENT_CEV, map_location="cpu").float().reshape(-1)
    assert physical_cev.shape == parent_cev.shape
    parent_delta = (physical_cev - parent_cev).abs()
    parent_delta_mean = float(parent_delta.mean())
    parent_delta_max = float(parent_delta.max())

    physical_identity = (
        source_bf16_exact
        and dtype_scalars == {"torch.float32": 487_931_904,
                              "torch.bfloat16": 57_970_998}
        and fingerprint == "7dabb830ac9ebb0d" and token_count == 675_457
        and factor_shapes == expected_factor_shapes
        and factor_scalars == L16_SCALARS and factor_bytes == L16_BYTES
        and factor_dtypes == ["torch.bfloat16"] and factor_no_dense
        and result["mlp16_hook_calls"] > 0
        and mlp_program_identity
        and mlp_program_scalars == MLP_PROGRAM_SCALARS
        and mlp_program_bytes == 2 * MLP_PROGRAM_SCALARS
        and result["selected_layers_ordered"] == [4, 0]
        and result["mlp_fit_rows_half_open"] == [24, 48]
        and result["qk_fit_rows_half_open"] == [72, 96]
        and result["mlp_input_program_observed"] == {"0": 768, "4": 768}
        and result["qk_storage_dtype"] == "float16"
        and result["qk_factor_tensor_dtypes"] == ["torch.float16"]
        and result["qk_factorized_maps"] == 440
        and result["literal_standalone_scalars"] == SCALARS
        and result["literal_source_format_bytes"] == BYTES
        and BYTES == 2 * SCALARS
    )
    pred_a = physical_identity
    pred_b = (
        result["physical_census_damage"] <= .070
        and result["physical_certificates"] >= 10
        and parent_delta_mean <= .010 and parent_delta_max <= .100)
    pred_c = (
        .90 <= result["composition_tax_ratio"] <= 1.35
        and result["physical_vs_additive_normalized_vector_cosine"] >= .95
        and result["physical_vs_additive_certificate_difference"] <= 7)
    pred_d = (
        shifted_mean <= .075 and shifted_p95 <= .140 and shifted_max <= .220
        and result["conditional_fresh8_max"] <= .040)
    null = (
        not physical_identity
        or result["physical_census_damage"] >= .10
        or result["physical_certificates"] <= 5
        or shifted_mean >= .10
        or result["mlp16_hook_calls"] == 0)

    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed64_physical_bf16_mlp04_mlp16_factored_complete",
        "rung": RUNG,
        "claim_level": "physical_all_two_byte_sub500_composition_screen",
        "global_storage_dtype": "source-fp32_to_bfloat16; source-bfloat16_exact",
        "global_compute_dtype": "float32_explicit_dequantization",
        "source_dtype_scalars": dtype_scalars,
        "source_bfloat16_tensors_bit_exact": source_bf16_exact,
        "rounded_fp32_tensors_changed": changed_tensors,
        "generated_mlp_program_dtypes": mlp_program_dtypes,
        "generated_mlp_program_scalars": mlp_program_scalars,
        "generated_mlp_program_bytes": mlp_program_bytes,
        "mlp16_physical_program": PHYSICAL_FACTOR.name,
        "mlp16_factor_shapes": {name: list(shape) for name, shape in factor_shapes.items()},
        "mlp16_factor_scalars": factor_scalars,
        "mlp16_factor_bytes": factor_bytes,
        "mlp16_factor_dtypes": factor_dtypes,
        "mlp16_no_dense_form": factor_no_dense,
        "parent_fp32_source_cev": PARENT_CEV.name,
        "parent_cev_delta_mean_abs": parent_delta_mean,
        "parent_cev_delta_max_abs": parent_delta_max,
        "shifted_broad_rounding_mean": float(broad_by_row.mean()),
        "shifted_full_original_native_mean": shifted_mean,
        "shifted_full_original_native_p95": shifted_p95,
        "shifted_full_original_native_max": shifted_max,
        "shifted_full_original_native_by_row": [
            float(value) for value in full_original_by_row],
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "literal_gibibytes": BYTES / (1024 ** 3),
        'pred_a_physical_dtypes_shapes_hooks_and_bill_hold': bool(pred_a),
        'pred_b_census_certificates_and_parent_delta_hold': bool(pred_b),
        'pred_c_additive_composition_law_holds': bool(pred_c),
        'pred_d_shifted_and_fresh_hold': bool(pred_d),
        "null_physical_sub500_tier_breaks_prediction_or_identity": bool(null),
        "signed_gate_licensed": bool(pred_a and pred_b and pred_c and pred_d and not null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "census_certificates": [result["physical_census_damage"],
                                result["physical_certificates"]],
        "parent_delta_mean_max": [parent_delta_mean, parent_delta_max],
        "composition": [result["composition_tax_ratio"],
                        result["physical_vs_additive_normalized_vector_cosine"],
                        result["physical_vs_additive_certificate_difference"]],
        "shifted": [shifted_mean, shifted_p95, shifted_max],
        "fresh_max": result["conditional_fresh8_max"],
        "bill": [SCALARS, BYTES, result["literal_gibibytes"]],
        "predicates": [pred_a, pred_b, pred_c, pred_d],
        "null": null,
    }, indent=2), flush=True)
    print("R414 PHYSICAL ALL-TWO-BYTE SUB500 TIER DONE", flush=True)


if __name__ == "__main__":
    main()
