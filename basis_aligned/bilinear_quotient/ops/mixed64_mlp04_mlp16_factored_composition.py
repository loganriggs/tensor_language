"""RUNG 392 -- PHYSICAL QK64 + MLP0/4 + FACTORED-MLP16 COMPOSITION.

Install the shipped 14,984-value eight-projection L16 artifact together with
the exact fp32 structural QK64 + split-B MLP{0,4}@p768 program from rung367.
The independently saved positionwise additive prediction is .0512222387 and
17/62 checks.  The exact shipped bill is 495,847,230 scalars and 1,867,449,228
source-format bytes (native mixed-dtype L16 removed; fp32 factors added).

Frozen predictions
------------------
pred_a: physical census <=.070 and at least 10/62 checks.
pred_b: census/additive ratio in [.90,1.35], normalized 62-vector cosine with
    the additive prediction >=.95, and certificate-count difference <=7.
pred_c: untouched WT103 [378304,409144) full native-relative mean/p95/max
    <=.075/.140/.220; conditional eight-window fresh increment <=.040.
pred_d: exact factor keys/shapes/no-dense/14,984 count, QK64/440 float32 maps,
    selected split-B {0,4}@p768, fit splits, live L16 hook, saved full CEV,
    and exact scalar/source-byte bills.

Strong null: census >=.10, <=5 checks, shifted mean >=.10, inert L16 hook,
or identity failure.  All positives with null false license one original-native
signed composite gate; otherwise retain the standalone L16 causal description
without rank, site, metric, precision, or population tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed64_mlp04_mlp16_factored_composition_results.json"
CEV = ROOT / "cev_mixed64_mlp04_mlp16_factored.pt"
PARENT = ROOT / "mixed64_context_qk_mlp04_context_p768_ood_results.json"
FACTOR_RESULT = ROOT / "mlp16_rank2_quadratic_factored_gate_results.json"
FACTOR_PROGRAM = ROOT / "mlp16_rank2_quadratic_factored.pt"
FRONTIER_CEV = ROOT / "cev_mixed64_context_qk_mlp04_context_p768.pt"
L16_CEV = ROOT / "cev_mlp16_rank2_quadratic_factored.pt"
WIKI_SKIP = 378_304
N_ROWS = 120
SCALARS = 495_847_230
BYTES = 1_867_449_228
NATIVE_MLP16_SCALARS = 15_926_400
NATIVE_MLP16_SOURCE_BYTES = 63_703_296
FACTORED_L16_SCALARS = 14_984
FACTORED_L16_BYTES = 59_936
ADDITIVE_DAMAGE = 0.051222238689661026
ADDITIVE_CERTIFICATES = 17
QK_RANK = 64
MLP_RANK = 768
D = 1152
R = 4
K = 2


def _factored_prediction(x: torch.Tensor,
                         program: dict[str, torch.Tensor]) -> torch.Tensor:
    projections = torch.einsum("...d,rkd->...rk", x.float(), program["form_vectors"])
    coefficients = (projections.square() * program["form_values"]).sum(-1)
    return coefficients @ program["output_directions"] + program["constant"]


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (PARENT, FACTOR_RESULT, FACTOR_PROGRAM, FRONTIER_CEV, L16_CEV,
                     ROOT / "census_state_diverse.pt", ROOT / "circuits/BATTERY.json"):
            assert path.exists(), path
        parent = json.loads(PARENT.read_text())
        factored = json.loads(FACTOR_RESULT.read_text())
        assert all(parent[key] for key in (
            "pred_a_census_and_certificate_frontier_improves",
            "pred_b_shifted_ood_and_fresh_hold",
            "pred_c_selection_program_identity_and_price_hold"))
        assert all(factored[key] for key in (
            "pred_a_physical_factorization_identity_holds",
            "pred_b_unablated_dense_reproduction_holds",
            "pred_c_signed_dense_and_native_reproduction_holds"))
        assert SCALARS == 511_758_646 - NATIVE_MLP16_SCALARS + FACTORED_L16_SCALARS
        assert BYTES == 1_931_092_588 - NATIVE_MLP16_SOURCE_BYTES + FACTORED_L16_BYTES
        assert WIKI_SKIP + N_ROWS * 257 == 409_144
        print("QK64 MLP04 + FACTORED L16 | dry run: parents, additive target, bills valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    import mixed64_context_qk_mlp04_context_p768_ood as harness
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp0_tail_robust_context_metric_screen import _score_rows
    from mlp16_tucker_physical_calibration import _certificate_metrics
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    program_cpu = torch.load(FACTOR_PROGRAM, map_location="cpu")
    program = {key: value.to(C.DEV) for key, value in program_cpu.items()}
    expected_shapes = {
        "output_directions": [R, D], "form_vectors": [R, K, D],
        "form_values": [R, K], "constant": [D]}
    factor_shapes = {key: list(value.shape) for key, value in program_cpu.items()}
    factor_scalars = sum(value.numel() for value in program_cpu.values())
    factor_dtypes = sorted({str(value.dtype) for value in program_cpu.values()})
    no_dense_form = "forms" not in program_cpu and set(program_cpu) == set(expected_shapes)

    rows_ood, fingerprint, token_count = _wikitext103_train_rows(
        n=N_ROWS, width=257, skip=WIKI_SKIP)
    original_ood = _score_rows(C.m, rows_ood, _manual_logits).double()
    observed = {"mlp16_factored": 0}

    def factor_hook(_module, args, output):
        observed["mlp16_factored"] += 1
        return _factored_prediction(args[0], program).to(output.dtype)

    handle = C.m.transformer.h[16].mlp.register_forward_hook(factor_hook)
    try:
        l16_ood = _score_rows(C.m, rows_ood, _manual_logits).double()
        harness.OUT = OUT
        harness.CEV = CEV
        harness.WIKI_SKIP = WIKI_SKIP
        harness.WIKI_STOP = WIKI_SKIP + N_ROWS * 257
        harness.N_ROWS = N_ROWS
        harness.QK_RANK = QK_RANK
        harness.SCALARS = SCALARS
        harness.BYTES = BYTES
        harness.QK_STORAGE_DTYPE = None
        harness.EXPECTED_QK_FACTOR_DTYPE = "torch.float32"
        harness.CENSUS_MAX = .070
        harness.CERTIFICATE_MIN = 10
        harness.OOD_MEAN_MAX = .075
        harness.OOD_P95_MAX = .140
        harness.OOD_MAX = .220
        harness.FRESH_MAX = .040
        harness.NULL_CENSUS = .10
        harness.NULL_CERTIFICATES = 5
        harness.main()
    finally:
        handle.remove()

    result = json.loads(OUT.read_text())
    l16_by_row = l16_ood - original_ood
    conditional_by_row = torch.tensor(result["shifted_damage_by_row"], dtype=torch.float64)
    assert l16_by_row.shape == conditional_by_row.shape == (N_ROWS,)
    total_by_row = l16_by_row + conditional_by_row
    shifted_mean = float(total_by_row.mean())
    shifted_p95 = float(torch.quantile(total_by_row, .95))
    shifted_max = float(total_by_row.max())

    CN.use_state(str(ROOT / "census_state_diverse.pt"))
    base = CN.base_ce().float().reshape(-1).cpu()
    physical_cev = torch.load(CEV, map_location="cpu").float().reshape(-1)
    frontier_cev = torch.load(FRONTIER_CEV, map_location="cpu").float().reshape(-1)
    l16_cev = torch.load(L16_CEV, map_location="cpu").float().reshape(-1)
    assert base.numel() == physical_cev.numel() == frontier_cev.numel() == l16_cev.numel()
    additive_cev = base + (frontier_cev - base) + (l16_cev - base)
    additive_damage = float((additive_cev - base).mean())
    physical_damage = float((physical_cev - base).mean())
    tax_ratio = physical_damage / additive_damage
    interaction = physical_damage - additive_damage

    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    ray = json.loads((ROOT / "certificate_damage_axis_transfer_results.json").read_text())
    physical_metrics = _certificate_metrics(CN, base, physical_cev, battery, ray)
    additive_metrics = _certificate_metrics(CN, base, additive_cev, battery, ray)
    physical_vector = torch.tensor(physical_metrics["normalized_vector"], dtype=torch.float64)
    additive_vector = torch.tensor(additive_metrics["normalized_vector"], dtype=torch.float64)
    vector_cosine = float(F.cosine_similarity(physical_vector[None], additive_vector[None]))
    certificate_difference = abs(
        physical_metrics["certificates"] - additive_metrics["certificates"])

    identity = (
        factor_shapes == expected_shapes and factor_scalars == FACTORED_L16_SCALARS
        and factor_dtypes == ["torch.float32"] and no_dense_form
        and observed["mlp16_factored"] > 0
        and result["selected_layers_ordered"] == [4, 0]
        and result["mlp_fit_rows_half_open"] == [24, 48]
        and result["qk_fit_rows_half_open"] == [72, 96]
        and result["mlp_input_program_observed"] == {"0": MLP_RANK, "4": MLP_RANK}
        and result["qk_metric"] == "context_rrr"
        and result["qk_storage_dtype"] == "float32"
        and result["qk_rank"] == QK_RANK and result["qk_factorized_maps"] == 440
        and result["qk_factor_tensor_dtypes"] == ["torch.float32"]
        and fingerprint == "7dabb830ac9ebb0d" and token_count == 675_457
        and result["saved_census_cev_file"] == CEV.name and CEV.exists()
        and result["literal_standalone_scalars"] == SCALARS
        and result["literal_raw_tensor_bytes"] == BYTES)

    pred_a = physical_damage <= .070 and physical_metrics["certificates"] >= 10
    pred_b = (
        .90 <= tax_ratio <= 1.35 and vector_cosine >= .95
        and certificate_difference <= 7)
    pred_c = (
        shifted_mean <= .075 and shifted_p95 <= .140 and shifted_max <= .220
        and result["max_fresh_damage"] <= .040)
    pred_d = identity
    strong_null = (
        physical_damage >= .10 or physical_metrics["certificates"] <= 5
        or shifted_mean >= .10 or observed["mlp16_factored"] == 0 or not identity)
    signed_gate_licensed = bool(pred_a and pred_b and pred_c and pred_d and not strong_null)

    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed64_mlp04_mlp16_factored_composition_complete",
        "rung": 392,
        "claim_level": "physical_structural_composition_census_certificate_ood_screen",
        "convention": "compiled CE minus original native CE; lower is better",
        "mlp16_factored_program": FACTOR_PROGRAM.name,
        "mlp16_factor_shapes": factor_shapes,
        "mlp16_factor_scalars": factor_scalars,
        "mlp16_factor_dtypes": factor_dtypes,
        "mlp16_no_dense_form": no_dense_form,
        "mlp16_hook_calls": observed["mlp16_factored"],
        "additive_prediction": {
            "census_damage": additive_damage,
            "certificates": additive_metrics["certificates"],
            "ray_cosine": additive_metrics["ray_cosine"],
        },
        "physical_census_damage": physical_damage,
        "physical_certificates": physical_metrics["certificates"],
        "composition_tax_ratio": tax_ratio,
        "composition_interaction_damage": interaction,
        "physical_vs_additive_normalized_vector_cosine": vector_cosine,
        "physical_vs_additive_certificate_difference": certificate_difference,
        "shifted_full_native_relative_mean": shifted_mean,
        "shifted_full_native_relative_p95": shifted_p95,
        "shifted_full_native_relative_max": shifted_max,
        "shifted_l16_only_mean": float(l16_by_row.mean()),
        "shifted_conditional_qk_mlp04_mean": float(conditional_by_row.mean()),
        "shifted_full_damage_by_row": [float(value) for value in total_by_row],
        "conditional_fresh8_max": result["max_fresh_damage"],
        "literal_standalone_scalars": SCALARS,
        "literal_source_format_bytes": BYTES,
        "saving_vs_native_scalars": 545_902_902 - SCALARS,
        "saving_vs_native_source_bytes": 2_067_669_612 - BYTES,
        'pred_a_physical_lower_fidelity_holds': bool(pred_a),
        'pred_b_additive_composition_law_holds': bool(pred_b),
        'pred_c_shifted_and_conditional_fresh_hold': bool(pred_c),
        'pred_d_shipped_program_identity_and_bills_hold': bool(pred_d),
        "null_composition_breaks_prediction_or_identity": bool(strong_null),
        "signed_composite_gate_licensed": signed_gate_licensed,
        "runtime_s_total": time.time() - started,
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("member_abs_dce", "shifted_damage_by_row",
                                     "shifted_full_damage_by_row", "fresh8")},
                     indent=2), flush=True)
    print("QK64 + MLP04 + FACTORED MLP16 COMPOSITION DONE", flush=True)


if __name__ == "__main__":
    main()
