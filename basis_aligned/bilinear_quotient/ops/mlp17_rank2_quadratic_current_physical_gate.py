"""RUNG 415 -- CLEAN-SPLIT PHYSICAL CURRENT-HARNESS MLP17 R4K2 SURROGATE.

Port the historical four-output/eight-signed-square whole-layer MLP17 object
after the L16 current-harness precedent. Dense forms are fit-time workspace
only; both clean and seed415 random-output controls execute from14,984 values.

Frozen predictions
------------------
pred_a_legacy_and_physical_identity_hold:
    Legacy CE within .005 of3.557555; exact four-tensor shapes/count/no-dense,
    fit/evaluation splits, factorization and live hooks.
pred_b_clean_prediction_and_transfer_hold:
    Heldout/fresh R2 >=.65; census <=.130 and >=5 certificates; WikiText
    mean/p95/max <=.15/.25/.40; FineWeb fresh mean/max <=.15/.30.
pred_c_historical_scale_and_random_specificity_hold:
    Census within .05 of historical +.101811 and clean beats random by .30
    heldout R2 and .05 census nat.
pred_d_damage_ray_transfers:
    Ray cosine >=.90, vector R2 >=.50, certificate-count error <=5.

Strong null: heldout R2<.40,census>=.20,zero certificates,inert hook,or random
within .05 R2 and .01 census. Pass licenses one signed gate; no tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp17_rank2_quadratic_current_physical_gate_results.json"
CEV_CLEAN = ROOT / "cev_mlp17_rank2_quadratic_clean_physical.pt"
CEV_RANDOM = ROOT / "cev_mlp17_rank2_quadratic_random_physical.pt"
PROGRAM = ROOT / "mlp17_rank2_quadratic_factored.pt"
LAYER = 17
R = 4
K = 2
D = 1152
PRICE = 14_984
PROGRAM_MODEL_PRICE = 529_991_486
RANDOM_SEED = 415
WIKI_SKIP = 470_824
HISTORICAL_CE = 3.55755478143692
HISTORICAL_BASELINE = 3.4557433128356934
HISTORICAL_DAMAGE = HISTORICAL_CE - HISTORICAL_BASELINE
RUNG = 415


class FactoredProgram(dict):
    """A factor-only program with a shape-only legacy forms view."""

    def __getitem__(self, key):
        if key == "forms":
            return torch.empty((R, D, D), device="meta")
        return super().__getitem__(key)


def _factored_prediction(x: torch.Tensor,
                         program: dict[str, torch.Tensor]) -> torch.Tensor:
    projections = torch.einsum(
        "...d,rkd->...rk", x.float(), program["form_vectors"].float())
    coefficients = (projections.square() * program["form_values"].float()).sum(-1)
    return (coefficients @ program["output_directions"].float()
            + program["constant"].float())


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for name in (
            ".rowcache/fineweb_n192_skip11000.pt",
            ".rowcache/fineweb_n192_skip7000.pt",
            "bilin18_eval_tokens.pt",
            "census_state_diverse.pt",
            "circuits/BATTERY.json",
            "certificate_damage_axis_transfer_results.json",
        ):
            assert (ROOT / name).exists(), name
        assert PRICE == R * D + R * K * D + R * K + D
        assert PROGRAM_MODEL_PRICE == 545_902_902 - 15_926_400 + PRICE
        assert WIKI_SKIP + 120 * 257 == 501_664
        print("R415 MLP17 PHYSICAL QUADRATIC | dry run: dossier, splits, price valid")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    import mlp16_rank2_quadratic_current_gate as parent

    original_build = parent._build_clean
    original_prediction = parent._prediction
    captured = {}
    factor_errors = {}

    def build_factored(model, x, y, random_output=False):
        dense = original_build(model, x, y, random_output=random_output)
        forms = dense["forms"].double()
        values, vectors = torch.linalg.eigh(forms)
        order = torch.argsort(values.abs(), dim=-1, descending=True)[:, :K]
        kept_values = torch.gather(values, 1, order)
        kept_vectors = torch.gather(
            vectors, 2, order[:, None, :].expand(R, D, K)).transpose(1, 2)
        reconstructed = torch.einsum(
            "rki,rk,rkj->rij", kept_vectors, kept_values, kept_vectors)
        relative_error = float(
            (reconstructed - forms).norm() / forms.norm().clamp_min(1e-30))
        program = FactoredProgram({
            "output_directions": dense["output_directions"].contiguous(),
            "form_vectors": kept_vectors.float().contiguous(),
            "form_values": kept_values.float().contiguous(),
            "constant": dense["constant"].contiguous(),
        })
        name = "random_output" if random_output else "clean"
        captured[name] = program
        factor_errors[name] = relative_error
        return program

    parent._build_clean = build_factored
    parent._prediction = _factored_prediction
    parent.OUT = OUT
    parent.CEV_CLEAN = CEV_CLEAN
    parent.CEV_RANDOM = CEV_RANDOM
    parent.LAYER = LAYER
    parent.RANDOM_SEED = RANDOM_SEED
    parent.WIKI_SKIP = WIKI_SKIP
    try:
        parent.main()
    finally:
        parent._build_clean = original_build
        parent._prediction = original_prediction

    result = json.loads(OUT.read_text())
    assert set(captured) == {"clean", "random_output"}
    expected_shapes = {
        "output_directions": (R, D),
        "form_vectors": (R, K, D),
        "form_values": (R, K),
        "constant": (D,),
    }
    physical = {}
    for name, program in captured.items():
        shapes = {key: tuple(value.shape) for key, value in program.items()}
        scalars = sum(value.numel() for value in program.values())
        dtypes = sorted({str(value.dtype) for value in program.values()})
        physical[name] = {
            "shapes": {key: list(shape) for key, shape in shapes.items()},
            "scalars": scalars,
            "dtypes": dtypes,
            "no_dense_form": "forms" not in program and set(program) == set(expected_shapes),
            "factorization_relative_error": factor_errors[name],
        }
        assert shapes == expected_shapes
        assert scalars == PRICE
        assert dtypes == ["torch.float32"]
        assert physical[name]["no_dense_form"]

    clean_cpu = {key: value.detach().cpu().contiguous()
                 for key, value in captured["clean"].items()}
    torch.save(clean_cpu, PROGRAM)

    clean_local = result["local_function_r2"]["clean"]
    random_local = result["local_function_r2"]["random_output"]
    clean_census = result["census"]["clean"]
    random_census = result["census"]["random_output"]
    clean_wiki = result["transfer"]["clean"]["wikitext103"]
    clean_fresh = result["transfer"]["clean"]["fineweb_fresh"]
    legacy_ce = result["legacy_overlap_reproduction"]["ce"]
    identity = (
        abs(legacy_ce - HISTORICAL_CE) <= .005
        and all(row["scalars"] == PRICE and row["no_dense_form"]
                and row["factorization_relative_error"] <= 1e-5
                for row in physical.values())
        and result["fit"] == {
            "cache": "fineweb_n192_skip11000.pt",
            "fit_b": [24, 48],
            "heldout_fit_a": [0, 24],
            "fresh_function": [48, 72],
        }
        and result["fresh_evaluation"] == {
            "cache": "fineweb_n192_skip7000.pt",
            "slice": [176, 188],
        }
        and result["hook_calls"].get("clean", 0) > 0
        and result["hook_calls"].get("random_output", 0) > 0
        and PROGRAM.exists()
    )
    pred_a = identity
    pred_b = (
        clean_local["heldout_r2"] >= .65
        and clean_local["fresh_r2"] >= .65
        and clean_census["damage"] <= .130
        and clean_census["certificates"] >= 5
        and clean_wiki["mean"] <= .15
        and clean_wiki["p95"] <= .25
        and clean_wiki["max"] <= .40
        and clean_fresh["mean"] <= .15
        and clean_fresh["max"] <= .30)
    pred_c = (
        abs(clean_census["damage"] - HISTORICAL_DAMAGE) <= .05
        and clean_local["heldout_r2"] >= random_local["heldout_r2"] + .30
        and clean_census["damage"] <= random_census["damage"] - .05)
    pred_d = (
        clean_census["ray_cosine"] >= .90
        and clean_census["ray_vector_r2"] >= .50
        and clean_census["ray_certificate_count_error"] <= 5)
    inert = clean_census["mean_absolute_position_damage"] < 1e-8
    random_matches = (
        clean_local["heldout_r2"] - random_local["heldout_r2"] <= .05
        and random_census["damage"] <= clean_census["damage"] + .01)
    null = (
        clean_local["heldout_r2"] < .40
        or clean_census["damage"] >= .20
        or clean_census["certificates"] == 0
        or inert or random_matches or not identity)

    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mlp17_rank2_quadratic_current_physical_gate_complete",
        "rung": RUNG,
        "claim_level": "clean_split_physical_current_harness_mlp17_quadratic_screen",
        "historical_authority": {
            "ce": HISTORICAL_CE,
            "baseline_ce": HISTORICAL_BASELINE,
            "damage": HISTORICAL_DAMAGE,
        },
        "physical_programs": physical,
        "saved_physical_clean_program": PROGRAM.name,
        "literal_layer_program_scalars": PRICE,
        "literal_whole_model_scalars": PROGRAM_MODEL_PRICE,
        'pred_a_legacy_and_physical_identity_hold': bool(pred_a),
        'pred_b_clean_prediction_and_transfer_hold': bool(pred_b),
        'pred_c_historical_scale_and_random_specificity_hold': bool(pred_c),
        'pred_d_damage_ray_transfers': bool(pred_d),
        "null_mlp17_quadratic_surrogate_fails_current_harness": bool(null),
        "signed_gate_licensed": bool(pred_a and pred_b and pred_c and pred_d and not null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "legacy_ce": legacy_ce,
        "local": result["local_function_r2"],
        "census": {
            name: {
                "damage": row["damage"],
                "certificates": row["certificates"],
                "ray_cosine": row["ray_cosine"],
                "ray_vector_r2": row["ray_vector_r2"],
            }
            for name, row in result["census"].items()
        },
        "transfer": {
            "wikitext": {key: clean_wiki[key] for key in ("mean", "p95", "max")},
            "fresh": {key: clean_fresh[key] for key in ("mean", "p95", "max")},
        },
        "physical": physical,
        "predicates": [pred_a, pred_b, pred_c, pred_d],
        "null": null,
    }, indent=2), flush=True)
    print("R415 PHYSICAL CURRENT-HARNESS MLP17 QUADRATIC DONE", flush=True)


if __name__ == "__main__":
    main()
