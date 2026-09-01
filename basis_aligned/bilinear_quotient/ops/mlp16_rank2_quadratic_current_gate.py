"""RUNG 389 -- CLEAN-SPLIT CURRENT-HARNESS GATE FOR THE OLD L16 QUADRATIC SURROGATE.

The duplicate-work audit found that §§9-10 already replaced MLP16 by four scalar
quadratic output coordinates, each truncated to rank two.  The old receipt reported
13,832 parameters and +.030734 CE, but (i) omitted the required 1,152-value constant
vector from its literal price and (ii) used overlapping rows to estimate output PCs,
the input metric, and CE.  The corrected standalone layer price is 14,984 values.

Arms
----
legacy_overlap: rebuild the exact old R=4,k=2 object on bilin18_eval_tokens.pt and
    reproduce only its historical overlapping-row CE; this is an implementation
    identity check, not evidence of transfer.
clean: fit output directions, input second moment, forms, and constant only on
    FineWeb skip11000 rows24:48; score rows0:24, rows48:72, the diverse census,
    untouched FineWeb, and shifted WikiText-103.
random_output: seed-389 random orthonormal output directions, with their exact native
    quadratic forms truncated under the identical fit metric and identical price.

Frozen predictions
------------------
pred_a: legacy CE reproduces 3.48647776 within .003; reported and corrected prices
    are exactly 13,832 and 14,984; clean program shapes/hook/splits are exact.
pred_b: clean heldout and fresh local function R2 each >=.70; census <=.060 and
    >=18/62 behavior checks; shifted mean/p95/max <=.070/.110/.160 and untouched
    FineWeb mean/max <=.060/.120.
pred_c: clean directly dominates rung388 Tucker on the common modern measures:
    census <=.04779619, certificates >=22, shifted mean <=.0545149, and untouched
    FineWeb mean <=.0539558, while using <1% of its layer values.
pred_d: clean heldout R2 exceeds random by >=.30, census damage is >=.05 lower than
    random, and its normalized 62-vector follows the frozen damage ray at cosine
    >=.95, vector R2 >=.70, and predicted-count error <=3.

Strong null: clean heldout R2 <.50, census >=.15, zero behavior checks, an inert hook,
or random comes within .05 heldout R2 and .01 census damage.  Decision: A/B/C/D
license one original-native signed gate and then a composition screen.  A/B/D without
C preserves the clean representation as a mapped lower-fidelity object but earns no
composition.  Strong null closes this old surrogate as a generalizing compiler route.
No R, k, layer, metric, or constant tuning follows this receipt.
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
STRUCTURE = ROOT / "experiments/structure_mapping"
OUT = ROOT / "mlp16_rank2_quadratic_current_gate_results.json"
FIT_CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FRESH_CACHE = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
LEGACY_ROWS = ROOT / "bilin18_eval_tokens.pt"
CEV_CLEAN = ROOT / "cev_mlp16_rank2_quadratic_clean.pt"
CEV_RANDOM = ROOT / "cev_mlp16_rank2_quadratic_random.pt"
LAYER = 16
D = 1152
H = 4608
R = 4
K = 2
FIT_A = (0, 24)
FIT_B = (24, 48)
FRESH_FUNCTION = (48, 72)
FRESH_EVAL = (176, 188)
RANDOM_SEED = 389
OLD_REPORTED_PRICE = R * D + R * K * D + R * K
CORRECTED_LAYER_PRICE = OLD_REPORTED_PRICE + D
NATIVE_LAYER_PRICE = 3 * D * H + D
NATIVE_MODEL_PRICE = 545_902_902
PROGRAM_MODEL_PRICE = NATIVE_MODEL_PRICE - NATIVE_LAYER_PRICE + CORRECTED_LAYER_PRICE
TUCKER_LAYER_PRICE = 2_065_536
TUCKER_CENSUS = 0.047796186
TUCKER_CERTIFICATES = 22
TUCKER_WIKI_MEAN = 0.0545149
TUCKER_FRESH_MEAN = 0.0539558
WIKI_N = 120
WIKI_SKIP = 0


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    error = (prediction - target).square().sum()
    total = (target - target.mean(0)).square().sum().clamp_min(1e-30)
    return float(1.0 - error / total)


def _prediction(x: torch.Tensor, program: dict[str, torch.Tensor]) -> torch.Tensor:
    coefficients = torch.einsum("...i,rij,...j->...r", x.float(), program["forms"], x.float())
    return coefficients @ program["output_directions"] + program["constant"]


@torch.no_grad()
def _build_clean(model, x: torch.Tensor, y: torch.Tensor,
                 random_output: bool = False) -> dict[str, torch.Tensor]:
    sys.path.insert(0, str(STRUCTURE))
    from bilin18_identifiable import form_for_direction
    from bilin18_whitened import sqrtm_psd, truncate

    device = next(model.parameters()).device
    xd = x.to(device).double()
    yd = y.to(device).double()
    mlp = model.transformer.h[LAYER].mlp
    mean = yd.mean(0)
    if random_output:
        generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
        raw = torch.randn(D, R, generator=generator, dtype=torch.float64).to(device)
        output_directions = torch.linalg.qr(raw, mode="reduced").Q.T
    else:
        _u, _s, vh = torch.linalg.svd(yd - mean, full_matrices=False)
        output_directions = vh[:R]
    covariance = xd.T @ xd / xd.shape[0]
    sh, sih = sqrtm_psd(covariance)
    exact = torch.stack([
        form_for_direction(mlp, output_directions[index].float()) for index in range(R)
    ])
    forms = torch.stack([
        sih @ truncate(sh @ exact[index] @ sh, K) @ sih for index in range(R)
    ])
    bias = mlp.Down_bias.detach().double()
    quadratic_mean = mean - bias
    mean_perpendicular = quadratic_mean - output_directions.T @ (
        output_directions @ quadratic_mean)
    constant = mean_perpendicular + bias
    return {
        "output_directions": output_directions.float().contiguous(),
        "forms": forms.float().contiguous(),
        "constant": constant.float().contiguous(),
    }


@torch.no_grad()
def _ce_vector(model, rows: torch.Tensor, program: dict[str, torch.Tensor] | None,
               observed: dict[str, int], name: str) -> torch.Tensor:
    from mlp16_tucker_physical_calibration import _manual_logits

    handle = None
    device = next(model.parameters()).device
    if program is not None:
        def replacement(_module, args, output):
            observed[name] = observed.get(name, 0) + 1
            return _prediction(args[0], program).to(output.dtype)
        handle = model.transformer.h[LAYER].mlp.register_forward_hook(replacement)
    losses = []
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index = batch[:, :-1].to(device)
            target = batch[:, 1:].to(device)
            logits = _manual_logits(model, index)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                reduction="none").cpu())
    finally:
        if handle is not None:
            handle.remove()
    return torch.cat(losses)


@torch.no_grad()
def _legacy_reproduction(model) -> dict[str, float]:
    sys.path.insert(0, str(STRUCTURE))
    from bilin18_depth_followup import build
    from bilin18_layer17 import Truncated
    from bilin18_whitened import truncate
    from tier2_model import eval_ce

    tokens = torch.load(LEGACY_ROWS, map_location="cpu")
    mlp = model.transformer.h[LAYER].mlp
    original = mlp.forward
    _mlp, directions, forms, mean_quadratic, bias, sh, sih = build(
        model, tokens, LAYER, R)
    whitened = torch.stack([
        sih @ truncate(sh @ forms[index] @ sh, K) @ sih for index in range(R)
    ])
    mlp.forward = Truncated(
        directions.float(), whitened.float(), mean_quadratic, bias).to(
            next(model.parameters()).device).forward
    try:
        ce = eval_ce(model, tokens, batch=4)
    finally:
        mlp.forward = original
    return {"ce": float(ce), "damage_from_recorded_baseline": float(ce - 3.4557433128356934)}


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        required = [FIT_CACHE, FRESH_CACHE, LEGACY_ROWS, ROOT / "census_state_diverse.pt",
                    ROOT / "circuits/BATTERY.json",
                    ROOT / "certificate_damage_axis_transfer_results.json",
                    ROOT / "mlp16_tucker_physical_calibration_results.json"]
        assert all(path.exists() for path in required)
        assert OLD_REPORTED_PRICE == 13_832 and CORRECTED_LAYER_PRICE == 14_984
        assert PROGRAM_MODEL_PRICE == 529_991_486
        assert CORRECTED_LAYER_PRICE < .01 * TUCKER_LAYER_PRICE
        assert FIT_A[1] == FIT_B[0] and FIT_B[1] == FRESH_FUNCTION[0]
        print("L16 RANK2 QUADRATIC CURRENT GATE | dry run: splits, prices, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp16_tucker_physical_calibration import (
        _capture, _certificate_metrics, _row_summary)
    from tier2_model import load_elriggs

    fit_cached = torch.load(FIT_CACHE, map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    fresh_cached = torch.load(FRESH_CACHE, map_location="cpu")
    fresh_cached = fresh_cached["rows"] if isinstance(fresh_cached, dict) else fresh_cached
    fit_a = fit_cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    fit_b = fit_cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    function_fresh = fit_cached[FRESH_FUNCTION[0]:FRESH_FUNCTION[1], :257].long().contiguous()
    fresh_rows = fresh_cached[FRESH_EVAL[0]:FRESH_EVAL[1], :257].long().contiguous()
    assert fit_a.shape == fit_b.shape == function_fresh.shape == (24, 257)
    assert fresh_rows.shape == (12, 257)

    model, config = load_elriggs("bilin18")
    model.eval()
    assert config["n_embd"] == D and len(model.transformer.h) == 18
    legacy = _legacy_reproduction(model)

    xb, yb = _capture(model, fit_b)
    clean = _build_clean(model, xb, yb, random_output=False)
    random_output = _build_clean(model, xb, yb, random_output=True)
    xa, ya = _capture(model, fit_a)
    xf, yf = _capture(model, function_fresh)
    device = next(model.parameters()).device
    xa, ya, xf, yf = (value.to(device) for value in (xa, ya, xf, yf))
    local = {
        name: {
            "heldout_r2": _r2(_prediction(xa, program), ya),
            "fresh_r2": _r2(_prediction(xf, program), yf),
        }
        for name, program in (("clean", clean), ("random_output", random_output))
    }

    CN.use_state(str(ROOT / "census_state_diverse.pt"))
    census_rows = CN.rows().cpu().long()[:, :257].contiguous()
    base = CN.base_ce().float().reshape(-1).cpu()
    assert base.numel() == CN.nflat() == census_rows.shape[0] * 256
    observed: dict[str, int] = {}
    clean_cev = _ce_vector(model, census_rows, clean, observed, "clean")
    random_cev = _ce_vector(model, census_rows, random_output, observed, "random_output")
    torch.save(clean_cev, CEV_CLEAN)
    torch.save(random_cev, CEV_RANDOM)
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    ray_receipt = json.loads((ROOT / "certificate_damage_axis_transfer_results.json").read_text())
    census = {}
    for name, cev in (("clean", clean_cev), ("random_output", random_cev)):
        census[name] = {
            "damage": float((cev.float() - base).mean()),
            "mean_absolute_position_damage": float((cev.float() - base).abs().mean()),
            **_certificate_metrics(CN, base, cev, battery, ray_receipt),
        }

    wiki_rows, wiki_fingerprint, wiki_token_count = _wikitext103_train_rows(
        n=WIKI_N, width=257, skip=WIKI_SKIP)
    native_wiki = _ce_vector(model, wiki_rows, None, observed, "native_wiki")
    native_fresh = _ce_vector(model, fresh_rows, None, observed, "native_fresh")
    transfer = {}
    for name, program in (("clean", clean), ("random_output", random_output)):
        candidate_wiki = _ce_vector(model, wiki_rows, program, observed, name)
        candidate_fresh = _ce_vector(model, fresh_rows, program, observed, name)
        transfer[name] = {
            "wikitext103": _row_summary(candidate_wiki, native_wiki, WIKI_N),
            "fineweb_fresh": _row_summary(candidate_fresh, native_fresh, len(fresh_rows)),
        }

    shapes = {key: list(clean[key].shape) for key in ("output_directions", "forms", "constant")}
    identity_ok = (
        shapes == {"output_directions": [R, D], "forms": [R, D, D], "constant": [D]}
        and OLD_REPORTED_PRICE == 13_832 and CORRECTED_LAYER_PRICE == 14_984
        and PROGRAM_MODEL_PRICE == 529_991_486
        and observed.get("clean", 0) > 0 and observed.get("random_output", 0) > 0
    )
    clean_census = census["clean"]
    random_census = census["random_output"]
    clean_wiki = transfer["clean"]["wikitext103"]
    clean_fresh = transfer["clean"]["fineweb_fresh"]
    pred_a = abs(legacy["ce"] - 3.4864777624607086) <= .003 and identity_ok
    pred_b = (
        local["clean"]["heldout_r2"] >= .70 and local["clean"]["fresh_r2"] >= .70
        and clean_census["damage"] <= .060 and clean_census["certificates"] >= 18
        and clean_wiki["mean"] <= .070 and clean_wiki["p95"] <= .110
        and clean_wiki["max"] <= .160 and clean_fresh["mean"] <= .060
        and clean_fresh["max"] <= .120
    )
    pred_c = (
        clean_census["damage"] <= TUCKER_CENSUS
        and clean_census["certificates"] >= TUCKER_CERTIFICATES
        and clean_wiki["mean"] <= TUCKER_WIKI_MEAN
        and clean_fresh["mean"] <= TUCKER_FRESH_MEAN
        and CORRECTED_LAYER_PRICE < .01 * TUCKER_LAYER_PRICE
    )
    pred_d = (
        local["clean"]["heldout_r2"] >= local["random_output"]["heldout_r2"] + .30
        and clean_census["damage"] <= random_census["damage"] - .05
        and clean_census["ray_cosine"] >= .95 and clean_census["ray_vector_r2"] >= .70
        and clean_census["ray_certificate_count_error"] <= 3
    )
    inert = float((clean_cev.float() - base).abs().mean()) < 1e-8
    random_matches = (
        local["clean"]["heldout_r2"] - local["random_output"]["heldout_r2"] <= .05
        and random_census["damage"] <= clean_census["damage"] + .01
    )
    strong_null = (
        local["clean"]["heldout_r2"] < .50 or clean_census["damage"] >= .15
        or clean_census["certificates"] == 0 or inert or random_matches
    )
    composition_licensed = bool(pred_a and pred_b and pred_c and pred_d and not strong_null)
    result = {
        "status": "mlp16_rank2_quadratic_current_gate_complete",
        "rung": 389,
        "claim_level": "clean_split_current_harness_old_l16_quadratic_surrogate_gate",
        "convention": "candidate CE minus original native CE; lower is better",
        "legacy_overlap_reproduction": legacy,
        "fit": {"cache": FIT_CACHE.name, "fit_b": list(FIT_B),
                "heldout_fit_a": list(FIT_A), "fresh_function": list(FRESH_FUNCTION)},
        "fresh_evaluation": {"cache": FRESH_CACHE.name, "slice": list(FRESH_EVAL)},
        "ranks": {"output_directions": R, "rank_per_quadratic_form": K,
                  "total_squared_projections": R * K},
        "local_function_r2": local,
        "program_shapes": shapes,
        "prices": {
            "old_reported_layer_scalars_omitting_constant": OLD_REPORTED_PRICE,
            "corrected_layer_scalars": CORRECTED_LAYER_PRICE,
            "native_layer_scalars": NATIVE_LAYER_PRICE,
            "program_fraction_of_native_layer": CORRECTED_LAYER_PRICE / NATIVE_LAYER_PRICE,
            "program_fraction_of_rung388_tucker_layer": CORRECTED_LAYER_PRICE / TUCKER_LAYER_PRICE,
            "native_model_scalars": NATIVE_MODEL_PRICE,
            "program_model_scalars": PROGRAM_MODEL_PRICE,
        },
        "census": census,
        "transfer": transfer,
        "shifted_population": {
            "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train",
            "source_rows_half_open": [100_000, 110_000], "n_rows": WIKI_N,
            "skip_tokens": WIKI_SKIP, "fingerprint": wiki_fingerprint,
            "token_count": wiki_token_count,
        },
        "rung388_tucker_comparator": {
            "layer_scalars": TUCKER_LAYER_PRICE, "census_damage": TUCKER_CENSUS,
            "certificates": TUCKER_CERTIFICATES, "wikitext_mean": TUCKER_WIKI_MEAN,
            "fineweb_fresh_mean": TUCKER_FRESH_MEAN,
        },
        "hook_calls": observed,
        "saved_cevs": {"clean": CEV_CLEAN.name, "random_output": CEV_RANDOM.name},
        'pred_a_legacy_identity_corrected_price_and_clean_identity': bool(pred_a),
        'pred_b_clean_split_predictive_behavior_and_transfer': bool(pred_b),
        'pred_c_clean_directly_dominates_rung388_tucker': bool(pred_c),
        'pred_d_output_directions_beat_random_and_ray_transfers': bool(pred_d),
        "null_old_quadratic_surrogate_fails_to_generalize": bool(strong_null),
        "composition_licensed": composition_licensed,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        "legacy": legacy, "local": local, "prices": result["prices"],
        "census": {name: {key: value for key, value in row.items()
                            if key not in ("normalized_vector", "member_abs_dce")}
                   for name, row in census.items()},
        "transfer": {
            name: {population: {key: value for key, value in summary.items() if key != "by_row"}
                   for population, summary in populations.items()}
            for name, populations in transfer.items()
        },
        "predicates": [pred_a, pred_b, pred_c, pred_d],
        "strong_null": strong_null, "composition_licensed": composition_licensed,
        "runtime_s": result["runtime_s"],
    }
    print(json.dumps(compact, indent=2), flush=True)
    print("L16 RANK2 QUADRATIC CURRENT GATE DONE", flush=True)


if __name__ == "__main__":
    main()
