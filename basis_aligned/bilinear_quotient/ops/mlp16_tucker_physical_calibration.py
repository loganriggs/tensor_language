"""RUNG 388 -- PHYSICAL L16 TUCKER CALIBRATION + EQUAL-CHEAPER CONTROL.

The paper-side program is complete through rung 387: the frozen
(input r512, product k576, output p512) importance-truncation core captures a
stable ~.82 of layer-16's function at 2,065,536 scalars.  This rung asks the
decision-grade question: is that omitted function downstream-benign when the
core physically replaces L16 in the whole model?

Arms
----
native: original model.
tucker: Q/A/B/U/C plus native bias, fit on FineWeb skip11000 rows 24:48.
product: the same selected 576 native product atoms, without Q/U projections;
    1,991,808 scalars, so it is an equal-cheaper native-coordinate control.
random_product: seed-388 random 576 native atoms at the same control price.

Frozen predictions
------------------
pred_a: Tucker census damage <= .025, >=43/62 certificates, damage <=80%
    of the selected-product control's nonnegative damage, and no fewer
    certificates than that control.
pred_b: Tucker WikiText-103 shifted row mean/p95/max damage
    <= .030/.070/.140 and untouched FineWeb mean/max <= .025/.060.
pred_c: exact split/rank/shape/price/hook identities; Tucker local heldout and
    fresh R2 each >=.765; selected-product local R2 exceeds random by >=.10.
pred_d: Tucker normalized 62-certificate vector projects onto the frozen
    rung-356 ray with cosine >=.95, vector R2 >=.70, count error <=3.

Strong null: Tucker local R2 <.70 or an inert candidate hook; OR both Tucker
and selected-product have census >=.08, both have <=20 certificates, or both
shifted means >=.08.  The selection instrument is also null if random is
within .02 local R2 of selected product and is no worse on census.

Decision: all positives license one original-native signed gate followed by a
composition test.  If the joint arm misses domination but selected product
itself reaches <=.025/43 and dominates, pivot to product-only.  Strong null
closes the L16 ~2M regime.  No rank/site tuning follows this receipt.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp16_tucker_physical_calibration_results.json"
FIT_CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FRESH_CACHE = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
CEV_TUCKER = ROOT / "cev_mlp16_tucker_r512_k576_p512.pt"
CEV_PRODUCT = ROOT / "cev_mlp16_product_k576.pt"
CEV_RANDOM = ROOT / "cev_mlp16_random_product_k576.pt"
LAYER = 16
D = 1152
H = 4608
R, K, P = 512, 576, 512
FIT_A = (0, 24)
FIT_B = (24, 48)
FRESH_FUNCTION = (48, 72)
FRESH_EVAL = (176, 188)
RANDOM_SEED = 388
NATIVE_LAYER_PRICE = 3 * D * H + D
TUCKER_LAYER_PRICE = R * D + 2 * K * R + P * K + D * P + D
PRODUCT_LAYER_PRICE = 3 * K * D + D
NATIVE_MODEL_PRICE = 545_902_902
TUCKER_MODEL_PRICE = NATIVE_MODEL_PRICE - NATIVE_LAYER_PRICE + TUCKER_LAYER_PRICE
PRODUCT_MODEL_PRICE = NATIVE_MODEL_PRICE - NATIVE_LAYER_PRICE + PRODUCT_LAYER_PRICE
WIKI_N = 120
WIKI_SKIP = 0


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    error = (prediction - target).square().sum()
    total = (target - target.mean(0)).square().sum().clamp_min(1e-30)
    return float(1.0 - error / total)


@torch.no_grad()
def _manual_logits(model, index: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def _capture(model, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, outputs = [], []

    def hook(_module, args, output):
        inputs.append(args[0].detach().reshape(-1, D).float().cpu())
        outputs.append(output.detach().reshape(-1, D).float().cpu())

    handle = model.transformer.h[LAYER].mlp.register_forward_hook(hook)
    device = next(model.parameters()).device
    try:
        for start in range(0, len(rows), 2):
            _manual_logits(model, rows[start:start + 2, :-1].to(device))
    finally:
        handle.remove()
    return torch.cat(inputs), torch.cat(outputs)


def _program_prediction(x: torch.Tensor, program: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "tucker":
        z = x @ program["q"].T
        hidden = (z @ program["a"].T) * (z @ program["b"].T)
        return (hidden @ program["c"].T) @ program["u"].T + program["bias"]
    indices = program[f"{mode}_indices"]
    left = program["native_left"][indices]
    right = program["native_right"][indices]
    down = program["native_down"][:, indices]
    hidden = (x @ left.T) * (x @ right.T)
    return hidden @ down.T + program["bias"]


@torch.no_grad()
def _build_program(model, fit_x: torch.Tensor, fit_y: torch.Tensor) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    x = fit_x.to(device)
    y = fit_y.to(device)
    mlp = model.transformer.h[LAYER].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()

    covariance = x.T @ x / x.shape[0]
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    q = vectors[:, torch.argsort(values, descending=True)[:R]].T.contiguous()
    lq, rq = left @ q.T, right @ q.T
    importance = down.norm(dim=0) * lq.norm(dim=1) * rq.norm(dim=1)
    selected = torch.argsort(importance, descending=True)[:K].contiguous()

    z = x @ q.T
    a, b = lq[selected].contiguous(), rq[selected].contiguous()
    products = (z @ a.T) * (z @ b.T)
    gram = products.T @ products + 1e-4 * torch.eye(K, device=device)
    output_map = torch.linalg.solve(gram, products.T @ (y - bias)).T
    uu, ss, vv = torch.linalg.svd(output_map, full_matrices=False)
    u = (uu[:, :P] * ss[:P]).contiguous()
    c = vv[:P].contiguous()

    generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    random_indices = torch.randperm(H, generator=generator)[:K].to(device).contiguous()
    return {
        "q": q, "a": a, "b": b, "u": u, "c": c, "bias": bias,
        "native_left": left, "native_right": right, "native_down": down,
        "product_indices": selected,
        "random_product_indices": random_indices,
    }


@torch.no_grad()
def _ce_vector(model, rows: torch.Tensor, program: dict[str, torch.Tensor] | None,
               mode: str | None, observed: dict[str, int]) -> torch.Tensor:
    handle = None
    device = next(model.parameters()).device
    if mode is not None:
        def replacement(_module, args, output):
            observed[mode] = observed.get(mode, 0) + 1
            return _program_prediction(args[0].float(), program, mode).to(output.dtype)
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


def _row_summary(candidate: torch.Tensor, native: torch.Tensor, n_rows: int) -> dict[str, object]:
    damage = (candidate - native).reshape(n_rows, -1).double().mean(1)
    return {
        "mean": float(damage.mean()),
        "p95": float(torch.quantile(damage, 0.95)),
        "max": float(damage.max()),
        "by_row": [float(value) for value in damage],
    }


def _certificate_metrics(cn, base: torch.Tensor, cev: torch.Tensor,
                         battery: dict, ray_receipt: dict) -> dict[str, object]:
    tags = ray_receipt["tags"]
    ray = torch.tensor(ray_receipt["qk"]["shape"], dtype=torch.float64)
    damage = cev.float().reshape(-1) - base.float().reshape(-1)
    values, member_abs = [], {}
    for tag in tags:
        member = cn.leaf(tag)["member"].long().cpu()
        threshold = 0.5 * float(battery[tag]["mean_ablation"]["top"][0]["abs_dce_members"])
        raw = float(damage[member].abs().mean())
        member_abs[tag] = raw
        values.append(raw / threshold)
    vector = torch.tensor(values, dtype=torch.float64)
    certificates = int((vector < 1.0).sum())
    scale = float(vector @ ray / ray.square().sum().clamp_min(1e-30))
    prediction = scale * ray
    cosine = float(F.cosine_similarity(vector[None], prediction[None]))
    denominator = (vector - vector.mean()).square().sum().clamp_min(1e-30)
    vector_r2 = float(1.0 - (vector - prediction).square().sum() / denominator)
    predicted_count = int((prediction < 1.0).sum())
    return {
        "certificates": certificates,
        "normalized_vector": [float(value) for value in vector],
        "member_abs_dce": member_abs,
        "ray_projection_scale": scale,
        "ray_cosine": cosine,
        "ray_vector_r2": vector_r2,
        "ray_predicted_certificates": predicted_count,
        "ray_certificate_count_error": abs(predicted_count - certificates),
    }


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        needed = [FIT_CACHE, FRESH_CACHE, ROOT / "census_state_diverse.pt",
                  ROOT / "circuits/BATTERY.json",
                  ROOT / "certificate_damage_axis_transfer_results.json"]
        assert all(path.exists() for path in needed)
        assert FIT_A[1] == FIT_B[0] and FIT_B[1] == FRESH_FUNCTION[0]
        assert TUCKER_LAYER_PRICE == 2_065_536 and PRODUCT_LAYER_PRICE == 1_991_808
        assert TUCKER_MODEL_PRICE == 532_042_038 and PRODUCT_MODEL_PRICE == 531_968_310
        print("L16 TUCKER PHYSICAL | dry run: splits, controls, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from tier2_model import load_elriggs

    fit_cached = torch.load(FIT_CACHE, map_location="cpu")
    fit_cached = fit_cached["rows"] if isinstance(fit_cached, dict) else fit_cached
    fresh_cached = torch.load(FRESH_CACHE, map_location="cpu")
    fresh_cached = fresh_cached["rows"] if isinstance(fresh_cached, dict) else fresh_cached
    fit_b = fit_cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    fit_a = fit_cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    function_fresh = fit_cached[FRESH_FUNCTION[0]:FRESH_FUNCTION[1], :257].long().contiguous()
    fresh_rows = fresh_cached[FRESH_EVAL[0]:FRESH_EVAL[1], :257].long().contiguous()
    assert fit_b.shape == fit_a.shape == function_fresh.shape == (24, 257)
    assert fresh_rows.shape == (12, 257)

    model, config = load_elriggs("bilin18")
    model.eval()
    assert config["n_embd"] == D and len(model.transformer.h) == 18
    xb, yb = _capture(model, fit_b)
    program = _build_program(model, xb, yb)
    xa, ya = _capture(model, fit_a)
    xf, yf = _capture(model, function_fresh)
    device = next(model.parameters()).device
    xa, ya, xf, yf = (value.to(device) for value in (xa, ya, xf, yf))
    local = {}
    for mode in ("tucker", "product", "random_product"):
        local[mode] = {
            "heldout_r2": _r2(_program_prediction(xa, program, mode), ya),
            "fresh_r2": _r2(_program_prediction(xf, program, mode), yf),
        }

    CN.use_state(str(ROOT / "census_state_diverse.pt"))
    census_rows = CN.rows().cpu().long()
    base = CN.base_ce().float().reshape(-1).cpu()
    assert base.numel() == CN.nflat() == census_rows.shape[0] * 256
    assert 2.0 <= float(base.mean()) <= 8.0
    observed: dict[str, int] = {}
    census_cevs = {
        mode: _ce_vector(model, census_rows, program, mode, observed)
        for mode in ("tucker", "product", "random_product")
    }
    torch.save(census_cevs["tucker"], CEV_TUCKER)
    torch.save(census_cevs["product"], CEV_PRODUCT)
    torch.save(census_cevs["random_product"], CEV_RANDOM)

    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    ray_receipt = json.loads((ROOT / "certificate_damage_axis_transfer_results.json").read_text())
    census = {}
    for mode, cev in census_cevs.items():
        certificate = _certificate_metrics(CN, base, cev, battery, ray_receipt)
        census[mode] = {
            "damage": float((cev.float() - base).mean()),
            "mean_absolute_position_damage": float((cev.float() - base).abs().mean()),
            **certificate,
        }

    wiki_rows, wiki_fingerprint, wiki_token_count = _wikitext103_train_rows(
        n=WIKI_N, width=257, skip=WIKI_SKIP)
    native_wiki = _ce_vector(model, wiki_rows, None, None, observed)
    native_fresh = _ce_vector(model, fresh_rows, None, None, observed)
    transfer = {}
    for mode in ("tucker", "product"):
        candidate_wiki = _ce_vector(model, wiki_rows, program, mode, observed)
        candidate_fresh = _ce_vector(model, fresh_rows, program, mode, observed)
        transfer[mode] = {
            "wikitext103": _row_summary(candidate_wiki, native_wiki, WIKI_N),
            "fineweb_fresh": _row_summary(candidate_fresh, native_fresh, len(fresh_rows)),
        }

    selected = program["product_indices"].detach().cpu().to(torch.int32).contiguous()
    random_indices = program["random_product_indices"].detach().cpu().to(torch.int32).contiguous()
    selected_hash = hashlib.sha256(selected.numpy().tobytes()).hexdigest()
    random_hash = hashlib.sha256(random_indices.numpy().tobytes()).hexdigest()
    shapes = {key: list(program[key].shape) for key in ("q", "a", "b", "u", "c")}
    identity_ok = (
        shapes == {"q": [R, D], "a": [K, R], "b": [K, R],
                   "u": [D, P], "c": [P, K]}
        and selected.unique().numel() == K and random_indices.unique().numel() == K
        and TUCKER_LAYER_PRICE == 2_065_536 and PRODUCT_LAYER_PRICE == 1_991_808
        and TUCKER_MODEL_PRICE == 532_042_038 and PRODUCT_MODEL_PRICE == 531_968_310
        and all(observed.get(mode, 0) > 0 for mode in ("tucker", "product", "random_product"))
    )

    tucker = census["tucker"]
    product = census["product"]
    random_control = census["random_product"]
    tucker_wiki = transfer["tucker"]["wikitext103"]
    tucker_fresh = transfer["tucker"]["fineweb_fresh"]
    pred_a = (
        tucker["damage"] <= .025 and tucker["certificates"] >= 43
        and tucker["damage"] <= .80 * max(product["damage"], 1e-12)
        and tucker["certificates"] >= product["certificates"]
    )
    pred_b = (
        tucker_wiki["mean"] <= .030 and tucker_wiki["p95"] <= .070
        and tucker_wiki["max"] <= .140 and tucker_fresh["mean"] <= .025
        and tucker_fresh["max"] <= .060
    )
    pred_c = (
        identity_ok and local["tucker"]["heldout_r2"] >= .765
        and local["tucker"]["fresh_r2"] >= .765
        and local["product"]["heldout_r2"] >= local["random_product"]["heldout_r2"] + .10
    )
    pred_d = (
        tucker["ray_cosine"] >= .95 and tucker["ray_vector_r2"] >= .70
        and tucker["ray_certificate_count_error"] <= 3
    )
    inert = float((census_cevs["tucker"].float() - base).abs().mean()) < 1e-8
    random_matches = (
        local["product"]["heldout_r2"] - local["random_product"]["heldout_r2"] <= .02
        and random_control["damage"] <= product["damage"]
    )
    strong_null = (
        local["tucker"]["heldout_r2"] < .70 or inert
        or (tucker["damage"] >= .08 and product["damage"] >= .08)
        or (tucker["certificates"] <= 20 and product["certificates"] <= 20)
        or (tucker_wiki["mean"] >= .08 and transfer["product"]["wikitext103"]["mean"] >= .08)
        or random_matches
    )
    product_pivot = (
        not pred_a and product["damage"] <= .025 and product["certificates"] >= 43
        and product["damage"] <= tucker["damage"]
    )

    result = {
        "status": "mlp16_tucker_physical_calibration_complete",
        "rung": 388,
        "claim_level": "physical_native_l16_tucker_census_certificate_ood_calibration",
        "convention": "candidate CE minus original native CE; lower is better",
        "fit": {"cache": FIT_CACHE.name, "fit_b": list(FIT_B),
                "heldout_fit_a": list(FIT_A), "fresh_function": list(FRESH_FUNCTION)},
        "fresh_evaluation": {"cache": FRESH_CACHE.name, "slice": list(FRESH_EVAL)},
        "ranks": {"input": R, "product": K, "output": P},
        "local_function_r2": local,
        "program_shapes": shapes,
        "selected_indices_sha256": selected_hash,
        "random_indices_sha256": random_hash,
        "selected_random_overlap": int(torch.isin(selected, random_indices).sum()),
        "random_seed": RANDOM_SEED,
        "hook_calls": observed,
        "prices": {
            "native_layer_scalars": NATIVE_LAYER_PRICE,
            "tucker_layer_scalars": TUCKER_LAYER_PRICE,
            "product_layer_scalars": PRODUCT_LAYER_PRICE,
            "native_model_scalars": NATIVE_MODEL_PRICE,
            "tucker_model_scalars": TUCKER_MODEL_PRICE,
            "product_model_scalars": PRODUCT_MODEL_PRICE,
        },
        "census": census,
        "transfer": transfer,
        "shifted_population": {
            "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train",
            "source_rows_half_open": [100_000, 110_000],
            "n_rows": WIKI_N, "skip_tokens": WIKI_SKIP,
            "fingerprint": wiki_fingerprint, "token_count": wiki_token_count,
        },
        "saved_cevs": {
            "tucker": CEV_TUCKER.name, "product": CEV_PRODUCT.name,
            "random_product": CEV_RANDOM.name,
        },
        "product_pivot_licensed": bool(product_pivot),
        'pred_a_tucker_predictive_and_beats_equal_cheaper_product': bool(pred_a),
        'pred_b_tucker_shifted_and_fresh_transfer': bool(pred_b),
        'pred_c_program_identity_local_transfer_and_selection_control': bool(pred_c),
        'pred_d_certificate_ray_transfer': bool(pred_d),
        'null_l16_two_million_regime_or_selection_instrument_fails': bool(strong_null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        "local_function_r2": local,
        "prices": result["prices"],
        "census": {mode: {key: value for key, value in row.items()
                            if key not in ("normalized_vector", "member_abs_dce")}
                   for mode, row in census.items()},
        "transfer": {
            mode: {population: {key: value for key, value in summary.items() if key != "by_row"}
                   for population, summary in populations.items()}
            for mode, populations in transfer.items()
        },
        "predicates": [pred_a, pred_b, pred_c, pred_d],
        "product_pivot": product_pivot,
        "strong_null": strong_null,
        "runtime_s": result["runtime_s"],
    }
    print(json.dumps(compact, indent=2), flush=True)
    print("L16 TUCKER PHYSICAL CALIBRATION DONE", flush=True)


if __name__ == "__main__":
    main()
