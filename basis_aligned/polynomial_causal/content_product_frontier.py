"""First matched-product frontier for the bilin18 content API.

This is a local interface diagnostic, not yet a whole-ship correction.  It asks
whether 32 learned shared products predict a frozen 64-dimensional content output
slice of MLP0-2 better than three honest controls:

* 32 native MLP products, selected on discovery rows and given a refitted decoder;
* a seeded 32-product random feature map with a refitted decoder;
* a linear map whose parameter count is within 3% of the paired program.

The content basis is fit only on the first 96 sequences from pooled residual
deviations at layers 8, 10, and 12.  Surrogate optimization uses those sequences,
checkpoint selection uses the next 48, and all reported headline R2 values use the
final untouched 48.  The output metric whitens each content coordinate by its
discovery standard deviation.

Registered before execution:
  A. the learned paired program beats both native products and the matched linear
     map by >= .05 held-out whitened R2 at at least one of MLP0-2;
  B. the winning paired program reaches held-out whitened R2 >= .60;
  C. its validation-to-heldout R2 drop is <= .10;
  D. reciprocal factor rescaling changes its output by relative RMSE <= 1e-5.

Passing this local stage licenses installation as a current-ship residual correction;
it does not satisfy the whole-model admission gates in CONTENT_API_COMPILER_SPEC.md.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(BQ))
sys.path.insert(0, "/workspace/rspd")

import census_lib as cl  # noqa: E402
from bilin18_joint_removal import DEV, m  # noqa: E402


D = 1152
M = 64
K = 32
NSEQ = 192
TRAIN_SEQ = 96
VALID_SEQ = 48
SEQ = 256
POSITIONS = torch.arange(64, SEQ - 1, 3)
SITES = (0, 1, 2)
CONTENT_LAYERS = (8, 10, 12)
STEPS = 500
BATCH = 512
LR = 2e-3
RIDGE = 1e-4
SEED = 314159
OUT = HERE / "content_product_frontier_results.json"
FACTORS = HERE / "content_product_frontier_factors.pt"


def linear(module, x: torch.Tensor) -> torch.Tensor:
    bias = getattr(module, "bias", None)
    return F.linear(x, module.weight.float(), None if bias is None else bias.float())


def ridge_decoder(features: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge regression with an unpenalized intercept."""
    xm = features.mean(0)
    ym = target.mean(0)
    xc = features - xm
    yc = target - ym
    gram = xc.T @ xc
    scale = float(gram.diag().mean().clamp_min(1e-12))
    gram.diagonal().add_(RIDGE * scale)
    weight = torch.linalg.solve(gram, xc.T @ yc)
    bias = ym - xm @ weight
    return weight, bias


def r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    denominator = (true - true.mean(0)).square().sum().clamp_min(1e-30)
    return 1.0 - float((pred - true).square().sum() / denominator)


@torch.no_grad()
def gather(rows: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    xs = {site: [] for site in SITES}
    ys = {site: [] for site in SITES}
    residuals = {layer: [] for layer in CONTENT_LAYERS}
    hooks = []

    for site in SITES:
        def make_hook(site_: int):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                xs[site_].append(args[0][:, POSITIONS].detach().float().cpu())
                ys[site_].append(value[:, POSITIONS].detach().float().cpu())
            return hook
        hooks.append(m.transformer.h[site].mlp.register_forward_hook(make_hook(site)))

    blocks = rows[:, :SEQ].contiguous()
    for start in range(0, NSEQ, 8):
        idx = blocks[start:start + 8, :-1].to(DEV)
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for layer, block in enumerate(m.transformer.h):
            x, v1 = block(x, v1, x0)
            if layer in CONTENT_LAYERS:
                residuals[layer].append(x.detach().float().cpu())

    for hook in hooks:
        hook.remove()
    return (
        {site: torch.cat(parts, 0) for site, parts in xs.items()},
        {site: torch.cat(parts, 0) for site, parts in ys.items()},
        {layer: torch.cat(parts, 0) for layer, parts in residuals.items()},
        blocks[:, :-1].cpu(),
    )


@torch.no_grad()
def fit_content_basis(residuals: dict[int, torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    tok = tokens[:TRAIN_SEQ].reshape(-1).to(DEV)
    counts = torch.bincount(tok, minlength=int(m.lm_head.weight.shape[0])).float()
    devsum = None
    for layer in CONTENT_LAYERS:
        values = residuals[layer][:TRAIN_SEQ].reshape(-1, D).to(DEV)
        means = torch.zeros(len(counts), D, device=DEV)
        means.index_add_(0, tok, values)
        means /= counts.clamp_min(1).unsqueeze(1)
        deviation = values - means[tok]
        devsum = deviation if devsum is None else devsum + deviation
        del values, means, deviation
    pooled = devsum / len(CONTENT_LAYERS)
    pooled -= pooled.mean(0)
    _, _, vh = torch.linalg.svd(pooled, full_matrices=False)
    basis = vh[:M].T.contiguous()
    del pooled, devsum, vh
    return basis


def split_sequences(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_end = TRAIN_SEQ
    valid_end = TRAIN_SEQ + VALID_SEQ
    return tuple(
        part.reshape(-1, part.shape[-1]).to(DEV)
        for part in (value[:train_end], value[train_end:valid_end], value[valid_end:])
    )


@torch.no_grad()
def native_features(mlp, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    left = F.linear(x, mlp.Left.weight.float()[indices])
    right = F.linear(x, mlp.Right.weight.float()[indices])
    return left * right


def paired_output(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, decoder: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return ((x @ a) * (x @ b)) @ decoder + bias


def train_paired(
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    a0: torch.Tensor,
    b0: torch.Tensor,
    decoder0: torch.Tensor,
    bias0: torch.Tensor,
    seed: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    decoder = decoder0.clone().requires_grad_(True)
    bias = bias0.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([a, b, decoder, bias], lr=LR, weight_decay=1e-5)
    generator = torch.Generator(device=DEV).manual_seed(seed)
    best = None
    curve = []
    for step in range(1, STEPS + 1):
        indices = torch.randint(len(xtr), (BATCH,), generator=generator, device=DEV)
        prediction = paired_output(xtr[indices], a, b, decoder, bias)
        loss = (prediction - ytr[indices]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a, b, decoder, bias], 10.0)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            with torch.no_grad():
                value = r2(paired_output(xva, a, b, decoder, bias), yva)
            curve.append([step, value])
            if best is None or value > best[0]:
                best = (value, {"a": a.detach().cpu(), "b": b.detach().cpu(),
                                "decoder": decoder.detach().cpu(), "bias": bias.detach().cpu()})

    state = {name: value.to(DEV) for name, value in best[1].items()}
    with torch.no_grad():
        test_prediction = paired_output(xte, **state)
        test_r2 = r2(test_prediction, yte)
        sample = xte[: min(512, len(xte))]
        base = paired_output(sample, **state)
        scales = torch.exp(torch.linspace(-1.0, 1.0, K, device=DEV))
        gauged = paired_output(sample, state["a"] * scales, state["b"] / scales,
                               state["decoder"], state["bias"])
        gauge_rel = math.sqrt(float((base - gauged).double().square().sum()) /
                              max(float(base.double().square().sum()), 1e-30))
    metrics = {
        "best_validation_r2": best[0],
        "heldout_r2": test_r2,
        "validation_to_heldout_drop": best[0] - test_r2,
        "gauge_relative_rmse": gauge_rel,
        "validation_curve": curve,
    }
    return metrics, best[1]


def evaluate_site(site: int, xseq: torch.Tensor, yseq: torch.Tensor, basis: torch.Tensor) -> tuple[dict, dict]:
    mlp = m.transformer.h[site].mlp
    with torch.no_grad():
        projected = yseq.to(DEV) @ basis
        xtr, xva, xte = split_sequences(xseq)
        ytr, yva, yte = split_sequences(projected.cpu())
        mean = ytr.mean(0)
        scale = ytr.std(0).clamp_min(1e-5)
        ytr = (ytr - mean) / scale
        yva = (yva - mean) / scale
        yte = (yte - mean) / scale

        # Parameter-matched linear control.
        linear_w, linear_b = ridge_decoder(xtr, ytr)
        linear_valid = r2(xva @ linear_w + linear_b, yva)
        linear_test = r2(xte @ linear_w + linear_b, yte)

        # Select native products by discovery contribution under the exact output decoder.
        all_phi = linear(mlp.Left, xtr) * linear(mlp.Right, xtr)
        exact_decoder = mlp.Down.weight.float().T @ basis
        contribution = all_phi.std(0) * exact_decoder.norm(dim=1)
        selected = contribution.topk(K).indices
        phi_tr = all_phi[:, selected]
        phi_va = native_features(mlp, xva, selected)
        phi_te = native_features(mlp, xte, selected)
        native_w, native_b = ridge_decoder(phi_tr, ytr)
        native_valid = r2(phi_va @ native_w + native_b, yva)
        native_test = r2(phi_te @ native_w + native_b, yte)

        # Seeded random product control, with exactly the same product count and decoder.
        generator = torch.Generator(device=DEV).manual_seed(SEED + site)
        random_a = torch.randn(D, K, generator=generator, device=DEV) / math.sqrt(D)
        random_b = torch.randn(D, K, generator=generator, device=DEV) / math.sqrt(D)
        random_tr = (xtr @ random_a) * (xtr @ random_b)
        random_w, random_bias = ridge_decoder(random_tr, ytr)
        random_valid = r2(((xva @ random_a) * (xva @ random_b)) @ random_w + random_bias, yva)
        random_test = r2(((xte @ random_a) * (xte @ random_b)) @ random_w + random_bias, yte)

        # Initialize the learned paired program at the native-product baseline.
        a0 = mlp.Left.weight.float()[selected].T.contiguous()
        b0 = mlp.Right.weight.float()[selected].T.contiguous()
        del all_phi, contribution, random_tr
    torch.cuda.empty_cache()
    paired, state = train_paired(xtr, ytr, xva, yva, xte, yte, a0, b0,
                                 native_w, native_b, SEED + 100 + site)
    paired["gain_over_native_heldout"] = paired["heldout_r2"] - native_test
    paired["gain_over_linear_heldout"] = paired["heldout_r2"] - linear_test

    standalone = 2 * D * K + K * M + M
    linear_parameters = D * M + M
    result = {
        "linear": {"validation_r2": linear_valid, "heldout_r2": linear_test,
                   "parameters": linear_parameters},
        "native_selected": {"validation_r2": native_valid, "heldout_r2": native_test,
                            "selected_indices": selected.cpu().tolist(),
                            "amortized_new_parameters": K * M + M,
                            "standalone_parameters": standalone},
        "random_products": {"validation_r2": random_valid, "heldout_r2": random_test,
                            "decoder_parameters": K * M + M,
                            "seed": SEED + site},
        "learned_paired": paired,
        "pricing": {
            "products": K,
            "paired_standalone_parameters": standalone,
            "linear_parameters": linear_parameters,
            "paired_to_linear_parameter_ratio": standalone / linear_parameters,
            "output_metric": "content coordinates whitened by discovery standard deviation",
        },
    }
    saved = {**state, "output_mean": mean.cpu(), "output_scale": scale.cpu(),
             "content_basis": basis.cpu(), "site": site}
    return result, saved


def main() -> None:
    torch.manual_seed(SEED)
    start = time.time()
    cl.use_state(str(BQ / "census_state_diverse.pt"))
    rows = cl.fineweb_rows(NSEQ)
    xseq, yseq, residuals, tokens = gather(rows)
    basis = fit_content_basis(residuals, tokens)
    del residuals, tokens, rows
    torch.cuda.empty_cache()

    results = {}
    factors = {}
    for site in SITES:
        results[str(site)], factors[str(site)] = evaluate_site(site, xseq[site], yseq[site], basis)
        row = results[str(site)]
        print(
            f"mlp{site}: linear={row['linear']['heldout_r2']:.3f} "
            f"native={row['native_selected']['heldout_r2']:.3f} "
            f"random={row['random_products']['heldout_r2']:.3f} "
            f"paired={row['learned_paired']['heldout_r2']:.3f}",
            flush=True,
        )

    winners = [site for site in SITES if
               results[str(site)]["learned_paired"]["gain_over_native_heldout"] >= 0.05 and
               results[str(site)]["learned_paired"]["gain_over_linear_heldout"] >= 0.05]
    best_site = max(SITES, key=lambda site: results[str(site)]["learned_paired"]["heldout_r2"])
    best = results[str(best_site)]["learned_paired"]
    predictions = {
        "A_paired_beats_native_and_linear": bool(winners),
        "B_paired_reaches_r2_0p60": best["heldout_r2"] >= 0.60,
        "C_validation_generalizes": best["validation_to_heldout_drop"] <= 0.10,
        "D_factor_gauge_invariant": all(results[str(site)]["learned_paired"]["gauge_relative_rmse"] <= 1e-5 for site in SITES),
    }
    output = {
        "config": {
            "model": "bilin18",
            "sites": list(SITES),
            "content_basis_layers": list(CONTENT_LAYERS),
            "content_dimension": M,
            "products": K,
            "sequences": {"train": TRAIN_SEQ, "validation": VALID_SEQ,
                          "heldout": NSEQ - TRAIN_SEQ - VALID_SEQ},
            "positions_per_sequence": len(POSITIONS),
            "steps": STEPS,
            "seed": SEED,
            "status": "local_interface_diagnostic_not_whole_ship_admission",
        },
        "sites": results,
        "best_paired_site": best_site,
        "predictions": predictions,
        "pricing_interpretation": {
            "primary_ledger": "standalone parameters for every factor and decoder",
            "native_amortized_status": "conditional only after factor projections are independently admitted, paid for, and reused",
            "rule": "provenance in the original model does not make a factor free",
        },
        "runtime_s": round(time.time() - start, 1),
    }
    torch.save({"config": output["config"], "sites": factors}, FACTORS)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(predictions, indent=2), flush=True)
    print(f"wrote {OUT} and {FACTORS} ({output['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
