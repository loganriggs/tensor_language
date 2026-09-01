"""RUNG 306 -- MULTI-LAYER ACTIVATION-PCA MLP COMPOSITION.

Rung 301 found that a rank-256 activation-PCA projection of MLP0's output costs
only +.0209 FineWeb / +.0114 WikiText while replacing native Down by a literal
factorized map.  Test whether this seed generalizes across layers and composes.

Fit a rank-256 output-PCA basis and mean independently for every one of the 18
MLPs on 16 frozen FineWeb skip80 rows.  A projected layer executes

    y_hat = mu + (y-mu) Q Q^T.

It does not need native Down at deployment: store Q^T Down, Q, and the adjusted
constant along with native Left/Right.  Per-layer price is

    2*4608*1152 + 256*(4608+1152) + 1152 = 12,092,544,

saving 3,833,856 scalars versus native MLP.  Four layers save 15,335,424.

Score every layer individually on eight FineWeb skip7000 calibration rows,
eight FineWeb skip11000 validation rows, and eight WikiText-2 test rows after
skip30000.  Choose the four smallest calibration damages only, freeze them,
and compose on both validation corpora.  Compare with fixed layers {0,5,11,17}
at identical price.  No validation result selects a layer.

Frozen predictions
------------------
pred_a_pca_generality:
    At least 9/18 layers have individual damage <=.04 on calibration FineWeb,
    validation FineWeb, and WikiText.
pred_b_calibration_selects_safe_layers:
    Calibration-vs-validation-FineWeb layer sensitivity Spearman >=.70 and
    every selected layer has individual damage <=.04 on BOTH validation corpora.
pred_c_four_layer_composition_is_predictive:
    Selected-four composition damage <=.08 FineWeb and <=.10 WikiText; each is
    <=1.5 times the sum of nonnegative selected single-layer damages; and it is
    >=25% smaller than the nonnegative fixed-four control damage on both.

Null: selected composition damage >=.15 on either validation corpus, or at most
three layers meet the individual <=.04 bar on all populations.  A pass is a
composition screen, not adoption: census, certificates, exact whole-program
billing, shifted OOD, and signed interventions remain mandatory.
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
OUT = ROOT / "mlp_activation_pca_four_layer_composition_results.json"
DEV = "cuda"
D = 1152
H = 4608
LAYERS = 18
RANK = 256
FIT_ROWS = 16
EVAL_ROWS = 8
SELECT_N = 4
FIXED_CONTROL = (0, 5, 11, 17)
WIKI_SKIP = 30000


def _load_rows(path: Path, n: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    rows = value["rows"] if isinstance(value, dict) else value
    assert rows.ndim == 2 and rows.shape[1] >= 257
    return rows[:n, :257].long().contiguous()


@torch.no_grad()
def _capture_outputs(model: torch.nn.Module, rows: torch.Tensor, manual_logits) -> list[torch.Tensor]:
    captured: list[list[torch.Tensor]] = [[] for _ in range(LAYERS)]
    handles = []
    for layer in range(LAYERS):
        def hook(_module, _args, output, layer=layer):
            captured[layer].append(output.detach().half().cpu().reshape(-1, D))
        handles.append(model.transformer.h[layer].mlp.register_forward_hook(hook))
    try:
        for start in range(0, len(rows), 2):
            index = rows[start:start + 2, :-1].to(DEV)
            manual_logits(model, index)
    finally:
        for handle in handles:
            handle.remove()
    outputs = [torch.cat(parts) for parts in captured]
    assert all(value.shape == (len(rows) * 256, D) for value in outputs)
    return outputs


def _fit_pca(outputs: list[torch.Tensor]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for layer, output_cpu in enumerate(outputs):
        output = output_cpu.float().to(DEV)
        mean = output.mean(0)
        centered = output - mean
        covariance = centered.T @ centered / len(centered)
        covariance = 0.5 * (covariance + covariance.T)
        assert bool(torch.isfinite(covariance).all())
        jitter = 1e-7 * float(torch.diagonal(covariance).mean().abs().clamp_min(1e-12))
        covariance = covariance + jitter * torch.eye(D, device=DEV)
        try:
            values, vectors = torch.linalg.eigh(covariance)
        except torch._C._LinAlgError:
            # The last native MLP has an extremely repeated fp16-captured
            # spectrum.  Float64 is a numerical fallback only; it does not
            # change the population, rank, basis objective, or bars.
            values64, vectors64 = torch.linalg.eigh(covariance.double())
            values, vectors = values64.float(), vectors64.float()
        basis = vectors[:, torch.argsort(values, descending=True)[:RANK]]
        result[layer] = (basis.cpu(), mean.cpu())
        print(f"fit layer {layer}: top{RANK} energy "
              f"{float(values.sort(descending=True).values[:RANK].sum()/values.clamp_min(0).sum()):.4f}",
              flush=True)
        del output, centered, covariance, values, vectors, basis
    return result


@torch.no_grad()
def _score(model: torch.nn.Module, rows: torch.Tensor,
           projectors: dict[int, tuple[torch.Tensor, torch.Tensor]], manual_logits) -> float:
    handles = []
    for layer, (basis_cpu, mean_cpu) in projectors.items():
        q, mean = basis_cpu.to(DEV), mean_cpu.to(DEV)

        def hook(_module, _args, output, q=q, mean=mean):
            centered = output.float() - mean
            return (mean + (centered @ q) @ q.T).to(output.dtype)

        handles.append(model.transformer.h[layer].mlp.register_forward_hook(hook))
    total, count = 0.0, 0
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index, target = batch[:, :-1].to(DEV), batch[:, 1:].to(DEV)
            logits = manual_logits(model, index)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="sum"))
            count += target.numel()
    finally:
        for handle in handles:
            handle.remove()
    return total / count


def _rankdata(values: list[float]) -> torch.Tensor:
    value = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(value, stable=True)
    rank = torch.empty_like(value)
    rank[order] = torch.arange(len(value), dtype=torch.float64)
    return rank


def _spearman(left: list[float], right: list[float]) -> float:
    a, b = _rankdata(left), _rankdata(right)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / torch.sqrt((a @ a) * (b @ b)))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for name in ("fineweb_n480_skip80.pt", "fineweb_n192_skip7000.pt",
                     "fineweb_n192_skip11000.pt"):
            assert (ROOT / ".rowcache" / name).exists()
        native = 3 * H * D + D
        factorized = 2 * H * D + RANK * (H + D) + D
        assert native - factorized == 3833856
        assert len(FIXED_CONTROL) == SELECT_N and len(set(FIXED_CONTROL)) == SELECT_N
        print("MLP ACTIVATION PCA FOUR-LAYER COMPOSITION | dry run: price, populations, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mlp0_signed_response_rank_screen import _manual_logits, _wikitext_rows
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == LAYERS
    fit = _load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    calibration = _load_rows(ROOT / ".rowcache/fineweb_n192_skip7000.pt", EVAL_ROWS)
    validation = _load_rows(ROOT / ".rowcache/fineweb_n192_skip11000.pt", EVAL_ROWS)
    wikitext, fingerprint = _wikitext_rows(EVAL_ROWS, skip=WIKI_SKIP)
    outputs = _capture_outputs(model, fit, _manual_logits)
    pca = _fit_pca(outputs)
    del outputs
    native = {
        "calibration_fineweb": _score(model, calibration, {}, _manual_logits),
        "validation_fineweb": _score(model, validation, {}, _manual_logits),
        "wikitext": _score(model, wikitext, {}, _manual_logits),
    }
    individual = {}
    for layer in range(LAYERS):
        projector = {layer: pca[layer]}
        row = {
            "calibration_fineweb_damage": _score(model, calibration, projector, _manual_logits)
            - native["calibration_fineweb"],
            "validation_fineweb_damage": _score(model, validation, projector, _manual_logits)
            - native["validation_fineweb"],
            "wikitext_damage": _score(model, wikitext, projector, _manual_logits) - native["wikitext"],
        }
        individual[str(layer)] = row
        print(f"layer {layer}: cal/val/wiki {row['calibration_fineweb_damage']:+.4f}/"
              f"{row['validation_fineweb_damage']:+.4f}/{row['wikitext_damage']:+.4f}", flush=True)

    selected = tuple(sorted(range(LAYERS), key=lambda layer:
                            individual[str(layer)]["calibration_fineweb_damage"])[:SELECT_N])
    selected_projectors = {layer: pca[layer] for layer in selected}
    fixed_projectors = {layer: pca[layer] for layer in FIXED_CONTROL}
    compositions = {}
    for name, projectors in (("calibration_selected", selected_projectors),
                             ("fixed_spaced_control", fixed_projectors)):
        row = {
            "layers": sorted(projectors),
            "validation_fineweb_damage": _score(model, validation, projectors, _manual_logits)
            - native["validation_fineweb"],
            "wikitext_damage": _score(model, wikitext, projectors, _manual_logits) - native["wikitext"],
        }
        compositions[name] = row
        print(f"{name} {row['layers']}: val/wiki {row['validation_fineweb_damage']:+.4f}/"
              f"{row['wikitext_damage']:+.4f}", flush=True)

    calibration_values = [individual[str(layer)]["calibration_fineweb_damage"] for layer in range(LAYERS)]
    validation_values = [individual[str(layer)]["validation_fineweb_damage"] for layer in range(LAYERS)]
    rho = _spearman(calibration_values, validation_values)
    broadly_safe = [layer for layer in range(LAYERS) if all(
        individual[str(layer)][key] <= 0.04 for key in
        ("calibration_fineweb_damage", "validation_fineweb_damage", "wikitext_damage"))]
    selected_safe = all(
        individual[str(layer)]["validation_fineweb_damage"] <= 0.04
        and individual[str(layer)]["wikitext_damage"] <= 0.04 for layer in selected)
    selected_row = compositions["calibration_selected"]
    fixed_row = compositions["fixed_spaced_control"]
    additive = {
        "validation_fineweb": sum(max(individual[str(layer)]["validation_fineweb_damage"], 0.0)
                                   for layer in selected),
        "wikitext": sum(max(individual[str(layer)]["wikitext_damage"], 0.0) for layer in selected),
    }
    ratio = {
        "validation_fineweb": selected_row["validation_fineweb_damage"]
        / max(additive["validation_fineweb"], 1e-12),
        "wikitext": selected_row["wikitext_damage"] / max(additive["wikitext"], 1e-12),
    }
    pred_a = len(broadly_safe) >= 9
    pred_b = rho >= 0.70 and selected_safe
    pred_c = bool(
        selected_row["validation_fineweb_damage"] <= 0.08
        and selected_row["wikitext_damage"] <= 0.10
        and ratio["validation_fineweb"] <= 1.5 and ratio["wikitext"] <= 1.5
        and fixed_row["validation_fineweb_damage"] >= 0
        and fixed_row["wikitext_damage"] >= 0
        and selected_row["validation_fineweb_damage"]
        <= 0.75 * fixed_row["validation_fineweb_damage"]
        and selected_row["wikitext_damage"] <= 0.75 * fixed_row["wikitext_damage"]
    )
    null = bool(
        selected_row["validation_fineweb_damage"] >= 0.15
        or selected_row["wikitext_damage"] >= 0.15
        or len(broadly_safe) <= 3
    )
    native_price = 3 * H * D + D
    factor_price = 2 * H * D + RANK * (H + D) + D
    result = {
        "status": "mlp_activation_pca_four_layer_composition_complete",
        "rung": 306,
        "claim_level": "calibration_selected_two_corpus_four_layer_composition_screen_only",
        "price": {"native_mlp_scalars_each": native_price,
                  "factorized_mlp_scalars_each": factor_price,
                  "saving_scalars_each": native_price - factor_price,
                  "selected_layers": SELECT_N,
                  "total_saving_scalars": SELECT_N * (native_price - factor_price)},
        "fit": {"rows": FIT_ROWS, "positions_per_layer": FIT_ROWS * 256,
                "rank": RANK, "validation_labels_used_for_selection": False},
        "evaluation": {"rows_per_population": EVAL_ROWS, "fineweb_calibration_skip": 7000,
                       "fineweb_validation_skip": 11000, "wikitext_skip": WIKI_SKIP,
                       "wikitext_fingerprint": fingerprint},
        "native": native,
        "individual": individual,
        "selection": {"selected_layers": list(selected), "fixed_control_layers": list(FIXED_CONTROL),
                      "broadly_safe_layers": broadly_safe,
                      "calibration_validation_spearman": rho},
        "compositions": compositions,
        "selected_additive_nonnegative_prediction": additive,
        "selected_composition_ratio": ratio,
        'pred_a_pca_generality': bool(pred_a),
        'pred_b_calibration_selects_safe_layers': bool(pred_b),
        'pred_c_four_layer_composition_is_predictive': bool(pred_c),
        "null_no_composable_pca_layers": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected": selected, "safe_count": len(broadly_safe), "rho": rho,
                      "composition_ratio": ratio, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP ACTIVATION PCA FOUR-LAYER COMPOSITION DONE", flush=True)


if __name__ == "__main__":
    main()
