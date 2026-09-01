"""RUNG 303 -- PROSPECTIVE EMPIRICAL ERROR CONTRACT FOR MLP0 PROJECTIONS.

This screen asks whether a cheap, local quantity can become an executable error
contract.  Fit the same three MLP0 output bases used by rung 301 on its frozen
32 FineWeb rows.  On eight disjoint skip7000 calibration rows, measure for every
basis in {activation PCA, Down-weight SVD, signed response} and rank in
{64,128,256}:

    epsilon = sum ||y - y_hat||^2 / sum ||y - mean(y)||^2
    delta   = CE(projected MLP0) - CE(native model).

Fit log(delta) = intercept + slope*log(epsilon), and enlarge the min/max
calibration residual envelope by 15 percent in log-space.  Freeze that interval
and validate it without refitting on eight skip11000 rows.  This is an empirical
family/corpus contract, not a theorem and not an adoption result.

Historical control
------------------
Report the corresponding power exponent for the old site-1 isotropic random
error curve (relative norms .25/.5/1.0).  It is not pooled into the fit: random
errors and structured projection residuals need not share a damage law.

Frozen predictions
------------------
pred_a_transfer_coverage:
    At least 8/9 skip11000 damages fall inside the frozen interval.
pred_b_contract_is_nonvacuous:
    The interval multiplicative width is <=4 and Spearman(epsilon, damage) is
    >=0.90 on both calibration and validation windows.
pred_c_lower_bound_is_decision_useful:
    The transferred lower bound is >.05 nats for all three rank-64 arms, while
    at least one rank-128 arm is not rejected by that lower-bound test.

Null: coverage <6/9, multiplicative width >10, or either Spearman correlation
<.60.  Literal factorized MLP0 prices are carried through unchanged from rung
301.  No certificate, composition, or full-census credit is awarded.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_projection_error_contract_results.json"
DEV = "cuda"
D = 1152
H = 4608
RANKS = (64, 128, 256)
FAMILIES = ("activation_pca", "weight_svd", "response")
FIT_ROWS = 32
CONTRACT_ROWS = 8


def _load_rows(path: Path, n: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    rows = value["rows"] if isinstance(value, dict) else value
    assert rows.ndim == 2 and rows.shape[1] >= 257
    return rows[:n, :257].long().contiguous()


def _rankdata(values: list[float]) -> torch.Tensor:
    x = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=torch.float64)
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / torch.sqrt((rx @ rx) * (ry @ ry)))


@torch.no_grad()
def _native_ce(model: torch.nn.Module, rows: torch.Tensor, manual_logits) -> float:
    total, count = 0.0, 0
    for start in range(0, len(rows), 2):
        batch = rows[start:start + 2]
        index, target = batch[:, :-1].to(DEV), batch[:, 1:].to(DEV)
        logits = manual_logits(model, index)
        total += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="sum"))
        count += target.numel()
    return total / count


@torch.no_grad()
def _score_projection(model: torch.nn.Module, rows: torch.Tensor, basis: torch.Tensor,
                      mean: torch.Tensor, manual_logits) -> dict[str, float]:
    q, mu = basis.to(DEV), mean.to(DEV)
    local = {"error_energy": 0.0, "centered_energy": 0.0}

    def hook(_module, _args, output):
        centered = output.float() - mu
        projected = (centered @ q) @ q.T
        local["error_energy"] += float((centered - projected).square().sum())
        local["centered_energy"] += float(centered.square().sum())
        return (mu + projected).to(output.dtype)

    handle = model.transformer.h[0].mlp.register_forward_hook(hook)
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
        handle.remove()
    return {
        "ce": total / count,
        "relative_omitted_energy": local["error_energy"] / local["centered_energy"],
    }


def _fit_contract(rows: list[dict[str, float]]) -> dict[str, float]:
    x = torch.log(torch.tensor([r["relative_omitted_energy"] for r in rows], dtype=torch.float64))
    y = torch.log(torch.tensor([r["damage"] for r in rows], dtype=torch.float64))
    design = torch.stack((torch.ones_like(x), x), dim=1)
    coef = torch.linalg.lstsq(design, y).solution
    residual = y - design @ coef
    pad = 0.15 * max(float(residual.max() - residual.min()), 1e-9)
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "log_residual_lower": float(residual.min() - pad),
        "log_residual_upper": float(residual.max() + pad),
    }


def _bounds(contract: dict[str, float], epsilon: float) -> tuple[float, float, float]:
    center = contract["intercept"] + contract["slope"] * math.log(epsilon)
    return tuple(math.exp(center + offset) for offset in (
        contract["log_residual_lower"], 0.0, contract["log_residual_upper"]))


def _historical_curve() -> dict[str, float | list[float]]:
    data = json.loads((ROOT.parent / "polynomial_causal/stream_error_price_v1_results.json").read_text())
    site = next(row for row in data["sites"] if row["site"] == 1)
    norms = [0.25, 0.5, 1.0]
    damages = [float(site[str(norm)]["random"]["mean"]) for norm in norms]
    x = torch.log(torch.tensor([norm * norm for norm in norms], dtype=torch.float64))
    y = torch.log(torch.tensor(damages, dtype=torch.float64))
    design = torch.stack((torch.ones_like(x), x), dim=1)
    coef = torch.linalg.lstsq(design, y).solution
    return {"relative_norms": norms, "damages": damages,
            "squared_error_power_exponent": float(coef[1])}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for name in ("fineweb_n480_skip80.pt", "fineweb_n192_skip7000.pt",
                     "fineweb_n192_skip11000.pt"):
            assert (ROOT / ".rowcache" / name).exists()
        assert (ROOT.parent / "polynomial_causal/stream_error_price_v1_results.json").exists()
        assert len(RANKS) * len(FAMILIES) == 9
        print("MLP0 PROJECTION ERROR CONTRACT | dry run: populations, prices, and bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    from mlp0_signed_response_rank_screen import _bases, _fit_moments, _manual_logits
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    fit = _load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    calibration = _load_rows(ROOT / ".rowcache/fineweb_n192_skip7000.pt", CONTRACT_ROWS)
    validation = _load_rows(ROOT / ".rowcache/fineweb_n192_skip11000.pt", CONTRACT_ROWS)
    moments = _fit_moments(model, fit)
    fitted = _bases(moments)
    down = model.transformer.h[0].mlp.Down.weight.detach().float()
    weight_vectors = torch.linalg.svd(down, full_matrices=False).U.cpu()
    vectors = {
        "activation_pca": fitted["pca_vectors"],
        "weight_svd": weight_vectors,
        "response": fitted["response_vectors"],
    }
    native = {
        "calibration_ce": _native_ce(model, calibration, _manual_logits),
        "validation_ce": _native_ce(model, validation, _manual_logits),
    }
    arms: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        for rank in RANKS:
            key = f"{family}_r{rank}"
            price = 2 * H * D + rank * (H + D) + D
            row: dict[str, object] = {
                "family": family,
                "rank": rank,
                "literal_mlp0_scalars": price,
                "storage_fraction_native_mlp0": price / (3 * H * D + D),
            }
            for split, data in (("calibration", calibration), ("validation", validation)):
                score = _score_projection(model, data, vectors[family][:, :rank], fitted["mean"], _manual_logits)
                score["damage"] = score["ce"] - native[f"{split}_ce"]
                assert score["damage"] > 0 and 0 < score["relative_omitted_energy"] < 1
                row[split] = score
            arms[key] = row
            print(f"{key}: eps {row['calibration']['relative_omitted_energy']:.4f}/"
                  f"{row['validation']['relative_omitted_energy']:.4f} damage "
                  f"{row['calibration']['damage']:+.4f}/{row['validation']['damage']:+.4f}", flush=True)

    calibration_rows = [row["calibration"] for row in arms.values()]
    contract = _fit_contract(calibration_rows)
    covered = 0
    for row in arms.values():
        score = row["validation"]
        lower, estimate, upper = _bounds(contract, score["relative_omitted_energy"])
        score.update({"contract_lower": lower, "contract_estimate": estimate,
                      "contract_upper": upper, "covered": lower <= score["damage"] <= upper})
        covered += int(score["covered"])

    width = math.exp(contract["log_residual_upper"] - contract["log_residual_lower"])
    cal_rho = _spearman([r["relative_omitted_energy"] for r in calibration_rows],
                        [r["damage"] for r in calibration_rows])
    validation_rows = [row["validation"] for row in arms.values()]
    val_rho = _spearman([r["relative_omitted_energy"] for r in validation_rows],
                        [r["damage"] for r in validation_rows])
    rank64 = [row["validation"] for row in arms.values() if row["rank"] == 64]
    rank128 = [row["validation"] for row in arms.values() if row["rank"] == 128]
    pred_a = covered >= 8
    pred_b = width <= 4 and cal_rho >= 0.90 and val_rho >= 0.90
    pred_c = all(row["contract_lower"] > 0.05 for row in rank64) and any(
        row["contract_lower"] <= 0.05 for row in rank128)
    null = covered < 6 or width > 10 or cal_rho < 0.60 or val_rho < 0.60
    result = {
        "status": "mlp0_projection_error_contract_complete",
        "rung": 303,
        "claim_level": "prospective_empirical_family_corpus_contract_only",
        "populations": {"fit_rows": FIT_ROWS, "calibration_skip7000_rows": CONTRACT_ROWS,
                        "validation_skip11000_rows": CONTRACT_ROWS},
        "native": native,
        "contract": {**contract, "multiplicative_width": width,
                     "calibration_spearman": cal_rho, "validation_spearman": val_rho,
                     "validation_coverage": covered, "validation_total": len(validation_rows)},
        "historical_isotropic_random_control": _historical_curve(),
        "arms": arms,
        'pred_a_transfer_coverage': bool(pred_a),
        'pred_b_contract_is_nonvacuous': bool(pred_b),
        'pred_c_lower_bound_is_decision_useful': bool(pred_c),
        "null_contract_failure": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"contract": result["contract"], "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP0 PROJECTION ERROR CONTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
