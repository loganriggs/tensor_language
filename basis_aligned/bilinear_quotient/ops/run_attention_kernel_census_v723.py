#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fitted_kernels_track_closed_form pred_b_early_delta_kernels pred_c_most_kernels_are_windows pred_d_deep_kernels_subtract pred_e_families_fit_well
# BQLANE: cpu
"""Attention lane, v723: the 162 positional kernels of the final program, read by family (CPU fold; no model).

The exact-rank program (v718) carries one kernel kappa_h(d), d = 1..512, per head — the offset lookup that replaces the head's positional
side. This rung classifies each kernel by least squares over d = 1..512 into: DELTA (d = 1 carries >= 50% of the kernel's energy — the
previous-token taps of v622/v623), WINDOW (best fit a exp(-d/tau) + c with tau <= 8 and R^2 >= 0.8), LONG (exp with tau > 8, or a d^-p + c,
R^2 >= 0.8 — the ~1/d and running-mean-like heads), FLAT (mean^2 / (mean^2 + var) >= 0.9), else OTHER; and records the sign of the mean and
the correlation with the closed-form kernel (v702: the per-offset mean of the native pattern) to see how far the fit moved each kernel.
FOLD only, 0 forwards.
PREDICTIONS (scored as written; failures preserved)
    pred_a_fitted_kernels_track_closed_form median corr(fitted kernel, closed-form kernel) over the 162 heads >= 0.9. Prior: likely
    pred_b_early_delta_kernels          layers 0-2 hold >= 5 DELTA kernels. Prior: likely (v622: 0.3 and the previous-token taps)
    pred_c_most_kernels_are_windows     >= 60% of the 162 kernels are WINDOW or LONG. Prior: unsure
    pred_d_deep_kernels_subtract        in layers 9-17, >= half of the kernels have negative mean over d (subtractive reads). Prior: unsure
    pred_e_families_fit_well            >= 120 of 162 kernels have best-family R^2 >= 0.8 (DELTA and FLAT count as fitting). Prior: unsure
PRICE (registered maximum): 0 forwards; 0 backwards; 0 fits; CPU only (162 x 25 least-squares problems of size 512).
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import torch
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_kernel_census_v723_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
PROGS702 = ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt"
CANDIDATE_ID = "attention.kernel_census_v723"
LAYERS = tuple(range(18)); H = 9
TAUS = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
POWERS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
CORR_MIN, N_DELTA, FRAC_WINDOW, FRAC_NEG, N_FIT, R2_MIN = 0.9, 5, 0.6, 0.5, 120, 0.8
PREDICTIONS = {"pred_a_fitted_kernels_track_closed_form": "median corr >= 0.9", "pred_b_early_delta_kernels": ">= 5 delta in layers 0-2", "pred_c_most_kernels_are_windows": ">= 60% window/long",
               "pred_d_deep_kernels_subtract": ">= 50% negative mean in layers 9-17", "pred_e_families_fit_well": ">= 120 with R^2 >= 0.8"}


def ls_fit(y, basis):
    """Least squares y ~ basis @ coef; returns R^2 and coef."""
    coef = torch.linalg.lstsq(basis, y[:, None]).solution[:, 0]; resid = y - basis @ coef
    return float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-30)), coef


def classify(kappa):
    d = torch.arange(1, 513, dtype=torch.float64); y = kappa[1:513].double()
    energy = float((y ** 2).sum()); delta_share = float(y[0] ** 2 / max(energy, 1e-30))
    mean = float(y.mean()); var = float(y.var(unbiased=False)); flat = mean ** 2 / max(mean ** 2 + var, 1e-30)
    best = {"family": "none", "r2": -1.0}
    for tau in TAUS:
        r2, coef = ls_fit(y, torch.stack([torch.exp(-d / tau), torch.ones_like(d)], 1))
        if r2 > best["r2"]:
            best = {"family": "window" if tau <= 8 else "long", "r2": r2, "tau": tau, "a": float(coef[0]), "c": float(coef[1])}
    for p in POWERS:
        r2, coef = ls_fit(y, torch.stack([d ** (-p), torch.ones_like(d)], 1))
        if r2 > best["r2"]:
            best = {"family": "long", "r2": r2, "p": p, "a": float(coef[0]), "c": float(coef[1])}
    if delta_share >= 0.5:
        label = "delta"
    elif flat >= 0.9:
        label = "flat"
    elif best["r2"] >= R2_MIN:
        label = best["family"]
    else:
        label = "other"
    return {"label": label, "delta_share": delta_share, "flat_share": flat, "mean": mean, "energy": energy, "best": best}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "lane": "cpu",
            "bars": {"corr_min": CORR_MIN, "n_delta": N_DELTA, "frac_window": FRAC_WINDOW, "frac_neg": FRAC_NEG, "n_fit": N_FIT, "r2_min": R2_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    fac = torch.load(PROGS718, map_location="cpu")["factored"]; cf = torch.load(PROGS702, map_location="cpu")
    heads = {}
    for l in LAYERS:
        for h in range(H):
            key = f"{l}.{h}"; k_fit = fac[key]["kappa"].float(); k_cf = cf[f"kappa_{l}_{h}"].float()
            corr = float(torch.corrcoef(torch.stack([k_fit[1:513], k_cf[1:513]]))[0, 1])
            info = classify(k_fit); info["corr_closed_form"] = corr; info["closed_form"] = classify(k_cf)["label"]; heads[key] = info
        print(f"layer {l}: " + " ".join(f"{h}:{heads[f'{l}.{h}']['label']}({heads[f'{l}.{h}']['best'].get('tau', heads[f'{l}.{h}']['best'].get('p', '-'))},{'+' if heads[f'{l}.{h}']['mean'] > 0 else '-'},r2 {heads[f'{l}.{h}']['best']['r2']:.2f},corr {heads[f'{l}.{h}']['corr_closed_form']:.2f})" for h in range(H)))
    corrs = sorted(v["corr_closed_form"] for v in heads.values()); med_corr = corrs[len(corrs) // 2]
    counts = {}
    for v in heads.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    n_delta_early = sum(1 for k, v in heads.items() if int(k.split(".")[0]) <= 2 and v["label"] == "delta")
    frac_window = sum(1 for v in heads.values() if v["label"] in ("window", "long")) / len(heads)
    deep = [v for k, v in heads.items() if int(k.split(".")[0]) >= 9]; frac_neg = sum(1 for v in deep if v["mean"] < 0) / len(deep)
    n_fit = sum(1 for v in heads.values() if v["label"] in ("delta", "flat") or v["best"]["r2"] >= R2_MIN)
    moved = sorted(heads, key=lambda k: heads[k]["corr_closed_form"])[:8]
    print(f"families {counts} | median corr with closed-form {med_corr:.3f} (least: {[(k, round(heads[k]['corr_closed_form'], 2)) for k in moved]}) | delta in layers 0-2: {n_delta_early} | window+long {frac_window:.2f} | deep negative-mean {frac_neg:.2f} | fit well {n_fit}/162")
    predictions = {"pred_a_fitted_kernels_track_closed_form": med_corr >= CORR_MIN, "pred_b_early_delta_kernels": n_delta_early >= N_DELTA, "pred_c_most_kernels_are_windows": frac_window >= FRAC_WINDOW,
                   "pred_d_deep_kernels_subtract": frac_neg >= FRAC_NEG, "pred_e_families_fit_well": n_fit >= N_FIT}
    OUT.write_text(json.dumps({"schema": "attention_kernel_census_result_v723", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"heads": heads, "families": counts, "median_corr_closed_form": med_corr, "n_delta_early": n_delta_early, "frac_window_long": frac_window, "frac_neg_deep": frac_neg, "n_fit_well": n_fit},
                               "predictions": predictions, "forwards": 0, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": 0}, indent=2))


if __name__ == "__main__":
    main()
