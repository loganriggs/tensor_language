#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_gain_uniform_within_every_class pred_c_scalar_reading_holds_every_class pred_d_uniformity_stable_with_size pred_e_gain_is_set_by_the_context
"""MLP 1: the context gain across context tokens, token classes and class size (v290). v289: after one context token ("The") MLP 1's write at the
noun is its single-token table entry times a scalar gain alpha ~ 0.54, uniform across 32 nouns (CV 0.09), residual 23%. Logan's request: fold the
single-token paths by class with EXPONENTIAL class sizes and describe what MLP 1 does. Here the two-token input [context, target] is folded exactly:
the block-1 input change c = n_ctx - n_table is the whole effect of the context through attention 0/1 and MLP 0 (recurrence exact), and the write
change Down[(L t)(R c) + (L c)(R t) + (L c)(R c)] is decomposed as in v289. Six contexts ("The", ",", " and", " of", " very", " 1") x seven
classes (v287 spec) x 64 targets, analysed at sizes 4 / 16 / 64. Per (context, class, size): alpha median, alpha CV across targets, residual 1 - cos^2.
PREDICTIONS (scored as written; failures preserved; priors from v289)
    pred_a_pair_closure                    Down[cross + context-only] = W_ctx - W_table within relative 1e-3 on every row
    pred_b_gain_uniform_within_every_class alpha's CV across the 64 targets <= 0.25 for every (context, class)
    pred_c_scalar_reading_holds_every_class median residual energy after the scalar <= 0.30 for every (context, class) at size 64
    pred_d_uniformity_stable_with_size      for every class, pooled over contexts, CV(64) <= 1.5 x CV(16) (uniformity is not a small-sample artefact)
    pred_e_gain_is_set_by_the_context       pooled over classes, alpha's spread ACROSS contexts (std of the six context medians) >= 2 x the median within-(context, class) std. Prior: unsure.
PRICE (registered maximum): 6 x 7 x 64 = 2688 two-token rows / 256 = 11 forwards + 448 single tokens / 256 = 2 forwards = 13; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_context_gain_scaling_v290_result.json"
CANDIDATE_ID = "mlp1.token_table.context_gain_scaling_v290"
CONTEXTS = ("The", ",", " and", " of", " very", " 1")
SIZES, N, BATCH = (4, 16, 64), 64, 256
CLOSURE_TOL, CV_MAX, RESID_MAX, SIZE_RATIO, SPREAD_RATIO = 1e-3, 0.25, 0.30, 1.5, 2.0
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_gain_uniform_within_every_class": "cv <= 0.25 x 42", "pred_c_scalar_reading_holds_every_class": "<= 0.30 x 42",
               "pred_d_uniformity_stable_with_size": "CV(64) <= 1.5 CV(16) x 7", "pred_e_gain_is_set_by_the_context": ">= 2.0"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    ctx_ids = [L._single(c) for c in CONTEXTS]
    targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "contexts": list(CONTEXTS), "context_ids": ctx_ids, "class_sizes": {k: len(v) for k, v in cls.items()}, "sizes": SIZES, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "cv_max": CV_MAX, "resid_max": RESID_MAX, "size_ratio": SIZE_RATIO, "spread_ratio": SPREAD_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    tab = {}
    for s in range(0, len(targets), BATCH):
        ids = torch.tensor(targets[s:s + BATCH], device="cuda").unsqueeze(1); c = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
        for k, v in c.items(): tab.setdefault(k, []).append(v)
    tab = {k: torch.cat(v) for k, v in tab.items()}; tindex = {t: i for i, t in enumerate(targets)}
    pairs = [(ci, t) for ci in ctx_ids for t in targets]; ctx = {}
    for s in range(0, len(pairs), BATCH):
        chunk = pairs[s:s + BATCH]; toks = torch.tensor(chunk, device="cuda"); c = v289.capture(backend, toks, torch.ones(len(chunk), dtype=torch.long, device="cuda")); forwards += 1
        for k, v in c.items(): ctx.setdefault(k, []).append(v)
    ctx = {k: torch.cat(v) for k, v in ctx.items()}
    D = ctx["x1"].shape[-1]; ti = torch.tensor([tindex[t] for _, t in pairs])
    W = ctx["mlp1"]; T = tab["mlp1"][ti]; n_ctx = F.rms_norm(ctx["x1"], (D,)); n_tab = F.rms_norm(tab["x1"][ti], (D,)); c = n_ctx - n_tab
    Lt, Rt, Lc, Rc = n_tab @ Lw.T, n_tab @ Rw.T, c @ Lw.T, c @ Rw.T
    cross = (Lt * Rc + Lc * Rt) @ Dw.T; only = (Lc * Rc) @ Dw.T; change = W - T
    closure = float((((cross + only) - change).norm(dim=1) / change.norm(dim=1).clamp_min(1e-6)).max())
    cross_share = cross.norm(dim=1) / (cross.norm(dim=1) + only.norm(dim=1))
    alpha = (W * T).sum(1) / (T * T).sum(1); resid = 1 - ((W * T).sum(1) ** 2) / ((W * W).sum(1) * (T * T).sum(1))
    pos = {p: i for i, p in enumerate(pairs)}
    table = {}
    for cname, ci in zip(CONTEXTS, ctx_ids):
        for k, toks in cls.items():
            for n in SIZES:
                idx = torch.tensor([pos[(ci, t)] for t in toks[:n]]); a = alpha[idx]
                table[f"{cname}|{k}|{n}"] = {"alpha_median": float(a.median()), "alpha_std": float(a.std()), "alpha_cv": float(a.std() / a.mean().abs().clamp_min(1e-6)), "resid_median": float(resid[idx].median()), "cross_share_median": float(cross_share[idx].median()),
                                             "write_over_table_norm": float((W[idx].norm(dim=1) / T[idx].norm(dim=1)).median())}
    cv64 = {key: v["alpha_cv"] for key, v in table.items() if key.endswith("|64")}; res64 = {key: v["resid_median"] for key, v in table.items() if key.endswith("|64")}
    by_class_cv = {k: {n: sum(table[f"{c_}|{k}|{n}"]["alpha_cv"] for c_ in CONTEXTS) / len(CONTEXTS) for n in SIZES} for k in cls}
    ctx_medians = {c_: float(torch.tensor([table[f"{c_}|{k}|64"]["alpha_median"] for k in cls]).median()) for c_ in CONTEXTS}
    spread_across = float(torch.tensor(list(ctx_medians.values())).std()); within = float(torch.tensor([v["alpha_std"] for key, v in table.items() if key.endswith("|64")]).median())
    report = {"closure_max": closure, "alpha_cv_max_64": max(cv64.values()), "alpha_cv_argmax_64": max(cv64, key=cv64.get), "resid_max_64": max(res64.values()), "resid_argmax_64": max(res64, key=res64.get), "context_alpha_medians": ctx_medians,
              "class_alpha_medians_pooled": {k: float(torch.tensor([table[f"{c_}|{k}|64"]["alpha_median"] for c_ in CONTEXTS]).median()) for k in cls}, "by_class_cv_by_size": by_class_cv, "spread_across_contexts": spread_across, "within_std_median": within,
              "spread_ratio": spread_across / max(within, 1e-6), "cross_share_median_all": float(cross_share.median()), "resid_median_all": float(resid.median())}
    print(json.dumps({k: v for k, v in report.items() if k != "by_class_cv_by_size"}, indent=1, default=float))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_gain_uniform_within_every_class": max(cv64.values()) <= CV_MAX, "pred_c_scalar_reading_holds_every_class": max(res64.values()) <= RESID_MAX,
                   "pred_d_uniformity_stable_with_size": all(v[64] <= SIZE_RATIO * v[16] for v in by_class_cv.values()), "pred_e_gain_is_set_by_the_context": report["spread_ratio"] >= SPREAD_RATIO}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_context_gain_scaling_result_v290", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "table": table, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
