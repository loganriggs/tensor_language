#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pr_monotone_160m pred_b_final_vs_step1000 pred_c_bilin_at_most_step4000 pred_d_pr_tracks_recovery pred_e_gpt2_at_least_pythia
"""Census of the read-side QK spectrum across the ladder models (v775), WEIGHTS-ONLY and INPUT-WEIGHTED (smoke test: the weights-only bilinear forms are nearly flat — GPT-2 median PR 58.7 of 64, Pythia-70m 52.1, OPT-125m 53.9 — while the input-weighted ones are 8.6, 7.4, 8.8: the low rank the fits exploit lives in the activation covariance, not in the weights; the input-weighted spectrum Sigma^1/2 Wq^T Wk Sigma^1/2, Sigma = the layer's attention-input second moment over 64 fit rows, is the measure the predictions are about). Mechanism question behind v760-v774: as training proceeds
and the kernel + low-rank program recovers less (Pythia-160m rank-32 recovery 1.009 -> 0.932 from 2B to 300B tokens), do the heads' QK bilinear
forms Wq^T Wk (D x D, rank <= head dim) actually spread over more directions? Measure per head: participation ratio PR = (sum s)^2 / sum s^2 of
the singular values of Wq^T Wk (weights only) and of Sigma^1/2 Wq^T Wk Sigma^1/2 (input-weighted; rotary/QK-norm/biases ignored; for the two bilinear-attention forms of bilin18, both forms counted as
heads), plus the fraction of spectral mass in the top 8 and top 32 singular values. Models: Pythia-160m steps 1000/4000/16000/64000/final,
Pythia-70m/410m/1b final, Pythia-160m-deduped, OPT-125m, SmolLM-135M, GPT-2 small, bilin18, the softmax twin. Loaded on GPU one at a time; 4 capture forwards (64 rows) per model for Sigma.
Cross-check: the median PR per model against that model's rank-32 recovery from the ladder results (v760-v774, v763-v765).
PREDICTIONS (scored as written; failures preserved)
    pred_a_pr_monotone_160m       median input-weighted PR increases monotonically across the five Pythia-160m checkpoints. Prior: likely
    pred_b_final_vs_step1000      Pythia-160m final median input-weighted PR >= 1.5 x step-1000's. Prior: unsure
    pred_c_bilin_at_most_step4000 bilin18's median input-weighted PR (over its 2 x 162 forms) <= Pythia-160m step-4000's. Prior: unsure
    pred_d_pr_tracks_recovery     Spearman between median input-weighted PR and rank-32 recovery across the models with both numbers <= -0.6. Prior: likely
    pred_e_gpt2_at_least_pythia   GPT-2 small median input-weighted PR >= Pythia-160m final's. Prior: unsure
PRICE (registered maximum): 17 models x 4 capture forwards (64 rows, batch 16) = 68 forwards, 0 backwards, 0 fits; ~4000 SVDs of D x D matrices. Bar <= 80.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, sys, time
import torch
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/qk_rank_census_v775_result.json"
CANDIDATE_ID = "attention.qk_rank_census_v775"
PREDICTIONS = {"pred_a_pr_monotone_160m": "monotone over 5 checkpoints", "pred_b_final_vs_step1000": ">= 1.5x", "pred_c_bilin_at_most_step4000": "<=", "pred_d_pr_tracks_recovery": "spearman <= -0.6", "pred_e_gpt2_at_least_pythia": ">="}
MODELS = [("pythia160m_step1000", "pythia", "EleutherAI/pythia-160m", "step1000", "pythia160m_step1000_v767"), ("pythia160m_step4000", "pythia", "EleutherAI/pythia-160m", "step4000", "pythia160m_step4000_v766"),
          ("pythia160m_step16000", "pythia", "EleutherAI/pythia-160m", "step16000", "pythia160m_step16000_v768"), ("pythia160m_step64000", "pythia", "EleutherAI/pythia-160m", "step64000", "pythia160m_step64000_v769"),
          ("pythia160m", "pythia", "EleutherAI/pythia-160m", None, "pythia160m_v760"), ("pythia70m", "pythia", "EleutherAI/pythia-70m", None, "pythia70m_v761"), ("pythia410m", "pythia", "EleutherAI/pythia-410m", None, "pythia410m_v762"),
          ("pythia1b", "pythia", "EleutherAI/pythia-1b", None, "pythia1b_v773"), ("pythia160m_deduped", "pythia", "EleutherAI/pythia-160m-deduped", None, "pythia160m_deduped_v771"),
          ("pythia410m_step4000", "pythia", "EleutherAI/pythia-410m", "step4000", "pythia410m_step4000_v770"), ("pythia70m_step4000", "pythia", "EleutherAI/pythia-70m", "step4000", "pythia70m_step4000_v772"), ("pythia1b_step4000", "pythia", "EleutherAI/pythia-1b", "step4000", "pythia1b_step4000_v774"),
          ("opt125m", "hf", "facebook/opt-125m", None, "opt125m_v763"), ("smollm135m", "hf", "HuggingFaceTB/SmolLM-135M", None, "smollm135m_v764"), ("gpt2", "gpt2", "gpt2", None, "gpt2_centred_v765"),
          ("bilin18", "bilin18", None, None, None), ("softmax_twin", "softmax", None, None, None)]
RECOVERY_FIXED = {"bilin18": 0.982, "softmax_twin": 0.974}     # v740 / v752 mixed-rank programs (ranks 16/64), from the softmax-replication note
PR_RATIO, SPEARMAN_MAX = 1.5, -0.6
FORWARDS_MAX, EBATCH = 80, 16
ROWS = {"pythia": ROOT / ".rowcache/pythia_fineweb_n480_skip80.pt", "gpt2": ROOT / ".rowcache/fineweb_n480_skip80.pt", "bilin18": ROOT / ".rowcache/fineweb_n480_skip80.pt", "softmax": ROOT / ".rowcache/fineweb_n480_skip80.pt", "facebook/opt-125m": ROOT / ".rowcache/opt_fineweb_n480_skip80.pt", "HuggingFaceTB/SmolLM-135M": ROOT / ".rowcache/smollm_fineweb_n480_skip80.pt"}


def _sqrtm(S):
    e, U = torch.linalg.eigh(S)
    return (U * e.clamp_min(0).sqrt()) @ U.T


def spectrum_stats(M):
    s = torch.linalg.svdvals(M.float()).double(); s = s[s > 0]     # float32 SVD: FP64 on this GPU stalled the 1b model (2048^2 x 256 SVDs)
    pr = float(s.sum() ** 2 / (s ** 2).sum()); tot = float(s.sum())
    return {"pr": pr, "top8": float(s[:8].sum() / tot), "top32": float(s[:32].sum() / tot), "rank": int(s.numel())}


def forms_pythia(model):
    import pythia_backend as PB
    L, H, D, hd, rot = PB.geometry(model)
    for l in range(L):
        for h in range(H):
            Wq, bq, Wk, bk = PB.head_qk(model, l, h)[:4]
            yield f"{l}.{h}", l, Wq.T @ Wk


def forms_hf(model):
    import hf_backend as HB
    L, H, D, hd, kv = HB.geometry(model)
    for l in range(L):
        for h in range(H):
            Wq, bq, Wk, bk, scale = HB.head_qk(model, l, h)
            yield f"{l}.{h}", l, Wq.T @ Wk


def forms_gpt2(model):
    import gpt2_backend as GB
    L, H, D, hd, _ = GB.geometry(model)
    for l in range(L):
        for h in range(H):
            Wq, bq, Wk, bk, scale = GB.head_qk(model, l, h)
            yield f"{l}.{h}", l, Wq.T @ Wk


def forms_tt(model, bilinear):
    H = model.config.n_head; D = model.config.n_embd; hd = D // H
    for l, blk in enumerate(model.transformer.h):
        at = blk.attn
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            yield f"{l}.{h}", l, at.c_q.weight[sl].detach().float().T @ at.c_k.weight[sl].detach().float()
            if bilinear:
                yield f"{l}.{h}b", l, at.c_q2.weight[sl].detach().float().T @ at.c_k2.weight[sl].detach().float()


def load_and_forms(kind, repo, rev):
    """Returns (model, forms iterator, list of per-layer attention modules, forward(idx_cpu) callable)."""
    if kind == "pythia":
        import pythia_backend as PB; m = PB.load(repo, "cuda", revision=rev)
        return m, forms_pythia(m), [l.attention for l in m.gpt_neox.layers], lambda idx: m(input_ids=idx.cuda())
    if kind == "hf":
        import hf_backend as HB; m = HB.load(repo, "cuda")
        return m, forms_hf(m), [HB.attn_of(l, m) for l in HB.layers(m)], lambda idx: m(input_ids=idx.cuda())
    if kind == "gpt2":
        import gpt2_backend as GB; m = GB.load(repo, "cuda")
        return m, forms_gpt2(m), [b.attn for b in m.transformer.h], lambda idx: m(input_ids=idx.cuda())
    if kind == "bilin18":
        import circuit_fast_screen_producer as producer; m = producer.Bilin18TorchBackend.load("cuda").model
        return m, forms_tt(m, True), [b.attn for b in m.transformer.h], lambda idx: m(idx[:, :-1].cuda(), idx[:, 1:].cuda())
    if kind == "softmax":
        import softmax_backend as SB; m, TT = SB.load("cuda")
        return m, forms_tt(m, False), [b.attn for b in m.transformer.h], lambda idx: m(idx[:, :-1].cuda(), idx[:, 1:].cuda())
    raise ValueError(kind)


def second_moments(attn_modules, forward, rows, batch):
    """Per-layer E[n n^T] of the attention input (double), from the first 64 rows in 2 forwards."""
    sig = {}; cnt = [0]
    def hook(l):
        def f(mod, a, kw):
            n = (a[0] if a else kw["hidden_states"]).detach().reshape(-1, (a[0] if a else kw["hidden_states"]).shape[-1]).double()
            sig[l] = sig.get(l, 0) + n.T @ n
            if l == 0: cnt[0] += n.shape[0]
        return f
    hs = [mod.register_forward_pre_hook(hook(l), with_kwargs=True) for l, mod in enumerate(attn_modules)]
    fw = 0
    with torch.no_grad():
        for s in range(0, 64, batch):
            forward(rows[s:s + batch].long()); fw += 1
    for h in hs: h.remove()
    return {l: v / cnt[0] for l, v in sig.items()}, fw


def spearman(x, y):
    import statistics
    rx = {v: i for i, v in enumerate(sorted(x))}; ry = {v: i for i, v in enumerate(sorted(y))}
    a = [rx[v] for v in x]; b = [ry[v] for v in y]; n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b)); den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "models": [m[0] for m in MODELS], "bars": {"pr_ratio": PR_RATIO, "spearman_max": SPEARMAN_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    per_model = {}; recovery = {}
    forwards = 0
    for name, kind, repo, rev, tag in MODELS:
        heads = {}
        model, forms, attn_modules, forward = load_and_forms(kind, repo, rev)
        rows = torch.load(ROWS.get(repo, ROWS.get(kind)), map_location="cpu")
        sig, fw = second_moments(attn_modules, forward, rows, EBATCH); forwards += fw
        half = {l: _sqrtm(v).float() for l, v in sig.items()}
        with torch.no_grad():
            for hid, l, M in forms:
                M = M.cuda(); w = spectrum_stats(M); Mw = half[l] @ M @ half[l]; v = spectrum_stats(Mw)
                heads[hid] = {"pr": w["pr"], "top8": w["top8"], "top32": w["top32"], "pr_w": v["pr"], "top8_w": v["top8"], "top32_w": v["top32"], "rank": w["rank"]}
        prs = sorted(v["pr"] for v in heads.values()); med = prs[len(prs) // 2]
        prw = sorted(v["pr_w"] for v in heads.values()); medw = prw[len(prw) // 2]
        t8 = sorted(v["top8_w"] for v in heads.values()); t32 = sorted(v["top32_w"] for v in heads.values())
        per_model[name] = {"n_forms": len(heads), "median_pr": med, "median_pr_w": medw, "mean_pr_w": sum(prw) / len(prw), "median_top8_w": t8[len(t8) // 2], "median_top32_w": t32[len(t32) // 2], "heads": heads}
        rec = None
        if tag is not None:
            p = ROOT / f"circuits/followups/{tag}_result.json"
            if p.exists():
                rec = json.load(open(p))["report"]["arms"]["32"]["recovery"]
        if name in RECOVERY_FIXED:
            rec = RECOVERY_FIXED[name]
        recovery[name] = rec
        print(f"{name:24s} forms {len(heads):4d} | median PR weights {med:6.2f} input-weighted {medw:6.2f} (mean {per_model[name]['mean_pr_w']:6.2f}) | median top-8 {per_model[name]['median_top8_w']:.3f} top-32 {per_model[name]['median_top32_w']:.3f} | rank-32 recovery {rec}")
        del model, forms, attn_modules, forward, sig, half; import gc; gc.collect(); torch.cuda.empty_cache()
    curve = ["pythia160m_step1000", "pythia160m_step4000", "pythia160m_step16000", "pythia160m_step64000", "pythia160m"]
    cm = [per_model[c]["median_pr_w"] for c in curve]
    both = [n for n in per_model if recovery.get(n) is not None]
    rho = spearman([per_model[n]["median_pr_w"] for n in both], [recovery[n] for n in both])
    print(f"160m curve median input-weighted PR: {' -> '.join(f'{v:.2f}' for v in cm)} | spearman(median PR, rank-32 recovery) over {len(both)} models: {rho:+.3f}")
    predictions = {"pred_a_pr_monotone_160m": all(cm[i + 1] > cm[i] for i in range(4)), "pred_b_final_vs_step1000": cm[-1] >= PR_RATIO * cm[0],
                   "pred_c_bilin_at_most_step4000": per_model["bilin18"]["median_pr_w"] <= per_model["pythia160m_step4000"]["median_pr_w"],
                   "pred_d_pr_tracks_recovery": rho <= SPEARMAN_MAX, "pred_e_gpt2_at_least_pythia": per_model["gpt2"]["median_pr_w"] >= per_model["pythia160m"]["median_pr_w"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "qk_rank_census_result_v775", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"per_model": per_model, "recovery_rank32": recovery, "curve_160m_median_pr": cm, "spearman_pr_recovery": rho, "n_models_with_recovery": len(both)},
                               "predictions": predictions, "forwards": forwards, "backwards": 0, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
