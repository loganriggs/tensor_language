#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_gain_falls_with_context_length pred_c_gain_tracks_self_share pred_d_gain_uniform_across_classes_at_every_length pred_e_direction_kept_at_every_length
"""MLP 1: is the context gain the attention self-share? Context LENGTH 1 / 2 / 4 / 8 (v291). v290: MLP 1's write after one context token is its
single-token table entry times alpha ~ 1/2 for EVERY context token and class (42 cells; spread across contexts 0.92x within). Content does not
set the gain, so position may: squared-bilinear attention (no softmax) sums over positions, and the target's own value share in x1 falls as the
context grows. Test: k = 1, 2, 4, 8 filler tokens (", and of the very , and of" prefix, truncated to k) before each of 7 classes x 32 targets; per k
measure alpha (median over targets, pooled over classes), the target's own-key weight in the bilinear attention pattern (q.k)(q2.k2)/D^2 at the target (absolute value) as a
share of the row's total absolute weight (self-share s_k, mean over the 9 heads and blocks 0 + 1; exact, read from the pattern; the pattern is causal, unnormalised), the residual 1 - cos^2, and alpha's CV across classes.
PREDICTIONS (scored as written; failures preserved; priors from v290)
    pred_a_pair_closure                            Down[cross + context-only] = W_ctx - W_table within relative 1e-3 on every row
    pred_b_gain_falls_with_context_length          alpha median strictly decreases 1 -> 2 -> 4 -> 8 filler tokens
    pred_c_gain_tracks_self_share                  alpha(k) / alpha(1) within 0.25 of s_k / s_1 for k = 2, 4, 8 (the gain scales as the self-share). Prior: unsure.
    pred_d_gain_uniform_across_classes_at_every_length  alpha's CV across the 7 class medians <= 0.25 at every k
    pred_e_direction_kept_at_every_length          median cosine(W_ctx, W_table) >= 0.80 at every k (the table direction survives 8 tokens of context)
PRICE (registered maximum): 4 lengths x 224 rows = 896 rows / 256 = 4 forwards + 224 tokens = 1 forward = 5; 0 backwards; 0 fits. Bar <= 8.
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
OUT = ROOT / "circuits/followups/mlp1_context_length_gain_v291_result.json"
CANDIDATE_ID = "mlp1.token_table.context_length_gain_v291"
FILLER = (",", " and", " of", " the", " very", ",", " and", " of")
LENGTHS, N, BATCH = (1, 2, 4, 8), 32, 256
CLOSURE_TOL, TRACK_TOL, CV_MAX, COS_MIN = 1e-3, 0.25, 0.25, 0.80
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_gain_falls_with_context_length": "strict decrease", "pred_c_gain_tracks_self_share": "within 0.25 x 3", "pred_d_gain_uniform_across_classes_at_every_length": "cv <= 0.25 x 4", "pred_e_direction_kept_at_every_length": ">= 0.80 x 4"}


def capture_self(backend, tokens, pos):
    """v289.capture plus the self weight of attention 0 + 1 at `pos`: the row-normalised squared pattern's own-key entry, mean over heads and the two blocks."""
    torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h; idx = torch.arange(tokens.shape[0]); out = {}; store = {}
    def wrap(l):
        orig = blocks[l].attn.squared_attention
        def f(q, k, v, q2, k2):
            B, T, H, D = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0)
            own = pat[idx, :, pos, pos].float().abs(); store[l] = (own / pat[idx, :, pos, :].float().abs().sum(-1).clamp_min(1e-9)).mean(1).cpu(); return orig(q, k, v, q2, k2)
        return f
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; out["x0"] = x0[idx, pos].float().cpu()
        for l in (0, 1):
            block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; orig = block.attn.squared_attention; block.attn.squared_attention = wrap(l)
            try: attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            finally: block.attn.squared_attention = orig
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            out[f"attn{l}"] = attention[idx, pos].float().cpu(); m = block.mlp(xin); out[f"mlp{l}"] = m[idx, pos].float().cpu()
            if l == 1: out["x1"] = x[idx, pos].float().cpu()
            x = x + m
    out["self_share"] = (store[0] + store[1]) / 2
    return out


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in FILLER]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "filler": list(FILLER), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "track_tol": TRACK_TOL, "cv_max": CV_MAX, "cos_min": COS_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = capture_self(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    ti = torch.tensor([tindex[t] for t in targets]); T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    closure, per_k = 0.0, {}
    for k in LENGTHS:
        toks = torch.tensor([fill[:k] + [t] for t in targets], device="cuda"); c = capture_self(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp1"]; cc = F.rms_norm(c["x1"], (T.shape[-1],)) - n_tab; Lc, Rc = cc @ Lw.T, cc @ Rw.T
        cross = (Lt * Rc + Lc * Rt) @ Dw.T; only = (Lc * Rc) @ Dw.T; change = W - T
        closure = max(closure, float((((cross + only) - change).norm(dim=1) / change.norm(dim=1).clamp_min(1e-6)).max()))
        alpha = (W * T).sum(1) / (T * T).sum(1); cos = (W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))
        cls_med = {name: float(alpha[torch.tensor([tindex[t] for t in v])].median()) for name, v in cls.items()}; cm = torch.tensor(list(cls_med.values()))
        per_k[k] = {"alpha_median": float(alpha.median()), "alpha_by_class": cls_med, "alpha_class_cv": float(cm.std() / cm.mean()), "self_share_median": float(c["self_share"].median()), "cos_median": float(cos.median()),
                    "resid_median": float((1 - cos ** 2).median()), "cross_share_median": float((cross.norm(dim=1) / (cross.norm(dim=1) + only.norm(dim=1))).median()), "write_over_table_norm": float((W.norm(dim=1) / T.norm(dim=1)).median())}
    a = [per_k[k]["alpha_median"] for k in LENGTHS]; s_ = [per_k[k]["self_share_median"] for k in LENGTHS]
    track = {k: abs(per_k[k]["alpha_median"] / a[0] - per_k[k]["self_share_median"] / s_[0]) for k in LENGTHS[1:]}
    report = {"closure_max": closure, "alpha_by_length": dict(zip(map(str, LENGTHS), a)), "self_share_by_length": dict(zip(map(str, LENGTHS), s_)), "self_share_single_token": float(tab["self_share"].median()), "track_gap": {str(k): v for k, v in track.items()},
              "cos_by_length": {str(k): per_k[k]["cos_median"] for k in LENGTHS}, "class_cv_by_length": {str(k): per_k[k]["alpha_class_cv"] for k in LENGTHS}, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps({k: v for k, v in report.items() if k != "per_k"}, indent=1, default=float))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_gain_falls_with_context_length": all(a[i + 1] < a[i] for i in range(len(a) - 1)), "pred_c_gain_tracks_self_share": all(v <= TRACK_TOL for v in track.values()),
                   "pred_d_gain_uniform_across_classes_at_every_length": all(per_k[k]["alpha_class_cv"] <= CV_MAX for k in LENGTHS), "pred_e_direction_kept_at_every_length": all(per_k[k]["cos_median"] >= COS_MIN for k in LENGTHS)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_context_length_gain_result_v291", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
