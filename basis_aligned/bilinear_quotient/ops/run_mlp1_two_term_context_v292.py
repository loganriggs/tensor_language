#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_remainder_is_the_context_write pred_c_two_term_model_explains_most pred_d_gain_floors pred_e_context_coefficient_rises
"""MLP 1: what is the remainder? Two-term model write = alpha x table(token) + beta x table(context), context length 1..64 (v292). v291: the
gain on the token's table entry tracks the attention own-key share (0.555 -> 0.335 for 1 -> 8 filler tokens vs share 0.525 -> 0.210) but flattens
above it at 8 tokens while the direction drifts (cos 0.77). Attention 0/1 mix the target's own value with the context's values, so the remainder
should be MLP 1's write on the context alone: the write at the LAST filler position of the filler sequence run by itself, W_ctx(k) (captured; no
target). Per k = 1, 2, 4, 8, 16, 32, 64 (filler = a cyclic ", and of the very" phrase) x 7 classes x 32 targets: least-squares-free projection
coefficients alpha, beta from the 2 x 2 Gram system (an exact projection onto a 2-d span, not a fitted model), explained energy of the two-term
span, cosine of the remainder (W - alpha T) with W_ctx(k), alpha and beta by k.
PREDICTIONS (scored as written; failures preserved; priors from v291)
    pred_a_pair_closure                 Down[cross + context-only] = W - T within relative 1e-3 on every row
    pred_b_remainder_is_the_context_write  median cosine(W - alpha T, W_ctx(k)) >= 0.50 at k >= 8. Prior: unsure.
    pred_c_two_term_model_explains_most median explained energy of span{T, W_ctx(k)} >= 0.85 at every k
    pred_d_gain_floors                  alpha(64) >= 0.20 (the token keeps a floor share; no decay to zero over the exponential range)
    pred_e_context_coefficient_rises    beta(k) median increases monotonically 1 -> 64
PRICE (registered maximum): 7 lengths x 224 rows = 1568 rows / 256 = 7 forwards + 224 tokens = 1 + 7 filler runs (1 forward, padded batch) = 9; 0 backwards; 0 fits. Bar <= 12.
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
OUT = ROOT / "circuits/followups/mlp1_two_term_context_v292_result.json"
CANDIDATE_ID = "mlp1.token_table.two_term_context_v292"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 2, 4, 8, 16, 32, 64), 32, 256
CLOSURE_TOL, REM_COS_MIN, EXPL_MIN, ALPHA_FLOOR = 1e-3, 0.50, 0.85, 0.20
FORWARDS_MAX = 12
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_remainder_is_the_context_write": ">= 0.50 at k >= 8", "pred_c_two_term_model_explains_most": ">= 0.85 x 7", "pred_d_gain_floors": "alpha(64) >= 0.20", "pred_e_context_coefficient_rises": "monotone"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "rem_cos_min": REM_COS_MIN, "expl_min": EXPL_MIN, "alpha_floor": ALPHA_FLOOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    # the context alone: one padded batch of the 7 filler sequences (right-padded; the write at the last real position is causal so padding is inert)
    pad = torch.tensor([filler(k) + [fill[0]] * (max(LENGTHS) - k) for k in LENGTHS], device="cuda"); cw = v289.capture(backend, pad, torch.tensor([k - 1 for k in LENGTHS], device="cuda")); forwards += 1
    Wctx = {k: cw["mlp1"][i] for i, k in enumerate(LENGTHS)}
    closure, per_k = 0.0, {}
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); c = v289.capture(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp1"]; cc = F.rms_norm(c["x1"], (T.shape[-1],)) - n_tab; Lc, Rc = cc @ Lw.T, cc @ Rw.T
        cross = (Lt * Rc + Lc * Rt) @ Dw.T; only = (Lc * Rc) @ Dw.T; change = W - T
        closure = max(closure, float((((cross + only) - change).norm(dim=1) / change.norm(dim=1).clamp_min(1e-6)).max()))
        alpha1 = (W * T).sum(1) / (T * T).sum(1); rem = W - alpha1[:, None] * T; C = Wctx[k].unsqueeze(0).expand_as(W)
        rem_cos = (rem * C).sum(1) / (rem.norm(dim=1) * C.norm(dim=1))
        g11, g12, g22 = (T * T).sum(1), (T * C).sum(1), (C * C).sum(1); b1, b2 = (W * T).sum(1), (W * C).sum(1); det = g11 * g22 - g12 ** 2
        al = (b1 * g22 - b2 * g12) / det; be = (g11 * b2 - g12 * b1) / det; proj = al[:, None] * T + be[:, None] * C
        expl = (proj * proj).sum(1) / (W * W).sum(1)
        per_k[k] = {"alpha_single": float(alpha1.median()), "alpha": float(al.median()), "beta": float(be.median()), "explained_median": float(expl.median()), "explained_p10": float(expl.quantile(0.1)), "rem_cos_median": float(rem_cos.median()),
                    "cos_table_median": float(((W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))).median()), "cos_context_median": float(((W * C).sum(1) / (W.norm(dim=1) * C.norm(dim=1))).median()), "context_write_norm": float(Wctx[k].norm()),
                    "write_norm_median": float(W.norm(dim=1).median()), "alpha_by_class": {name: float(al[torch.tensor([tindex[t] for t in v])].median()) for name, v in cls.items()}}
    ks = list(LENGTHS); be = [per_k[k]["beta"] for k in ks]
    report = {"closure_max": closure, "alpha_by_length": {str(k): per_k[k]["alpha"] for k in ks}, "beta_by_length": {str(k): per_k[k]["beta"] for k in ks}, "explained_by_length": {str(k): per_k[k]["explained_median"] for k in ks},
              "rem_cos_by_length": {str(k): per_k[k]["rem_cos_median"] for k in ks}, "cos_table_by_length": {str(k): per_k[k]["cos_table_median"] for k in ks}, "cos_context_by_length": {str(k): per_k[k]["cos_context_median"] for k in ks}, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps({k: v for k, v in report.items() if k != "per_k"}, indent=1, default=float))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_remainder_is_the_context_write": all(per_k[k]["rem_cos_median"] >= REM_COS_MIN for k in ks if k >= 8), "pred_c_two_term_model_explains_most": all(per_k[k]["explained_median"] >= EXPL_MIN for k in ks),
                   "pred_d_gain_floors": per_k[64]["alpha"] >= ALPHA_FLOOR, "pred_e_context_coefficient_rises": all(be[i + 1] > be[i] for i in range(len(be) - 1))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_two_term_context_result_v292", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
