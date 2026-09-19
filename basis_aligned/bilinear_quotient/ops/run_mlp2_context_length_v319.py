#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_mlp2_gain_falls_with_length pred_c_mlp2_cross_negative pred_d_mlp2_lookup_term_weaker_than_mlp1 pred_e_mlp2_gain_floors
"""MLP 2 vs context length (v319; v301's instrument at layer 2). v318 (text): MLP 2 obeys MLP 1's cancellation law from a weaker lookup (gamma^2 0.40,
cross -0.43, alpha_2 0.20). Controlled version: phrase A, lengths 1 / 8 / 64, 7 classes x 32 targets; MLP 2's write vs its single-token table entry
(block-2 output for the token alone): alpha_2, cosine, and the exact three-term expansion (gamma^2 T_2 + cross + context^2) in MLP 2's normalised input.
PREDICTIONS (scored as written; failures preserved; priors from v291 / v318)
    pred_a_pair_closure                    gamma^2 T_2 + cross + context^2 = W_2 within relative 2e-2 on every row (float32 over 4,608 units; v301 reached 1.1%)
    pred_b_mlp2_gain_falls_with_length     median alpha_2 strictly decreases 1 -> 8 -> 64
    pred_c_mlp2_cross_negative             the cross term projects negatively on T_2 for >= 0.90 of rows at every length
    pred_d_mlp2_lookup_term_weaker_than_mlp1  gamma^2 median <= 0.70 at every length (MLP 1's was 0.86-0.93)
    pred_e_mlp2_gain_floors                alpha_2 (64) >= 0.10 (a floor, as MLP 1's 0.29)
PRICE (registered maximum): 1 table batch + 3 length batches = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
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
OUT = ROOT / "circuits/followups/mlp2_context_length_v319_result.json"
CANDIDATE_ID = "mlp2.token_table.context_length_v319"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, NEG_MIN, GAMMA_MAX, FLOOR_MIN = 2e-2, 0.90, 0.70, 0.10
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_pair_closure": "<= 2e-2", "pred_b_mlp2_gain_falls_with_length": "strict decrease", "pred_c_mlp2_cross_negative": ">= 0.90 x 3", "pred_d_mlp2_lookup_term_weaker_than_mlp1": "<= 0.70 x 3", "pred_e_mlp2_gain_floors": ">= 0.10"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "neg_min": NEG_MIN, "gamma_max": GAMMA_MAX, "floor_min": FLOOR_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp2 = model.transformer.h[2].mlp; Lw, Rw, Dw = mlp2.Left.weight.detach().float().cpu(), mlp2.Right.weight.detach().float().cpu(), mlp2.Down.weight.detach().float().cpu()
    def capture2(tokens, pos):
        idx = torch.arange(tokens.shape[0]); out = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1, 2):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if l == 2: out["x2"] = x[idx, pos].float().cpu(); out["mlp2"] = m[idx, pos].float().cpu()
                x = x + m
        return out
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = capture2(ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
    T = tab["mlp2"]; n_tab = F.rms_norm(tab["x2"], (T.shape[-1],))
    closure, per_k = 0.0, {}
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); c = capture2(toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp2"]; n = F.rms_norm(c["x2"], (T.shape[-1],)); gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
        Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T
        quad = (Lt2 * Rt2) @ Dw.T; cross = (Lt2 * Rc + Lc * Rt2) @ Dw.T; only = (Lc * Rc) @ Dw.T
        closure = max(closure, float((((quad + cross + only) - W).norm(dim=1) / W.norm(dim=1)).max()))
        pT = lambda A: (A * T).sum(1) / (T * T).sum(1); alpha, g2, pc, po = pT(W), pT(quad), pT(cross), pT(only)
        per_k[k] = {"alpha2_median": float(alpha.median()), "cos_median": float(((W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))).median()), "gamma2_median": float(g2.median()), "cross_proj_median": float(pc.median()), "cross_negative_fraction": float((pc < 0).float().mean()), "only_proj_median": float(po.median()), "c_norm_median": float(cc.norm(dim=1).median())}
    report = {"closure_max": closure, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps(report, indent=1))
    a_ = [per_k[k]["alpha2_median"] for k in LENGTHS]
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_mlp2_gain_falls_with_length": all(a_[i + 1] < a_[i] for i in range(len(a_) - 1)), "pred_c_mlp2_cross_negative": all(v["cross_negative_fraction"] >= NEG_MIN for v in per_k.values()),
                   "pred_d_mlp2_lookup_term_weaker_than_mlp1": all(v["gamma2_median"] <= GAMMA_MAX for v in per_k.values()), "pred_e_mlp2_gain_floors": per_k[64]["alpha2_median"] >= FLOOR_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_context_length_result_v319", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
