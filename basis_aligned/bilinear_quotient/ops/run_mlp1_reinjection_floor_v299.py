#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_floor_is_the_reinjection pred_c_share_law_exact_without_reinjection pred_d_reinjection_share_of_x1_grows pred_e_natural_alpha_floor_matches
"""MLP 1: is the gain floor the embedding re-injection? (v299). v291 / v292 / v297: the gain alpha on MLP 1's token lookup tracks the token's own-key
attention share at short context but FLOORS (~0.29 at 16-64 filler tokens; ~0.23 from position 6 on text) while the share keeps falling (0.12-0.21).
Every block re-injects the token's own normalised embedding: live = lambda0 x + lambda1 x0. That path carries token identity into MLP 1's input
past attention entirely, so alpha should floor at the re-injection's share of x1. Test: lengths 1 / 8 / 64 (phrase A, 224 targets), native vs an
edit that zeroes lambda1 in blocks 0 and 1 for the fold (x0 still seeds x; only the re-injection term is removed; the table entries are recomputed
under the same edit so alpha is well defined). Measures: alpha native vs edited; own-key share s (unchanged by the edit except through q/k of block 1);
the re-injection term's share of x1 at the target (norm of lambda1 x0 summed over the two blocks over the norm of x1).
PREDICTIONS (scored as written; failures preserved; priors from v291 / v297)
    pred_a_pair_closure                       Down[cross + context-only] = W - T within relative 1e-3 on every row, native and edited
    pred_b_floor_is_the_reinjection           at 64 tokens, alpha(edited) <= alpha(native) - 0.10 (removing the re-injection removes the floor)
    pred_c_share_law_exact_without_reinjection  |alpha(edited) - s(edited)| <= 0.10 at every length (without the bypass, alpha IS the share)
    pred_d_reinjection_share_of_x1_grows      the re-injection's norm share of x1 at the target rises monotonically 1 -> 8 -> 64
    pred_e_natural_alpha_floor_matches        alpha(native, 64) is within 0.10 of the re-injection share of x1 at 64 plus s(64) x (1 - that share). Prior: unsure.
PRICE (registered maximum): 2 conditions x (1 table + 3 length batches) = 8 forwards; 0 backwards; 0 fits. Bar <= 10.
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
OUT = ROOT / "circuits/followups/mlp1_reinjection_floor_v299_result.json"
CANDIDATE_ID = "mlp1.token_table.reinjection_floor_v299"
FILLER = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, DROP_MIN, LAW_TOL, MATCH_TOL = 1e-3, 0.10, 0.10, 0.10
FORWARDS_MAX = 10
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_floor_is_the_reinjection": "drop >= 0.10 at 64", "pred_c_share_law_exact_without_reinjection": "<= 0.10 x 3", "pred_d_reinjection_share_of_x1_grows": "monotone", "pred_e_natural_alpha_floor_matches": "<= 0.10"}


def capture_self(backend, tokens, pos, reinject=True):
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
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; out["x0"] = x0[idx, pos].float().cpu(); rn = 0.0
        for l in (0, 1):
            block = blocks[l]; rein = (block.lambdas[1] * x0) if reinject else 0.0; live = block.lambdas[0] * x + rein; rn = rn + (rein[idx, pos].float().cpu() if reinject else 0.0); orig = block.attn.squared_attention; block.attn.squared_attention = wrap(l)
            try: attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            finally: block.attn.squared_attention = orig
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            out[f"attn{l}"] = attention[idx, pos].float().cpu(); m = block.mlp(xin); out[f"mlp{l}"] = m[idx, pos].float().cpu()
            if l == 1: out["x1"] = x[idx, pos].float().cpu()
            x = x + m
    out["self_share"] = (store[0] + store[1]) / 2; out["reinjection"] = rn if reinject else torch.zeros_like(out["x0"])
    return out


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in FILLER]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "filler": list(FILLER), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "drop_min": DROP_MIN, "law_tol": LAW_TOL, "match_tol": MATCH_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    closure, per = 0.0, {}
    for cond, rein in (("native", True), ("no_reinjection", False)):
        ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = capture_self(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda"), reinject=rein); forwards += 1
        T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
        for k in LENGTHS:
            toks = torch.tensor([fill[:0] + [fill[i % len(fill)] for i in range(k)] + [t] for t in targets], device="cuda"); c = capture_self(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda"), reinject=rein); forwards += 1
            W = c["mlp1"]; cc = F.rms_norm(c["x1"], (T.shape[-1],)) - n_tab; Lc, Rc = cc @ Lw.T, cc @ Rw.T
            cross = (Lt * Rc + Lc * Rt) @ Dw.T; only = (Lc * Rc) @ Dw.T; change = W - T
            closure = max(closure, float((((cross + only) - change).norm(dim=1) / change.norm(dim=1).clamp_min(1e-6)).max()))
            alpha = (W * T).sum(1) / (T * T).sum(1)
            per[f"{cond}|{k}"] = {"alpha_median": float(alpha.median()), "self_share_median": float(c["self_share"].median()), "gap_median": float((alpha - c["self_share"]).abs().median()),
                                  "reinjection_share_of_x1": float((c["reinjection"].norm(dim=1) / c["x1"].norm(dim=1)).median()) if rein else 0.0, "cos_table_median": float(((W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))).median())}
    nat, ed = {k: per[f"native|{k}"] for k in LENGTHS}, {k: per[f"no_reinjection|{k}"] for k in LENGTHS}
    r64 = nat[64]["reinjection_share_of_x1"]; predicted64 = r64 + nat[64]["self_share_median"] * (1 - r64)
    report = {"closure_max": closure, "per": per, "alpha_drop_at_64": nat[64]["alpha_median"] - ed[64]["alpha_median"], "predicted_alpha_64": predicted64, "reinjection_share_by_length": {str(k): nat[k]["reinjection_share_of_x1"] for k in LENGTHS}}
    print(json.dumps(report, indent=1))
    rs = [nat[k]["reinjection_share_of_x1"] for k in LENGTHS]
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_floor_is_the_reinjection": report["alpha_drop_at_64"] >= DROP_MIN, "pred_c_share_law_exact_without_reinjection": all(ed[k]["gap_median"] <= LAW_TOL for k in LENGTHS),
                   "pred_d_reinjection_share_of_x1_grows": all(rs[i + 1] > rs[i] for i in range(len(rs) - 1)), "pred_e_natural_alpha_floor_matches": abs(nat[64]["alpha_median"] - predicted64) <= MATCH_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_reinjection_floor_result_v299", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
