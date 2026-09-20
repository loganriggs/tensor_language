#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_context_out_beats_coefficient pred_c_context_in_beats_coefficient pred_d_context_beats_single_token pred_e_mean_matters
"""Embedding-forward folding, rung 8 (v616): WHY did the token metric lose under CE (v615)? Context-derived frames vs single-token frames vs weights.

v615: at equal rank, plain coefficient-metric subspaces of MLP-0 cost the same or less CE than the single-token token-metric subspaces (IN r=256:
0.095 vs 0.14 / 0.48; whitened oblique projections worst). Two hypotheses: (a) single-token statistics misrepresent contextual inputs (24% of
MLP-0's input is the previous token, v612) so table-derived frames transfer badly; (c) CE is governed by what downstream readers pick up, so no
input-statistics metric can win. Discriminator: frames from the CONTEXT statistics of MLP-0's actual normalised input and actual write on the
480 x 513 FIT rows (skip80; disjoint from the 192 skip7000 eval rows) — Euclidean PCA of mean-centred data, no whitening — installed as the same
pure projections as v615 and priced in CE at r in {32, 64, 128, 256}.
    OUT arms: coefficient (v611 frame, keep bias) | single_token (v611 frequency-centred frame, keep bias) | context_mean (context PCA of the
              write, keep the context mean write) | context_bias (context PCA frame, keep only the bias)
    IN arms:  coefficient (v611 frame, P P^T x) | single_token (Euclidean eigenframe of v610's frequency-weighted token covariance, mean kept)
              | context (context PCA of the normalised input, context mean kept)
Sign convention: CE ADDED above the native forward of the same run; lower is better. If context frames win big: (a). If they tie the coefficient
frames: (c) — the output metric must come from the readers (Codex's side), not from input statistics.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays               native CE on the 192 eval rows within 0.002 of v615's 3.13241 (instrument)
    pred_b_context_out_beats_coefficient OUT r=128: context_mean CE added <= 0.5 x coefficient, with coefficient >= 0.02. Prior: unsure — this is (a) vs (c)
    pred_c_context_in_beats_coefficient  IN r=128: context CE added <= 0.5 x coefficient, with coefficient >= 0.05. Prior: likely
    pred_d_context_beats_single_token    IN r=128: context CE added <= 0.5 x single_token (transfer of table statistics to contexts). Prior: unsure
    pred_e_mean_matters                  OUT r=128: context_mean <= context_bias (keeping the context mean of the write does not hurt). Prior: likely
PRICE (registered maximum): capture 480 fit rows x 512 tokens in 15 batches (15 forwards) + 29 configs (28 edits + native) x 6 eval batches = 174;
total 189 forwards; 0 backwards; 0 fits (PCA of captured statistics is an eigendecomposition, not a fit against the loss). Bar <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_context_frame_edit_v616_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_context_frame_edit_v616_frames.pt"
FRAMES = ROOT / "circuits/followups/embedding_forward_tucker_energy_v611_frames.pt"
COVS = ROOT / "circuits/followups/embedding_forward_degree_census_v610_tensors.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.context_frame_edit_v616"
FORWARDS_MAX = 200
BATCH = 32
RANKS = (32, 64, 128, 256)
REPLAY_TOL, RATIO, FLOOR_OUT, FLOOR_IN = 0.002, 0.5, 0.02, 0.05
PREDICTIONS = {"pred_a_native_replays": "+-0.002 of 3.13241", "pred_b_context_out_beats_coefficient": "<= 0.5x, coef >= 0.02", "pred_c_context_in_beats_coefficient": "<= 0.5x, coef >= 0.05",
               "pred_d_context_beats_single_token": "<= 0.5x", "pred_e_mean_matters": "context_mean <= context_bias"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "ranks": list(RANKS),
            "bars": {"replay_tol": REPLAY_TOL, "ratio": RATIO, "floor_out": FLOOR_OUT, "floor_in": FLOOR_IN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; dev = "cuda"; b0 = model.transformer.h[0]; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    v611 = torch.load(FRAMES, map_location=dev); v610 = torch.load(COVS, map_location=dev)
    with torch.no_grad():
        bias = b0.mlp.Down_bias.detach().float()
        # ---- context statistics of MLP-0's normalised input and write on the fit rows ------------------------------------------
        stat = {k: {"n": 0, "s": torch.zeros(D, dtype=torch.float64, device=dev), "ss": torch.zeros(D, D, dtype=torch.float64, device=dev)} for k in ("x", "w")}

        def acc(key, val):
            v = val.reshape(-1, D).double(); stat[key]["n"] += v.shape[0]; stat[key]["s"] += v.sum(0); stat[key]["ss"] += v.T @ v

        h1 = b0.mlp.register_forward_pre_hook(lambda m, a: acc("x", a[0]))
        h2 = b0.mlp.register_forward_hook(lambda m, a, o: acc("w", o))
        for s in range(0, fit.shape[0], BATCH):
            idx = fit[s:s + BATCH, :-1].to(dev); model(idx, fit[s:s + BATCH, 1:].to(dev)); forwards += 1
        h1.remove(); h2.remove()
        ctx = {}
        for k in ("x", "w"):
            n = stat[k]["n"]; mean = stat[k]["s"] / n; cov = stat[k]["ss"] / n - torch.outer(mean, mean)
            evals, U = torch.linalg.eigh(cov); ctx[k] = (mean.float(), U.flip(1)[:, :max(RANKS)].float(), evals.flip(0))
            c = evals.flip(0).cumsum(0) / evals.sum()
            print(f"context {k}: n={n}, 90/95/99% PCA ranks = {int((c < 0.9).sum()) + 1}/{int((c < 0.95).sum()) + 1}/{int((c < 0.99).sum()) + 1}, mean norm {float(mean.norm()):.2f}")
        tok_cov = v610["cov_mlp0_frequency"].double(); tok_mu = v610["mu_mlp0_frequency"].float()
        tok_P = torch.linalg.eigh(tok_cov)[1].flip(1)[:, :max(RANKS)].float()

        def ce(hooks):
            total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], BATCH):
                idx = ev[s:s + BATCH, :-1].to(dev); tgt = ev[s:s + BATCH, 1:].to(dev)
                total += float(model(idx, tgt)) * idx.numel(); n += idx.numel(); fw += 1
            for h in hooks:
                h.remove()
            return total / n, fw

        native, fw = ce([]); forwards += fw
        print(f"native CE {native:.5f} (v615: {NATIVE_V615})")
        edits = {}
        out_arms = {"coefficient": (v611["mlp0_coefficient"]["out_frame"].float(), bias), "single_token": (v611["mlp0_frequency_centered"]["out_frame"].float(), bias),
                    "context_mean": (ctx["w"][1], ctx["w"][0]), "context_bias": (ctx["w"][1], bias)}
        in_arms = {"coefficient": (v611["mlp0_coefficient"]["in_frame"].float(), None), "single_token": (tok_P, tok_mu), "context": (ctx["x"][1], ctx["x"][0])}
        for r in RANKS:
            for name, (frame, c) in out_arms.items():
                W = frame[:, :r].to(dev); c = c.to(dev)
                h = b0.mlp.register_forward_hook(lambda m, a, o, W=W, c=c: c + (o - c) @ W @ W.T)
                val, fw = ce([h]); forwards += fw; edits[f"out|{name}|{r}"] = val - native
            for name, (frame, mu) in in_arms.items():
                P = frame[:, :r].to(dev)
                if mu is None:
                    proj = lambda x, P=P: x @ P @ P.T
                else:
                    mu = mu.to(dev); proj = lambda x, P=P, mu=mu: mu + (x - mu) @ P @ P.T
                h = b0.mlp.register_forward_pre_hook(lambda m, a, proj=proj: (proj(a[0]),))
                val, fw = ce([h]); forwards += fw; edits[f"in|{name}|{r}"] = val - native
            print(f"r={r:3d}: OUT " + " ".join(f"{k}={edits[f'out|{k}|{r}']:+.4f}" for k in out_arms) + " | IN " + " ".join(f"{k}={edits[f'in|{k}|{r}']:+.4f}" for k in in_arms))
        disk_guard.guard_torch_save({"ctx_x_mean": ctx["x"][0].cpu(), "ctx_x_frame": ctx["x"][1].cpu(), "ctx_x_evals": ctx["x"][2].float().cpu(),
                                     "ctx_w_mean": ctx["w"][0].cpu(), "ctx_w_frame": ctx["w"][1].cpu(), "ctx_w_evals": ctx["w"][2].float().cpu()}, str(OUT_PT), "v616 context frames")
    e = edits
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_context_out_beats_coefficient": e["out|coefficient|128"] >= FLOOR_OUT and e["out|context_mean|128"] <= RATIO * e["out|coefficient|128"],
                   "pred_c_context_in_beats_coefficient": e["in|coefficient|128"] >= FLOOR_IN and e["in|context|128"] <= RATIO * e["in|coefficient|128"],
                   "pred_d_context_beats_single_token": e["in|context|128"] <= RATIO * e["in|single_token|128"],
                   "pred_e_mean_matters": e["out|context_mean|128"] <= e["out|context_bias|128"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_context_frame_edit_result_v616", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": edits, "context_pca_ranks": {k: {"evals_top8": ctx[k][2][:8].tolist()} for k in ctx}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
