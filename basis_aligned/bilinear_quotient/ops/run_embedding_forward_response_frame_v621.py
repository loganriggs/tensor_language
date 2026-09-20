#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_response_beats_pca_128 pred_c_response_beats_pca_64 pred_d_fisher_concentrated pred_e_frames_overlap_weakly
"""Embedding-forward folding, rung 13 (v621): RESPONSE-weighted output frames for MLP-0 — do the readers, not the write statistics, pick the metric?

v615/v616 left hypothesis (c) open: CE is governed by what downstream readers pick up, so the right output metric for compressing a write is the
loss's sensitivity, not the write's covariance. Test with gradient STATISTICS (no parameter is fitted): on the 480 x 512 skip80 fit rows, capture
per-token gradients g_t of the summed loss with respect to MLP-0's write (60 forward + 60 backward passes, batch 8) and the write covariance
Cov_w; form the empirical Fisher F = E[g g^T] (damped by 1e-3 x its mean eigenvalue). Three rank-r output projections, mean kept:
    pca       write' = mu + W W^T (write - mu),  W = top-r eigenvectors of Cov_w                (v616's frame, re-measured)
    fisher    W = top-r eigenvectors of F                                                        (keep what the loss is most sensitive to)
    response  Pi = F^-1/2 U U^T F^1/2 with U = top-r eigenvectors of F^1/2 Cov_w F^1/2           (minimises E[(g . (I - Pi) dw)^2]: the
              generalised-eigenvector, Fisher-whitened PCA; oblique)
priced in CE on the 192 x 512 skip7000 rows at r in {64, 128, 256}. CE ADDED, lower is better. Also: the Fisher spectrum and the overlap of the
top-128 Fisher frame with the top-128 PCA frame.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_response_beats_pca_128 response r=128 CE added <= 0.5 x pca r=128 (v616: 0.036). Prior: unsure — this is (c)
    pred_c_response_beats_pca_64  response r=64 <= 0.5 x pca r=64 (v616: 0.082). Prior: unsure
    pred_d_fisher_concentrated   the empirical Fisher's 90%-energy rank <= 256 (the loss is sensitive to few write directions). Prior: unsure
    pred_e_frames_overlap_weakly  fraction of the top-128 Fisher frame's energy inside the top-128 PCA subspace <= 0.5. Prior: likely
PRICE (registered maximum): capture 60 forwards + 60 BACKWARDS (batch 8, gradient statistics only); eval 10 configs x 6 batches = 60 forwards;
total 120 forwards, 60 backwards; 0 fits (no parameter is optimised; the Fisher is a statistic of the native model). Bars: forwards <= 130,
backwards <= 60.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_response_frame_v621_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_response_frame_v621_tensors.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.response_frame_v621"
FORWARDS_MAX, BACKWARDS_MAX = 130, 60
CAP_BATCH, BATCH = 8, 32
RANKS = (64, 128, 256)
DAMP = 1e-3
REPLAY_TOL, RATIO, FISHER_RANK_MAX, OVERLAP_MAX = 0.002, 0.5, 256, 0.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_response_beats_pca_128": "<= 0.5x", "pred_c_response_beats_pca_64": "<= 0.5x",
               "pred_d_fisher_concentrated": "90% rank <= 256", "pred_e_frames_overlap_weakly": "<= 0.5"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "ranks": list(RANKS), "damp": DAMP,
            "bars": {"replay_tol": REPLAY_TOL, "ratio": RATIO, "fisher_rank_max": FISHER_RANK_MAX, "overlap_max": OVERLAP_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; dev = "cuda"; b0 = model.transformer.h[0]; forwards = 0; backwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    # ---- gradient statistics at MLP-0's write (the write becomes the graph's leaf; parameters stay frozen) ---------------------
    F = torch.zeros(D, D, dtype=torch.float64, device=dev); gsum = torch.zeros(D, dtype=torch.float64, device=dev); n_tok = 0
    Sw = torch.zeros(D, dtype=torch.float64, device=dev); SSw = torch.zeros(D, D, dtype=torch.float64, device=dev)
    leaf = {}

    def make_leaf(m, a, o):
        w = o.detach().requires_grad_(True); leaf["w"] = w; return w

    h = b0.mlp.register_forward_hook(make_leaf)
    for s in range(0, fit.shape[0], CAP_BATCH):
        idx = fit[s:s + CAP_BATCH, :-1].to(dev); tgt = fit[s:s + CAP_BATCH, 1:].to(dev)
        with torch.enable_grad():
            loss = model(idx, tgt) * idx.numel()                                           # summed loss -> per-token gradients
            loss.backward()
        forwards += 1; backwards += 1
        with torch.no_grad():
            g = leaf["w"].grad.reshape(-1, D).double(); w = leaf["w"].detach().reshape(-1, D).double()
            F += g.T @ g; gsum += g.sum(0); n_tok += g.shape[0]; Sw += w.sum(0); SSw += w.T @ w
        leaf.clear()
    h.remove()
    with torch.no_grad():
        F /= n_tok; mu_w = Sw / n_tok; Cov_w = SSw / n_tok - torch.outer(mu_w, mu_w)
        fe, fU = torch.linalg.eigh(F); fe, fU = fe.flip(0), fU.flip(1)
        c = fe.clamp_min(0).cumsum(0) / fe.clamp_min(0).sum(); fisher_ranks = {q: int((c < q).sum()) + 1 for q in (0.9, 0.95, 0.99)}
        we, wU = torch.linalg.eigh(Cov_w); we, wU = we.flip(0), wU.flip(1)
        lam = DAMP * float(fe.mean()); Fd = F + lam * torch.eye(D, dtype=torch.float64, device=dev)
        de, dU = torch.linalg.eigh(Fd); F_half = (dU * de.sqrt()) @ dU.T; F_ihalf = (dU / de.sqrt()) @ dU.T
        S = F_half @ Cov_w @ F_half; se, sU = torch.linalg.eigh(S); sU = sU.flip(1)
        overlap = float((wU[:, :128].T @ fU[:, :128]).square().sum() / 128)
        print(f"tokens {n_tok}; Fisher 90/95/99% ranks {fisher_ranks}; |mean g| / sqrt(tr F) = {float(gsum.norm() / n_tok / fe.sum().sqrt()):.4f}; overlap(F128, PCA128) = {overlap:.3f}")
        mu = mu_w.float()
        frames = {"pca": lambda r: (wU[:, :r].float(), None), "fisher": lambda r: (fU[:, :r].float(), None),
                  "response": lambda r: (None, (F_ihalf @ sU[:, :r] @ sU[:, :r].T @ F_half).float())}

        def ce(hs):
            total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], BATCH):
                idx = ev[s:s + BATCH, :-1].to(dev); total += float(model(idx, ev[s:s + BATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            for hh in hs:
                hh.remove()
            return total / n, fw

        native, fw = ce([]); forwards += fw; print(f"native CE {native:.5f}")
        edits = {}
        for r in RANKS:
            for name, fn in frames.items():
                W, Pi = fn(r)
                if W is not None:
                    hh = b0.mlp.register_forward_hook(lambda m, a, o, W=W: mu + (o - mu) @ W @ W.T)
                else:
                    hh = b0.mlp.register_forward_hook(lambda m, a, o, Pi=Pi: mu + (o - mu) @ Pi.T)
                v, fw = ce([hh]); forwards += fw; edits[f"{name}|{r}"] = v - native
            print(f"r={r:3d}: " + " ".join(f"{k}={edits[f'{k}|{r}']:+.4f}" for k in frames))
        disk_guard.guard_torch_save({"fisher": F.float().cpu(), "cov_w": Cov_w.float().cpu(), "mu_w": mu.cpu(), "fisher_evals": fe.float().cpu()}, str(OUT_PT), "v621 Fisher")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_response_beats_pca_128": edits["response|128"] <= RATIO * edits["pca|128"],
                   "pred_c_response_beats_pca_64": edits["response|64"] <= RATIO * edits["pca|64"],
                   "pred_d_fisher_concentrated": fisher_ranks[0.9] <= FISHER_RANK_MAX,
                   "pred_e_frames_overlap_weakly": overlap <= OVERLAP_MAX}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "embedding_forward_response_frame_result_v621", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": edits, "fisher_ranks": {str(k): v for k, v in fisher_ranks.items()}, "overlap_f128_pca128": overlap,
                                          "fisher_evals_top16": fe[:16].tolist(), "n_tokens": n_tok},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
