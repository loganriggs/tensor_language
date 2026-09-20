#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_token_beats_coefficient_out pred_c_token_beats_coefficient_in pred_d_frequency_le_uniform pred_e_out512_cheap
"""Embedding-forward folding, rung 7 (v615): the lane's first EDIT — install v611's token-metric HOSVD subspaces in MLP-0 and price them in CE.

v611 measured (Frobenius, single-token tables) that the token-metric frames of MLP-0's centred tensor retain far more energy than the
coefficient-metric frames at equal rank (out-mode r=256: 0.68 freq / 0.60 uniform vs 0.46; two-input-mode r=256: 0.79 / 0.61 vs 0.19). This
rung asks the question that matters: do subspaces DERIVED FROM SINGLE TOKENS transfer to real contexts (where 24% of MLP-0's input is the
previous token, v612), and is the token metric the right metric for compressing MLP-0 under the model's own loss?
Two surgery arms, each a pure projection with no fitting:
    OUT  write' = bias + W W^T (write - bias)                             (W = top-r output frame; Euclidean orthogonal projection)
    IN   x^' = mu + C^1/2 P P^T C^-1/2 (x^ - mu)                          (P = top-r whitened input frame; mean kept exactly; oblique projection)
         coefficient metric: x^' = P P^T x^ (Euclidean frame of the raw tensor, no mean)
for metrics {coefficient, uniform_centered, frequency_centered} x r in {128, 256, 512}, scored as mean next-token CE on the 192 x 513 held-out
FineWeb rows (skip7000; native 3.29205 on record) — rows disjoint from the skip80 rows that produced the unigram weights. Sign convention: CE ADDED
above the native model, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_replays          native CE on the 192 rows within 0.002 of 3.29205 (instrument)
    pred_b_token_beats_coefficient_out  OUT arm, r=256: CE added (frequency_centered) <= 0.5 x CE added (coefficient), with the coefficient arm
                                     >= 0.05 nats so the ratio is meaningful. Prior: likely
    pred_c_token_beats_coefficient_in   IN arm, r=256: same bar. Prior: unsure — the oblique whitened projection may not transfer to contexts
    pred_d_frequency_le_uniform      at every (arm, r): CE added frequency_centered <= uniform_centered. Prior: likely
    pred_e_out512_cheap              OUT arm, frequency_centered, r=512: CE added <= 0.10 nats. Prior: unsure
PRICE (registered maximum): 19 configs (18 edits + native) x 6 batches of 32 rows = 114 forwards of 512 tokens; 0 backwards; 0 fits. Bar <= 130.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_subspace_edit_v615_result.json"
FRAMES = ROOT / "circuits/followups/embedding_forward_tucker_energy_v611_frames.pt"
COVS = ROOT / "circuits/followups/embedding_forward_degree_census_v610_tensors.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_CE = 3.29205
CANDIDATE_ID = "embedding_forward.subspace_edit_v615"
FORWARDS_MAX = 130
BATCH = 32
RANKS = (128, 256, 512)
METRICS = ("coefficient", "uniform_centered", "frequency_centered")
REPLAY_TOL, RATIO_MAX, FLOOR, CHEAP_MAX = 0.002, 0.5, 0.05, 0.10
PREDICTIONS = {"pred_a_baseline_replays": "+-0.002", "pred_b_token_beats_coefficient_out": "<= 0.5x and coef >= 0.05", "pred_c_token_beats_coefficient_in": "<= 0.5x and coef >= 0.05",
               "pred_d_frequency_le_uniform": "freq <= uniform everywhere", "pred_e_out512_cheap": "<= 0.10"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "ranks": list(RANKS), "metrics": list(METRICS),
            "bars": {"replay_tol": REPLAY_TOL, "ratio_max": RATIO_MAX, "floor": FLOOR, "cheap_max": CHEAP_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    D = model.config.n_embd; dev = "cuda"; b0 = model.transformer.h[0]; forwards = 0
    rows = torch.load(EVAL_ROWS, map_location="cpu").long()
    frames = torch.load(FRAMES, map_location=dev); covs = torch.load(COVS, map_location=dev)
    with torch.no_grad():
        bias = b0.mlp.Down_bias.detach().float()
        # whitening operators for the centred metrics (pinv with a relative floor: directions with no token variance are dropped)
        white = {}
        for name, key in (("uniform_centered", "uniform"), ("frequency_centered", "frequency")):
            C = covs[f"cov_mlp0_{key}"].double(); mu = covs[f"mu_mlp0_{key}"].double()
            w, U = torch.linalg.eigh(C); keep = w > 1e-8 * w.max()
            half = (U[:, keep] * w[keep].sqrt()) @ U[:, keep].T; ihalf = (U[:, keep] / w[keep].sqrt()) @ U[:, keep].T
            white[name] = (mu.float(), half.float(), ihalf.float(), int(keep.sum()))

        def ce(hooks):
            total = 0.0; n = 0; fw = 0
            for s in range(0, rows.shape[0], BATCH):
                idx = rows[s:s + BATCH, :-1].to(dev); tgt = rows[s:s + BATCH, 1:].to(dev)
                loss = model(idx, tgt); total += float(loss) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        native, fw = ce([]); forwards += fw
        print(f"native CE {native:.5f} (record {NATIVE_CE})")
        results = {"native": native, "edits": {}}
        for metric in METRICS:
            fr = frames[f"mlp0_{metric}"]
            for r in RANKS:
                W = fr["out_frame"][:, :r].float().to(dev); P = fr["in_frame"][:, :r].float().to(dev)
                # OUT arm
                h = b0.mlp.register_forward_hook(lambda m, a, o, W=W: bias + (o - bias) @ W @ W.T)
                c_out, fw = ce([h]); h.remove(); forwards += fw
                # IN arm
                if metric == "coefficient":
                    proj = lambda x, P=P: x @ P @ P.T
                else:
                    mu, half, ihalf, _ = white[metric]
                    M = (half @ P @ P.T @ ihalf).T                                            # right-multiply form of C^1/2 P P^T C^-1/2
                    proj = lambda x, M=M, mu=mu: mu + (x - mu) @ M
                h = b0.mlp.register_forward_pre_hook(lambda m, a, proj=proj: (proj(a[0]),))
                c_in, fw = ce([h]); h.remove(); forwards += fw
                results["edits"][f"{metric}|out|{r}"] = c_out - native; results["edits"][f"{metric}|in|{r}"] = c_in - native
                print(f"{metric:19s} r={r:3d}: CE added OUT {c_out - native:+.4f}  IN {c_in - native:+.4f}")
    e = results["edits"]
    predictions = {"pred_a_baseline_replays": abs(native - NATIVE_CE) <= REPLAY_TOL,
                   "pred_b_token_beats_coefficient_out": e["coefficient|out|256"] >= FLOOR and e["frequency_centered|out|256"] <= RATIO_MAX * e["coefficient|out|256"],
                   "pred_c_token_beats_coefficient_in": e["coefficient|in|256"] >= FLOOR and e["frequency_centered|in|256"] <= RATIO_MAX * e["coefficient|in|256"],
                   "pred_d_frequency_le_uniform": all(e[f"frequency_centered|{arm}|{r}"] <= e[f"uniform_centered|{arm}|{r}"] for arm in ("out", "in") for r in RANKS),
                   "pred_e_out512_cheap": e["frequency_centered|out|512"] <= CHEAP_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_subspace_edit_result_v615", "candidate_id": CANDIDATE_ID, "plan": plan, "report": results,
                               "whitening_kept_dims": {k: v[3] for k, v in white.items()}, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
