#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_offset1_reproduces_v612 pred_b_prev_heads_decay pred_c_content_heads_flat pred_d_prev_heads_stay_separable pred_e_some_head_peaks_later
"""Embedding-forward folding, rung 14 (v622): POSITIONAL KERNELS of the nine layer-0 heads from weights and tables alone.

At layer 0 every query and key is an exact function of its token, so the squared pattern between a query token t at position p and a key token s
at position p - d is an exact function of (t, s, d):
    S_h^(d)(t,s) = (rot_d(q^_h(t)) . k^_h(s) / 128) x (rot_d(q^2_h(t)) . k^2_h(s) / 128)     (rotary is relative: only the offset d matters;
                                                                                             the model's bf16 cos/sin at each d are replicated)
v612 measured d = 1 only and found two families: rank-one separable patterns (heads 0.3 / 0.4 / 0.7, and 0.6 / 0.8 at rank 2) with large rms,
and high-rank content patterns (0.0 / 0.1 / 0.2 / 0.5) with small rms. This rung sweeps d in {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256,
512} on the same 4096 x 4096 unigram grid and reports per head and offset: rms energy, the separable fraction (top singular value^2 / total),
mean sign, and the 90%-energy rank. This is Logan's 19 Sep "shift operator" template read directly off the weights: a previous-token head is a
kernel concentrated at d = 1, an n-back head peaks at d = n, a content head is flat in d.
PREDICTIONS (scored as written; failures preserved)
    pred_a_offset1_reproduces_v612   at d = 1 the 90%-energy ranks match v612's per head (0.3 / 0.4 / 0.7 -> 1; 0.6 / 0.8 -> 2; others >= 300) (instrument)
    pred_b_prev_heads_decay          heads 0.3, 0.4, 0.7: energy at d = 2 <= 0.5 x energy at d = 1 (previous-token kernels). Prior: likely
    pred_c_content_heads_flat        heads 0.0, 0.1, 0.2, 0.5: max / min energy over d in [1, 64] <= 2 (content matching is position-insensitive). Prior: unsure
    pred_d_prev_heads_stay_separable heads 0.3, 0.4, 0.7: separable fraction >= 0.9 at every d <= 64 (separability is a property of the head, not of
                                     the offset). Prior: unsure
    pred_e_some_head_peaks_later     at least one head's energy over d is maximal at some d >= 2 (an n-back or broad-context head exists at layer 0).
                                     Prior: unsure
PRICE (registered maximum): 13 batched T=1 manual table forwards (block-0 tables); 9 x 15 SVDs of 4096^2 fp32 matrices. 0 real model forwards;
0 backwards; 0 fits. Bar <= 14.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_positional_kernels_v622_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_positional_kernels_v622_tensors.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.positional_kernels_v622"
FORWARDS_MAX = 14
BATCH = 4096
N_SCORE = 4096
OFFSETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512)
PREV, CONTENT, RANK2 = (3, 4, 7), (0, 1, 2, 5), (6, 8)
DECAY, FLAT_MAX, SEP_MIN = 0.5, 2.0, 0.9
PREDICTIONS = {"pred_a_offset1_reproduces_v612": "ranks 1 / 2 / >= 300 by family", "pred_b_prev_heads_decay": "E(2) <= 0.5 E(1) x 3", "pred_c_content_heads_flat": "max/min <= 2 over d <= 64 x 4",
               "pred_d_prev_heads_stay_separable": ">= 0.9 at all d <= 64 x 3", "pred_e_some_head_peaks_later": "argmax d >= 2 for some head"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "offsets": list(OFFSETS), "n_score": N_SCORE,
            "bars": {"decay": DECAY, "flat_max": FLAT_MAX, "sep_min": SEP_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    b0 = model.transformer.h[0]; a = b0.attn; rms = EF.rms; forwards = 0
    with torch.no_grad():
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))

        def rot(x, d):                                                            # the model's Rotary at position d (bf16 cos/sin), applied to x at position 0
            cos, sin = (d * inv_freq).cos().bfloat16().float(), (d * inv_freq).sin().bfloat16().float()
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)

        E = model.transformer.wte.weight.detach().float(); lam0 = b0.lambdas.detach().float()
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("q", "q2", "k", "k2")}
        for s0 in range(0, V, BATCH):
            ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
            x0 = rms(E[ids]); n = rms(lam0[0] * x0 + lam0[1] * x0)
            T["q"][ids], T["k"][ids] = rms(a.c_q(n).view(-1, H, hd)), rms(a.c_k(n).view(-1, H, hd))
            T["q2"][ids], T["k2"][ids] = rms(a.c_q2(n).view(-1, H, hd)), rms(a.c_k2(n).view(-1, H, hd))
            forwards += 1
        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)                      # the v612 grid
        ts = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen); ss = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen)
        # rotary is relative: rotate the query by d (query at position d, key at position 0) -- identical to query at p, key at p - d
        kernels = {h: {} for h in range(H)}
        for d in OFFSETS:
            qr, q2r = rot(T["q"][ts], d), rot(T["q2"][ts], d)                     # [N, H, hd]
            for h in range(H):
                S = ((qr[:, h] @ T["k"][ss, h].T) / hd) * ((q2r[:, h] @ T["k2"][ss, h].T) / hd)
                sv = torch.linalg.svdvals(S); e = sv.square(); c = e.cumsum(0) / e.sum()
                kernels[h][d] = {"rms": float(S.square().mean().sqrt()), "separable_fraction": float(e[0] / e.sum()), "mean": float(S.mean()),
                                 "rank90": int((c < 0.9).sum()) + 1, "rank99": int((c < 0.99).sum()) + 1}
            print(f"d={d:3d}: " + " ".join(f"0.{h}:{kernels[h][d]['rms']:.4f}/{kernels[h][d]['separable_fraction']:.2f}/r{kernels[h][d]['rank90']}" for h in range(H)))
        disk_guard.guard_torch_save({"grid_t": ts.cpu(), "grid_s": ss.cpu(), "kernels": kernels}, str(OUT_PT), "v622 kernels")
    E_ = {h: {d: kernels[h][d]["rms"] ** 2 for d in OFFSETS} for h in range(H)}
    argmax = {h: max(OFFSETS, key=lambda d: E_[h][d]) for h in range(H)}
    predictions = {"pred_a_offset1_reproduces_v612": all(kernels[h][1]["rank90"] == 1 for h in PREV) and all(kernels[h][1]["rank90"] == 2 for h in RANK2) and all(kernels[h][1]["rank90"] >= 300 for h in CONTENT),
                   "pred_b_prev_heads_decay": all(E_[h][2] <= DECAY * E_[h][1] for h in PREV),
                   "pred_c_content_heads_flat": all(max(E_[h][d] for d in OFFSETS if d <= 64) / min(E_[h][d] for d in OFFSETS if d <= 64) <= FLAT_MAX for h in CONTENT),
                   "pred_d_prev_heads_stay_separable": all(kernels[h][d]["separable_fraction"] >= SEP_MIN for h in PREV for d in OFFSETS if d <= 64),
                   "pred_e_some_head_peaks_later": any(argmax[h] >= 2 for h in range(H))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_positional_kernels_result_v622", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"kernels": {str(h): {str(d): v for d, v in kd.items()} for h, kd in kernels.items()}, "argmax_offset": {str(h): d for h, d in argmax.items()}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
