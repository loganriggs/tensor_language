#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_kernel_only_cheap pred_c_offdiag_matters pred_d_program_shape_right pred_e_mass_grows_with_position
"""Embedding-forward folding, rung 18 (v626): what is head 1.8? Real pattern vs the token program, and three edits that name it.

v624/v625: head 1.8's single-token pattern is a broad rank-one window (rms 0.17 flat to d ~ 8, still 0.03 at d = 64) and the token program
kappa(d) A(t) B(s) fits it to 0.01 on the grid — yet installing it costs +0.558 nats, and it is not a position-0 sink (v625). So in context the
head's pattern is determined by something the single-token tables do not carry. This rung captures the REAL pattern on 64 fit rows (skip80,
2 forwards) and compares it with the token program on the same rows (off-diagonal entries, query positions >= 8): entrywise Pearson correlation,
least-squares scale (real ~ c x program), the real mean signed pattern per offset d (the context kernel), and the mean |pattern| per entry as a
function of ABSOLUTE query position (does the head's per-entry weight change with how much context there is? — unnormalised attention sums
more entries as the sequence grows). Then three edits on skip7000: (a) pattern := real context kernel only, kbar(d), no token dependence;
(b) pattern := c x token program; (c) off-diagonal := 0 (keep the native diagonal). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_kernel_only_cheap       the kernel-only program (a) costs <= 0.10 (the head is mostly a positional averager). Prior: unsure
    pred_c_offdiag_matters         zeroing the off-diagonal (c) costs >= 0.30 (the head's context part is load-bearing). Prior: likely
    pred_d_program_shape_right     entrywise Pearson(real, program) >= 0.5 on off-diagonal entries with query >= 8 — the program has the right
                                   shape and fails on scale or context modulation. Prior: unsure
    pred_e_mass_grows_with_position mean |pattern| per entry for queries in [256, 512) >= 1.5 x that for queries in [8, 64). Prior: unsure
PRICE (registered maximum): 2 capture forwards (64 rows, batch 32) + 4 configs x 6 eval batches = 26 forwards; 0 backwards; 0 fits (c is one
closed-form scalar). Bar <= 30.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_head18_v626_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_head18_v626_tensors.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.head18_v626"
FORWARDS_MAX = 30
EBATCH, N_CAP = 32, 64
HEAD, Q_MIN = 8, 8
REPLAY_TOL, KERNEL_MAX, OFF_MIN, CORR_MIN, GROW_RATIO = 0.002, 0.10, 0.30, 0.5, 1.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_kernel_only_cheap": "<= 0.10", "pred_c_offdiag_matters": ">= 0.30", "pred_d_program_shape_right": "Pearson >= 0.5",
               "pred_e_mass_grows_with_position": ">= 1.5x"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "head": HEAD, "n_capture": N_CAP,
            "bars": {"replay_tol": REPLAY_TOL, "kernel_max": KERNEL_MAX, "off_min": OFF_MIN, "corr_min": CORR_MIN, "grow_ratio": GROW_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    dev = "cuda"; b1 = model.transformer.h[1]; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    t1_ = torch.load(T1, map_location=dev); A, Bt, kappa = t1_[f"head{HEAD}_A"], t1_[f"head{HEAD}_B"], t1_[f"head{HEAD}_kappa"]
    with torch.no_grad():
        state = {"idx": None, "mode": None, "capture": None, "kbar": None, "scale": 1.0}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        native_sq = b1.attn.squared_attention

        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
            if state["capture"] is not None:
                state["capture"].append(pat[:, HEAD].detach().clone())
            if state["mode"]:
                idx = state["idx"]; pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
                if state["mode"] == "kernel":
                    prog = state["kbar"][dmat][None].expand(Bn, -1, -1)
                elif state["mode"] == "program":
                    prog = state["scale"] * kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                else:
                    prog = torch.zeros_like(pat[:, HEAD])
                pat[:, HEAD] = torch.where(off, prog, pat[:, HEAD])
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)

        b1.attn.squared_attention = patched
        # ---- capture the real pattern on 64 fit rows ----------------------------------------------------------------------------
        state["capture"] = []; reals, idxs = [], []
        for s in range(0, N_CAP, EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1; idxs.append(idx)
        reals = torch.cat(state["capture"]); idx_all = torch.cat(idxs); state["capture"] = None                          # [N, T, T]
        Tn = reals.shape[-1]; pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0)
        prog = kappa[dmat][None] * A[idx_all][:, :, None] * Bt[idx_all][:, None, :]
        qmask = (pos >= Q_MIN)[:, None] & off                                                                            # [T, T]
        r = reals[:, qmask]; p = prog[:, qmask]
        corr = float(torch.corrcoef(torch.stack([r.flatten(), p.flatten()]))[0, 1]); scale = float((r * p).sum() / (p * p).sum())
        resid = float(((r - scale * p).square().sum()) / r.square().sum())
        kbar = torch.zeros(Tn + 1, device=dev)
        for d in range(1, Tn):
            m = (dmat == d) & (pos >= Q_MIN)[:, None]; kbar[d] = float(reals[:, m].mean())
        kbar_prog = torch.zeros(Tn + 1, device=dev)
        for d in range(1, Tn):
            m = (dmat == d) & (pos >= Q_MIN)[:, None]; kbar_prog[d] = float(prog[:, m].mean())
        per_entry = (reals.abs() * off[None]).sum(-1) / off.sum(-1).clamp_min(1)[None]                                    # [N, T] mean |pattern| per entry
        early = float(per_entry[:, 8:64].mean()); late = float(per_entry[:, 256:512].mean())
        row_sum = (reals * off[None]).sum(-1); early_sum = float(row_sum[:, 8:64].mean()); late_sum = float(row_sum[:, 256:512].mean())
        sv = torch.linalg.svdvals(reals[:, Q_MIN:, :].reshape(-1, Tn)[:4096].double()); e = sv.square(); rank90 = int((e.cumsum(0) / e.sum() < 0.9).sum()) + 1
        print(f"real vs program (queries >= {Q_MIN}): Pearson {corr:.3f}, LS scale {scale:.3f}, residual after scale {resid:.3f}")
        print(f"real kernel kbar(1..8) {[round(x, 4) for x in kbar[1:9].tolist()]} | program kappa-mean(1..8) {[round(x, 4) for x in kbar_prog[1:9].tolist()]} | kbar(16,32,64,128,256) {[round(float(kbar[d]), 4) for d in (16, 32, 64, 128, 256)]}")
        print(f"mean |pattern| per entry: queries 8-64 {early:.4f}, 256-512 {late:.4f} (ratio {late / early:.2f}); signed row sums {early_sum:+.3f} -> {late_sum:+.3f}; rank90 of real query rows {rank90}")

        def ce(mode):
            state["mode"] = mode; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        state["kbar"] = kbar; state["scale"] = scale
        native, fw = ce(None); forwards += fw
        edits = {}
        for mode in ("kernel", "program", "zero"):
            v_, fw = ce(mode); forwards += fw; edits[mode] = v_ - native
        b1.attn.squared_attention = native_sq; pre.remove()
        print(f"native {native:.5f} | CE added: kernel-only {edits['kernel']:+.4f} scaled program {edits['program']:+.4f} off-diagonal zero {edits['zero']:+.4f}")
        disk_guard.guard_torch_save({"kbar": kbar.cpu(), "kbar_prog": kbar_prog.cpu(), "per_entry_mean_over_rows": per_entry.mean(0).cpu()}, str(OUT_PT), "v626 kernels")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_kernel_only_cheap": edits["kernel"] <= KERNEL_MAX,
                   "pred_c_offdiag_matters": edits["zero"] >= OFF_MIN, "pred_d_program_shape_right": corr >= CORR_MIN, "pred_e_mass_grows_with_position": late >= GROW_RATIO * early}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_head18_result_v626", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": edits, "pearson": corr, "ls_scale": scale, "residual_after_scale": resid, "kbar_1_16": kbar[1:17].tolist(),
                                          "kbar_prog_1_16": kbar_prog[1:17].tolist(), "per_entry_early": early, "per_entry_late": late, "row_sum_early": early_sum, "row_sum_late": late_sum, "rank90_real_rows": rank90},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
