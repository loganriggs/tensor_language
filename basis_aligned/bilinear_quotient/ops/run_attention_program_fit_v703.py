#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_step0_replays_v702 pred_c_fitted_x09_recovery pred_d_fitted_x075_recovery pred_e_rank32_helps_content_heads
"""Attention lane, v703: the v702 programs with their kernels FITTED jointly (native heads fixed), and higher-rank hybrids for the content heads.

v702: the X = 0.9 program (129 closed-form kernels, 5 rank-16 hybrids, 28 native) costs +0.507 jointly (recovery 0.873 of the 3.996 joint value);
X = 0.75 (133 / 3 / 12 / 14 native) costs +0.805. v646 showed that closed-form kernels compound and that a joint refit against CE removes much of
it. Here each program's kernels (the kappa of every kernel and hybrid head; hybrid low-rank maps fixed; native heads fixed) are refit jointly:
Adam 0.02 -> 0.002 cosine on kappa = kappa0 (1 + c), 200 steps, batch 8, train 576 / validation 96 rows (skip80 + skip11000 split, seed 703),
validation minimum every 25 steps chooses the step, held-out 192 x 512 skip7000 at that step. Separately, single-head hybrids at rank 32 and 64 for
the two big content matchers 2.5 and 3.8 on the 64 fit rows (recovery relative to their v702 values .031 / .020). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays       native CE within 0.002 of 3.13241 (instrument)
    pred_b_step0_replays_v702   the X = 0.9 program at step 0 replays v702's +0.507 within 0.01 (instrument)
    pred_c_fitted_x09_recovery  fitted X = 0.9 program: held-out joint recovery >= 0.94 (cost <= 0.24). Prior: unsure
    pred_d_fitted_x075_recovery fitted X = 0.75 program: held-out joint recovery >= 0.90 (cost <= 0.40). Prior: unsure
    pred_e_rank32_helps_content_heads for both 2.5 and 3.8, rank-32 recovery >= 0.75 on fit rows. Prior: unsure
PRICE (registered maximum): 2 arms x 200 steps = 400 forwards + 400 BACKWARDS; validation 2 x 9 x 3 = 54; held-out 2 x 9 x 6 + 6 = 114; content
hybrids and their kappa_r captures ~60; total ~630 forwards, 400 backwards; fit parameters 134 x 513 = 69k (X=0.9) / 148 x 513 = 76k (X=0.75).
Bars: forwards <= 650, backwards <= 400.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_program_fit_v703_result.json"
OUT_PT = ROOT / "circuits/followups/attention_program_fit_v703_kernels.pt"
V702 = ROOT / "circuits/followups/attention_program_ladder_v702_result.json"
PROGS = ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, V702_X09, JOINT_VALUE = 3.13241, 0.5065, 3.9961
CANDIDATE_ID = "attention.program_fit_v703"
FORWARDS_MAX, BACKWARDS_MAX = 650, 400
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 200, 8, 32, 25, 96
LR, LR_MIN = 0.02, 0.002
LAYERS = tuple(range(18)); H = 9
CONTENT = ((2, 5), (3, 8)); CONTENT_RANKS = (32, 64)
REPLAY_TOL, STEP0_TOL, REC09, REC075, REC_CONTENT = 0.002, 0.01, 0.94, 0.90, 0.75
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_step0_replays_v702": "+-0.01 of 0.507", "pred_c_fitted_x09_recovery": ">= 0.94",
               "pred_d_fitted_x075_recovery": ">= 0.90", "pred_e_rank32_helps_content_heads": ">= 0.75 x 2"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 148 * 513,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN],
            "bars": {"replay_tol": REPLAY_TOL, "step0_tol": STEP0_TOL, "rec09": REC09, "rec075": REC075, "rec_content": REC_CONTENT}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    v702 = json.load(open(V702))["report"]; progs_pt = torch.load(PROGS, map_location=dev)
    kappa0 = {(l, h): progs_pt[f"kappa_{l}_{h}"].to(dev) for l in LAYERS for h in range(H)}
    value_fit = {k: v["value_fit"] for k, v in v702["heads"].items()}
    state = {"idx": None, "programs": {}, "n": {}}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]

    def make_patched(l):
        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
            pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            ctx = {"dmat": dmat, "off": off, "pos": pos, "idx": state["idx"], "attn": blocks[l].attn, "n": state["n"][l], "hd": hd, "causal": causal}
            cols = []
            for h in range(Hn):
                prog = state["programs"].get((l, h))
                cols.append(pat[:, h] if prog is None else AP.apply_program(pat[:, h], prog() if callable(prog) else prog, ctx))
            pat = torch.stack(cols, 1)
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(rows, progs):
        state["programs"] = progs; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rows.shape[0], EBATCH):
                idx = rows[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rows[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["programs"] = {}
        return total / n, fw

    native, fw = ce(ev, {}); forwards += fw
    # ---- content-head hybrids at rank 32 / 64 (single, fit rows) ---------------------------------------------------------------
    nat64, fw = ce(fit64, {}); forwards += fw; content = {}
    with torch.no_grad():
        for (l, h) in CONTENT:
            for r in CONTENT_RANKS:
                maps = AP.truncated_maps(blocks[l].attn, h, r, hd)
                # kappa_r: positional mean of the rank-r pattern on the 64 rows (one capture through the patched pattern)
                acc = 0; nb = 0
                for s in range(0, 64, EBATCH):
                    idx = fit64[s:s + EBATCH, :-1].to(dev); state["programs"] = {}; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); fw_ = 1; forwards += fw_
                    Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                    pr = AP.lowrank_pattern(blocks[l].attn, state["n"][l], maps, hd, causal); acc = acc + AP.offset_mean(pr, dmat, off, Tn); nb += 1
                prog = {"kind": "lowrank", "kappa": kappa0[(l, h)], "kappa_r": acc / nb, "maps": maps}
                c_, fw = ce(fit64, {(l, h): prog}); forwards += fw; content[f"{l}.{h}|r{r}"] = {"cost": c_ - nat64, "recovery": 1 - (c_ - nat64) / value_fit[f"{l}.{h}"]}
        print("content-head hybrids (fit rows): " + " ".join(f"{k}: rec {v['recovery']:.2f}" for k, v in content.items()))

    # ---- joint kernel refits of the X = 0.9 and X = 0.75 programs --------------------------------------------------------------
    results, saved = {}, {}
    for X in ("0.9", "0.75"):
        sel = v702["selections"][X]
        fixed, fitted = {}, {}
        for key, name in sel.items():
            l, h = map(int, key.split(".")); k = (l, h)
            if name == "native":
                continue
            if name in ("kernel", "lowrank4", "lowrank16"):
                c = torch.zeros(513, device=dev, requires_grad=True); fitted[k] = (name, c)
            elif name == "runmean":
                fixed[k] = {"kind": "runmean", "m": float(progs_pt[f"m_{l}_{h}"])}
        lr_maps = {k: AP.truncated_maps(blocks[k[0]].attn, k[1], 4 if name == "lowrank4" else 16, hd) for k, (name, c) in fitted.items() if name != "kernel"}
        # kappa_r for hybrids: positional mean of the rank-r pattern on the training rows' first 64
        kr = {}
        with torch.no_grad():
            for k, maps in lr_maps.items():
                acc = 0; nb = 0
                for s in range(0, 64, EBATCH):
                    idx = fit64[s:s + EBATCH, :-1].to(dev); state["programs"] = {}; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
                    Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                    acc = acc + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps, hd, causal), dmat, off, Tn); nb += 1
                kr[k] = acc / nb

        def build():
            progs = dict(fixed)
            for k, (name, c) in fitted.items():
                kap = kappa0[k] * (1 + c)
                progs[k] = {"kind": "kernel", "kappa": kap} if name == "kernel" else {"kind": "lowrank", "kappa": kap, "kappa_r": kr[k], "maps": lr_maps[k]}
            return progs

        params = [c for (_, c) in fitted.values()]
        step0, fw = ce(ev, build()); forwards += fw
        opt = torch.optim.Adam(params, lr=LR); val_curve, ho_curve = {0: None}, {0: step0 - native}
        v0, fw = ce(val, build()); forwards += fw; val_curve[0] = v0
        gen2 = torch.Generator().manual_seed(int(float(X) * 1000)); order = torch.randperm(fit.shape[0], generator=gen2)
        for step in range(1, STEPS + 1):
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (step - 1) / STEPS))
            for g in opt.param_groups:
                g["lr"] = lr
            s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel_idx = order[s0:s0 + TBATCH]
            if len(sel_idx) < TBATCH:
                sel_idx = order[:TBATCH]
            idx = fit[sel_idx, :-1].to(dev); tgt = fit[sel_idx, 1:].to(dev); state["programs"] = build()
            with torch.enable_grad():
                loss = model(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            state["programs"] = {}; forwards += 1; backwards += 1
            if step % EVAL_EVERY == 0:
                v_, fw = ce(val, build()); forwards += fw; h_, fw = ce(ev, build()); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - native
        chosen = min(val_curve, key=val_curve.get); ho_at = ho_curve[chosen]
        results[X] = {"step0_added": step0 - native, "chosen_step": chosen, "heldout_at_chosen": ho_at, "recovery_at_chosen": 1 - ho_at / JOINT_VALUE, "heldout_curve": {str(k): v for k, v in ho_curve.items()},
                      "n_fitted_kernels": len(fitted), "n_native": sum(1 for v in sel.values() if v == "native")}
        saved[X] = {f"{k[0]}.{k[1]}": (kappa0[k] * (1 + c)).detach().cpu() for k, (_, c) in fitted.items()}
        print(f"X={X}: step 0 {step0 - native:+.4f} -> fitted (validation step {chosen}) {ho_at:+.4f}, joint recovery {1 - ho_at / JOINT_VALUE:.3f}; native heads {results[X]['n_native']}, fitted kernels {len(fitted)}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    disk_guard.guard_torch_save(saved, str(OUT_PT), "v703 fitted kernels (endpoint of training)")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_step0_replays_v702": abs(results["0.9"]["step0_added"] - V702_X09) <= STEP0_TOL,
                   "pred_c_fitted_x09_recovery": results["0.9"]["recovery_at_chosen"] >= REC09, "pred_d_fitted_x075_recovery": results["0.75"]["recovery_at_chosen"] >= REC075,
                   "pred_e_rank32_helps_content_heads": all(content[f"{l}.{h}|r32"]["recovery"] >= REC_CONTENT for (l, h) in CONTENT)}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_program_fit_result_v703", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "joint_value": JOINT_VALUE, "fits": results, "content_hybrids": content},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
