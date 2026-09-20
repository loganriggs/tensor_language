#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_fitted_write_cost pred_c_no_overfit pred_d_pattern_write_compose pred_e_values_preserved_scale
"""Attention lane, v712: the write side FITTED — every head's write as its mean plus a fitted rank-32 map — and the whole attention simplified
(pattern program + write program together).

v710: closed-form centered rank-32 write projections cost 0.124 singly but +0.666 jointly (they stack). This rung fits them jointly, as the
pattern side was fitted (v703/v706): z_h := mean_h + (z_h - mean_h) A_h B_h with A_h (128 x 32), B_h (32 x 128) initialised at the
closed-form projection (A = W_o^T Q_r, B = Q_r^T pinv(W_o)^T; Q_r the top-32 eigenvectors of the head's write covariance on 64 fit rows;
mean from v701). All 162 heads; patterns native during the fit. Adam 300 steps (lr 3e-4 -> 3e-5 cosine, batch 8; 576 fit / 96 validation
rows, seed 703 as v706), validation minimum every 25 steps chooses the reported step. Then (i) manipulability: the 24 most valuable heads
mean-ablated inside the fitted write program (endpoint), (ii) COMPOSITION: the fitted write program together with the v706 pattern program
(endpoint maps, the five heads 8.3/5.5/1.4/3.5/1.1 native as in v707 — held-out cost 0.084 alone). Write numbers: 162 x 32 x 1280 = 6.6M
vs 47.8M native (c_v + c_proj). CE ADDED, lower is better; joint recovery = 1 - cost / 3.996.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_fitted_write_cost     held-out cost at the validation-chosen step <= 0.2 (from 0.666 closed-form). Prior: unsure
    pred_c_no_overfit            held-out at every evaluation after step 50 <= the step-50 value + 0.003. Prior: unsure
    pred_d_pattern_write_compose combined pattern + write cost <= pattern-only (0.084 replayed here) + write-only + 0.03. Prior: unsure
    pred_e_values_preserved_scale median ratio in-program / native value over the 24 heads within [0.5, 2.0] (Spearman reported). Prior: likely
PRICE (registered maximum): native 6; covariance capture 2; 300 forwards + 300 BACKWARDS; validation 13 x 3 = 39; held-out 13 x 6 = 78; endpoint 6;
manipulability 24 x 6 = 144; composition: kappa_r 2 + pattern-only 6 + combined 6; total ~590 forwards, 300 backwards; fit parameters 1.3M. Bars: forwards <= 610, backwards <= 300.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_write_fit_v712_result.json"
OUT_PT = ROOT / "circuits/followups/attention_write_fit_v712_maps.pt"
PROGS706 = ROOT / "circuits/followups/attention_mixed16_manip_v706_programs.pt"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.write_fit_v712"
FORWARDS_MAX, BACKWARDS_MAX = 610, 300
R = 32
FIVE = ((8, 3), (5, 5), (1, 4), (3, 5), (1, 1))
N_MANIP = 24
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 300, 8, 32, 25, 96
LR, LR_MIN = 3e-4, 3e-5
LAYERS = tuple(range(18)); H = 9
REPLAY_TOL, COST_MAX, UPTURN_TOL, COMPOSE_TOL, RATIO = 0.002, 0.2, 0.003, 0.03, (0.5, 2.0)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_fitted_write_cost": "<= 0.2", "pred_c_no_overfit": "no upturn > 0.003", "pred_d_pattern_write_compose": "<= pattern + write + 0.03",
               "pred_e_values_preserved_scale": "median ratio in [0.5, 2]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 162 * 2 * 128 * R,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN], "rank": R,
            "bars": {"replay_tol": REPLAY_TOL, "cost_max": COST_MAX, "upturn_tol": UPTURN_TOL, "compose_tol": COMPOSE_TOL, "ratio": RATIO}, "n_manip": N_MANIP}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    means = torch.load(MEANS, map_location=dev); v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    state = {"idx": None, "programs": {}, "writes": {}, "n": {}, "ablate": set(), "capture": False}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]
    cov = {k: torch.zeros(hd, hd, dtype=torch.float64, device=dev) for k in ALL}; n_tok = [0]

    def make_patched(l):
        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
            if state["programs"]:
                pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                ctx = {"dmat": dmat, "off": off, "pos": pos, "idx": state["idx"], "attn": blocks[l].attn, "n": state["n"][l], "hd": hd, "causal": causal}
                cols = []
                for h in range(Hn):
                    prog = state["programs"].get((l, h))
                    cols.append(pat[:, h] if prog is None else AP.apply_program(pat[:, h], prog, ctx))
                pat = torch.stack(cols, 1)
            z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
            if state["capture"]:
                for h in range(Hn):
                    zc = (z[:, h].float() - means[f"mean_{l}_{h}"]).reshape(-1, Dn).double(); cov[(l, h)] += zc.T @ zc
                if l == 0:
                    n_tok[0] += Bn * Tn
            if state["writes"]:
                cols = []
                for h in range(Hn):
                    w = state["writes"].get((l, h))
                    if w is None:
                        cols.append(z[:, h])
                    else:
                        m = means[f"mean_{l}_{h}"]; cols.append((m + ((z[:, h].float() - m) @ w[0]) @ w[1]).to(z.dtype))
                z = torch.stack(cols, 1)
            for (al, h) in state["ablate"]:
                if al == l:
                    z = z.clone(); z[:, h] = means[f"mean_{l}_{h}"]
            return z
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(rows, writes, progs=None):
        state["writes"] = writes; state["programs"] = progs or {}; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rows.shape[0], EBATCH):
                idx = rows[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rows[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["writes"] = {}; state["programs"] = {}
        return total / n, fw

    native, fw = ce(ev, {}); forwards += fw
    with torch.no_grad():
        state["capture"] = True
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
        state["capture"] = False
        A, B = {}, {}
        for (l, h) in ALL:
            Wo = blocks[l].attn.c_proj.weight[:, h * hd:(h + 1) * hd].detach().double()
            G = Wo @ (cov[(l, h)] / n_tok[0]) @ Wo.T; evals, Q = torch.linalg.eigh(G); Qr = Q[:, torch.argsort(evals, descending=True)][:, :R]
            Wp = torch.linalg.pinv(Wo)
            A[(l, h)] = (Wo.T @ Qr).float().contiguous().requires_grad_(True); B[(l, h)] = (Qr.T @ Wp.T).float().contiguous().requires_grad_(True)

    def build():
        return {k: (A[k], B[k]) for k in ALL}

    opt = torch.optim.Adam([m for k in ALL for m in (A[k], B[k])], lr=LR)
    step0, fw = ce(ev, build()); forwards += fw; v0, fw = ce(val, build()); forwards += fw
    val_curve, ho_curve = {0: v0}, {0: step0 - native}
    gen2 = torch.Generator().manual_seed(712); order = torch.randperm(fit.shape[0], generator=gen2)
    for step in range(1, STEPS + 1):
        f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS)); opt.param_groups[0]["lr"] = LR_MIN + (LR - LR_MIN) * f
        s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel_idx = order[s0:s0 + TBATCH]
        if len(sel_idx) < TBATCH:
            sel_idx = order[:TBATCH]
        idx = fit[sel_idx, :-1].to(dev); tgt = fit[sel_idx, 1:].to(dev); state["writes"] = build()
        with torch.enable_grad():
            loss = model(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        state["writes"] = {}; forwards += 1; backwards += 1
        if step % EVAL_EVERY == 0:
            v_, fw = ce(val, build()); forwards += fw; h_, fw = ce(ev, build()); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - native
    chosen = min(val_curve, key=val_curve.get); ho_at = ho_curve[chosen]
    print(f"[write r{R}] closed-form {step0 - native:+.4f} -> fitted (validation step {chosen}) {ho_at:+.4f}, joint recovery {1 - ho_at / JOINT_VALUE:.3f}; write numbers {162 * R * (hd + D) / 1e6:.1f}M vs 47.8M; curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}")
    final = build(); c_end, fw = ce(ev, final); forwards += fw
    top = sorted(v701, key=v701.get, reverse=True)[:N_MANIP]; manip = {}
    for key in top:
        l, h = map(int, key.split(".")); state["ablate"] = {(l, h)}; c_, fw = ce(ev, final); forwards += fw; state["ablate"] = set()
        manip[key] = {"native_value": v701[key], "in_program_value": c_ - c_end, "ratio": (c_ - c_end) / v701[key]}
    rx = torch.tensor([manip[k]["native_value"] for k in top]).argsort().argsort().double(); ry = torch.tensor([manip[k]["in_program_value"] for k in top]).argsort().argsort().double()
    rx -= rx.mean(); ry -= ry.mean(); rho = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)); rs = sorted(manip[k]["ratio"] for k in top); med_ratio = rs[len(rs) // 2]
    print("[write] manipulability: " + " ".join(f"{k}:{manip[k]['native_value']:.3f}->{manip[k]['in_program_value']:.3f}" for k in top) + f"; Spearman {rho:.3f}, median ratio {med_ratio:.2f}")
    # ---- composition with the v706 pattern program (five heads native) --------------------------------------------------------------
    saved = torch.load(PROGS706, map_location=dev)["mixed16"]; simplified = [k for k in ALL if k not in FIVE]
    maps = {k: {n_: saved[f"{k[0]}.{k[1]}"][n_].to(dev).float() for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in simplified}
    acc = {k: 0 for k in simplified}; nb = 0
    with torch.no_grad():
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
            Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            for k in simplified:
                acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps[k], hd, causal), dmat, off, Tn)
            nb += 1
    pattern = {k: {"kind": "lowrank", "kappa": saved[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float(), "kappa_r": acc[k] / nb, "maps": maps[k]} for k in simplified}
    c_pat, fw = ce(ev, {}, pattern); forwards += fw; c_both, fw = ce(ev, final, pattern); forwards += fw
    print(f"[compose] pattern-only {c_pat - native:+.4f} | write-only (endpoint) {c_end - native:+.4f} | pattern + write {c_both - native:+.4f} (sum {c_pat + c_end - 2 * native:+.4f}); joint recovery {1 - (c_both - native) / JOINT_VALUE:.3f}; numbers pattern 22.1M + write 6.6M")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    disk_guard.guard_torch_save({f"{k[0]}.{k[1]}": {"A": A[k].detach().cpu(), "B": B[k].detach().cpu()} for k in ALL}, str(OUT_PT), "v712 fitted write maps (endpoint)")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_fitted_write_cost": ho_at <= COST_MAX,
                   "pred_c_no_overfit": all(ho_curve[st] <= ho_curve[50] + UPTURN_TOL for st in range(75, STEPS + 1, EVAL_EVERY)),
                   "pred_d_pattern_write_compose": (c_both - native) <= (c_pat - native) + (c_end - native) + COMPOSE_TOL, "pred_e_values_preserved_scale": RATIO[0] <= med_ratio <= RATIO[1]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_write_fit_result_v712", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "joint_value": JOINT_VALUE, "closed_form_cost": step0 - native, "chosen_step": chosen, "heldout_at_chosen": ho_at, "recovery_at_chosen": 1 - ho_at / JOINT_VALUE,
                                          "heldout_curve": {str(k): v for k, v in ho_curve.items()}, "endpoint_cost": c_end - native, "manipulability": manip, "spearman": rho, "median_ratio": med_ratio,
                                          "pattern_only_cost": c_pat - native, "combined_cost": c_both - native, "combined_recovery": 1 - (c_both - native) / JOINT_VALUE, "write_numbers": 162 * R * (hd + D)},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
