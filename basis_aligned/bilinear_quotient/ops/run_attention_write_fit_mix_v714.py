#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_all64_cost pred_c_mix_close_to_all64 pred_d_pattern_write_compose pred_e_values_preserved_scale
"""Attention lane, v714: where does the write side become cheap? Fitted write maps at rank 64 for all heads, and a 32/64 mix by v710's
single-head recoveries; composition with the pattern program; the 24-head check.

v712: fitted rank-32 writes for all 162 heads cost +0.275 and double the heads' values; the QK side at rank 16/64 costs 0.075. Two arms,
each v712's protocol (centered write maps A_h B_h initialised at the head's own write-covariance projection, Adam 300 steps, validation
stopping, patterns native): ALL64 (rank 64 everywhere, 13.3M numbers vs 47.8M) and MIX (rank 32 where v710's single rank-32 recovery >= 0.9
or the head's value < 0.005, else 64). Then for ALL64: manipulability over the 24 most valuable heads and composition with the v706
pattern program (five heads native; 0.084 alone). CE ADDED, lower is better; joint recovery = 1 - cost / 3.996.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_all64_cost            ALL64 held-out at the validation-chosen step <= 0.12. Prior: unsure
    pred_c_mix_close_to_all64    MIX held-out at its chosen step <= ALL64's + 0.03. Prior: unsure
    pred_d_pattern_write_compose combined pattern + ALL64 write cost <= pattern-only + write-only (endpoint) + 0.03 (v712 failed by 0.07 at rank 32). Prior: unsure
    pred_e_values_preserved_scale ALL64: median ratio in-program / native value over the 24 heads within [0.5, 1.5] (v712 at rank 32: 1.91). Prior: unsure
PRICE (registered maximum): native 6; covariance capture 2; 2 arms x (300 forwards + 300 BACKWARDS + validation 39 + held-out 78 + step-0 9); endpoint 6;
manipulability 144; composition 2 + 6 + 6; total ~1030 forwards, 600 backwards; fit parameters 2.7M. Bars: forwards <= 1050, backwards <= 600.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_write_fit_mix_v714_result.json"
OUT_PT = ROOT / "circuits/followups/attention_write_fit_mix_v714_maps.pt"
V710 = ROOT / "circuits/followups/attention_write_rank_centered_v710_result.json"
PROGS706 = ROOT / "circuits/followups/attention_mixed16_manip_v706_programs.pt"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.write_fit_mix_v714"
FORWARDS_MAX, BACKWARDS_MAX = 1050, 600
ARMS = ("all64", "mix")
FIVE = ((8, 3), (5, 5), (1, 4), (3, 5), (1, 1))
N_MANIP = 24
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 300, 8, 32, 25, 96
LR, LR_MIN = 3e-4, 3e-5
LAYERS = tuple(range(18)); H = 9
REPLAY_TOL, COST_MAX, MIX_TOL, COMPOSE_TOL, RATIO = 0.002, 0.12, 0.03, 0.03, (0.5, 1.5)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_all64_cost": "<= 0.12", "pred_c_mix_close_to_all64": "<= all64 + 0.03", "pred_d_pattern_write_compose": "<= pattern + write + 0.03",
               "pred_e_values_preserved_scale": "median ratio in [0.5, 1.5]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 162 * 2 * 128 * 64 + 162 * 2 * 128 * 48,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN], "arms": list(ARMS),
            "bars": {"replay_tol": REPLAY_TOL, "cost_max": COST_MAX, "mix_tol": MIX_TOL, "compose_tol": COMPOSE_TOL, "ratio": RATIO}, "n_manip": N_MANIP}
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
        eig = {}
        for (l, h) in ALL:
            Wo = blocks[l].attn.c_proj.weight[:, h * hd:(h + 1) * hd].detach().double()
            G = Wo @ (cov[(l, h)] / n_tok[0]) @ Wo.T; evals, Q = torch.linalg.eigh(G); eig[(l, h)] = (Wo, Q[:, torch.argsort(evals, descending=True)], torch.linalg.pinv(Wo))
    v710 = json.load(open(V710))["report"]["heads"]

    def rank_for(arm, k):
        if arm == "all64":
            return 64
        info = v710[f"{k[0]}.{k[1]}"]
        return 32 if (info["value"] < 0.005 or (info["recovery_32"] or 0) >= 0.9) else 64

    results, saved, finals = {}, {}, {}
    for arm in ARMS:
        ranks = {k: rank_for(arm, k) for k in ALL}; counts = {str(r): sum(1 for k in ALL if ranks[k] == r) for r in (32, 64)}
        A, B = {}, {}
        with torch.no_grad():
            for k in ALL:
                Wo, Q, Wp = eig[k]; Qr = Q[:, :ranks[k]]
                A[k] = (Wo.T @ Qr).float().contiguous().requires_grad_(True); B[k] = (Qr.T @ Wp.T).float().contiguous().requires_grad_(True)

        def build():
            return {k: (A[k], B[k]) for k in ALL}

        opt = torch.optim.Adam([m for k in ALL for m in (A[k], B[k])], lr=LR)
        step0, fw = ce(ev, build()); forwards += fw; v0, fw = ce(val, build()); forwards += fw
        val_curve, ho_curve = {0: v0}, {0: step0 - native}
        gen2 = torch.Generator().manual_seed(714); order = torch.randperm(fit.shape[0], generator=gen2)
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
        chosen = min(val_curve, key=val_curve.get); ho_at = ho_curve[chosen]; numbers = sum(ranks[k] * (hd + D) for k in ALL)
        results[arm] = {"ranks": {f"{k[0]}.{k[1]}": ranks[k] for k in ALL}, "rank_counts": counts, "closed_form_cost": step0 - native, "chosen_step": chosen, "heldout_at_chosen": ho_at,
                        "recovery_at_chosen": 1 - ho_at / JOINT_VALUE, "heldout_curve": {str(k): v for k, v in ho_curve.items()}, "write_numbers": numbers}
        saved[arm] = {f"{k[0]}.{k[1]}": {"A": A[k].detach().cpu(), "B": B[k].detach().cpu()} for k in ALL}; finals[arm] = build()
        print(f"[{arm}] ranks {counts}: closed-form {step0 - native:+.4f} -> fitted (validation step {chosen}) {ho_at:+.4f}, joint recovery {1 - ho_at / JOINT_VALUE:.3f}; write numbers {numbers / 1e6:.1f}M vs 47.8M; curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}")
    final = finals["all64"]; c_end, fw = ce(ev, final); forwards += fw
    top = sorted(v701, key=v701.get, reverse=True)[:N_MANIP]; manip = {}
    for key in top:
        l, h = map(int, key.split(".")); state["ablate"] = {(l, h)}; c_, fw = ce(ev, final); forwards += fw; state["ablate"] = set()
        manip[key] = {"native_value": v701[key], "in_program_value": c_ - c_end, "ratio": (c_ - c_end) / v701[key]}
    rx = torch.tensor([manip[k]["native_value"] for k in top]).argsort().argsort().double(); ry = torch.tensor([manip[k]["in_program_value"] for k in top]).argsort().argsort().double()
    rx -= rx.mean(); ry -= ry.mean(); rho = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)); rs = sorted(manip[k]["ratio"] for k in top); med_ratio = rs[len(rs) // 2]
    print("[all64] manipulability: " + " ".join(f"{k}:{manip[k]['native_value']:.3f}->{manip[k]['in_program_value']:.3f}" for k in top) + f"; Spearman {rho:.3f}, median ratio {med_ratio:.2f}")
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
    print(f"[compose] pattern-only {c_pat - native:+.4f} | write-only (endpoint) {c_end - native:+.4f} | pattern + write {c_both - native:+.4f} (sum {c_pat + c_end - 2 * native:+.4f}); joint recovery {1 - (c_both - native) / JOINT_VALUE:.3f}; numbers pattern 22.1M + write 13.3M")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    disk_guard.guard_torch_save(saved, str(OUT_PT), "v714 fitted write maps (endpoint), arms all64 / mix")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_all64_cost": results["all64"]["heldout_at_chosen"] <= COST_MAX,
                   "pred_c_mix_close_to_all64": results["mix"]["heldout_at_chosen"] <= results["all64"]["heldout_at_chosen"] + MIX_TOL,
                   "pred_d_pattern_write_compose": (c_both - native) <= (c_pat - native) + (c_end - native) + COMPOSE_TOL, "pred_e_values_preserved_scale": RATIO[0] <= med_ratio <= RATIO[1]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_write_fit_mix_result_v714", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "joint_value": JOINT_VALUE, "arms": results, "all64_endpoint_cost": c_end - native, "manipulability": manip, "spearman": rho, "median_ratio": med_ratio,
                                          "pattern_only_cost": c_pat - native, "combined_cost": c_both - native, "combined_recovery": 1 - (c_both - native) / JOINT_VALUE},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
