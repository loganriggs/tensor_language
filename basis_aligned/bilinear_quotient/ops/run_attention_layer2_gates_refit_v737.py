#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_refit_halves_excess pred_c_no_overfit pred_d_twenty_readable_heads pred_e_tables_stay_close
"""Attention lane, v737: the five layer-2 gate tables REFITTED in the context of the fifteen readable heads (the fix for v736).

v736: the closed-form layer-2 tables (fitted with native layer-0/1 inputs) cost +0.091 on top of the fifteen readable heads vs +0.042 on
the base program. Fit: inside the fifteen-program, the five heads' A (50304), B (50304) and kappa (as kappa0 (1 + c)) are free parameters,
initialised at v628's tables; Adam 200 steps (tables lr 3e-3 -> 3e-4, kernels 0.01 -> 0.001, batch 8, seed-703 split), validation every 25,
parameters SNAPSHOTTED at the validation minimum; the twenty-head cost and the tables' drift from their initialisation. CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays     native within 0.002 of 3.13241 (instrument)
    pred_b_refit_halves_excess the refitted five cost <= 0.045 on top of the fifteen (from 0.091). Prior: unsure
    pred_c_no_overfit         held-out at every evaluation after step 50 <= the step-50 value + 0.003. Prior: unsure
    pred_d_twenty_readable_heads twenty readable heads (refit) cost <= 0.19. Prior: unsure
    pred_e_tables_stay_close  median over the five heads of corr(A_fit, A_0) and corr(B_fit, B_0) >= 0.9 (the refit adjusts, it does not replace). Prior: likely
PRICE (registered maximum): native 6; kappa_r 2; program 6; fifteen 6; 200 forwards + 200 BACKWARDS; validation 9 x 3 = 27; held-out 9 x 6 = 54; step-0 9;
snapshot 6; total ~320 forwards, 200 backwards; fit parameters 5 x (2 x 50304 + 513). Bars: forwards <= 340, backwards <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_layer2_gates_refit_v737_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
READINGS = {(1, 4): ("same", 0.05, 0.0037), (5, 5): ("induction", 0.10, 0.0071), (5, 7): ("sink", 0.55, 0.0058)}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.layer2_gates_refit_v737"
FORWARDS_MAX, BACKWARDS_MAX = 340, 200
STEPS, TBATCH, EVAL_EVERY, N_VAL = 200, 8, 25, 96
LR_T, LR_T_MIN, LR_K, LR_K_MIN = 3e-3, 3e-4, 0.01, 0.001
FIT_ROWS_ALL = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
T2 = ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"
L2 = {0: 0.0013, 2: 0.0094, 3: 0.0096, 4: 0.0022, 8: 0.0090}
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
L0_SET, L1_SET, MEAN_HEAD = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, EXCESS_MAX, UPTURN_TOL, TWENTY, CORR_MIN = 0.002, 0.045, 0.003, 0.19, 0.9
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_refit_halves_excess": "<= fifteen + 0.045", "pred_c_no_overfit": "no upturn > 0.003", "pred_d_twenty_readable_heads": "<= 0.19", "pred_e_tables_stay_close": "median corr >= 0.9"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_updates": 0, "fit_parameters": 5 * (2 * 50304 + 513),
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS), "model_backwards": BACKWARDS_MAX,
            "bars": {"replay_tol": REPLAY_TOL, "excess_max": EXCESS_MAX, "upturn_tol": UPTURN_TOL, "twenty": TWENTY, "corr_min": CORR_MIN}, "layer2": {str(k): v for k, v in L2.items()}, "steps": STEPS, "readings": {f"{k[0]}.{k[1]}": list(v) for k, v in READINGS.items()}, "gates": {"0": list(L0_SET), "1": list(L1_SET)}, "mean_head": MEAN_HEAD}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    rows = {name: torch.load(p, map_location="cpu").long() for name, p in SETS.items()}
    fit64 = torch.load(FIT_ROWS, map_location="cpu").long()[:64]
    means = torch.load(MEANS, map_location=dev); v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    state = {"idx": None, "programs": {}, "writes": {}, "n": {}, "ablate": set()}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]

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
                    if prog is not None and prog["kind"] in ("same", "induction", "sink"):
                        idx_ = state["idx"]; base = prog["kappa"][dmat][None].expand(Bn, Tn, Tn).clone()
                        if prog["kind"] == "same":
                            base = base + prog["beta"] * (idx_[:, :, None] == idx_[:, None, :]).float()
                        elif prog["kind"] == "induction":
                            prev = torch.cat([torch.full_like(idx_[:, :1], -1), idx_[:, :-1]], 1); base = base + prog["beta"] * (idx_[:, :, None] == prev[:, None, :]).float()
                        else:
                            base[:, :, 0] = prog["beta"]
                        cols.append(torch.where(off[None], base, pat[:, h]))
                    elif prog is not None and prog["kind"] == "gatetab":
                        idx_ = state["idx"]; base = prog["kappa"][dmat][None] * prog["A"][idx_][:, :, None] * prog["B"][idx_][:, None, :]
                        cols.append(torch.where(off[None], base, pat[:, h]))
                    elif prog is not None and prog["kind"] == "runmean":
                        cols.append(torch.where(off[None], (-1.0 / pos.clamp_min(1).float())[None, :, None].expand(Bn, Tn, Tn), pat[:, h]))
                    else:
                        cols.append(pat[:, h] if prog is None else AP.apply_program(pat[:, h], prog, ctx))
                pat = torch.stack(cols, 1)
            z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
            if state["writes"]:
                cols = []
                for h in range(Hn):
                    w = state["writes"].get((l, h)); m = means[f"mean_{l}_{h}"]
                    cols.append(z[:, h] if w is None else (m + ((z[:, h].float() - m) @ w[0]) @ w[1]).to(z.dtype))
                z = torch.stack(cols, 1)
            if state["ablate"]:
                z = z.clone()
                for (al, h) in state["ablate"]:
                    if al == l:
                        z[:, h] = means[f"mean_{l}_{h}"]
            return z
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(rs, progs=None, writes=None):
        state["programs"] = progs or {}; state["writes"] = writes or {}; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rs.shape[0], EBATCH):
                idx = rs[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rs[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["programs"] = {}; state["writes"] = {}
        return total / n, fw

    fac = torch.load(PROGS718, map_location=dev)["factored"]
    maps = {k: {n_: (fac[f"{k[0]}.{k[1]}"][n_][0].to(dev).float() @ fac[f"{k[0]}.{k[1]}"][n_][1].to(dev).float()) for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in ALL}
    kappa = {k: fac[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float() for k in ALL}
    acc = {k: 0 for k in ALL}; nb = 0
    with torch.no_grad():
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
            Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            for k in ALL:
                acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps[k], hd, causal), dmat, off, Tn)
            nb += 1
    kr = {k: acc[k] / nb for k in ALL}

    def program(kap):
        return {k: {"kind": "lowrank", "kappa": kap[k], "kappa_r": kr[k], "maps": maps[k]} for k in ALL}

    ev7 = rows["skip7000"]; nat, fw = ce(ev7); forwards += fw
    base_prog = program(kappa); c_prog, fw = ce(ev7, base_prog); forwards += fw
    kappa702 = torch.load(ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt", map_location=dev)

    def with_readings(keys):
        p = dict(base_prog)
        for k in keys:
            kind, beta, _ = READINGS[k]; p[k] = {"kind": kind, "beta": beta, "kappa": kappa702[f"kappa_{k[0]}_{k[1]}"].float()}
        return p

    t0_ = torch.load(T0, map_location=dev); t1_ = torch.load(T1, map_location=dev)
    gates = {(0, h): {"kind": "gatetab", "A": t0_[f"head{h}_A"].float(), "B": t0_[f"head{h}_B"].float(), "kappa": t0_[f"head{h}_kappa"].float()} for h in L0_SET}
    gates.update({(1, h): {"kind": "gatetab", "A": t1_[f"head{h}_A"].float(), "B": t1_[f"head{h}_B"].float(), "kappa": t1_[f"head{h}_kappa"].float()} for h in L1_SET})
    gates[(1, MEAN_HEAD)] = {"kind": "runmean"}

    def with_gates(keys, readings=()):
        p = with_readings(list(readings))
        for k in keys:
            p[k] = gates[k]
        return p

    import math
    t2_ = torch.load(T2, map_location=dev)
    A0 = {h: t2_[f"head{h}_A"].float() for h in L2}; B0 = {h: t2_[f"head{h}_B"].float() for h in L2}; K0 = {h: t2_[f"head{h}_kappa"].float() for h in L2}
    A = {h: A0[h].clone().requires_grad_(True) for h in L2}; Bt = {h: B0[h].clone().requires_grad_(True) for h in L2}; cm = {h: torch.zeros(513, device=dev, requires_grad=True) for h in L2}
    fifteen = with_gates(list(gates), READINGS); c15, fw = ce(ev7, fifteen); forwards += fw

    def twenty():
        p = dict(fifteen)
        for h in L2:
            p[(2, h)] = {"kind": "gatetab", "A": A[h], "B": Bt[h], "kappa": K0[h] * (1 + cm[h])}
        return p

    allfit = torch.cat([torch.load(p_, map_location="cpu").long() for p_ in FIT_ROWS_ALL]); gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fitr = allfit[perm[N_VAL:]]
    params = [A[h] for h in L2] + [Bt[h] for h in L2]; opt = torch.optim.Adam([{"params": params, "lr": LR_T}, {"params": [cm[h] for h in L2], "lr": LR_K}])
    step0, fw = ce(ev7, twenty()); forwards += fw; v0, fw = ce(val, twenty()); forwards += fw
    val_curve, ho_curve = {0: v0}, {0: step0 - nat}; best = (v0, 0, [p_.detach().clone() for p_ in params + [cm[h] for h in L2]])
    gen2 = torch.Generator().manual_seed(737); order = torch.randperm(fitr.shape[0], generator=gen2); backwards = 0
    for step in range(1, STEPS + 1):
        f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS)); opt.param_groups[0]["lr"] = LR_T_MIN + (LR_T - LR_T_MIN) * f; opt.param_groups[1]["lr"] = LR_K_MIN + (LR_K - LR_K_MIN) * f
        s0 = ((step - 1) * TBATCH) % fitr.shape[0]; sel = order[s0:s0 + TBATCH]
        if len(sel) < TBATCH:
            sel = order[:TBATCH]
        idx = fitr[sel, :-1].to(dev); tgt = fitr[sel, 1:].to(dev); state["programs"] = twenty()
        with torch.enable_grad():
            loss = model(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        state["programs"] = {}; forwards += 1; backwards += 1
        if step % EVAL_EVERY == 0:
            v_, fw = ce(val, twenty()); forwards += fw; h_, fw = ce(ev7, twenty()); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - nat
            if v_ < best[0]:
                best = (v_, step, [p_.detach().clone() for p_ in params + [cm[h] for h in L2]])
    with torch.no_grad():
        for p_, q_ in zip(params + [cm[h] for h in L2], best[2]):
            p_.copy_(q_)
    c20, fw = ce(ev7, twenty()); forwards += fw
    corrs = [float(torch.corrcoef(torch.stack([A[h].detach(), A0[h]]))[0, 1]) for h in L2] + [float(torch.corrcoef(torch.stack([Bt[h].detach(), B0[h]]))[0, 1]) for h in L2]
    med_corr = sorted(corrs)[len(corrs) // 2]
    report = {"native": nat, "program_cost": c_prog - nat, "fifteen_cost": c15 - nat, "twenty_step0": step0 - nat, "chosen_step": best[1], "twenty_snapshot": c20 - nat, "excess_over_fifteen": c20 - c15,
              "heldout_curve": {str(k): v for k, v in ho_curve.items()}, "table_corrs": corrs, "median_corr": med_corr, "recovery": 1 - (c20 - nat) / 3.9961}
    print(f"fifteen {c15 - nat:+.4f} | twenty at step 0 {step0 - nat:+.4f} -> snapshot (validation step {best[1]}) {c20 - nat:+.4f} (excess over fifteen {c20 - c15:+.4f}; recovery {1 - (c20 - nat) / 3.9961:.3f}); curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}; median table corr {med_corr:.3f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_refit_halves_excess": (c20 - c15) <= EXCESS_MAX,
                   "pred_c_no_overfit": all(ho_curve[st] <= ho_curve[50] + UPTURN_TOL for st in range(75, STEPS + 1, EVAL_EVERY)), "pred_d_twenty_readable_heads": (c20 - nat) <= TWENTY, "pred_e_tables_stay_close": med_corr >= CORR_MIN}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_layer2_gates_refit_result_v737", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
