#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_write64_snapshot_values pred_c_joint_removes_superadditivity pred_d_joint_recovery pred_e_joint_values_preserved_rank
"""Attention lane, v716: the WHOLE attention program — pattern program (v713) and write program (rank 64) fitted together — with every follow-up
measured on the validation-minimum SNAPSHOT (fixing the endpoint defect of v706-v714).

Arm W: the all-heads rank-64 centered write maps (v714 init: closed-form covariance projections), Adam 200 steps at lr 1e-4 -> 1e-5 (v714 at
3e-4 overfit from step 75), validation every 25, parameters SNAPSHOTTED at the validation minimum; manipulability (24 heads) on the snapshot.
Arm J: pattern program initialised at v713's fitted maps (111 heads rank 16, 51 rank 64; kernels kappa) and write maps at W's snapshot, fitted
JOINTLY for 200 steps (pattern lr 1e-4 -> 1e-5 maps / 0.01 -> 0.001 kernels; write lr 1e-4 -> 1e-5), snapshot at the validation minimum;
held-out cost, recovery, manipulability on the snapshot; and the EXACT-RANK number: the pattern maps re-truncated to their nominal ranks
(the fitted maps drift by ~0.1% of their energy) and re-evaluated. Numbers: pattern 25.9M + write 13.3M = 39.2M vs native 95.6M (2.4x).
CE ADDED, lower is better; joint recovery = 1 - cost / 3.996; ratio = in-program / native mean-ablation value.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_write64_snapshot_values W snapshot: median ratio over the 24 heads in [0.67, 1.5] and Spearman >= 0.7 (endpoint in v714: 1.34 / 0.56). Prior: unsure
    pred_c_joint_removes_superadditivity J snapshot cost <= pattern-only (v713 snapshot-less 0.060; replayed here from its endpoint maps) + W snapshot cost + 0.03. Prior: likely
    pred_d_joint_recovery          J snapshot recovery >= 0.96 (cost <= 0.16). Prior: unsure
    pred_e_joint_values_preserved_rank J snapshot: Spearman over the 24 heads >= 0.7. Prior: unsure
PRICE (registered maximum): native 6; covariance capture 2; W: 200 forwards + 200 BACKWARDS + validation 9 x 3 + held-out 9 x 6 + step-0 9; snapshot eval 6;
manipulability 144; pattern-only replay 6 + kappa_r 2; J: 200 + 200 BACKWARDS + kappa_r 9 x 2 + validation 27 + held-out 54 + step-0 9; snapshot 6 + 2; manipulability 144;
exact-rank 2 + 6; total ~1030 forwards, 400 backwards. Bars: forwards <= 1060, backwards <= 400.
"""
from __future__ import annotations
from datetime import datetime, timezone
import copy, json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_whole_program_v716_result.json"
OUT_PT = ROOT / "circuits/followups/attention_whole_program_v716_snapshot.pt"
PROGS713 = ROOT / "circuits/followups/attention_band68_r64_fit_v713_programs.pt"
V713 = ROOT / "circuits/followups/attention_band68_r64_fit_v713_result.json"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.whole_program_v716"
FORWARDS_MAX, BACKWARDS_MAX = 1060, 400
R_WRITE = 64
N_MANIP = 24
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 200, 8, 32, 25, 96
LR_W, LR_W_MIN = 1e-4, 1e-5
LR_MAP, LR_MAP_MIN, LR_K, LR_K_MIN = 1e-4, 1e-5, 0.01, 0.001
LAYERS = tuple(range(18)); H = 9
REPLAY_TOL, RATIO_W, RHO_W, COMPOSE_TOL, COST_J, RHO_J = 0.002, (0.67, 1.5), 0.7, 0.03, 0.16, 0.7
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_write64_snapshot_values": "median ratio in [0.67, 1.5] and Spearman >= 0.7", "pred_c_joint_removes_superadditivity": "<= pattern + write + 0.03",
               "pred_d_joint_recovery": "cost <= 0.16", "pred_e_joint_values_preserved_rank": "Spearman >= 0.7"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 162 * 2 * 128 * R_WRITE + 162 * 4 * 128 * 1152 + 162 * 513,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "snapshot": "validation minimum",
            "bars": {"replay_tol": REPLAY_TOL, "ratio_w": RATIO_W, "rho_w": RHO_W, "compose_tol": COMPOSE_TOL, "cost_j": COST_J, "rho_j": RHO_J}, "n_manip": N_MANIP}
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

    def ce(rows, writes=None, progs=None):
        state["writes"] = writes or {}; state["programs"] = progs or {}; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rows.shape[0], EBATCH):
                idx = rows[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rows[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["writes"] = {}; state["programs"] = {}
        return total / n, fw

    def manipulability(writes, progs, base):
        top = sorted(v701, key=v701.get, reverse=True)[:N_MANIP]; out = {}; fw = 0
        for key in top:
            l, h = map(int, key.split(".")); state["ablate"] = {(l, h)}; c_, f_ = ce(ev, writes, progs); fw += f_; state["ablate"] = set()
            out[key] = {"native_value": v701[key], "in_program_value": c_ - base, "ratio": (c_ - base) / v701[key]}
        rx = torch.tensor([out[k]["native_value"] for k in top]).argsort().argsort().double(); ry = torch.tensor([out[k]["in_program_value"] for k in top]).argsort().argsort().double()
        rx -= rx.mean(); ry -= ry.mean(); rho = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)); rs = sorted(out[k]["ratio"] for k in top)
        return out, rho, rs[len(rs) // 2], fw

    def kappa_r_now(maps_by_head):
        acc = {k: 0 for k in maps_by_head}; nb = 0; fw = 0
        with torch.no_grad():
            for s in range(0, 64, EBATCH):
                idx = fit64[s:s + EBATCH, :-1].to(dev); state["programs"] = {}; state["writes"] = {}; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); fw += 1
                Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                for k, maps in maps_by_head.items():
                    acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], {n_: m.detach() for n_, m in maps.items()}, hd, causal), dmat, off, Tn)
                nb += 1
        return {k: v / nb for k, v in acc.items()}, fw

    native, fw = ce(ev); forwards += fw
    with torch.no_grad():
        state["capture"] = True
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
        state["capture"] = False
        A, B = {}, {}
        for (l, h) in ALL:
            Wo = blocks[l].attn.c_proj.weight[:, h * hd:(h + 1) * hd].detach().double()
            G = Wo @ (cov[(l, h)] / n_tok[0]) @ Wo.T; evals, Q = torch.linalg.eigh(G); Qr = Q[:, torch.argsort(evals, descending=True)][:, :R_WRITE]
            Wp = torch.linalg.pinv(Wo)
            A[(l, h)] = (Wo.T @ Qr).float().contiguous().requires_grad_(True); B[(l, h)] = (Qr.T @ Wp.T).float().contiguous().requires_grad_(True)

    def writes_now():
        return {k: (A[k], B[k]) for k in ALL}

    def fit_loop(tag, params_groups, build_writes, build_progs, refresh, seed):
        """Generic fit with validation-minimum snapshot. refresh() is called every EVAL_EVERY steps (kappa_r). Returns curves, chosen step, snapshot (list of tensors)."""
        nonlocal forwards, backwards
        opt = torch.optim.Adam(params_groups)
        w0 = build_writes(); p0 = build_progs()
        step0, fw = ce(ev, w0, p0); forwards += fw; v0, fw = ce(val, w0, p0); forwards += fw
        val_curve, ho_curve = {0: v0}, {0: step0 - native}; best = (v0, 0, [p.detach().clone() for g in params_groups for p in g["params"]])
        gen2 = torch.Generator().manual_seed(seed); order = torch.randperm(fit.shape[0], generator=gen2)
        for step in range(1, STEPS + 1):
            f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
            for g in opt.param_groups:
                g["lr"] = g["lr_min"] + (g["lr_max"] - g["lr_min"]) * f
            s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel_idx = order[s0:s0 + TBATCH]
            if len(sel_idx) < TBATCH:
                sel_idx = order[:TBATCH]
            idx = fit[sel_idx, :-1].to(dev); tgt = fit[sel_idx, 1:].to(dev); state["writes"] = build_writes(); state["programs"] = build_progs()
            with torch.enable_grad():
                loss = model(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            state["writes"] = {}; state["programs"] = {}; forwards += 1; backwards += 1
            if step % EVAL_EVERY == 0:
                fw = refresh(); forwards += fw
                v_, fw = ce(val, build_writes(), build_progs()); forwards += fw; h_, fw = ce(ev, build_writes(), build_progs()); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - native
                if v_ < best[0]:
                    best = (v_, step, [p.detach().clone() for g in params_groups for p in g["params"]])
        with torch.no_grad():
            for p, q in zip([p for g in params_groups for p in g["params"]], best[2]):
                p.copy_(q)
        fw = refresh(); forwards += fw
        print(f"[{tag}] step 0 {step0 - native:+.4f} -> validation minimum at step {best[1]}, held-out there {ho_curve[best[1]]:+.4f}; curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}")
        return {"step0": step0 - native, "chosen_step": best[1], "heldout_at_chosen": ho_curve[best[1]], "heldout_curve": {str(k): v for k, v in ho_curve.items()}}

    # ---- Arm W: write maps alone -------------------------------------------------------------------------------------------------------
    groups_w = [{"params": [m for k in ALL for m in (A[k], B[k])], "lr": LR_W, "lr_max": LR_W, "lr_min": LR_W_MIN}]
    resW = fit_loop("W", groups_w, writes_now, dict, lambda: 0, 716)
    cW, fw = ce(ev, writes_now()); forwards += fw
    manW, rhoW, medW, fw = manipulability(writes_now(), {}, cW); forwards += fw
    resW.update({"snapshot_cost": cW - native, "recovery": 1 - (cW - native) / JOINT_VALUE, "manipulability": manW, "spearman": rhoW, "median_ratio": medW, "numbers": 162 * R_WRITE * (hd + D)})
    print(f"[W] snapshot cost {cW - native:+.4f}, recovery {1 - (cW - native) / JOINT_VALUE:.3f}; Spearman {rhoW:.3f}, median ratio {medW:.2f}")
    # ---- Arm J: pattern (v713 init) + write (W snapshot) jointly ----------------------------------------------------------------------------
    saved = torch.load(PROGS713, map_location=dev)["band68_r64"]; ranks = json.load(open(V713))["report"]["arms"]["band68_r64"]["ranks"]
    kappa0 = {k: saved[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float() for k in ALL}
    cmul = {k: torch.zeros(513, device=dev, requires_grad=True) for k in ALL}
    maps = {k: {n_: saved[f"{k[0]}.{k[1]}"][n_].to(dev).float().clone().requires_grad_(True) for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in ALL}
    kr_box = {}
    kr_box["kr"], fw = kappa_r_now(maps); forwards += fw

    def progs_now():
        return {k: {"kind": "lowrank", "kappa": kappa0[k] * (1 + cmul[k]), "kappa_r": kr_box["kr"][k], "maps": maps[k]} for k in ALL}

    def refresh_kr():
        kr_box["kr"], fw = kappa_r_now(maps); return fw

    cP, fw = ce(ev, None, progs_now()); forwards += fw
    groups_j = [{"params": list(cmul.values()), "lr": LR_K, "lr_max": LR_K, "lr_min": LR_K_MIN},
                {"params": [m for mm in maps.values() for m in mm.values()], "lr": LR_MAP, "lr_max": LR_MAP, "lr_min": LR_MAP_MIN},
                {"params": [m for k in ALL for m in (A[k], B[k])], "lr": LR_W, "lr_max": LR_W, "lr_min": LR_W_MIN}]
    resJ = fit_loop("J", groups_j, writes_now, progs_now, refresh_kr, 717)
    cJ, fw = ce(ev, writes_now(), progs_now()); forwards += fw
    manJ, rhoJ, medJ, fw = manipulability(writes_now(), progs_now(), cJ); forwards += fw
    # exact-rank number: pattern maps re-truncated to nominal ranks
    with torch.no_grad():
        for k in ALL:
            r = ranks[f"{k[0]}.{k[1]}"]
            for n_ in maps[k]:
                U, S, Vh = torch.linalg.svd(maps[k][n_], full_matrices=False); maps[k][n_].copy_((U[:, :r] * S[:r]) @ Vh[:r])
    fw = refresh_kr(); forwards += fw
    cX, fw = ce(ev, writes_now(), progs_now()); forwards += fw
    numbers = sum(4 * ranks[f"{k[0]}.{k[1]}"] * (hd + D) + 513 for k in ALL) + 162 * R_WRITE * (hd + D)
    resJ.update({"snapshot_cost": cJ - native, "recovery": 1 - (cJ - native) / JOINT_VALUE, "manipulability": manJ, "spearman": rhoJ, "median_ratio": medJ, "pattern_only_init_cost": cP - native,
                 "exact_rank_cost": cX - native, "numbers": numbers})
    print(f"[J] pattern-only (v713 endpoint maps) {cP - native:+.4f} | W snapshot {cW - native:+.4f} | joint at step 0 {resJ['step0']:+.4f} -> snapshot {cJ - native:+.4f} (recovery {1 - (cJ - native) / JOINT_VALUE:.3f}); exact-rank {cX - native:+.4f}; numbers {numbers / 1e6:.1f}M vs 95.6M")
    print("[J] manipulability: " + " ".join(f"{k}:{manJ[k]['native_value']:.3f}->{manJ[k]['in_program_value']:.3f}" for k in manJ) + f"; Spearman {rhoJ:.3f}, median ratio {medJ:.2f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    disk_guard.guard_torch_save({f"{k[0]}.{k[1]}": {"A": A[k].detach().cpu(), "B": B[k].detach().cpu(), "kappa": (kappa0[k] * (1 + cmul[k])).detach().cpu(), **{n_: m.detach().cpu() for n_, m in maps[k].items()}} for k in ALL},
                                str(OUT_PT), "v716 whole-attention program (snapshot; pattern maps re-truncated)")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_write64_snapshot_values": RATIO_W[0] <= medW <= RATIO_W[1] and rhoW >= RHO_W,
                   "pred_c_joint_removes_superadditivity": (cJ - native) <= (cP - native) + (cW - native) + COMPOSE_TOL, "pred_d_joint_recovery": (cJ - native) <= COST_J, "pred_e_joint_values_preserved_rank": rhoJ >= RHO_J}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_whole_program_result_v716", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "joint_value": JOINT_VALUE, "W": resW, "J": resJ},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
