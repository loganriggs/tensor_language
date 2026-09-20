#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_composition_additive pred_c_whole_program_cost pred_d_whole_values_preserved_rank pred_e_whole_values_preserved_scale
"""Attention lane, v720: the WHOLE attention program with both sides exact-rank — v718's factored pattern maps (111 heads at 16, 51 at 64;
25.9M) and v716's rank-64 write maps A_h B_h (13.3M) fitted jointly, validation-minimum snapshot, every follow-up on the snapshot.

v716's joint number (+0.161) used free pattern maps that leaned on off-rank components (+0.241 when truncated); v718 showed the factored
form reaches the pattern cost (0.072) exactly. Arm J: initialise pattern factors U, V at v718's snapshot and the write maps at v716's
snapshot A, B; fit all jointly 200 steps (factors lr 1e-4 -> 1e-5, kernels 0.01 -> 0.001, write 1e-4 -> 1e-5), snapshot at the validation
minimum; held-out cost, recovery, manipulability over the 24 most valuable heads. Total numbers 39.2M vs native 95.6M (2.4x), all exact.
CE ADDED, lower is better; joint recovery = 1 - cost / 3.996; ratio = in-program / native mean-ablation value.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_composition_additive  step-0 joint cost <= pattern-only (v718 snapshot, replayed) + write-only (v716 A/B, replayed) + 0.03. Prior: likely
    pred_c_whole_program_cost    snapshot held-out cost <= 0.17 (recovery >= 0.957). Prior: unsure
    pred_d_whole_values_preserved_rank Spearman over the 24 heads >= 0.75. Prior: unsure
    pred_e_whole_values_preserved_scale median ratio in [0.67, 1.5]. Prior: unsure
PRICE (registered maximum): native 6; kappa_r 2; pattern-only 6; write-only 6; J: 200 forwards + 200 BACKWARDS + kappa_r 9 x 2 + validation 27 + held-out 54 + step-0 9;
snapshot kappa_r 2 + eval 6; manipulability 144; total ~490 forwards, 200 backwards. Bars: forwards <= 520, backwards <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import copy, json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_whole_exact_v720_result.json"
OUT_PT = ROOT / "circuits/followups/attention_whole_exact_v720_snapshot.pt"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
SNAP716 = ROOT / "circuits/followups/attention_whole_program_v716_snapshot.pt"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.whole_exact_v720"
FORWARDS_MAX, BACKWARDS_MAX = 520, 200
R_WRITE = 64
N_MANIP = 24
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 200, 8, 32, 25, 96
LR_W, LR_W_MIN = 1e-4, 1e-5
LR_MAP, LR_MAP_MIN, LR_K, LR_K_MIN = 1e-4, 1e-5, 0.01, 0.001
LAYERS = tuple(range(18)); H = 9
REPLAY_TOL, COMPOSE_TOL, COST_J, RHO_J, RATIO_J = 0.002, 0.03, 0.17, 0.75, (0.67, 1.5)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_composition_additive": "<= pattern + write + 0.03", "pred_c_whole_program_cost": "<= 0.17",
               "pred_d_whole_values_preserved_rank": "Spearman >= 0.75", "pred_e_whole_values_preserved_scale": "median ratio in [0.67, 1.5]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 162 * 2 * 128 * R_WRITE + 111 * 4 * 16 * 1280 + 51 * 4 * 64 * 1280 + 162 * 513,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "snapshot": "validation minimum",
            "bars": {"replay_tol": REPLAY_TOL, "compose_tol": COMPOSE_TOL, "cost_j": COST_J, "rho_j": RHO_J, "ratio_j": RATIO_J}, "n_manip": N_MANIP}
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
    snap = torch.load(SNAP716, map_location=dev); fac = torch.load(PROGS718, map_location=dev)["factored"]
    A = {k: snap[f"{k[0]}.{k[1]}"]["A"].to(dev).float().clone().requires_grad_(True) for k in ALL}
    B = {k: snap[f"{k[0]}.{k[1]}"]["B"].to(dev).float().clone().requires_grad_(True) for k in ALL}
    kappa0 = {k: fac[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float() for k in ALL}
    cmul = {k: torch.zeros(513, device=dev, requires_grad=True) for k in ALL}
    facs = {k: {n_: (fac[f"{k[0]}.{k[1]}"][n_][0].to(dev).float().clone().requires_grad_(True), fac[f"{k[0]}.{k[1]}"][n_][1].to(dev).float().clone().requires_grad_(True)) for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in ALL}

    def prod(k):
        return {n_: facs[k][n_][0] @ facs[k][n_][1] for n_ in facs[k]}

    def writes_now():
        return {k: (A[k], B[k]) for k in ALL}

    def fit_loop(tag, params_groups, build_writes, build_progs, refresh, seed):
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

    kr_box = {}
    kr_box["kr"], fw = kappa_r_now({k: prod(k) for k in ALL}); forwards += fw

    def progs_now():
        return {k: {"kind": "lowrank", "kappa": kappa0[k] * (1 + cmul[k]), "kappa_r": kr_box["kr"][k], "maps": prod(k)} for k in ALL}

    def refresh_kr():
        kr_box["kr"], fw = kappa_r_now({k: prod(k) for k in ALL}); return fw

    cP, fw = ce(ev, None, progs_now()); forwards += fw; cW, fw = ce(ev, writes_now()); forwards += fw
    groups_j = [{"params": list(cmul.values()), "lr": LR_K, "lr_max": LR_K, "lr_min": LR_K_MIN},
                {"params": [f for k in ALL for pair in facs[k].values() for f in pair], "lr": LR_MAP, "lr_max": LR_MAP, "lr_min": LR_MAP_MIN},
                {"params": [m for k in ALL for m in (A[k], B[k])], "lr": LR_W, "lr_max": LR_W, "lr_min": LR_W_MIN}]
    resJ = fit_loop("J", groups_j, writes_now, progs_now, refresh_kr, 720)
    cJ, fw = ce(ev, writes_now(), progs_now()); forwards += fw
    manJ, rhoJ, medJ, fw = manipulability(writes_now(), progs_now(), cJ); forwards += fw
    numbers = sum(4 * facs[k]["c_q"][0].shape[1] * (hd + D) + 513 for k in ALL) + 162 * R_WRITE * (hd + D)
    resJ.update({"snapshot_cost": cJ - native, "recovery": 1 - (cJ - native) / JOINT_VALUE, "manipulability": manJ, "spearman": rhoJ, "median_ratio": medJ, "pattern_only_cost": cP - native, "write_only_cost": cW - native, "numbers": numbers})
    print(f"[J] pattern-only (v718 factored snapshot) {cP - native:+.4f} | write-only (v716 A/B) {cW - native:+.4f} | joint at step 0 {resJ['step0']:+.4f} -> snapshot {cJ - native:+.4f} (recovery {1 - (cJ - native) / JOINT_VALUE:.3f}); numbers {numbers / 1e6:.1f}M vs 95.6M, all exact")
    print("[J] manipulability: " + " ".join(f"{k}:{manJ[k]['native_value']:.3f}->{manJ[k]['in_program_value']:.3f}" for k in manJ) + f"; Spearman {rhoJ:.3f}, median ratio {medJ:.2f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    disk_guard.guard_torch_save({f"{k[0]}.{k[1]}": {"A": A[k].detach().cpu(), "B": B[k].detach().cpu(), "kappa": (kappa0[k] * (1 + cmul[k])).detach().cpu(), **{n_: (facs[k][n_][0].detach().cpu(), facs[k][n_][1].detach().cpu()) for n_ in facs[k]}} for k in ALL},
                                str(OUT_PT), "v720 whole-attention program, both sides exact-rank (snapshot)")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_composition_additive": resJ["step0"] <= (cP - native) + (cW - native) + COMPOSE_TOL,
                   "pred_c_whole_program_cost": (cJ - native) <= COST_J, "pred_d_whole_values_preserved_rank": rhoJ >= RHO_J, "pred_e_whole_values_preserved_scale": RATIO_J[0] <= medJ <= RATIO_J[1]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_whole_exact_result_v720", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "joint_value": JOINT_VALUE, "J": resJ},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
