#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_rank16_cost pred_c_rank32_cost pred_d_rank32_values_preserved_rank pred_e_rank32_values_preserved_scale
"""GPT-2 small replication, v754: the fitted program at UNIFORM rank 16 and rank 32 for all 144 heads (true compression of the 64-wide heads;
v753's 'rank 64' was full rank for the valuable heads).

Same program as v753 (kernel + rank-r content on the pre-softmax logits, column 0 native, diagonal native; kappa0 = v751's kernels; exact-rank
factors; Adam 300 steps; snapshot at the validation minimum). Two arms: RANK16 (144 x 2 x 16 x 832 + 144 x 513 = 3.9M numbers vs 14.2M native
QK, 3.6x fewer) and RANK32 (7.7M, 1.8x). Manipulability over the 24 most valuable heads on the RANK32 snapshot. Held-out skip7000 (native
3.39828); joint value 3.345. CE ADDED. References: bilin18 0.072 at 3.7x fewer (v718); softmax sibling 0.088 at 3.4x fewer (v752).
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        explicit-softmax native within 0.002 of 3.39828 (instrument)
    pred_b_rank16_cost           RANK16 snapshot <= 0.25. Prior: unsure
    pred_c_rank32_cost           RANK32 snapshot <= 0.15. Prior: unsure
    pred_d_rank32_values_preserved_rank RANK32: Spearman over the 24 heads >= 0.6. Prior: unsure
    pred_e_rank32_values_preserved_scale RANK32: median ratio in [0.5, 2]. Prior: unsure
PRICE (registered maximum): native 6; 2 arms x (kappa_r 26 + step-0 9 + 300 forwards + 300 BACKWARDS + validation 36 + held-out 72 + snapshot 8); manipulability 144;
total ~1052 forwards, 600 backwards. Bars: forwards <= 1080, backwards <= 600.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import torch
import torch.nn.functional as F
import dod_battery
import disk_guard
import gpt2_backend as GB

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gpt2_program_fit_uniform_v754_result.json"
OUT_PT = ROOT / "circuits/followups/gpt2_program_fit_uniform_v754_programs.pt"
V750 = ROOT / "circuits/followups/gpt2_values_kernels_v751_result.json"
T750 = ROOT / "circuits/followups/gpt2_values_kernels_v751_tensors.pt"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_REF, JOINT_VALUE = 3.39828, 3.345
CANDIDATE_ID = "gpt2.program_fit_uniform_v754"
FORWARDS_MAX, BACKWARDS_MAX = 1080, 600
ARMS = (16, 32)
N_MANIP = 24
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 300, 8, 32, 25, 96
LR_MAP, LR_MAP_MIN, LR_K, LR_K_MIN = 3e-4, 3e-5, 0.01, 0.001
LAYERS = tuple(range(12)); H = 12
VALUE_FLOOR = 0.005
REPLAY_TOL, R16_MAX, R32_MAX, RHO_MIN, RATIO = 0.002, 0.25, 0.15, 0.6, (0.5, 2.0)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_rank16_cost": "<= 0.25", "pred_c_rank32_cost": "<= 0.15", "pred_d_rank32_values_preserved_rank": "Spearman >= 0.6", "pred_e_rank32_values_preserved_scale": "median ratio in [0.5, 2]"}


def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 73 * 2 * 16 * 832 + 71 * 2 * 64 * 832 + 144 * 513,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "model": GB.REPO, "steps": STEPS,
            "bars": {"replay_tol": REPLAY_TOL, "r16_max": R16_MAX, "r32_max": R32_MAX, "rho_min": RHO_MIN, "ratio": RATIO}, "arms": list(ARMS), "n_manip": N_MANIP}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    model = GB.load("cuda"); dev = "cuda"; D = model.config.n_embd; hd = D // H; blocks = model.transformer.h; forwards = 0; backwards = 0
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    v750 = json.load(open(V750))["report"]; values = v750["mean_ablation_cost"]; t750 = torch.load(T750, map_location=dev)
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    means = {k: t750["means"][f"{k[0]}.{k[1]}"].to(dev).float() for k in ALL}; kappa0 = {k: t750["kappa"][f"{k[0]}.{k[1]}"].to(dev).float() for k in ALL}
    state = GB.instrument(model)
    nbox = {}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: nbox.__setitem__(l, a[0])) for l in LAYERS]
    native, fw = GB.ce(model, ev, dev, EBATCH); forwards += fw
    results, saved_all, manip_all = {}, {}, {}
    for R in ARMS:
        ranks = {k: R for k in ALL}
        # factored maps U (128 x r), V (r x 1152) for c_q, c_k of every head, SVD-initialised (balanced)
        fac = {}; bias = {}
        for (l, h) in ALL:
            at = blocks[l].attn; Wfull = at.c_attn.weight.detach().float(); bfull = at.c_attn.bias.detach().float(); fac[(l, h)] = {}
            for n_, o in (("c_q", 0), ("c_k", D)):
                W = Wfull[:, o + h * hd:o + (h + 1) * hd].T.contiguous(); bias[(l, h, n_)] = bfull[o + h * hd:o + (h + 1) * hd]        # W: [hd, D] as in bilin18's rows
                U_, S_, Vh_ = torch.linalg.svd(W, full_matrices=False); r = min(ranks[(l, h)], hd)
                fac[(l, h)][n_] = ((U_[:, :r] * S_[:r].sqrt()).contiguous().requires_grad_(True), (S_[:r].sqrt()[:, None] * Vh_[:r]).contiguous().requires_grad_(True))
        cmul = {k: torch.zeros(513, device=dev, requires_grad=True) for k in ALL}

        def lowrank_logits(l, h, n):
            """[B, T, T] pre-softmax logits of head (l, h) from its factored maps, rms-norm and rotary as the model."""
            Uq, Vq = fac[(l, h)]["c_q"]; Uk, Vk = fac[(l, h)]["c_k"]
            q = (n @ Vq.T) @ Uq.T + bias[(l, h, "c_q")]; k = (n @ Vk.T) @ Uk.T + bias[(l, h, "c_k")]
            return torch.einsum("bqd,bkd->bqk", q, k) / (hd ** 0.5)

        def offset_mean(P, T):
            pos = torch.arange(T, device=dev); dmat = pos[:, None] - pos[None, :]; qm = (pos >= 8)[:, None] & (dmat > 0); dflat = dmat[qm]
            X = P[:, qm]; counts = torch.bincount(dflat, minlength=513).double() * X.shape[0]
            sums = torch.zeros(513, dtype=torch.float64, device=dev).index_add_(0, dflat, X.double().sum(0))
            return torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums)).float()

        kr = {}

        def kappa_r_now():
            acc = {k: 0 for k in ALL}; nb = 0; fw = 0
            state["logit_edit"] = None
            with torch.no_grad():
                for s in range(0, 64, EBATCH):
                    idx = fit64[s:s + EBATCH, :-1].contiguous().to(dev); model(input_ids=idx); fw += 1
                    T = idx.shape[1]
                    for (l, h) in ALL:
                        acc[(l, h)] = acc[(l, h)] + offset_mean(lowrank_logits(l, h, nbox[l]), T)
                    nb += 1
            for k in ALL:
                kr[k] = acc[k] / nb
            return fw

        prog_on = {"v": False}

        def logit_program(l, logits):
            B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0) & (pos[None, :] > 0)   # column 0 native (the sink)
            cols = []
            for h in range(Hn):
                kap = kappa0[(l, h)] * (1 + cmul[(l, h)]); pr = lowrank_logits(l, h, nbox[l])
                prog = kap[dmat][None] + (pr - kr[(l, h)][dmat][None]); cols.append(torch.where(off[None], prog, logits[:, h]))
            return torch.stack(cols, 1)

        abl = {"set": set()}

        def z_ablate(l, z):
            for (al, h) in abl["set"]:
                if al == l:
                    z = z.clone(); z[:, h] = means[(l, h)]
            return z

        def ce(rows, program):
            state["logit_edit"] = logit_program if program else None; state["z_edit"] = z_ablate
            out = GB.ce(model, rows, dev, EBATCH); state["logit_edit"] = None
            return out

        forwards += kappa_r_now()
        params = [f for k in ALL for pair in fac[k].values() for f in pair]
        opt = torch.optim.Adam([{"params": list(cmul.values()), "lr": LR_K}, {"params": params, "lr": LR_MAP}])
        step0, fw = ce(ev, True); forwards += fw; v0, fw = ce(val, True); forwards += fw
        val_curve, ho_curve = {0: v0}, {0: step0 - native}; best = (v0, 0, [p.detach().clone() for p in list(cmul.values()) + params])
        gen2 = torch.Generator().manual_seed(752); order = torch.randperm(fit.shape[0], generator=gen2)
        for step in range(1, STEPS + 1):
            f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS)); opt.param_groups[0]["lr"] = LR_K_MIN + (LR_K - LR_K_MIN) * f; opt.param_groups[1]["lr"] = LR_MAP_MIN + (LR_MAP - LR_MAP_MIN) * f
            s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel = order[s0:s0 + TBATCH]
            if len(sel) < TBATCH:
                sel = order[:TBATCH]
            ids = fit[sel].contiguous().to(dev); state["logit_edit"] = logit_program; state["z_edit"] = None
            with torch.enable_grad():
                loss = model(input_ids=ids, labels=ids).loss; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            state["logit_edit"] = None; forwards += 1; backwards += 1
            if step % EVAL_EVERY == 0:
                forwards += kappa_r_now()
                v_, fw = ce(val, True); forwards += fw; h_, fw = ce(ev, True); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - native
                if v_ < best[0]:
                    best = (v_, step, [p.detach().clone() for p in list(cmul.values()) + params])
        with torch.no_grad():
            for p, q in zip(list(cmul.values()) + params, best[2]):
                p.copy_(q)
        forwards += kappa_r_now(); c_snap, fw = ce(ev, True); forwards += fw
        results[R] = {"closed_form_cost": step0 - native, "chosen_step": best[1], "snapshot_cost": c_snap - native, "recovery": 1 - (c_snap - native) / JOINT_VALUE, "heldout_curve": {str(k): v for k, v in ho_curve.items()}, "values": sum(2 * min(R, hd) * (hd + D) + 513 for k in ALL)}
        print(f"[gpt2 rank {R}] closed-form (step 0) {step0 - native:+.4f} -> snapshot (validation step {best[1]}) {c_snap - native:+.4f}, joint recovery {1 - (c_snap - native) / JOINT_VALUE:.3f}; curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}")
        if R != 32:
            saved_all[R] = {f"{k[0]}.{k[1]}": {"kappa": (kappa0[k] * (1 + cmul[k])).detach().cpu(), **{n_: (fac[k][n_][0].detach().cpu(), fac[k][n_][1].detach().cpu()) for n_ in fac[k]}} for k in ALL}; continue
        top = sorted(values, key=values.get, reverse=True)[:N_MANIP]; manip = {}
        for key in top:
            l, h = map(int, key.split(".")); abl["set"] = {(l, h)}; c_, fw = ce(ev, True); forwards += fw; abl["set"] = set()
            manip[key] = {"native_value": values[key], "in_program_value": c_ - c_snap, "ratio": (c_ - c_snap) / values[key]}
        rx = torch.tensor([manip[k]["native_value"] for k in top]).argsort().argsort().double(); ry = torch.tensor([manip[k]["in_program_value"] for k in top]).argsort().argsort().double()
        rx -= rx.mean(); ry -= ry.mean(); rho = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)); rs = sorted(manip[k]["ratio"] for k in top); med = rs[len(rs) // 2]
        results[R].update({"manipulability": manip, "spearman": rho, "median_ratio": med}); saved_all[R] = {f"{k[0]}.{k[1]}": {"kappa": (kappa0[k] * (1 + cmul[k])).detach().cpu(), **{n_: (fac[k][n_][0].detach().cpu(), fac[k][n_][1].detach().cpu()) for n_ in fac[k]}} for k in ALL}
        print("manipulability: " + " ".join(f"{k}:{manip[k]['native_value']:.3f}->{manip[k]['in_program_value']:.3f}" for k in top) + f"; Spearman {rho:.3f}, median ratio {med:.2f}")
    state["_restore"]()
    for hh in nhooks:
        hh.remove()
    disk_guard.guard_torch_save({str(R): saved_all[R] for R in ARMS}, str(OUT_PT), "v754 gpt2 uniform-rank programs (snapshots)")
    r16, r32 = results[16], results[32]
    predictions = {"pred_a_native_replays": abs(native - NATIVE_REF) <= REPLAY_TOL, "pred_b_rank16_cost": r16["snapshot_cost"] <= R16_MAX, "pred_c_rank32_cost": r32["snapshot_cost"] <= R32_MAX,
                   "pred_d_rank32_values_preserved_rank": r32["spearman"] >= RHO_MIN, "pred_e_rank32_values_preserved_scale": RATIO[0] <= r32["median_ratio"] <= RATIO[1]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "gpt2_program_fit_uniform_result_v754", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "joint_value": JOINT_VALUE, "arms": {str(R): results[R] for R in ARMS}},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
