#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_joint_costs_replay pred_b_isolated_r32_halves pred_c_full_rank_near_zero pred_d_r8_worse_than_r32 pred_e_low_value_head_fixed
"""Is the diffuse descent a limit of the JOINT fit or of the PROGRAM CLASS? (v777). v776: the fitted rank-32 programs of Pythia-160m (final)
cost +0.253 summed over heads, spread over dozens of heads, including head 1.10 (+0.024, worth 0.006) and 0.2 (+0.014, worth 0.090). Here the
six costliest heads (1.10, 0.2, 0.0, 9.3, 4.1, 7.11) are each fitted IN ISOLATION (all other heads native) at rank 8, 32 and 64 (= full head
dim: the content term can be exact), Adam 150 steps from the SVD-init closed form, snapshot at the validation minimum, priced on the 193
held-out rows; kappa_r recomputed each evaluation as in the joint fit. Row-centred protocol. If isolated rank-32 fits reach a fraction of the
joint-fit cost, the joint fit (144 heads, 300 steps, 8 rows per step) is what limits the ladder; if rank 64 is also costly, the kernel +
low-rank content class is what the late heads escape.
PREDICTIONS (scored as written; failures preserved)
    pred_a_joint_costs_replay     the saved joint program applied to each of the six heads alone replays v776's isolated costs within 0.003. Prior: likely
    pred_b_isolated_r32_halves    isolated rank-32 fit cost <= 0.5 x the joint-fit isolated cost for >= 4 of 6 heads. Prior: unsure
    pred_c_full_rank_near_zero    isolated rank-64 fit cost <= 0.003 for all six heads. Prior: likely
    pred_d_r8_worse_than_r32      isolated rank-8 cost > isolated rank-32 cost for >= 5 of 6 heads. Prior: likely
    pred_e_low_value_head_fixed   head 1.10 (value 0.006): isolated rank-32 cost <= 0.006 (no worse than deleting it). Prior: unsure
PRICE (registered maximum): native 7 + kappa_r 4 + replay 6 x 7 = 53; fits: 6 heads x 3 ranks x (kappa_r 4 + step-0 val 7 + ho 7 + 150 steps + 3 x (4 + 7 + 7) + final 4 + 7) = 18 x 233 = 4194;
total ~4250 forwards, 2700 backwards. Bars: forwards <= 4400, backwards <= 2700.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import torch
import dod_battery
import pythia_backend as PB

ROOT = dod_battery.ROOT
TAG = "pythia160m_isolated_v777"
OUT = ROOT / f"circuits/followups/{TAG}_result.json"
REPO = "EleutherAI/pythia-160m"; PREV = "pythia160m_v760"; PERHEAD = "pythia160m_perhead_v776"
HEADS = [(1, 10), (0, 2), (0, 0), (9, 3), (4, 1), (7, 11)]
RANKS = (8, 32, 64)
FIT_ROWS = (ROOT / ".rowcache/pythia_fineweb_n480_skip80.pt", ROOT / ".rowcache/pythia_fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/pythia_fineweb_n192_skip7000.pt"
CANDIDATE_ID = f"pythia.isolated_{TAG}"
FORWARDS_MAX, BACKWARDS_MAX = 4400, 2700
EBATCH, KBATCH, TBATCH, STEPS, EVAL_EVERY, N_VAL = 32, 16, 8, 150, 50, 96
LR_MAP, LR_MAP_MIN, LR_K, LR_K_MIN = 3e-4, 3e-5, 0.01, 0.001
REPLAY_TOL, HALVE, N_HALVE, ZERO, N_R8, LOWV = 0.003, 0.5, 4, 0.003, 5, 0.006
PREDICTIONS = {"pred_a_joint_costs_replay": "+-0.003 x6", "pred_b_isolated_r32_halves": "<= 0.5x joint for >= 4/6", "pred_c_full_rank_near_zero": "<= 0.003 x6", "pred_d_r8_worse_than_r32": ">= 5/6", "pred_e_low_value_head_fixed": "1.10 r32 <= 0.006"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": "per-head factored q/k maps + kernel multiplier, 18 fits",
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "model": REPO, "heads": [f"{l}.{h}" for l, h in HEADS], "ranks": list(RANKS),
            "bars": {"replay_tol": REPLAY_TOL, "halve": HALVE, "n_halve": N_HALVE, "zero": ZERO, "n_r8": N_R8, "lowv": LOWV}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter(); dev = "cuda"; forwards = 0; backwards = 0
    prev = torch.load(ROOT / f"circuits/followups/{PREV}_programs.pt", map_location="cpu"); kappa0 = {k: prev["kappa0"][f"{k[0]}.{k[1]}"].to(dev) for k in HEADS}; joint32 = prev["programs"]["32"]
    per = json.load(open(ROOT / f"circuits/followups/{PERHEAD}_result.json"))["report"]["final"]; iso_joint = {f"{l}.{h}": per["isolated_cost"][f"{l}.{h}"] for l, h in HEADS}; value = {f"{l}.{h}": per["value"][f"{l}.{h}"] for l, h in HEADS}
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    model = PB.load(REPO, dev); L, H, D, hd, rot = PB.geometry(model); scaling = model.gpt_neox.layers[0].attention.scaling
    state = PB.instrument(model)
    native, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw
    print(f"[{TAG}] native {native:.5f} | heads {[f'{l}.{h}' for l, h in HEADS]} | joint-fit isolated costs {[round(iso_joint[f'{l}.{h}'], 4) for l, h in HEADS]}")

    def row_centre(X):
        T = X.shape[-1]; pos = torch.arange(T, device=X.device); rep = ((pos[:, None] > pos[None, :]) & (pos[None, :] > 0)).float()
        m = (X * rep[None]).sum(-1) / rep.sum(-1).clamp_min(1)[None]
        return X - m[:, :, None], m

    def offset_mean(Pm, T):
        Pm, _ = row_centre(Pm)
        pos = torch.arange(T, device=dev); dmat = pos[:, None] - pos[None, :]; qm = (pos >= 8)[:, None] & (dmat > 0) & (pos[None, :] > 0); dflat = dmat[qm]
        X = Pm[:, qm]; counts = torch.bincount(dflat, minlength=513).double() * X.shape[0]
        sums_ = torch.zeros(513, dtype=torch.float64, device=dev).index_add_(0, dflat, X.double().sum(0))
        return torch.where(counts > 0, sums_ / counts.clamp_min(1), torch.zeros_like(sums_)).float()

    cur = {}       # the single active program: {"head": (l,h), "q": (U,V), "k": (U,V), "bq", "bk", "kappa": tensor or (kappa0, cmul), "kr": tensor}

    def lowrank_logits(n):
        Uq, Vq = cur["q"]; Uk, Vk = cur["k"]
        q = (n @ Vq.T) @ Uq.T + cur["bq"]; k = (n @ Vk.T) @ Uk.T + cur["bk"]
        q, k = PB.rotary_qk(q, k, state["cos"], state["sin"], rot)
        return torch.einsum("bqd,bkd->bqk", q, k) * scaling

    def kappa_now():
        return cur["kappa"] if torch.is_tensor(cur["kappa"]) else cur["kappa"][0] * (1 + cur["kappa"][1])

    def kappa_r_now():
        acc = 0; nb = 0; fw = 0; state["logit_edit"] = None; l = cur["head"][0]
        with torch.no_grad():
            for s in range(0, 64, KBATCH):
                idx = fit64[s:s + KBATCH, :-1].contiguous().to(dev); model(input_ids=idx); fw += 1
                acc = acc + offset_mean(lowrank_logits(state["n"][l]), idx.shape[1]); nb += 1
        cur["kr"] = acc / nb
        return fw

    def logit_program(l, logits):
        if l != cur["head"][0]:
            return logits
        h = cur["head"][1]; B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0) & (pos[None, :] > 0)
        pr, _ = row_centre(lowrank_logits(state["n"][l])); _, m = row_centre(logits[:, h])
        out = logits.clone(); out[:, h] = torch.where(off[None], m[:, :, None] + kappa_now()[dmat][None] + (pr - cur["kr"][dmat][None]), logits[:, h])
        return out

    def ce_prog(rows):
        state["logit_edit"] = logit_program; out = PB.ce(model, rows, dev, EBATCH); state["logit_edit"] = None; return out

    # ---- replay of the joint-fit programs, one head at a time -----------------------------------------------------------------------
    replay = {}
    for (l, h) in HEADS:
        key = f"{l}.{h}"; jp = joint32[key]; _, bq, _, bk = PB.head_qk(model, l, h)
        cur.update({"head": (l, h), "q": tuple(t.to(dev) for t in jp["q"]), "k": tuple(t.to(dev) for t in jp["k"]), "bq": bq.to(dev), "bk": bk.to(dev), "kappa": jp["kappa"].to(dev)})
        forwards += kappa_r_now(); c_, fw = ce_prog(ev); forwards += fw; replay[key] = c_ - native
    print(f"[{TAG}] replay of joint-fit programs: " + " ".join(f"{k}:{replay[k]:+.4f} (v776 {iso_joint[k]:+.4f})" for k in replay))
    # ---- isolated fits --------------------------------------------------------------------------------------------------------------
    fits = {}
    for (l, h) in HEADS:
        key = f"{l}.{h}"; Wq, bq, Wk, bk = PB.head_qk(model, l, h); fits[key] = {}
        for R in RANKS:
            r = min(R, hd); fac = {}
            for n_, W in (("q", Wq), ("k", Wk)):
                U_, S_, Vh_ = torch.linalg.svd(W.to(dev), full_matrices=False)
                fac[n_] = ((U_[:, :r] * S_[:r].sqrt()).contiguous().requires_grad_(True), (S_[:r].sqrt()[:, None] * Vh_[:r]).contiguous().requires_grad_(True))
            cmul = torch.zeros(513, device=dev, requires_grad=True)
            cur.update({"head": (l, h), "q": fac["q"], "k": fac["k"], "bq": bq.to(dev), "bk": bk.to(dev), "kappa": (kappa0[(l, h)], cmul)})
            params = [fac["q"][0], fac["q"][1], fac["k"][0], fac["k"][1]]
            opt = torch.optim.Adam([{"params": [cmul], "lr": LR_K}, {"params": params, "lr": LR_MAP}])
            forwards += kappa_r_now(); v0, fw = ce_prog(val); forwards += fw; h0, fw = ce_prog(ev); forwards += fw
            curve = {0: h0 - native}; best = (v0, 0, [p.detach().clone() for p in [cmul] + params])
            gen2 = torch.Generator().manual_seed(777 + R + 100 * l + h); order = torch.randperm(fit.shape[0], generator=gen2)
            for step in range(1, STEPS + 1):
                f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS)); opt.param_groups[0]["lr"] = LR_K_MIN + (LR_K - LR_K_MIN) * f; opt.param_groups[1]["lr"] = LR_MAP_MIN + (LR_MAP - LR_MAP_MIN) * f
                s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel = order[s0:s0 + TBATCH]
                if len(sel) < TBATCH:
                    sel = order[:TBATCH]
                ids = fit[sel].contiguous().to(dev); state["logit_edit"] = logit_program
                with torch.enable_grad():
                    loss = model(input_ids=ids, labels=ids).loss; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                state["logit_edit"] = None; forwards += 1; backwards += 1
                if step % EVAL_EVERY == 0:
                    forwards += kappa_r_now(); v_, fw = ce_prog(val); forwards += fw; h_, fw = ce_prog(ev); forwards += fw; curve[step] = h_ - native
                    if v_ < best[0]:
                        best = (v_, step, [p.detach().clone() for p in [cmul] + params])
            with torch.no_grad():
                for p, q in zip([cmul] + params, best[2]):
                    p.copy_(q)
            forwards += kappa_r_now(); c_snap, fw = ce_prog(ev); forwards += fw
            fits[key][str(R)] = {"closed_form_cost": h0 - native, "chosen_step": best[1], "snapshot_cost": c_snap - native, "curve": {str(k): v for k, v in curve.items()}}
            print(f"[{TAG}] head {key} (value {value[key]:.4f}, joint-fit isolated {iso_joint[key]:+.4f}) rank {R}: closed-form {h0 - native:+.4f} -> snapshot (step {best[1]}) {c_snap - native:+.4f}")
    state["_restore"]()
    a = all(abs(replay[k] - iso_joint[k]) <= REPLAY_TOL for k in replay)
    halves = sum(1 for k in fits if fits[k]["32"]["snapshot_cost"] <= HALVE * iso_joint[k]); zero = all(fits[k]["64"]["snapshot_cost"] <= ZERO for k in fits)
    r8w = sum(1 for k in fits if fits[k]["8"]["snapshot_cost"] > fits[k]["32"]["snapshot_cost"])
    predictions = {"pred_a_joint_costs_replay": a, "pred_b_isolated_r32_halves": halves >= N_HALVE, "pred_c_full_rank_near_zero": zero, "pred_d_r8_worse_than_r32": r8w >= N_R8, "pred_e_low_value_head_fixed": fits["1.10"]["32"]["snapshot_cost"] <= LOWV}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX} or {backwards} > {BACKWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": f"pythia_isolated_result_{TAG}", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "value": value, "joint_fit_isolated_cost": iso_joint, "replay": replay, "fits": fits, "n_halved": halves, "n_r8_worse": r8w},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
