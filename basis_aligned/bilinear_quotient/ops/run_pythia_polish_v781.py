#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_joint_replays pred_b_sum_isolated_halves pred_c_joint_recovery pred_d_beats_joint_fit pred_e_values_preserved
"""Does a joint polish of the per-head-refined programs beat either alone? (v781). v780 refined all 144 rank-32 head programs one at a time and
every head improved in isolation (sum of isolated costs +0.2526 -> +0.1676, 40 heads better, none worse) -- but applied TOGETHER they are WORSE
than the jointly fitted programs they started from (+0.2131 -> +0.2659, recovery 0.932 -> 0.915). The head programs' errors are not independent:
the joint fit finds a mutually compensating solution that per-head optimality destroys. This rung asks whether the two can be combined: a short
joint fit (100 Adam steps, evaluation every 25, snapshot at the validation minimum) run twice with the same seed and budget, once warm-started
from v780's refined programs and once from v760's joint snapshot. Arm "refined" tests whether per-head refinement puts the joint optimiser in a
better basin; arm "joint" is the control for the extra 100 steps (v778/v779 showed the joint fit overfits past its minimum, so the control should
not improve). Row-centred protocol; CE ADDED on the 193 held-out rows; recovery = 1 - cost / 3.118.
PREDICTIONS (scored as written; failures preserved)
    pred_a_joint_replays          the v780 refined programs applied jointly replay +0.2659 within 0.003 (instrument + warm start). Prior: likely
    pred_b_refined_start_beats_v760 the refined-start arm's snapshot cost < +0.2131 (the v760 joint fit). Prior: unsure
    pred_c_joint_recovery         the refined-start arm reaches recovery >= 0.95 (cost <= 0.156). Prior: unsure
    pred_d_refined_beats_control  the refined-start arm's snapshot cost < the joint-start arm's. Prior: unsure
    pred_e_values_preserved       median value ratio over the 24 most valuable heads, best arm, in [0.5, 2]. Prior: likely
PRICE (registered maximum): native 7 + 2 arms x (kappa_r 4 + step-0 validation 3 + held-out 7 + 100 steps + 4 x (kappa_r 4 + validation 3 + held-out 7) + final kappa_r 4 + held-out 7) = 2 x 188 = 376;
means 4; manipulability 168; total ~560 forwards, 200 backwards. Bars: forwards <= 900, backwards <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import torch
import dod_battery
import disk_guard
import pythia_backend as PB

ROOT = dod_battery.ROOT
TAG = "pythia160m_polish_v781"
OUT = ROOT / f"circuits/followups/{TAG}_result.json"
OUT_PT = ROOT / f"circuits/followups/{TAG}_programs.pt"
REPO = "EleutherAI/pythia-160m"; PREV = "pythia160m_v760"; REFINED = "pythia160m_sequential_v780"; PERHEAD = "pythia160m_perhead_v776"
FIT_ROWS = (ROOT / ".rowcache/pythia_fineweb_n480_skip80.pt", ROOT / ".rowcache/pythia_fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/pythia_fineweb_n192_skip7000.pt"
CANDIDATE_ID = f"pythia.sequential_{TAG}"
FORWARDS_MAX, BACKWARDS_MAX = 900, 200
EBATCH, KBATCH, TBATCH, STEPS, EVAL_AT, N_VAL, N_MANIP, RANK = 32, 16, 8, 100, (25, 50, 75, 100), 96, 24, 32
LR_MAP, LR_MAP_MIN, LR_K, LR_K_MIN = 1e-4, 1e-5, 0.003, 0.0003
REPLAY_TOL, REC_MIN, RATIO, JOINT_BEFORE = 0.003, 0.95, (0.5, 2.0), 0.2131
PREDICTIONS = {"pred_a_joint_replays": "+-0.003 of +0.2659", "pred_b_refined_start_beats_v760": "< +0.2131", "pred_c_joint_recovery": "recovery >= 0.95", "pred_d_refined_beats_control": "refined < joint start", "pred_e_values_preserved": "median ratio in [0.5, 2]"}

def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": "joint polish of 144 rank-32 programs from two starts",
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "model": REPO, "rank": RANK, "steps": STEPS, "arms": ["refined", "joint"],
            "bars": {"replay_tol": REPLAY_TOL, "rec_min": REC_MIN, "ratio": RATIO, "joint_before": JOINT_BEFORE}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter(); dev = "cuda"; forwards = 0; backwards = 0
    starts = {"refined": torch.load(ROOT / f"circuits/followups/{REFINED}_programs.pt", map_location="cpu"),
              "joint": torch.load(ROOT / f"circuits/followups/{PREV}_programs.pt", map_location="cpu")["programs"]["32"]}
    per = json.load(open(ROOT / f"circuits/followups/{PERHEAD}_result.json"))["report"]["final"]; value = per["value"]; joint_value = per["joint_value"]
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    model = PB.load(REPO, dev); L, H, D, hd, rot = PB.geometry(model); ALL = [(l, h) for l in range(L) for h in range(H)]
    scaling = model.gpt_neox.layers[0].attention.scaling
    bias = {}
    for (l, h) in ALL:
        _, bq, _, bk = PB.head_qk(model, l, h); bias[(l, h)] = (bq.to(dev), bk.to(dev))
    state = PB.instrument(model)
    native, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw
    P = {}

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

    def lowrank_logits(l, h, n):
        p = P[(l, h)]; Uq, Vq = p["q"]; Uk, Vk = p["k"]; bq, bk = bias[(l, h)]
        q = (n @ Vq.T) @ Uq.T + bq; k = (n @ Vk.T) @ Uk.T + bk
        q, k = PB.rotary_qk(q, k, state["cos"], state["sin"], rot)
        return torch.einsum("bqd,bkd->bqk", q, k) * scaling

    def kappa_r():
        acc = {k: 0 for k in ALL}; nb = 0; fw = 0; state["logit_edit"] = None
        with torch.no_grad():
            for s in range(0, 64, KBATCH):
                idx = fit64[s:s + KBATCH, :-1].contiguous().to(dev); model(input_ids=idx); fw += 1; T = idx.shape[1]
                for k in ALL:
                    acc[k] = acc[k] + offset_mean(lowrank_logits(k[0], k[1], state["n"][k[0]]), T)
                nb += 1
        for k in ALL:
            P[k]["kr"] = acc[k] / nb
        return fw

    abl = {"set": set()}; means = {}

    def logit_program(l, logits):
        B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0) & (pos[None, :] > 0)
        cols = []
        for h in range(Hn):
            p = P[(l, h)]; pr, _ = row_centre(lowrank_logits(l, h, state["n"][l])); _, m = row_centre(logits[:, h])
            kap = p["kappa0"] * (1 + p["cmul"]) if p["cmul"] is not None else p["kappa0"]
            cols.append(torch.where(off[None], m[:, :, None] + kap[dmat][None] + (pr - p["kr"][dmat][None]), logits[:, h]))
        return torch.stack(cols, 1)

    def z_edit(l, z):
        for (al, h) in abl["set"]:
            if al == l:
                z = z.clone(); z[:, h] = means[(al, h)]
        return z

    def ce_prog(rows):
        state["logit_edit"] = logit_program; out = PB.ce(model, rows, dev, EBATCH); state["logit_edit"] = None; return out

    results = {}
    for arm, src in starts.items():
        P.clear()
        for (l, h) in ALL:
            sp = src[f"{l}.{h}"]
            P[(l, h)] = {"kappa0": sp["kappa"].to(dev), "cmul": torch.zeros(513, device=dev, requires_grad=True),
                         "q": [t.to(dev).clone().requires_grad_(True) for t in sp["q"]], "k": [t.to(dev).clone().requires_grad_(True) for t in sp["k"]], "kr": None}
        params = [t for k in ALL for t in P[k]["q"] + P[k]["k"]]; cmuls = [P[k]["cmul"] for k in ALL]
        opt = torch.optim.Adam([{"params": cmuls, "lr": LR_K}, {"params": params, "lr": LR_MAP}])
        forwards += kappa_r(); v0, fw = ce_prog(val); forwards += fw; h0, fw = ce_prog(ev); forwards += fw
        curve = {0: h0 - native}; best = (v0, 0, [t.detach().clone() for t in cmuls + params])
        print(f"[{TAG}] arm {arm}: start {h0 - native:+.4f} (recovery {1 - (h0 - native) / joint_value:.3f})")
        gen2 = torch.Generator().manual_seed(781); order = torch.randperm(fit.shape[0], generator=gen2)
        for step in range(1, STEPS + 1):
            f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS)); opt.param_groups[0]["lr"] = LR_K_MIN + (LR_K - LR_K_MIN) * f; opt.param_groups[1]["lr"] = LR_MAP_MIN + (LR_MAP - LR_MAP_MIN) * f
            s0 = ((step - 1) * TBATCH) % fit.shape[0]; ids = fit[order[s0:s0 + TBATCH]].contiguous().to(dev)
            if ids.shape[0] < TBATCH:
                ids = fit[order[:TBATCH]].contiguous().to(dev)
            state["logit_edit"] = logit_program
            with torch.enable_grad():
                loss = model(input_ids=ids, labels=ids).loss; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            state["logit_edit"] = None; forwards += 1; backwards += 1
            if step in EVAL_AT:
                forwards += kappa_r(); v_, fw = ce_prog(val); forwards += fw; h_, fw = ce_prog(ev); forwards += fw; curve[step] = h_ - native
                if v_ < best[0]:
                    best = (v_, step, [t.detach().clone() for t in cmuls + params])
        with torch.no_grad():
            for p_, q_ in zip(cmuls + params, best[2]):
                p_.copy_(q_)
        forwards += kappa_r(); c_, fw = ce_prog(ev); forwards += fw
        results[arm] = {"start_cost": h0 - native, "chosen_step": best[1], "snapshot_cost": c_ - native, "recovery": 1 - (c_ - native) / joint_value, "curve": {str(k): v for k, v in curve.items()}}
        print(f"[{TAG}] arm {arm}: snapshot (step {best[1]}) {c_ - native:+.4f}, recovery {1 - (c_ - native) / joint_value:.3f}; curve {[round(curve[k], 3) for k in sorted(curve)]}")
        if arm == "refined":
            keep = {f"{k[0]}.{k[1]}": {"kappa": (P[k]["kappa0"] * (1 + P[k]["cmul"])).detach().cpu(), "q": [t.detach().cpu() for t in P[k]["q"]], "k": [t.detach().cpu() for t in P[k]["k"]]} for k in ALL}
    # ---- manipulability on the better arm ---------------------------------------------------------------------------------------------
    best_arm = min(results, key=lambda a: results[a]["snapshot_cost"])
    src = starts[best_arm] if best_arm == "joint" else keep
    P.clear()
    for (l, h) in ALL:
        sp = src[f"{l}.{h}"]
        P[(l, h)] = {"kappa0": sp["kappa"].to(dev), "cmul": None, "q": [t.to(dev) for t in sp["q"]], "k": [t.to(dev) for t in sp["k"]], "kr": None}
    forwards += kappa_r()
    capt = {"sums": {k: 0 for k in ALL}, "n": 0, "on": True}

    def z_capture(l, z):
        if capt["on"]:
            for h in range(z.shape[1]):
                capt["sums"][(l, h)] = capt["sums"][(l, h)] + z[:, h].double().sum(dim=(0, 1))
            if l == 0:
                capt["n"] += z.shape[0] * z.shape[2]
            return z
        return z_edit(l, z)

    state["z_edit"] = z_capture; state["logit_edit"] = None
    with torch.no_grad():
        for s in range(0, 64, KBATCH):
            model(input_ids=fit64[s:s + KBATCH, :-1].contiguous().to(dev)); forwards += 1
    capt["on"] = False
    for k in ALL:
        means[k] = (capt["sums"][k] / capt["n"]).float()
    base, fw = ce_prog(ev); forwards += fw
    manip = {}; top = sorted(value, key=value.get, reverse=True)[:N_MANIP]
    for key in top:
        l, h = map(int, key.split(".")); abl["set"] = {(l, h)}; c_, fw = ce_prog(ev); forwards += fw; abl["set"] = set()
        manip[key] = {"native_value": value[key], "in_program_value": c_ - base, "ratio": (c_ - base) / value[key] if value[key] else float("nan")}
    rx = torch.tensor([manip[k]["native_value"] for k in top]).argsort().argsort().double(); ry = torch.tensor([manip[k]["in_program_value"] for k in top]).argsort().argsort().double()
    rx -= rx.mean(); ry -= ry.mean(); rho = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))
    rs = sorted(manip[k]["ratio"] for k in top); med = rs[len(rs) // 2]
    print(f"[{TAG}] best arm {best_arm}: manipulability Spearman {rho:.3f}, median ratio {med:.2f}")
    state["_restore"]()
    disk_guard.guard_torch_save(keep, str(OUT_PT), f"{TAG} polished programs")
    predictions = {"pred_a_joint_replays": abs(results["refined"]["start_cost"] - 0.2659) <= REPLAY_TOL, "pred_b_refined_start_beats_v760": results["refined"]["snapshot_cost"] < JOINT_BEFORE,
                   "pred_c_joint_recovery": results["refined"]["recovery"] >= REC_MIN, "pred_d_refined_beats_control": results["refined"]["snapshot_cost"] < results["joint"]["snapshot_cost"],
                   "pred_e_values_preserved": RATIO[0] <= med <= RATIO[1]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX} or {backwards} > {BACKWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": f"pythia_polish_result_{TAG}", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "joint_value": joint_value, "arms": results, "best_arm": best_arm, "manipulability": manip, "spearman": rho, "median_ratio": med},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
