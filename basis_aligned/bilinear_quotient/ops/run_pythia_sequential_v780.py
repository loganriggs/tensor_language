#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_joint_replays pred_b_sum_isolated_halves pred_c_joint_recovery pred_d_beats_joint_fit pred_e_values_preserved
"""Refit ALL 144 heads one at a time and re-evaluate the joint program (v780). The mechanism chapter so far: the rank-32 kernel + low-rank
program recovers 0.932 of Pythia-160m's head value when the 144 heads are fitted jointly (v760); the per-head isolated costs sum to +0.253 and
are spread over dozens of heads, many worth almost nothing (v776); the six costliest heads fitted ONE AT A TIME reach 60% lower cost at the same
rank (v777); and more joint steps only overfit (v778). If the joint fit is gradient-starved per head, then refining every head in isolation --
warm-started from the v760 snapshot, 50 Adam steps each, all other heads native during its own fit -- and then applying all 144 refined programs
together should close most of the gap. Per head: kappa multiplier + factored rank-32 q/k maps, validation eval at steps 25 and 50 (96 rows),
best-of {0, 25, 50} kept. Then kappa_r is recomputed for all heads from the refined maps and the joint program is priced on the 193 held-out rows,
with the 24-head manipulability check. Row-centred protocol (kernels and kappa_r on row-centred logits; native row mean added back; column 0 and
the diagonal native). CE ADDED, lower is better; recovery = 1 - cost / joint value (3.118).
PREDICTIONS (scored as written; failures preserved)
    pred_a_joint_replays          the v760 snapshot programs applied jointly replay +0.2131 within 0.003 (instrument + warm start). Prior: likely
    pred_b_sum_isolated_halves    after refinement, the sum of the 144 isolated costs <= 0.5 x v776's +0.2526. Prior: likely (v777 got -60% on six heads)
    pred_c_joint_recovery         refined joint program recovery >= 0.96 (v760: 0.932; i.e. cost <= 0.125). Prior: unsure
    pred_d_beats_joint_fit        refined joint cost < the v760 joint cost +0.2131. Prior: likely
    pred_e_values_preserved       median value ratio over the 24 most valuable heads in [0.5, 2]. Prior: likely
PRICE (registered maximum): native 7 + kappa_r 4 + joint replay 7; 144 heads x (kappa_r 4 + 50 steps + 2 x (kappa_r 4 + validation 3) + final kappa_r 4 + held-out 7) = 144 x 79 = 11376;
final kappa_r 4 + joint 7 + isolated census 144 x 7 = 1008; manipulability 168; total ~12580 forwards, 7200 backwards. Bars: forwards <= 13000, backwards <= 7300.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import torch
import dod_battery
import disk_guard
import pythia_backend as PB

ROOT = dod_battery.ROOT
TAG = "pythia160m_sequential_v780"
OUT = ROOT / f"circuits/followups/{TAG}_result.json"
OUT_PT = ROOT / f"circuits/followups/{TAG}_programs.pt"
REPO = "EleutherAI/pythia-160m"; PREV = "pythia160m_v760"; PERHEAD = "pythia160m_perhead_v776"
FIT_ROWS = (ROOT / ".rowcache/pythia_fineweb_n480_skip80.pt", ROOT / ".rowcache/pythia_fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/pythia_fineweb_n192_skip7000.pt"
CANDIDATE_ID = f"pythia.sequential_{TAG}"
FORWARDS_MAX, BACKWARDS_MAX = 13000, 7300
EBATCH, KBATCH, TBATCH, STEPS, EVAL_AT, N_VAL, N_MANIP, RANK = 32, 16, 8, 50, (25, 50), 96, 24, 32
LR_MAP, LR_MAP_MIN, LR_K, LR_K_MIN = 1e-4, 1e-5, 0.003, 0.0003
REPLAY_TOL, HALVE, REC_MIN, RATIO = 0.003, 0.5, 0.96, (0.5, 2.0)
PREDICTIONS = {"pred_a_joint_replays": "+-0.003", "pred_b_sum_isolated_halves": "<= 0.5 x 0.2526", "pred_c_joint_recovery": "recovery >= 0.96", "pred_d_beats_joint_fit": "< +0.2131", "pred_e_values_preserved": "median ratio in [0.5, 2]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": "144 per-head refinements of rank-32 q/k maps + kernel multipliers",
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "model": REPO, "rank": RANK, "steps_per_head": STEPS,
            "bars": {"replay_tol": REPLAY_TOL, "halve": HALVE, "rec_min": REC_MIN, "ratio": RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter(); dev = "cuda"; forwards = 0; backwards = 0
    prev = torch.load(ROOT / f"circuits/followups/{PREV}_programs.pt", map_location="cpu"); joint32 = prev["programs"]["32"]
    per = json.load(open(ROOT / f"circuits/followups/{PERHEAD}_result.json"))["report"]["final"]
    value = per["value"]; iso_before = per["isolated_cost"]; joint_before = 0.2131; joint_value = per["joint_value"]; sum_before = per["sum_isolated"]
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    model = PB.load(REPO, dev); L, H, D, hd, rot = PB.geometry(model); ALL = [(l, h) for l in range(L) for h in range(H)]
    scaling = model.gpt_neox.layers[0].attention.scaling
    P = {}
    for (l, h) in ALL:
        jp = joint32[f"{l}.{h}"]; _, bq, _, bk = PB.head_qk(model, l, h)
        P[(l, h)] = {"kappa": jp["kappa"].to(dev), "q": [t.to(dev) for t in jp["q"]], "k": [t.to(dev) for t in jp["k"]], "bq": bq.to(dev), "bk": bk.to(dev), "kr": None}
    state = PB.instrument(model)
    native, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw

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
        p = P[(l, h)]; Uq, Vq = p["q"]; Uk, Vk = p["k"]
        q = (n @ Vq.T) @ Uq.T + p["bq"]; k = (n @ Vk.T) @ Uk.T + p["bk"]
        q, k = PB.rotary_qk(q, k, state["cos"], state["sin"], rot)
        return torch.einsum("bqd,bkd->bqk", q, k) * scaling

    def kappa_r(keys):
        """Recompute kappa_r for the given heads (4 forwards; all heads share the same forwards)."""
        acc = {k: 0 for k in keys}; nb = 0; fw = 0; state["logit_edit"] = None
        with torch.no_grad():
            for s in range(0, 64, KBATCH):
                idx = fit64[s:s + KBATCH, :-1].contiguous().to(dev); model(input_ids=idx); fw += 1; T = idx.shape[1]
                for k in keys:
                    acc[k] = acc[k] + offset_mean(lowrank_logits(k[0], k[1], state["n"][k[0]]), T)
                nb += 1
        for k in keys:
            P[k]["kr"] = acc[k] / nb
        return fw

    sel = {"heads": None}; abl = {"set": set()}
    means = {k: None for k in ALL}

    def logit_program(l, logits):
        B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0) & (pos[None, :] > 0)
        cols = []
        for h in range(Hn):
            if sel["heads"] is not None and (l, h) not in sel["heads"]:
                cols.append(logits[:, h]); continue
            p = P[(l, h)]; pr, _ = row_centre(lowrank_logits(l, h, state["n"][l])); _, m = row_centre(logits[:, h])
            cols.append(torch.where(off[None], m[:, :, None] + p["kappa"][dmat][None] + (pr - p["kr"][dmat][None]), logits[:, h]))
        return torch.stack(cols, 1)

    def ce_prog(rows):
        state["logit_edit"] = logit_program; out = PB.ce(model, rows, dev, EBATCH); state["logit_edit"] = None; return out

    forwards += kappa_r(ALL)
    sel["heads"] = None; c_joint0, fw = ce_prog(ev); forwards += fw
    print(f"[{TAG}] native {native:.5f} | joint replay of the v760 snapshot {c_joint0 - native:+.4f} (v760 {joint_before:+.4f}) | joint value {joint_value:.3f}")
    # ---- sequential per-head refinement ---------------------------------------------------------------------------------------------
    refined = {}
    for i, (l, h) in enumerate(ALL):
        key = f"{l}.{h}"; p = P[(l, h)]
        base = [p["kappa"].clone(), p["q"][0].clone(), p["q"][1].clone(), p["k"][0].clone(), p["k"][1].clone()]
        cmul = torch.zeros(513, device=dev, requires_grad=True)
        fq = [p["q"][0].clone().requires_grad_(True), p["q"][1].clone().requires_grad_(True)]
        fk = [p["k"][0].clone().requires_grad_(True), p["k"][1].clone().requires_grad_(True)]
        p["q"], p["k"] = fq, fk; kap0 = base[0]
        p["kappa"] = kap0 * (1 + cmul)
        params = fq + fk; opt = torch.optim.Adam([{"params": [cmul], "lr": LR_K}, {"params": params, "lr": LR_MAP}])
        sel["heads"] = {(l, h)}
        forwards += kappa_r([(l, h)]); v_best, fw = ce_prog(val); forwards += fw
        best = (v_best, 0, [t.detach().clone() for t in [p["kappa"]] + params])
        gen2 = torch.Generator().manual_seed(780 + 100 * l + h); order = torch.randperm(fit.shape[0], generator=gen2)
        for step in range(1, STEPS + 1):
            f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS)); opt.param_groups[0]["lr"] = LR_K_MIN + (LR_K - LR_K_MIN) * f; opt.param_groups[1]["lr"] = LR_MAP_MIN + (LR_MAP - LR_MAP_MIN) * f
            s0 = ((step - 1) * TBATCH) % fit.shape[0]; ids = fit[order[s0:s0 + TBATCH]].contiguous().to(dev)
            if ids.shape[0] < TBATCH:
                ids = fit[order[:TBATCH]].contiguous().to(dev)
            p["kappa"] = kap0 * (1 + cmul); state["logit_edit"] = logit_program
            with torch.enable_grad():
                loss = model(input_ids=ids, labels=ids).loss; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            state["logit_edit"] = None; forwards += 1; backwards += 1
            if step in EVAL_AT:
                with torch.no_grad():
                    p["kappa"] = kap0 * (1 + cmul)
                forwards += kappa_r([(l, h)]); v_, fw = ce_prog(val); forwards += fw
                if v_ < best[0]:
                    best = (v_, step, [t.detach().clone() for t in [p["kappa"]] + params])
        with torch.no_grad():
            p["kappa"] = best[2][0]; p["q"] = [best[2][1], best[2][2]]; p["k"] = [best[2][3], best[2][4]]
        forwards += kappa_r([(l, h)]); c_, fw = ce_prog(ev); forwards += fw
        refined[key] = {"chosen_step": best[1], "isolated_cost": c_ - native, "isolated_cost_before": iso_before[key], "value": value[key]}
        if i % 24 == 0 or refined[key]["isolated_cost_before"] > 0.004:
            print(f"[{TAG}] head {key} ({i + 1}/{len(ALL)}) value {value[key]:.4f}: isolated {iso_before[key]:+.4f} -> {c_ - native:+.4f} (step {best[1]})")
    sel["heads"] = None
    # ---- joint evaluation of the refined programs -----------------------------------------------------------------------------------
    forwards += kappa_r(ALL); c_joint, fw = ce_prog(ev); forwards += fw
    sum_iso = sum(v["isolated_cost"] for v in refined.values()); rec = 1 - (c_joint - native) / joint_value
    print(f"[{TAG}] refined: sum of isolated costs {sum_before:+.4f} -> {sum_iso:+.4f} | JOINT {c_joint0 - native:+.4f} -> {c_joint - native:+.4f} | recovery {1 - (c_joint0 - native) / joint_value:.3f} -> {rec:.3f}")
    # ---- manipulability on the refined joint program ---------------------------------------------------------------------------------
    capt = {"sums": {k: 0 for k in ALL}, "n": 0, "on": False}

    def z_hook(l, z):
        if capt["on"]:
            for h in range(z.shape[1]):
                capt["sums"][(l, h)] = capt["sums"][(l, h)] + z[:, h].double().sum(dim=(0, 1))
            if l == 0:
                capt["n"] += z.shape[0] * z.shape[2]
            return z
        for (al, h) in abl["set"]:
            if al == l:
                z = z.clone(); z[:, h] = means[(al, h)]
        return z

    state["z_edit"] = z_hook; capt["on"] = True; state["logit_edit"] = None
    with torch.no_grad():
        for s in range(0, 64, KBATCH):
            model(input_ids=fit64[s:s + KBATCH, :-1].contiguous().to(dev)); forwards += 1
    capt["on"] = False
    for k in ALL:
        means[k] = (capt["sums"][k] / capt["n"]).float()
    manip = {}
    top = sorted(value, key=value.get, reverse=True)[:N_MANIP]
    for key in top:
        l, h = map(int, key.split(".")); abl["set"] = {(l, h)}; c_, fw = ce_prog(ev); forwards += fw; abl["set"] = set()
        manip[key] = {"native_value": value[key], "in_program_value": c_ - c_joint, "ratio": (c_ - c_joint) / value[key] if value[key] else float("nan")}
    rx = torch.tensor([manip[k]["native_value"] for k in top]).argsort().argsort().double(); ry = torch.tensor([manip[k]["in_program_value"] for k in top]).argsort().argsort().double()
    rx -= rx.mean(); ry -= ry.mean(); rho = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))
    rs = sorted(manip[k]["ratio"] for k in top); med = rs[len(rs) // 2]
    print(f"[{TAG}] refined manipulability: Spearman {rho:.3f}, median ratio {med:.2f}")
    state["_restore"]()
    disk_guard.guard_torch_save({f"{k[0]}.{k[1]}": {"kappa": P[k]["kappa"].cpu(), "q": [t.cpu() for t in P[k]["q"]], "k": [t.cpu() for t in P[k]["k"]]} for k in ALL}, str(OUT_PT), f"{TAG} refined programs")
    predictions = {"pred_a_joint_replays": abs((c_joint0 - native) - joint_before) <= REPLAY_TOL, "pred_b_sum_isolated_halves": sum_iso <= HALVE * sum_before,
                   "pred_c_joint_recovery": rec >= REC_MIN, "pred_d_beats_joint_fit": (c_joint - native) < joint_before, "pred_e_values_preserved": RATIO[0] <= med <= RATIO[1]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX} or {backwards} > {BACKWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": f"pythia_sequential_result_{TAG}", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "joint_value": joint_value, "joint_replay_cost": c_joint0 - native, "joint_refined_cost": c_joint - native, "recovery": rec,
                                          "sum_isolated_before": sum_before, "sum_isolated_after": sum_iso, "heads": refined, "manipulability": manip, "spearman": rho, "median_ratio": med},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
