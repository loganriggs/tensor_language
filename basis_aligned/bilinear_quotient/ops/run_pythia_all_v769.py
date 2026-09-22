#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays pred_b_kernel_energy_top4 pred_c_rank32_cost pred_d_rank16_cost pred_e_rank32_values_preserved_scale
"""Pythia-160m EARLY CHECKPOINT (revision step64000; training-progress curve between step 4000 and the final 143000) — tests whether training amount drives read-side compressibility (v760 final checkpoint: rank 32 rec 0.932, rank 16 0.876). All-in-one, ROW-CENTRED logits (v755's instrument failure: Pythia's raw logits carry per-query-row constants of 1e4-1e5 — massive activations — to which softmax is invariant but which contaminated the per-offset kernels; kernels, kappa_r and the program now act on row-centred logits and add the native row mean back) (Logan, 22 Sep: 'try a few sets of pythia models, then other models, to test the specific hypothesis of why
we're seeing what we're seeing'). MODEL = EleutherAI/pythia-160m (12 x 12 x 768, softmax, ROTARY on 16 of 64 head dims, GPT-NeoX tokenizer;
rows: the same FineWeb text as the GPT-2 caches re-tokenized, 481 fit / 193 skip11000 / 193 held-out skip7000 rows of 513 tokens).

Hypothesis under test: GPT-2 small's poor read-side compressibility (rank 32: +0.160, recovery 0.952; rank 16: +0.322) comes from its
learned ABSOLUTE positions (+ first-token sink), not from softmax or from its size; Pythia-160m has GPT-2's size and softmax but rotary
positions. Stages, same protocol as v751 / v754: native (stock vs explicit instrument); mean-ablation value of every head (means over the fit
rows) and the joint value; closed-form per-offset logit kernels (queries >= 8, 64 rows) as the kernels-only program, with and without the
position-0 column native; the SVD shape ladder (1, 2, 4, 8, 16) inside kernels-only; then the FITTED kernel + exact-rank content program at
uniform rank 16 and rank 32 (column 0 native, diagonal native; Adam 300 steps, snapshot at the validation minimum) and the 24-head
manipulability check on the rank-32 snapshot. CE ADDED on the Pythia-tokenized held-out rows (not directly comparable in nats to the GPT-2
numbers; recoveries and compression ratios are).
PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays   explicit native CE within 0.002 of the stock native CE (instrument)
    pred_b_kernel_energy_top4   top-4 SVD shapes carry >= 0.90 of the kernels' energy. Prior: likely
    pred_c_rank32_cost          rank-32 snapshot recovery >= 0.96 (GPT-2: 0.952; bilinear twin at half rank: 0.974). Prior: unsure
    pred_d_rank16_cost          rank-16 snapshot recovery >= 0.92 (GPT-2: 0.904). Prior: unsure
    pred_e_rank32_values_preserved_scale rank-32: median value ratio in [0.5, 2] over the 24 most valuable heads. Prior: likely
PRICE (registered maximum): stock 7 + explicit 7; means 16; kernels 2; 144 x 7 = 1008; all-ablated 7; kernels-only 7 + col0 7; ladder 35; 2 arms x (kappa_r 26 +
step-0 10 + 300 + validation 39 + held-out 91 + snapshot 9) = 950; manipulability 168; total ~2214 forwards, 600 backwards. Bars: forwards <= 2300, backwards <= 600.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import torch
import dod_battery
import disk_guard
import pythia_backend as PB

REPO = "EleutherAI/pythia-160m"
REVISION = "step64000"
TAG = "pythia160m_step64000_v769"
ROOT = dod_battery.ROOT
OUT = ROOT / f"circuits/followups/{TAG}_result.json"
OUT_PT = ROOT / f"circuits/followups/{TAG}_programs.pt"
FIT_ROWS = (ROOT / ".rowcache/pythia_fineweb_n480_skip80.pt", ROOT / ".rowcache/pythia_fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/pythia_fineweb_n192_skip7000.pt"
CANDIDATE_ID = f"pythia.all_{TAG}"
FORWARDS_MAX, BACKWARDS_MAX = 2300, 600
EBATCH, TBATCH, STEPS, EVAL_EVERY, N_VAL, N_MANIP = 32, 8, 300, 25, 96, 24
LR_MAP, LR_MAP_MIN, LR_K, LR_K_MIN = 3e-4, 3e-5, 0.01, 0.001
KS, ARMS = (1, 2, 4, 8, 16), (16, 32)
REPLAY_TOL, E4_MIN, REC32, REC16, RATIO = 0.002, 0.90, 0.96, 0.92, (0.5, 2.0)
PREDICTIONS = {"pred_a_instrument_replays": "+-0.002", "pred_b_kernel_energy_top4": ">= 0.90", "pred_c_rank32_cost": "recovery >= 0.96", "pred_d_rank16_cost": "recovery >= 0.92", "pred_e_rank32_values_preserved_scale": "median ratio in [0.5, 2]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": "2 arms of factored q/k maps + kernels",
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "model": REPO, "revision": REVISION, "arms": list(ARMS), "ks": list(KS),
            "bars": {"replay_tol": REPLAY_TOL, "e4_min": E4_MIN, "rec32": REC32, "rec16": REC16, "ratio": RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    model = PB.load(REPO, "cuda", revision=REVISION); dev = "cuda"; L, H, D, hd, rot = PB.geometry(model); forwards = 0; backwards = 0
    LAYERS = tuple(range(L)); ALL = [(l, h) for l in LAYERS for h in range(H)]
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit_all = torch.load(FIT_ROWS[0], map_location="cpu").long(); fit64 = fit_all[:64]
    native_stock, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw
    state = PB.instrument(model)
    native, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw
    print(f"[{TAG}] native: stock {native_stock:.5f} | explicit {native:.5f} | geometry L={L} H={H} D={D} hd={hd} rot={rot}")

    def run_rows(rows):
        fw = 0
        with torch.no_grad():
            for s in range(0, rows.shape[0], EBATCH):
                model(input_ids=rows[s:s + EBATCH].contiguous().to(dev)); fw += 1
        return fw

    # ---- capture: head means and per-offset logit kernels ------------------------------------------------------------------------------
    sums = {k: torch.zeros(hd, dtype=torch.float64, device=dev) for k in ALL}; n_tok = [0]
    ksum = {k: torch.zeros(513, dtype=torch.float64, device=dev) for k in ALL}; kcnt = torch.zeros(513, dtype=torch.float64, device=dev); cap = {"mode": None}

    def z_capture(l, z):
        if cap["mode"] == "means":
            for h in range(H):
                sums[(l, h)] += z[:, h].double().sum(dim=(0, 1))
            if l == 0:
                n_tok[0] += z.shape[0] * z.shape[2]
        return z

    def row_centre(X):
        """X: [B, T, T] logits. Subtract each query row's mean over the keys 1 <= j < i (the entries a program replaces); softmax is invariant
        to per-row constants, and in models with massive activations (Pythia) the raw logits carry row constants of 1e4-1e5."""
        T = X.shape[-1]; pos = torch.arange(T, device=X.device); rep = ((pos[:, None] > pos[None, :]) & (pos[None, :] > 0)).float()
        m = (X * rep[None]).sum(-1) / rep.sum(-1).clamp_min(1)[None]
        return X - m[:, :, None], m

    def logit_capture(l, logits):
        if cap["mode"] == "kernels":
            B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = pos[:, None] - pos[None, :]; qm = (pos >= 8)[:, None] & (dmat > 0) & (pos[None, :] > 0); dflat = dmat[qm]
            for h in range(Hn):
                Xc, _ = row_centre(logits[:, h]); ksum[(l, h)].index_add_(0, dflat, Xc[:, qm].double().sum(0))
            if l == 0:
                kcnt.index_add_(0, dflat, torch.full_like(dflat, B, dtype=torch.float64))
        return logits

    state["z_edit"] = z_capture; state["logit_edit"] = logit_capture
    cap["mode"] = "means"; forwards += run_rows(fit_all[:, :-1])
    cap["mode"] = "kernels"; forwards += run_rows(fit64[:, :-1]); cap["mode"] = None
    means = {k: (sums[k] / n_tok[0]).float() for k in ALL}
    kappa0 = {k: torch.where(kcnt > 0, ksum[k] / kcnt.clamp_min(1), torch.zeros_like(ksum[k])).float() for k in ALL}
    # ---- values -----------------------------------------------------------------------------------------------------------------------
    abl = {"set": set()}

    def z_ablate(l, z):
        for (al, h) in abl["set"]:
            if al == l:
                z = z.clone(); z[:, h] = means[(l, h)]
        return z

    state["z_edit"] = z_ablate; state["logit_edit"] = None
    value = {}
    for k in ALL:
        abl["set"] = {k}; c_, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw; value[f"{k[0]}.{k[1]}"] = c_ - native
    abl["set"] = set(ALL); c_all, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw; abl["set"] = set(); joint = c_all - native
    top = sorted(value, key=value.get, reverse=True)[:12]; vals = sorted(value.values()); median = vals[len(vals) // 2]
    print(f"[{TAG}] values: top " + " ".join(f"{k}:{value[k]:.3f}" for k in top) + f" | median {median:.4f} | sum {sum(vals):.3f} | joint {joint:.3f}")
    # ---- kernels-only program, with and without column 0 native; shape ladder --------------------------------------------------------
    kap_box = {"kap": kappa0, "col0": False}

    def logit_kernel(l, logits):
        B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = dmat > 0
        if kap_box["col0"]:
            off = off & (pos[None, :] > 0)
        out = logits.clone()
        for h in range(Hn):
            _, m = row_centre(logits[:, h])
            out[:, h] = torch.where(off[None], m[:, :, None] + kap_box["kap"][(l, h)][dmat][None].expand(B, T, T), logits[:, h])
        return out

    state["logit_edit"] = logit_kernel
    c_ko, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw
    kap_box["col0"] = True; c_ko0, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw; kap_box["col0"] = False
    Kmat = torch.stack([kappa0[k][1:512] for k in ALL]).double(); U, S, Vh = torch.linalg.svd(Kmat, full_matrices=False); energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    ladder = {}
    for kk in KS:
        P = (Kmat @ Vh[:kk].T) @ Vh[:kk]; kap_k = {}
        for i_, k in enumerate(ALL):
            v = kappa0[k].clone(); v[1:512] = P[i_].float(); kap_k[k] = v
        kap_box["kap"] = kap_k; c_, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw; ladder[str(kk)] = {"energy": float(energy[kk - 1]), "cost": c_ - native, "over_kernels_only": c_ - c_ko}
    kap_box["kap"] = kappa0
    print(f"[{TAG}] kernels-only {c_ko - native:+.4f} | with column 0 native {c_ko0 - native:+.4f} | SVD energy cum {[round(float(energy[i]), 3) for i in range(8)]} | ladder " + " ".join(f"k={kk}:{ladder[str(kk)]['cost']:+.3f}" for kk in KS))
    # ---- fitted programs at uniform rank (column 0 native) ------------------------------------------------------------------------------
    scaling = model.gpt_neox.layers[0].attention.scaling
    results, saved = {}, {}
    for R in ARMS:
        fac, bias = {}, {}
        for (l, h) in ALL:
            Wq, bq, Wk, bk = PB.head_qk(model, l, h); fac[(l, h)] = {}; bias[(l, h, "q")] = bq; bias[(l, h, "k")] = bk
            for n_, W in (("q", Wq), ("k", Wk)):
                U_, S_, Vh_ = torch.linalg.svd(W, full_matrices=False); r = min(R, hd)
                fac[(l, h)][n_] = ((U_[:, :r] * S_[:r].sqrt()).contiguous().requires_grad_(True), (S_[:r].sqrt()[:, None] * Vh_[:r]).contiguous().requires_grad_(True))
        cmul = {k: torch.zeros(513, device=dev, requires_grad=True) for k in ALL}
        kr = {}

        def lowrank_logits(l, h, n):
            Uq, Vq = fac[(l, h)]["q"]; Uk, Vk = fac[(l, h)]["k"]
            q = (n @ Vq.T) @ Uq.T + bias[(l, h, "q")]; k = (n @ Vk.T) @ Uk.T + bias[(l, h, "k")]
            q, k = PB.rotary_qk(q, k, state["cos"], state["sin"], rot)
            return torch.einsum("bqd,bkd->bqk", q, k) * scaling

        def offset_mean(P, T):
            P, _ = row_centre(P)
            pos = torch.arange(T, device=dev); dmat = pos[:, None] - pos[None, :]; qm = (pos >= 8)[:, None] & (dmat > 0) & (pos[None, :] > 0); dflat = dmat[qm]
            X = P[:, qm]; counts = torch.bincount(dflat, minlength=513).double() * X.shape[0]
            sums_ = torch.zeros(513, dtype=torch.float64, device=dev).index_add_(0, dflat, X.double().sum(0))
            return torch.where(counts > 0, sums_ / counts.clamp_min(1), torch.zeros_like(sums_)).float()

        def kappa_r_now():
            acc = {k: 0 for k in ALL}; nb = 0; fw = 0; state["logit_edit"] = None
            with torch.no_grad():
                for s in range(0, 64, EBATCH):
                    idx = fit64[s:s + EBATCH, :-1].contiguous().to(dev); model(input_ids=idx); fw += 1; T = idx.shape[1]
                    for (l, h) in ALL:
                        acc[(l, h)] = acc[(l, h)] + offset_mean(lowrank_logits(l, h, state["n"][l]), T)
                    nb += 1
            for k in ALL:
                kr[k] = acc[k] / nb
            return fw

        def logit_program(l, logits):
            B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0) & (pos[None, :] > 0)
            cols = []
            for h in range(Hn):
                kap = kappa0[(l, h)] * (1 + cmul[(l, h)]); pr, _ = row_centre(lowrank_logits(l, h, state["n"][l])); _, m = row_centre(logits[:, h])
                cols.append(torch.where(off[None], m[:, :, None] + kap[dmat][None] + (pr - kr[(l, h)][dmat][None]), logits[:, h]))
            return torch.stack(cols, 1)

        def ce_prog(rows):
            state["logit_edit"] = logit_program; state["z_edit"] = z_ablate; out = PB.ce(model, rows, dev, EBATCH); state["logit_edit"] = None; return out

        forwards += kappa_r_now()
        params = [f for k in ALL for pair in fac[k].values() for f in pair]
        opt = torch.optim.Adam([{"params": list(cmul.values()), "lr": LR_K}, {"params": params, "lr": LR_MAP}])
        step0, fw = ce_prog(ev); forwards += fw; v0, fw = ce_prog(val); forwards += fw
        val_curve, ho_curve = {0: v0}, {0: step0 - native}; best = (v0, 0, [p.detach().clone() for p in list(cmul.values()) + params])
        gen2 = torch.Generator().manual_seed(755 + R); order = torch.randperm(fit.shape[0], generator=gen2)
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
                forwards += kappa_r_now(); v_, fw = ce_prog(val); forwards += fw; h_, fw = ce_prog(ev); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - native
                if v_ < best[0]:
                    best = (v_, step, [p.detach().clone() for p in list(cmul.values()) + params])
        with torch.no_grad():
            for p, q in zip(list(cmul.values()) + params, best[2]):
                p.copy_(q)
        forwards += kappa_r_now(); c_snap, fw = ce_prog(ev); forwards += fw
        results[R] = {"closed_form_cost": step0 - native, "chosen_step": best[1], "snapshot_cost": c_snap - native, "recovery": 1 - (c_snap - native) / joint, "heldout_curve": {str(k): v for k, v in ho_curve.items()},
                      "numbers": sum(2 * min(R, hd) * (hd + D) + 513 for k in ALL), "native_qk_numbers": 2 * len(ALL) * hd * D}
        print(f"[{TAG}] rank {R}: closed-form {step0 - native:+.4f} -> snapshot (step {best[1]}) {c_snap - native:+.4f}, recovery {1 - (c_snap - native) / joint:.3f}; curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}")
        if R == 32:
            manip = {}
            for key in top[:N_MANIP] if len(top) >= N_MANIP else sorted(value, key=value.get, reverse=True)[:N_MANIP]:
                l, h = map(int, key.split(".")); abl["set"] = {(l, h)}; c_, fw = ce_prog(ev); forwards += fw; abl["set"] = set()
                manip[key] = {"native_value": value[key], "in_program_value": c_ - c_snap, "ratio": (c_ - c_snap) / value[key]}
            keys = list(manip); rx = torch.tensor([manip[k]["native_value"] for k in keys]).argsort().argsort().double(); ry = torch.tensor([manip[k]["in_program_value"] for k in keys]).argsort().argsort().double()
            rx -= rx.mean(); ry -= ry.mean(); rho = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)); rs = sorted(manip[k]["ratio"] for k in keys); med = rs[len(rs) // 2]
            results[R].update({"manipulability": manip, "spearman": rho, "median_ratio": med})
            print(f"[{TAG}] rank 32 manipulability: Spearman {rho:.3f}, median ratio {med:.2f}: " + " ".join(f"{k}:{manip[k]['native_value']:.3f}->{manip[k]['in_program_value']:.3f}" for k in keys[:8]))
        saved[R] = {f"{k[0]}.{k[1]}": {"kappa": (kappa0[k] * (1 + cmul[k])).detach().cpu(), **{n_: (fac[k][n_][0].detach().cpu(), fac[k][n_][1].detach().cpu()) for n_ in fac[k]}} for k in ALL}
    state["_restore"]()
    disk_guard.guard_torch_save({"means": {f"{k[0]}.{k[1]}": means[k].cpu() for k in ALL}, "kappa0": {f"{k[0]}.{k[1]}": kappa0[k].cpu() for k in ALL}, "svd_S": S.cpu(), "svd_Vh": Vh[:16].cpu(), "programs": {str(R): saved[R] for R in ARMS}}, str(OUT_PT), f"{TAG} tensors")
    r32, r16 = results[32], results[16]
    predictions = {"pred_a_instrument_replays": abs(native - native_stock) <= REPLAY_TOL, "pred_b_kernel_energy_top4": float(energy[3]) >= E4_MIN, "pred_c_rank32_cost": r32["recovery"] >= REC32,
                   "pred_d_rank16_cost": r16["recovery"] >= REC16, "pred_e_rank32_values_preserved_scale": RATIO[0] <= r32["median_ratio"] <= RATIO[1]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": f"pythia_all_result_{TAG}", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"model": REPO, "revision": REVISION, "revision": REVISION, "geometry": {"L": L, "H": H, "D": D, "hd": hd, "rot": rot}, "native_stock": native_stock, "native": native, "mean_ablation_cost": value, "median_head": median, "joint_value": joint,
                                          "kernels_only_cost": c_ko - native, "kernels_only_col0_native_cost": c_ko0 - native, "svd_energy_cum": [float(energy[i]) for i in range(16)], "ladder": ladder, "arms": {str(R): results[R] for R in ARMS}},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
