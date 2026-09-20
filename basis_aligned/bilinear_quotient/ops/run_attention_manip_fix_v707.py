#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_over_weight_is_own_program pred_c_frozen_cost_not_worse pred_d_refit_values_preserved_rank pred_e_residual_spread
"""Attention lane, v707: the manipulability fix — v706's program with the five over-weighted heads native, frozen and refitted; and the
per-head RESIDUAL inside the joint program.

v706: the rank-16/64 all-hybrid program costs +0.075 (recovery 0.981) but leans harder than the model on five heads — mean-ablation value
in-program / native: 8.3 4.0x, 5.5 2.3x, 1.4 1.9x, 3.5 1.6x, 1.1 1.3x — Spearman over the 24 most valuable heads 0.65 (bar 0.8 failed).
Two questions. (i) Whose fault: is a head over-weighted because ITS OWN program makes it so, or because the OTHER heads' programs route
through it? Arm FROZEN: v706's endpoint programs unchanged, the five heads restored to native, no refit; mean-ablate each of the 24 again.
If the five ratios fall to ~1 the over-weighting sat in their own programs; if they stay, the rest of the program leans on them. (ii) Does
the fix hold under refit: arm REFIT warm-starts from the frozen arm (157 hybrids, 5 native), 200 steps at half the v706 rates, validation
stopping (same seed-703 split), then the manipulability check and, new, the per-head RESIDUAL: for each of the 24, cost(program) -
cost(program with that head native) — what each head's approximation costs inside the joint program, which is 'how much was recovered,
relative to its value' read from the joint side. CE ADDED, lower is better; joint recovery = 1 - cost / 3.996.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays            native CE within 0.002 of 3.13241 (instrument)
    pred_b_over_weight_is_own_program in the FROZEN arm, all five restored heads have in-program / native value ratio <= 2 (8.3 falls from 4x). Prior: unsure
    pred_c_frozen_cost_not_worse     FROZEN held-out cost <= v706's endpoint cost 0.0852 + 0.005 (restoring native heads does not hurt). Prior: likely
    pred_d_refit_values_preserved_rank REFIT arm: Spearman(native value, in-program value) over the 24 >= 0.8 (v706: 0.65). Prior: unsure
    pred_e_residual_spread           REFIT arm: no single head's residual exceeds 25% of the program's held-out cost. Prior: likely
PRICE (registered maximum): native 6; frozen 6 + manipulability 24 x 6 = 144; refit 200 forwards + 200 BACKWARDS; kappa_r 9 x 2 = 18;
validation 9 x 3 = 27; held-out 9 x 6 = 54; endpoint 6; manipulability 144; residuals 24 x 6 = 144; total ~750 forwards, 200 backwards.
Bars: forwards <= 780, backwards <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_manip_fix_v707_result.json"
OUT_PT = ROOT / "circuits/followups/attention_manip_fix_v707_programs.pt"
V706 = ROOT / "circuits/followups/attention_mixed16_manip_v706_result.json"
PROGS706 = ROOT / "circuits/followups/attention_mixed16_manip_v706_programs.pt"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, JOINT_VALUE, V706_ENDPOINT = 3.13241, 3.9961, 0.0852
CANDIDATE_ID = "attention.manip_fix_v707"
FORWARDS_MAX, BACKWARDS_MAX = 780, 200
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_HEADS = ((8, 3), (5, 5), (1, 4), (3, 5), (1, 1))
N_MANIP = 24
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 200, 8, 32, 25, 96
LR_MAP, LR_MAP_MIN = 1.5e-4, 1.5e-5
LR, LR_MIN = 0.01, 0.001
LAYERS = tuple(range(18)); H = 9
REPLAY_TOL, RATIO_MAX, COST_TOL, RHO_MIN, RESID_SHARE = 0.002, 2.0, 0.005, 0.8, 0.25
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_over_weight_is_own_program": "five ratios <= 2 (frozen)", "pred_c_frozen_cost_not_worse": "<= 0.0852 + 0.005",
               "pred_d_refit_values_preserved_rank": "Spearman >= 0.8 (refit)", "pred_e_residual_spread": "max residual share <= 0.25 (refit)"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 157 * 4 * 24 * 1280 + 157 * 513,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN], "native_heads": [list(k) for k in NATIVE_HEADS],
            "bars": {"replay_tol": REPLAY_TOL, "ratio_max": RATIO_MAX, "cost_tol": COST_TOL, "rho_min": RHO_MIN, "resid_share": RESID_SHARE}, "n_manip": N_MANIP}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    v706 = json.load(open(V706))["report"]["arms"]["mixed16"]; saved706 = torch.load(PROGS706, map_location=dev)["mixed16"]
    ranks = {(l, h): v706["ranks"][f"{l}.{h}"] for l in LAYERS for h in range(H)}
    ALL = [(l, h) for l in LAYERS for h in range(H)]; simplified = [k for k in ALL if k not in NATIVE_HEADS]
    kappa0 = {k: saved706[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float() for k in simplified}
    state = {"idx": None, "programs": {}, "n": {}, "ablate": set(), "restore": set()}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]
    means = torch.load(MEANS, map_location=dev); v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    top = sorted(v701, key=v701.get, reverse=True)[:N_MANIP]

    def make_patched(l):
        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
            pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            ctx = {"dmat": dmat, "off": off, "pos": pos, "idx": state["idx"], "attn": blocks[l].attn, "n": state["n"][l], "hd": hd, "causal": causal}
            cols = []
            for h in range(Hn):
                prog = state["programs"].get((l, h))
                cols.append(pat[:, h] if prog is None or (l, h) in state["restore"] else AP.apply_program(pat[:, h], prog, ctx))
            pat = torch.stack(cols, 1)
            z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
            for (al, h) in state["ablate"]:
                if al == l:
                    z = z.clone(); z[:, h] = means[f"mean_{l}_{h}"]
            return z
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(rows, progs):
        state["programs"] = progs; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rows.shape[0], EBATCH):
                idx = rows[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rows[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["programs"] = {}
        return total / n, fw

    def kappa_r_now(maps_by_head):
        acc = {k: 0 for k in maps_by_head}; nb = 0; fw = 0
        with torch.no_grad():
            for s in range(0, 64, EBATCH):
                idx = fit64[s:s + EBATCH, :-1].to(dev); state["programs"] = {}; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); fw += 1
                Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                for k, maps in maps_by_head.items():
                    acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], {n_: m.detach() for n_, m in maps.items()}, hd, causal), dmat, off, Tn)
                nb += 1
        return {k: v / nb for k, v in acc.items()}, fw

    def manipulability(prog, base):
        out = {}; fw = 0
        for key in top:
            l, h = map(int, key.split(".")); state["ablate"] = {(l, h)}
            c_, f_ = ce(ev, prog); fw += f_; state["ablate"] = set()
            out[key] = {"native_value": v701[key], "in_program_value": c_ - base, "ratio": (c_ - base) / v701[key]}
        return out, fw

    def spearman(x, y):
        rx = torch.tensor(x).argsort().argsort().double(); ry = torch.tensor(y).argsort().argsort().double(); rx -= rx.mean(); ry -= ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))

    def summarize(m):
        rho = spearman([m[k]["native_value"] for k in top], [m[k]["in_program_value"] for k in top]); rs = sorted(m[k]["ratio"] for k in top)
        return rho, rs[len(rs) // 2]

    native, fw = ce(ev, {}); forwards += fw
    cmul = {k: torch.zeros(513, device=dev, requires_grad=True) for k in simplified}
    maps = {k: {n_: saved706[f"{k[0]}.{k[1]}"][n_].to(dev).float().clone().requires_grad_(True) for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in simplified}
    kr, fw = kappa_r_now(maps); forwards += fw

    def build():
        return {k: {"kind": "lowrank", "kappa": kappa0[k] * (1 + cmul[k]), "kappa_r": kr[k], "maps": maps[k]} for k in simplified}

    results = {}
    # ---- FROZEN: v706 endpoint programs, five heads native, no refit ------------------------------------------------------------------
    frozen_prog = build(); c_frozen, fw = ce(ev, frozen_prog); forwards += fw
    manip_f, fw = manipulability(frozen_prog, c_frozen); forwards += fw; rho_f, med_f = summarize(manip_f)
    five = [f"{l}.{h}" for (l, h) in NATIVE_HEADS]
    results["frozen"] = {"heldout_added": c_frozen - native, "recovery": 1 - (c_frozen - native) / JOINT_VALUE, "manipulability": manip_f, "spearman": rho_f, "median_ratio": med_f}
    print(f"[frozen] cost {c_frozen - native:+.4f} (v706 endpoint {V706_ENDPOINT:+.4f}); restored heads ratio: " + " ".join(f"{k}:{manip_f[k]['ratio']:.2f}" for k in five) + f"; Spearman {rho_f:.3f}, median ratio {med_f:.2f}")
    # ---- REFIT: warm start, 200 steps, validation stopping ----------------------------------------------------------------------------
    opt = torch.optim.Adam([{"params": list(cmul.values()), "lr": LR}, {"params": [m for mm in maps.values() for m in mm.values()], "lr": LR_MAP}])
    v0, fw = ce(val, build()); forwards += fw
    val_curve, ho_curve = {0: v0}, {0: c_frozen - native}
    gen2 = torch.Generator().manual_seed(707); order = torch.randperm(fit.shape[0], generator=gen2)
    for step in range(1, STEPS + 1):
        f = 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        opt.param_groups[0]["lr"] = LR_MIN + (LR - LR_MIN) * f; opt.param_groups[1]["lr"] = LR_MAP_MIN + (LR_MAP - LR_MAP_MIN) * f
        s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel_idx = order[s0:s0 + TBATCH]
        if len(sel_idx) < TBATCH:
            sel_idx = order[:TBATCH]
        idx = fit[sel_idx, :-1].to(dev); tgt = fit[sel_idx, 1:].to(dev); state["programs"] = build()
        with torch.enable_grad():
            loss = model(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        state["programs"] = {}; forwards += 1; backwards += 1
        if step % EVAL_EVERY == 0:
            kr, fw = kappa_r_now(maps); forwards += fw
            v_, fw = ce(val, build()); forwards += fw; h_, fw = ce(ev, build()); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - native
    chosen = min(val_curve, key=val_curve.get); ho_at = ho_curve[chosen]
    values = sum(513 + 4 * ranks[k] * 1280 for k in simplified) + sum(4 * 128 * 1280 for _ in NATIVE_HEADS)
    prog_final = build(); c_end, fw = ce(ev, prog_final); forwards += fw
    manip_r, fw = manipulability(prog_final, c_end); forwards += fw; rho_r, med_r = summarize(manip_r)
    resid = {}
    for key in top:
        l, h = map(int, key.split(".")); state["restore"] = {(l, h)}
        c_, fw = ce(ev, prog_final); forwards += fw; state["restore"] = set()
        resid[key] = {"residual": c_end - c_, "share": (c_end - c_) / max(c_end - native, 1e-9), "native_value": v701[key]}
    max_share = max(r["share"] for r in resid.values()); top_resid = sorted(resid, key=lambda k: resid[k]["residual"], reverse=True)[:8]
    results["refit"] = {"chosen_step": chosen, "heldout_at_chosen": ho_at, "recovery_at_chosen": 1 - ho_at / JOINT_VALUE, "heldout_curve": {str(k): v for k, v in ho_curve.items()}, "endpoint_cost": c_end - native,
                        "manipulability": manip_r, "spearman": rho_r, "median_ratio": med_r, "residuals": resid, "max_residual_share": max_share, "values": values}
    print(f"[refit] frozen {c_frozen - native:+.4f} -> fitted (validation step {chosen}) {ho_at:+.4f}, recovery {1 - ho_at / JOINT_VALUE:.3f}; endpoint {c_end - native:+.4f}; values {values / 1e6:.1f}M; curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}")
    print("[refit] manipulability: " + " ".join(f"{k}:{manip_r[k]['native_value']:.3f}->{manip_r[k]['in_program_value']:.3f}" for k in top) + f"; Spearman {rho_r:.3f}, median ratio {med_r:.2f}")
    print("[refit] residual inside the program (cost - cost with head native): " + " ".join(f"{k}:{resid[k]['residual']:+.4f}({resid[k]['share']:.2f})" for k in top_resid) + f"; max share {max_share:.2f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    disk_guard.guard_torch_save({f"{k[0]}.{k[1]}": {"kappa": (kappa0[k] * (1 + cmul[k])).detach().cpu(), **{n_: m.detach().cpu() for n_, m in maps[k].items()}} for k in simplified}, str(OUT_PT), "v707 refit hybrids (endpoint)")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_over_weight_is_own_program": all(manip_f[k]["ratio"] <= RATIO_MAX for k in five),
                   "pred_c_frozen_cost_not_worse": (c_frozen - native) <= V706_ENDPOINT + COST_TOL,
                   "pred_d_refit_values_preserved_rank": rho_r >= RHO_MIN, "pred_e_residual_spread": max_share <= RESID_SHARE}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_manip_fix_result_v707", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "joint_value": JOINT_VALUE, "native_heads": five, "arms": results},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
