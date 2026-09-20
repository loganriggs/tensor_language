#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_cost_not_worse pred_c_values_preserved_rank pred_d_no_head_over_2p5 pred_e_no_overfit
"""Attention lane, v713: the fitted program with band 6-8 promoted to rank 64 — does the per-band rank budget fix manipulability under a fit?

v706's rank-16/64 all-hybrid program (+0.075, recovery 0.981) leans 2-4x on 8.3 / 5.5 / 1.4 / 3.5 / 1.1; v709-v711 localized the leaning to
each head's own layer band and, for 8.3 / 5.5, to head 7.3's rank-16 program; closed-form rank 64 for band 6-8 cut the cost to 0.061 and 8.3's
ratio to 1.3. This rung re-runs v706's protocol (SVD init, Adam 300 steps, validation stopping, seed-703 split) with every band-6-8 head at
rank 64 (the 28 content heads stay at 64, the rest at 16; no native heads), then the manipulability check over the 24 most valuable heads.
Values 20.2M -> 25.9M. CE ADDED, lower is better; joint recovery = 1 - cost / 3.996; ratio = in-program / native mean-ablation value.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_cost_not_worse        held-out at the validation-chosen step <= 0.075 (v706). Prior: likely
    pred_c_values_preserved_rank Spearman(native value, in-program value) over the 24 heads >= 0.8 (v706: 0.65). Prior: unsure
    pred_d_no_head_over_2p5      no head among the 24 has ratio > 2.5 (v706: 8.3 4.0, 5.5 2.3). Prior: unsure
    pred_e_no_overfit            held-out at every evaluation after step 50 <= the step-50 value + 0.003 (v706 failed by 0.007). Prior: unsure
PRICE (registered maximum): 1 arm x 300 steps = 300 forwards + 300 BACKWARDS; kappa_r 13 x 2 = 26; validation 13 x 3 = 39; held-out 13 x 6 + 6 = 84;
manipulability 6 + 24 x 6 = 150; total ~605 forwards, 300 backwards; fit parameters ~26M. Bars: forwards <= 625, backwards <= 300.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_band68_r64_fit_v713_result.json"
OUT_PT = ROOT / "circuits/followups/attention_band68_r64_fit_v713_programs.pt"
V702 = ROOT / "circuits/followups/attention_program_ladder_v702_result.json"
PROGS = ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, V702_X09, JOINT_VALUE = 3.13241, 0.5065, 3.9961
CANDIDATE_ID = "attention.band68_r64_fit_v713"
FORWARDS_MAX, BACKWARDS_MAX = 625, 300
BAND = (6, 7, 8)
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
N_MANIP = 24
STEPS, TBATCH, EBATCH, EVAL_EVERY, N_VAL = 300, 8, 32, 25, 96
LR_MAP, LR_MAP_MIN = 3e-4, 3e-5
LR, LR_MIN = 0.02, 0.002
LAYERS = tuple(range(18)); H = 9
ARMS = ("band68_r64",)
REPLAY_TOL, UPTURN_TOL, COST_MAX, RHO_MIN, RATIO_MAX = 0.002, 0.003, 0.075, 0.8, 2.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_cost_not_worse": "<= 0.075", "pred_c_values_preserved_rank": "Spearman >= 0.8",
               "pred_d_no_head_over_2p5": "max ratio <= 2.5", "pred_e_no_overfit": "no upturn > 0.003"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": 111 * 4 * 16 * 1280 + 51 * 4 * 64 * 1280 + 162 * 513,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN],
            "bars": {"replay_tol": REPLAY_TOL, "upturn_tol": UPTURN_TOL, "cost_max": COST_MAX, "rho_min": RHO_MIN, "ratio_max": RATIO_MAX}, "band": list(BAND), "arms": list(ARMS), "n_manip": N_MANIP}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(703); perm = torch.randperm(allfit.shape[0], generator=gen); val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fit64 = torch.load(FIT_ROWS[0], map_location="cpu").long()[:64]
    v702 = json.load(open(V702))["report"]; progs_pt = torch.load(PROGS, map_location=dev)
    kappa0 = {(l, h): progs_pt[f"kappa_{l}_{h}"].to(dev) for l in LAYERS for h in range(H)}
    value_fit = {k: v["value_fit"] for k, v in v702["heads"].items()}
    state = {"idx": None, "programs": {}, "n": {}}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]

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
                cols.append(pat[:, h] if prog is None else AP.apply_program(pat[:, h], prog() if callable(prog) else prog, ctx))
            pat = torch.stack(cols, 1)
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)
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

    native, fw = ce(ev, {}); forwards += fw
    heads_info = v702["heads"]; ALL = [(l, h) for l in LAYERS for h in range(H)]

    def rank_for(k):
        info = heads_info[f"{k[0]}.{k[1]}"]; rec = info["recovery"]
        if k[0] in BAND:
            return 64
        if info["value_fit"] < 0.005 or (rec.get("lowrank4") or 0) >= 0.9 or (rec.get("lowrank16") or 0) >= 0.9:
            return 16
        return 64

    def kappa_r_now(maps_by_head):
        """Positional mean of each head's current rank-r pattern on 64 rows (a statistic of the maps)."""
        acc = {k: 0 for k in maps_by_head}; nb = 0; fw = 0
        with torch.no_grad():
            for s in range(0, 64, EBATCH):
                idx = fit64[s:s + EBATCH, :-1].to(dev); state["programs"] = {}; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); fw += 1
                Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                for k, maps in maps_by_head.items():
                    acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], {n_: m.detach() for n_, m in maps.items()}, hd, causal), dmat, off, Tn)
                nb += 1
        return {k: v / nb for k, v in acc.items()}, fw

    results, saved = {}, {}
    for arm in ARMS:
        simplified = ALL; ranks = {k: rank_for(k) for k in ALL}
        cmul = {k: torch.zeros(513, device=dev, requires_grad=True) for k in simplified}
        maps = {k: {n_: m.clone().requires_grad_(True) for n_, m in AP.truncated_maps(blocks[k[0]].attn, k[1], ranks[k], hd).items()} for k in simplified}
        counts = {r_: sum(1 for k in ALL if ranks[k] == r_) for r_ in (4, 16, 64)}; print(f"[{arm}] rank counts {counts}")
        kr, fw = kappa_r_now(maps); forwards += fw

        def build():
            return {k: {"kind": "lowrank", "kappa": kappa0[k] * (1 + cmul[k]), "kappa_r": kr[k], "maps": maps[k]} for k in simplified}

        opt = torch.optim.Adam([{"params": list(cmul.values()), "lr": LR}, {"params": [m for mm in maps.values() for m in mm.values()], "lr": LR_MAP}])
        step0, fw = ce(ev, build()); forwards += fw; v0, fw = ce(val, build()); forwards += fw
        val_curve, ho_curve = {0: v0}, {0: step0 - native}
        gen2 = torch.Generator().manual_seed(706); order = torch.randperm(fit.shape[0], generator=gen2)
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
        values = sum(513 + 4 * ranks[k] * 1280 for k in ALL)
        results[arm] = {"step0_added": step0 - native, "chosen_step": chosen, "heldout_at_chosen": ho_at, "recovery_at_chosen": 1 - ho_at / JOINT_VALUE, "heldout_curve": {str(k): v for k, v in ho_curve.items()},
                        "rank_counts": {str(k_): v_ for k_, v_ in counts.items()}, "ranks": {f"{k[0]}.{k[1]}": ranks[k] for k in ALL}, "values": values}
        saved[arm] = {f"{k[0]}.{k[1]}": {"kappa": (kappa0[k] * (1 + cmul[k])).detach().cpu(), **{n_: m.detach().cpu() for n_, m in maps[k].items()}} for k in simplified}
        print(f"[{arm}]: step 0 {step0 - native:+.4f} -> fitted (validation step {chosen}) {ho_at:+.4f}, joint recovery {1 - ho_at / JOINT_VALUE:.3f}; values {values / 1e6:.1f}M; curve {[round(ho_curve[k], 3) for k in sorted(ho_curve)]}")
    # ---- manipulability under the fitted program (endpoint maps; native heads none) ---------------------------------------------
    means = torch.load(MEANS, map_location=dev); v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    top = sorted(v701, key=v701.get, reverse=True)[:N_MANIP]
    state["ablate"] = set()
    orig_patched = {l: blocks[l].attn.squared_attention for l in LAYERS}

    def make_ablating(l):
        inner = orig_patched[l]
        def patched(q, k, v, q2, k2):
            z = inner(q, k, v, q2, k2)
            for (al, h) in state["ablate"]:
                if al == l:
                    z = z.clone(); z[:, h] = means[f"mean_{l}_{h}"]
            return z
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_ablating(l)
    prog_final = build()
    base_prog, fw = ce(ev, prog_final); forwards += fw
    manip = {}
    for key in top:
        l, h = map(int, key.split(".")); state["ablate"] = {(l, h)}
        c_, fw = ce(ev, prog_final); forwards += fw; state["ablate"] = set()
        manip[key] = {"native_value": v701[key], "in_program_value": c_ - base_prog, "ratio": (c_ - base_prog) / v701[key]}
    def spearman(x, y):
        rx = torch.tensor(x).argsort().argsort().double(); ry = torch.tensor(y).argsort().argsort().double(); rx -= rx.mean(); ry -= ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))
    rho = spearman([manip[k]["native_value"] for k in top], [manip[k]["in_program_value"] for k in top])
    ratios = sorted(manip[k]["ratio"] for k in top); med_ratio = ratios[len(ratios) // 2]
    print("manipulability (top-24 heads): " + " ".join(f"{k}:{manip[k]['native_value']:.3f}->{manip[k]['in_program_value']:.3f}" for k in top))
    print(f"Spearman(native value, in-program value) = {rho:.3f}; median ratio {med_ratio:.2f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    disk_guard.guard_torch_save(saved, str(OUT_PT), "v713 fitted hybrids, band 6-8 at rank 64 (endpoint)")
    M = results["band68_r64"]; max_ratio = max(manip[k]["ratio"] for k in top)
    print(f"max ratio {max_ratio:.2f} at {max(top, key=lambda k: manip[k]['ratio'])}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_cost_not_worse": M["heldout_at_chosen"] <= COST_MAX,
                   "pred_c_values_preserved_rank": rho >= RHO_MIN, "pred_d_no_head_over_2p5": max_ratio <= RATIO_MAX,
                   "pred_e_no_overfit": all(M["heldout_curve"][str(st)] <= M["heldout_curve"]["50"] + UPTURN_TOL for st in range(75, STEPS + 1, EVAL_EVERY))}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "attention_band68_r64_fit_result_v713", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "joint_value": JOINT_VALUE, "arms": results, "manipulability": manip, "spearman": rho, "median_ratio": med_ratio, "program_endpoint_cost": base_prog - native},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
