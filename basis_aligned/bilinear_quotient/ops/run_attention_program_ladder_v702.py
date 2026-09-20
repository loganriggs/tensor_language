#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_lowrank_rescues_valuable_heads pred_c_selection_keeps_few_native pred_d_joint_cost_bounded pred_e_joint_recovery
"""Attention lane, v702: a per-head PROGRAM LADDER chosen by recovery relative to mean-ablation value, priced jointly.

Logan (20 Sep): simplify all the attention; some heads need a hybrid; value heads by mean-ablation CE and report recovery relative to it.
Programs per head (attention_program_lib): runmean (-m/i, 1 number) < kernel (kappa(d), 513) < lowrank4 (kernel + rank-4 content deviation,
~21k) < lowrank16 (~82k) < gate (kappa A B tables, blocks 0-2 only, ~101k) < native (590k QK weights). On 64 FIT rows (skip80): the
mean-ablation value of every head and the single-head cost of every program; recovery = 1 - cost / value. Selection at thresholds X in
{0.5, 0.75, 0.9}: for each head the cheapest program with fit-row recovery >= X (heads valued < 0.005 on fit rows get the kernel). Then, on the
192 x 512 skip7000 rows: native, ALL heads mean-ablated (the joint value), and the three selections priced JOINTLY; joint recovery = 1 - joint cost /
all-mean-ablated cost. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays              native CE within 0.002 of 3.13241 (instrument)
    pred_b_lowrank_rescues_valuable_heads among heads valued >= 0.02 on fit rows, lowrank16 recovers >= 0.8 for at least half of those where the kernel
                                       recovers < 0.5 (hybrids rescue what kernels cannot). Prior: unsure
    pred_c_selection_keeps_few_native  the X = 0.9 selection keeps <= 40 heads native. Prior: unsure
    pred_d_joint_cost_bounded          the X = 0.9 selection costs <= 0.30 jointly on held-out. Prior: unsure (compounding)
    pred_e_joint_recovery              the X = 0.9 selection's joint recovery >= 0.8. Prior: unsure
PRICE (registered maximum): capture 2 forwards; fit-row singles: 162 mean ablations + 162 x 4 programs + 27 gates = 837 configs x 2 batches = 1674;
held-out: 5 configs x 6 = 30; total 1706 forwards; 0 backwards; 0 fits (m and kappa are means; truncations are SVDs). Bar <= 1750.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_program_ladder_v702_result.json"
OUT_PT = ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
TAB = {0: ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt", 1: ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt",
       2: ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"}
GATE_HEADS = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7, 8), 2: (0, 2, 3, 4, 6, 7, 8)}
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.program_ladder_v702"
FORWARDS_MAX = 1750
EBATCH, N_FIT = 32, 64
LAYERS = tuple(range(18))
RANKS = (4, 16)
THRESH = (0.5, 0.75, 0.9)
PRICE = {"runmean": 1, "kernel": 513, "lowrank4": 513 + 4 * 4 * (1152 + 128), "lowrank16": 513 + 4 * 16 * (1152 + 128), "gate": 2 * 50304 + 513, "native": 4 * 1152 * 128}
ORDER = ("runmean", "kernel", "lowrank4", "lowrank16", "gate", "native")
REPLAY_TOL, VALUE_FLOOR, RESCUE, NATIVE_MAX, JOINT_MAX, REC_MIN = 0.002, 0.02, 0.8, 40, 0.30, 0.8
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_lowrank_rescues_valuable_heads": ">= half rescued", "pred_c_selection_keeps_few_native": "<= 40 native at X=0.9",
               "pred_d_joint_cost_bounded": "<= 0.30", "pred_e_joint_recovery": ">= 0.8"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "ranks": list(RANKS), "thresholds": list(THRESH), "prices": PRICE,
            "bars": {"replay_tol": REPLAY_TOL, "value_floor": VALUE_FLOOR, "rescue": RESCUE, "native_max": NATIVE_MAX, "joint_max": JOINT_MAX, "rec_min": REC_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long()[:N_FIT]; ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    means = torch.load(MEANS, map_location=dev)
    gates = {}
    for l, p in TAB.items():
        t_ = torch.load(p, map_location=dev)
        for h in GATE_HEADS[l]:
            gates[(l, h)] = {"kind": "gate", "kappa": t_[f"head{h}_kappa"], "A": t_[f"head{h}_A"], "B": t_[f"head{h}_B"]}
    with torch.no_grad():
        state = {"idx": None, "programs": {}, "ablate": set(), "capture": None, "n": {}}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
        nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]
        trunc = {(l, h, r): AP.truncated_maps(blocks[l].attn, h, r, hd) for l in LAYERS for h in range(H) for r in RANKS}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                if state["capture"] is not None:
                    state["capture"][l] = (pat.detach().clone(), causal, dmat, off)
                ctx = {"dmat": dmat, "off": off, "pos": pos, "idx": state["idx"], "attn": blocks[l].attn, "n": state["n"][l], "hd": hd, "causal": causal}
                for (pl, h), prog in state["programs"].items():
                    if pl == l:
                        pat[:, h] = AP.apply_program(pat[:, h], prog, ctx)
                z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
                for (al, h) in state["ablate"]:
                    if al == l:
                        z = z.clone(); z[:, h] = means[f"mean_{l}_{h}"]
                return z
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)
        # ---- capture on the fit rows: kernels, running-mean m, low-rank kernels ---------------------------------------------------
        kappa = {}; kappa_r = {}; msum = {}
        acc_k = {}; acc_kr = {}; acc_m = {}; nb = 0
        for s in range(0, N_FIT, EBATCH):
            state["capture"] = {}; idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1; nb += 1
            for l in LAYERS:
                pat, causal, dmat, off = state["capture"][l]; Tn = pat.shape[-1]
                for h in range(H):
                    acc_k[(l, h)] = acc_k.get((l, h), 0) + AP.offset_mean(pat[:, h], dmat, off, Tn)
                    acc_m[(l, h)] = acc_m.get((l, h), 0) + float(-(pat[:, h] * off[None]).sum(-1)[:, 8:].mean())
                    for r in RANKS:
                        pr = AP.lowrank_pattern(blocks[l].attn, state["n"][l], trunc[(l, h, r)], hd, causal)
                        acc_kr[(l, h, r)] = acc_kr.get((l, h, r), 0) + AP.offset_mean(pr, dmat, off, Tn)
            state["capture"] = None
        for key in acc_k:
            kappa[key] = acc_k[key] / nb; msum[key] = acc_m[key] / nb
        for key in acc_kr:
            kappa_r[key] = acc_kr[key] / nb
        programs = {}
        for l in LAYERS:
            for h in range(H):
                programs[(l, h)] = {"runmean": {"kind": "runmean", "m": msum[(l, h)]}, "kernel": {"kind": "kernel", "kappa": kappa[(l, h)]},
                                    "lowrank4": {"kind": "lowrank", "kappa": kappa[(l, h)], "kappa_r": kappa_r[(l, h, 4)], "maps": trunc[(l, h, 4)]},
                                    "lowrank16": {"kind": "lowrank", "kappa": kappa[(l, h)], "kappa_r": kappa_r[(l, h, 16)], "maps": trunc[(l, h, 16)]}, "native": {"kind": "native"}}
                if (l, h) in gates:
                    programs[(l, h)]["gate"] = gates[(l, h)]
        print("programs built; running fit-row singles")

        def ce(rows, progs, ablate=()):
            state["programs"] = dict(progs); state["ablate"] = set(ablate); total = 0.0; n = 0; fw = 0
            for s in range(0, rows.shape[0], EBATCH):
                idx = rows[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rows[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            state["programs"] = {}; state["ablate"] = set()
            return total / n, fw

        nat_f, fw = ce(fit, {}); forwards += fw
        value_f, cost_f = {}, {}
        for l in LAYERS:
            for h in range(H):
                v_, fw = ce(fit, {}, ablate=[(l, h)]); forwards += fw; value_f[(l, h)] = v_ - nat_f
                cost_f[(l, h)] = {}
                for name in ("runmean", "kernel", "lowrank4", "lowrank16", "gate"):
                    if name in programs[(l, h)]:
                        c_, fw = ce(fit, {(l, h): programs[(l, h)][name]}); forwards += fw; cost_f[(l, h)][name] = c_ - nat_f
            print(f"layer {l}: " + " ".join(f"{l}.{h}: val {value_f[(l, h)]:+.3f} k {cost_f[(l, h)]['kernel']:+.3f} r4 {cost_f[(l, h)]['lowrank4']:+.3f} r16 {cost_f[(l, h)]['lowrank16']:+.3f}" for h in range(H)))
        # ---- selection ------------------------------------------------------------------------------------------------------------
        def recovery(key, name):
            v = value_f[key]; c = cost_f[key].get(name)
            return None if c is None or v < 0.005 else 1 - c / v

        selections = {}
        for X in THRESH:
            sel = {}
            for key in programs:
                if value_f[key] < 0.005:
                    sel[key] = "kernel"; continue
                chosen = "native"
                for name in ORDER[:-1]:
                    rec = recovery(key, name)
                    if rec is not None and rec >= X:
                        chosen = name; break
                sel[key] = chosen
            selections[X] = sel
        # ---- held-out pricing -----------------------------------------------------------------------------------------------------
        native, fw = ce(ev, {}); forwards += fw
        allmean, fw = ce(ev, {}, ablate=[(l, h) for l in LAYERS for h in range(H)]); forwards += fw; joint_value = allmean - native
        joint = {}
        for X, sel in selections.items():
            progs = {key: programs[key][name] for key, name in sel.items() if name != "native"}
            c_, fw = ce(ev, progs); forwards += fw
            counts = {name: sum(1 for v in sel.values() if v == name) for name in ORDER}; price = sum(PRICE[v] for v in sel.values())
            joint[X] = {"cost": c_ - native, "recovery": 1 - (c_ - native) / joint_value, "counts": counts, "price_values": price}
            print(f"X={X}: joint cost {c_ - native:+.4f}, recovery {joint[X]['recovery']:.3f} of the all-mean-ablated {joint_value:.3f}; counts {counts}; price {price / 1e6:.1f}M vs {162 * PRICE['native'] / 1e6:.1f}M")
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        for hh in nhooks:
            hh.remove()
        pre.remove()
        disk_guard.guard_torch_save({f"kappa_{l}_{h}": kappa[(l, h)].cpu() for l in LAYERS for h in range(H)} | {f"m_{l}_{h}": msum[(l, h)] for l in LAYERS for h in range(H)}, str(OUT_PT), "v702 programs")
    big = [k for k in programs if value_f[k] >= VALUE_FLOOR]
    weak = [k for k in big if (recovery(k, "kernel") or 0) < 0.5]; rescued = [k for k in weak if (recovery(k, "lowrank16") or 0) >= RESCUE]
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_lowrank_rescues_valuable_heads": bool(weak) and len(rescued) >= 0.5 * len(weak),
                   "pred_c_selection_keeps_few_native": joint[0.9]["counts"]["native"] <= NATIVE_MAX,
                   "pred_d_joint_cost_bounded": joint[0.9]["cost"] <= JOINT_MAX, "pred_e_joint_recovery": joint[0.9]["recovery"] >= REC_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_program_ladder_result_v702", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "native_fit": nat_f, "joint_value_all_mean_ablated": joint_value,
                                          "heads": {f"{l}.{h}": {"value_fit": value_f[(l, h)], "cost_fit": cost_f[(l, h)], "recovery": {n: recovery((l, h), n) for n in cost_f[(l, h)]}} for l in LAYERS for h in range(H)},
                                          "selections": {str(X): {f"{l}.{h}": sel[(l, h)] for l in LAYERS for h in range(H)} for X, sel in selections.items()},
                                          "joint": {str(X): v for X, v in joint.items()}, "n_big": len(big), "n_weak_kernel": len(weak), "n_rescued_lowrank16": len(rescued)},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
