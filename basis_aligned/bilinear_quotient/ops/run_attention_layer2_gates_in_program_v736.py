#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_layer2_five_hold pred_c_each_layer2_gate_holds pred_d_twenty_readable_heads pred_e_layer2_composes_with_fifteen
"""Attention lane, v736: the five separable layer-2 readings (v628 tables: 2.0 / 2.2 / 2.3 / 2.4 / 2.8 as kappa(d) A(t_i) B(t_j)) inside the joint
program, and TWENTY readable heads.

v628 found seven layer-2 heads separable on the tables but two of the programs broken (2.6 +0.16, 2.7 +0.95); the other five cost 0.0013 /
0.0094 / 0.0096 / 0.0022 / 0.0090 alone in the native model. This rung installs each of the five inside the v718 program (0.0723), the five
together, and the five on top of v735's fifteen readable heads (0.142). Response only. CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native within 0.002 of 3.13241 (instrument)
    pred_b_layer2_five_hold        the five together cost <= 0.0315 + 0.005 above the program (sum of standalone singles). Prior: unsure
    pred_c_each_layer2_gate_holds  each single costs <= its standalone cost + 0.003 above the program. Prior: likely
    pred_d_twenty_readable_heads   the program with twenty readable heads costs <= 0.185. Prior: unsure
    pred_e_layer2_composes_with_fifteen the five's excess on top of the fifteen is within 0.005 of their excess on top of the base program. Prior: unsure
PRICE (registered maximum): native 6; kappa_r 2; program 6; five singles 30; five together 6; fifteen 6; twenty 6; total 62 forwards; 0 backwards; 0 fits. Bar <= 70.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_layer2_gates_in_program_v736_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
READINGS = {(1, 4): ("same", 0.05, 0.0037), (5, 5): ("induction", 0.10, 0.0071), (5, 7): ("sink", 0.55, 0.0058)}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.layer2_gates_in_program_v736"
FORWARDS_MAX = 70
T2 = ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"
L2 = {0: 0.0013, 2: 0.0094, 3: 0.0096, 4: 0.0022, 8: 0.0090}
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
L0_SET, L1_SET, MEAN_HEAD = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, FIVE_TOL, CTX_TOL, TWENTY, COMPOSE_TOL = 0.002, 0.005, 0.003, 0.185, 0.005
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_layer2_five_hold": "<= program + 0.0365", "pred_c_each_layer2_gate_holds": "single <= standalone + 0.003", "pred_d_twenty_readable_heads": "<= 0.185", "pred_e_layer2_composes_with_fifteen": "within 0.005"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "five_tol": FIVE_TOL, "ctx_tol": CTX_TOL, "twenty": TWENTY, "compose_tol": COMPOSE_TOL}, "layer2": {str(k): v for k, v in L2.items()}, "readings": {f"{k[0]}.{k[1]}": list(v) for k, v in READINGS.items()}, "gates": {"0": list(L0_SET), "1": list(L1_SET)}, "mean_head": MEAN_HEAD}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    rows = {name: torch.load(p, map_location="cpu").long() for name, p in SETS.items()}
    fit64 = torch.load(FIT_ROWS, map_location="cpu").long()[:64]
    means = torch.load(MEANS, map_location=dev); v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    state = {"idx": None, "programs": {}, "writes": {}, "n": {}, "ablate": set()}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]

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
                    if prog is not None and prog["kind"] in ("same", "induction", "sink"):
                        idx_ = state["idx"]; base = prog["kappa"][dmat][None].expand(Bn, Tn, Tn).clone()
                        if prog["kind"] == "same":
                            base = base + prog["beta"] * (idx_[:, :, None] == idx_[:, None, :]).float()
                        elif prog["kind"] == "induction":
                            prev = torch.cat([torch.full_like(idx_[:, :1], -1), idx_[:, :-1]], 1); base = base + prog["beta"] * (idx_[:, :, None] == prev[:, None, :]).float()
                        else:
                            base[:, :, 0] = prog["beta"]
                        cols.append(torch.where(off[None], base, pat[:, h]))
                    elif prog is not None and prog["kind"] == "gatetab":
                        idx_ = state["idx"]; base = prog["kappa"][dmat][None] * prog["A"][idx_][:, :, None] * prog["B"][idx_][:, None, :]
                        cols.append(torch.where(off[None], base, pat[:, h]))
                    elif prog is not None and prog["kind"] == "runmean":
                        cols.append(torch.where(off[None], (-1.0 / pos.clamp_min(1).float())[None, :, None].expand(Bn, Tn, Tn), pat[:, h]))
                    else:
                        cols.append(pat[:, h] if prog is None else AP.apply_program(pat[:, h], prog, ctx))
                pat = torch.stack(cols, 1)
            z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
            if state["writes"]:
                cols = []
                for h in range(Hn):
                    w = state["writes"].get((l, h)); m = means[f"mean_{l}_{h}"]
                    cols.append(z[:, h] if w is None else (m + ((z[:, h].float() - m) @ w[0]) @ w[1]).to(z.dtype))
                z = torch.stack(cols, 1)
            if state["ablate"]:
                z = z.clone()
                for (al, h) in state["ablate"]:
                    if al == l:
                        z[:, h] = means[f"mean_{l}_{h}"]
            return z
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(rs, progs=None, writes=None):
        state["programs"] = progs or {}; state["writes"] = writes or {}; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rs.shape[0], EBATCH):
                idx = rs[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rs[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["programs"] = {}; state["writes"] = {}
        return total / n, fw

    fac = torch.load(PROGS718, map_location=dev)["factored"]
    maps = {k: {n_: (fac[f"{k[0]}.{k[1]}"][n_][0].to(dev).float() @ fac[f"{k[0]}.{k[1]}"][n_][1].to(dev).float()) for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in ALL}
    kappa = {k: fac[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float() for k in ALL}
    acc = {k: 0 for k in ALL}; nb = 0
    with torch.no_grad():
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
            Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            for k in ALL:
                acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps[k], hd, causal), dmat, off, Tn)
            nb += 1
    kr = {k: acc[k] / nb for k in ALL}

    def program(kap):
        return {k: {"kind": "lowrank", "kappa": kap[k], "kappa_r": kr[k], "maps": maps[k]} for k in ALL}

    ev7 = rows["skip7000"]; nat, fw = ce(ev7); forwards += fw
    base_prog = program(kappa); c_prog, fw = ce(ev7, base_prog); forwards += fw
    kappa702 = torch.load(ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt", map_location=dev)

    def with_readings(keys):
        p = dict(base_prog)
        for k in keys:
            kind, beta, _ = READINGS[k]; p[k] = {"kind": kind, "beta": beta, "kappa": kappa702[f"kappa_{k[0]}_{k[1]}"].float()}
        return p

    t0_ = torch.load(T0, map_location=dev); t1_ = torch.load(T1, map_location=dev)
    gates = {(0, h): {"kind": "gatetab", "A": t0_[f"head{h}_A"].float(), "B": t0_[f"head{h}_B"].float(), "kappa": t0_[f"head{h}_kappa"].float()} for h in L0_SET}
    gates.update({(1, h): {"kind": "gatetab", "A": t1_[f"head{h}_A"].float(), "B": t1_[f"head{h}_B"].float(), "kappa": t1_[f"head{h}_kappa"].float()} for h in L1_SET})
    gates[(1, MEAN_HEAD)] = {"kind": "runmean"}

    def with_gates(keys, readings=()):
        p = with_readings(list(readings))
        for k in keys:
            p[k] = gates[k]
        return p

    t2_ = torch.load(T2, map_location=dev)
    g2 = {(2, h): {"kind": "gatetab", "A": t2_[f"head{h}_A"].float(), "B": t2_[f"head{h}_B"].float(), "kappa": t2_[f"head{h}_kappa"].float()} for h in L2}

    def with_l2(keys, base):
        p = dict(base)
        for k in keys:
            p[k] = g2[k]
        return p

    report = {"native": nat, "program_cost": c_prog - nat, "singles": {}, "arms": {}}
    for k in g2:
        c_, fw = ce(ev7, with_l2([k], base_prog)); forwards += fw
        report["singles"][f"{k[0]}.{k[1]}"] = {"cost": c_ - nat, "excess": c_ - c_prog, "standalone": L2[k[1]]}
        print(f"layer-2 gate {k[0]}.{k[1]} inside the program: excess {c_ - c_prog:+.4f} (standalone {L2[k[1]]:+.4f})")
    c5, fw = ce(ev7, with_l2(list(g2), base_prog)); forwards += fw
    fifteen = with_gates(list(gates), READINGS); c15, fw = ce(ev7, fifteen); forwards += fw
    c20, fw = ce(ev7, with_l2(list(g2), fifteen)); forwards += fw
    report["arms"] = {"layer2_five": {"cost": c5 - nat, "excess": c5 - c_prog}, "fifteen": {"cost": c15 - nat}, "twenty": {"cost": c20 - nat, "excess_over_fifteen": c20 - c15, "recovery": 1 - (c20 - nat) / 3.9961}}
    print(f"five layer-2 gates together: excess {c5 - c_prog:+.4f} (sum of standalone {sum(L2.values()):+.4f}) | fifteen {c15 - nat:+.4f} | twenty readable heads {c20 - nat:+.4f} (excess over fifteen {c20 - c15:+.4f}; recovery {1 - (c20 - nat) / 3.9961:.3f})")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_layer2_five_hold": (c5 - c_prog) <= sum(L2.values()) + FIVE_TOL,
                   "pred_c_each_layer2_gate_holds": all(v["excess"] <= v["standalone"] + CTX_TOL for v in report["singles"].values()),
                   "pred_d_twenty_readable_heads": (c20 - nat) <= TWENTY, "pred_e_layer2_composes_with_fifteen": abs((c20 - c15) - (c5 - c_prog)) <= COMPOSE_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_layer2_gates_in_program_result_v736", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
