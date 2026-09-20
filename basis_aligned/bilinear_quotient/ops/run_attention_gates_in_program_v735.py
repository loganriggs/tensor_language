#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_twelve_hold_in_program pred_c_layer0_gates_hold pred_d_layer1_gates_hold pred_e_fifteen_readable_heads
"""Attention lane, v735: the twelve gated-filter readings of blocks 0-1 (v623 / v624 / v627) installed INSIDE the exact-rank joint program.

The embedding-forward chapter read eleven early heads as gated positional filters kappa(d) A(tok_i) B(tok_j) (layer 0: 0.3 / 0.4 / 0.6 /
0.7 / 0.8; layer 1: 1.0 / 1.1 / 1.3 / 1.5 / 1.6 / 1.7) and 1.8 as the running-mean subtractor -1/i; together they cost +0.036 in the native
model (v627). This rung installs them in place of those heads' rank-16 content terms inside the v718 program (0.0723): layer-0 five alone,
layer-1 seven alone, all twelve; and, with v734's three one-number readings, all FIFTEEN readable heads. Response only. CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays       native within 0.002 of 3.13241 (instrument)
    pred_b_twelve_hold_in_program the twelve inside the program cost <= 0.036 + 0.005 above the program. Prior: likely (v734's pattern)
    pred_c_layer0_gates_hold    the layer-0 five alone cost <= 0.0094 + 0.003 above the program (v623 standalone). Prior: likely
    pred_d_layer1_gates_hold    the layer-1 seven alone cost <= 0.0126 + 0.0027 + 0.003 above the program (v624 + v627 standalone). Prior: likely
    pred_e_fifteen_readable_heads the program with all fifteen readable heads costs <= 0.130 (recovery >= 0.967). Prior: unsure
PRICE (registered maximum): native 6; kappa_r 2; program 6; four arms 24; total 38 forwards; 0 backwards; 0 fits. Bar <= 45.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_gates_in_program_v735_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
READINGS = {(1, 4): ("same", 0.05, 0.0037), (5, 5): ("induction", 0.10, 0.0071), (5, 7): ("sink", 0.55, 0.0058)}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.gates_in_program_v735"
FORWARDS_MAX = 45
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
L0_SET, L1_SET, MEAN_HEAD = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, TWELVE, L0, L1, FIFTEEN = 0.002, 0.036 + 0.005, 0.0094 + 0.003, 0.0126 + 0.0027 + 0.003, 0.130
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_twelve_hold_in_program": "<= program + 0.041", "pred_c_layer0_gates_hold": "<= program + 0.0124", "pred_d_layer1_gates_hold": "<= program + 0.0183", "pred_e_fifteen_readable_heads": "<= 0.130"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "twelve": TWELVE, "l0": L0, "l1": L1, "fifteen": FIFTEEN}, "readings": {f"{k[0]}.{k[1]}": list(v) for k, v in READINGS.items()}, "gates": {"0": list(L0_SET), "1": list(L1_SET)}, "mean_head": MEAN_HEAD}
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

    arms = {"layer0_five": [k for k in gates if k[0] == 0], "layer1_seven": [k for k in gates if k[0] == 1], "twelve": list(gates)}
    report = {"native": nat, "program_cost": c_prog - nat, "arms": {}}
    for name, keys in arms.items():
        c_, fw = ce(ev7, with_gates(keys)); forwards += fw; report["arms"][name] = {"cost": c_ - nat, "excess": c_ - c_prog, "heads": [f"{k[0]}.{k[1]}" for k in keys]}
        print(f"{name} inside the program: cost {c_ - nat:+.4f}, excess {c_ - c_prog:+.4f}")
    c15, fw = ce(ev7, with_gates(list(gates), READINGS)); forwards += fw
    report["arms"]["fifteen"] = {"cost": c15 - nat, "excess": c15 - c_prog, "recovery": 1 - (c15 - nat) / 3.9961}
    print(f"fifteen readable heads (twelve gates + 1.4 / 5.5 / 5.7): cost {c15 - nat:+.4f}, excess {c15 - c_prog:+.4f}, recovery {1 - (c15 - nat) / 3.9961:.3f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    ar = report["arms"]
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_twelve_hold_in_program": ar["twelve"]["excess"] <= TWELVE,
                   "pred_c_layer0_gates_hold": ar["layer0_five"]["excess"] <= L0, "pred_d_layer1_gates_hold": ar["layer1_seven"]["excess"] <= L1, "pred_e_fifteen_readable_heads": (c15 - nat) <= FIFTEEN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_gates_in_program_result_v735", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
