#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_program_replays pred_c_readable_replays pred_d_dictionary_replays pred_e_fresh_replays
"""Attention lane, v739: CANARY 2 — one script that replays every registered number of the chapter from saved files (no fit).

Registered on skip7000 (192 x 512; native 3.13241): exact-rank pattern program (v718) 0.0723; four-shape kernel dictionary inside it (v724)
0.0734; fifteen readable heads (v735) 0.1420; twenty readable heads with the v738 tables 0.1683; joint value (all 162 mean-ablated) 3.9961.
On the fresh 512-row window (native 3.5061): v718 program not registered before (reported here), fifteen 0.1225, twenty 0.1481.
Anyone can re-run this; a drift beyond the bars means a file or an instrument changed. CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays    skip7000 native within 0.002 of 3.13241 and fresh native within 0.002 of 3.5061
    pred_b_program_replays   v718 program within 0.003 of 0.0723; joint value within 0.02 of 3.9961
    pred_c_readable_replays  fifteen within 0.003 of 0.1420; twenty within 0.003 of 0.1683
    pred_d_dictionary_replays four-shape dictionary program within 0.003 of 0.0734
    pred_e_fresh_replays     fresh fifteen within 0.003 of 0.1225 and fresh twenty within 0.003 of 0.1481
PRICE (registered maximum): skip7000: native 6, kappa_r 2, program 6, dictionary 6, fifteen 6, twenty 6, all-ablated 6; fresh: native 16, program 16, fifteen 16, twenty 16;
total 108 forwards; 0 backwards; 0 fits. Bar <= 120.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_canary2_v739_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
READINGS = {(1, 4): ("same", 0.05, 0.0037), (5, 5): ("induction", 0.10, 0.0071), (5, 7): ("sink", 0.55, 0.0058)}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt", "fresh": ROOT / "bilin18_eval_tokens_large.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.canary2_v739"
FORWARDS_MAX = 120
TABLES738 = ROOT / "circuits/followups/attention_readable_rowsets_v738_tables.pt"
REG = {"native": 3.13241, "fresh_native": 3.5061, "program": 0.0723, "joint": 3.9961, "fifteen": 0.1420, "twenty": 0.1683, "dictionary4": 0.0734, "fresh_fifteen": 0.1225, "fresh_twenty": 0.1481}
STEPS, TBATCH, EVAL_EVERY, N_VAL = 200, 8, 25, 96
LR_T, LR_T_MIN, LR_K, LR_K_MIN = 3e-3, 3e-4, 0.01, 0.001
FIT_ROWS_ALL = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
T2 = ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"
L2 = {0: 0.0013, 2: 0.0094, 3: 0.0096, 4: 0.0022, 8: 0.0090}
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
L0_SET, L1_SET, MEAN_HEAD = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, COST_TOL, JOINT_TOL = 0.002, 0.003, 0.02
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_program_replays": "+-0.003 / +-0.02", "pred_c_readable_replays": "+-0.003", "pred_d_dictionary_replays": "+-0.003", "pred_e_fresh_replays": "+-0.003"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "cost_tol": COST_TOL, "joint_tol": JOINT_TOL}, "registered": REG, "layer2": {str(k): v for k, v in L2.items()}, "steps": STEPS, "readings": {f"{k[0]}.{k[1]}": list(v) for k, v in READINGS.items()}, "gates": {"0": list(L0_SET), "1": list(L1_SET)}, "mean_head": MEAN_HEAD}
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

    tb = torch.load(TABLES738, map_location=dev)
    fifteen = with_gates(list(gates), READINGS)
    twenty = dict(fifteen)
    for h in L2:
        twenty[(2, h)] = {"kind": "gatetab", "A": tb[f"2.{h}"]["A"].to(dev).float(), "B": tb[f"2.{h}"]["B"].to(dev).float(), "kappa": tb[f"2.{h}"]["kappa"].to(dev).float()}
    Kmat = torch.stack([kappa[k][1:513] for k in ALL]).double(); U, S, Vh = torch.linalg.svd(Kmat, full_matrices=False); K4 = (U[:, :4] * S[:4]) @ Vh[:4]
    kap4 = {}
    for i_, k in enumerate(ALL):
        v = kappa[k].clone(); v[1:513] = K4[i_].float(); kap4[k] = v
    c_d4, fw = ce(ev7, program(kap4)); forwards += fw
    state["ablate"] = set(ALL); c_all, fw = ce(ev7); forwards += fw; state["ablate"] = set()
    sets = {}
    for sname, rs in rows.items():
        n_, fw = ce(rs); forwards += fw; p_, fw = ce(rs, base_prog); forwards += fw; f_, fw = ce(rs, fifteen); forwards += fw; t_, fw = ce(rs, twenty); forwards += fw
        sets[sname] = {"rows": int(rs.shape[0]), "native": n_, "program": p_ - n_, "fifteen": f_ - n_, "twenty": t_ - n_}
        print(f"{sname} ({rs.shape[0]} rows): native {n_:.5f} | program {p_ - n_:+.4f} | fifteen {f_ - n_:+.4f} | twenty {t_ - n_:+.4f}")
    print(f"skip7000: four-shape dictionary {c_d4 - nat:+.4f} (reg {REG['dictionary4']}) | joint value {c_all - nat:.4f} (reg {REG['joint']})")
    report = {"sets": sets, "dictionary4": c_d4 - nat, "joint_value": c_all - nat, "registered": REG}
    s7, frs = sets["skip7000"], sets["fresh"]
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    predictions = {"pred_a_native_replays": abs(s7["native"] - REG["native"]) <= REPLAY_TOL and abs(frs["native"] - REG["fresh_native"]) <= REPLAY_TOL,
                   "pred_b_program_replays": abs(s7["program"] - REG["program"]) <= COST_TOL and abs(c_all - nat - REG["joint"]) <= JOINT_TOL,
                   "pred_c_readable_replays": abs(s7["fifteen"] - REG["fifteen"]) <= COST_TOL and abs(s7["twenty"] - REG["twenty"]) <= COST_TOL,
                   "pred_d_dictionary_replays": abs(c_d4 - nat - REG["dictionary4"]) <= COST_TOL,
                   "pred_e_fresh_replays": abs(frs["fifteen"] - REG["fresh_fifteen"]) <= COST_TOL and abs(frs["twenty"] - REG["fresh_twenty"]) <= COST_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_canary2_result_v739", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
