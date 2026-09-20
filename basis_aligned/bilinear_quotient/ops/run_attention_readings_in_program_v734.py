#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_readings_in_program_cheap pred_c_each_reading_holds_in_context pred_d_readings_additive pred_e_reading_heads_values_preserved
"""Attention lane, v734: do the three one-number readings hold INSIDE the joint program? (single edits nominate, joint edits certify)

v729 / v731 / v733 read three heads as one number each, alone inside the native model: 1.4 = kappa + 0.05 [tok_i == tok_j]; 5.5 = kappa +
0.10 [tok_i == tok_{j-1}]; 5.7 = kappa with 0.55 on the position-0 column. This rung installs each reading in place of the head's rank-r
content term inside the exact-rank program (v718; all other 159-161 heads as fitted), one at a time and all three together, and mean-ablates
the three heads inside the all-three program. Response only. CE ADDED; program reference 0.0723; single-head standalone costs 0.0037 /
0.0071 / 0.0058.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays            native within 0.002 of 3.13241 (instrument)
    pred_b_readings_in_program_cheap the all-three program costs <= 0.0723 + 0.015. Prior: unsure
    pred_c_each_reading_holds_in_context each single replacement costs <= its standalone single-head cost + 0.003 above the program. Prior: likely
    pred_d_readings_additive         the all-three cost above the program <= the sum of the three single excesses + 0.003. Prior: unsure
    pred_e_reading_heads_values_preserved inside the all-three program, each of the three heads' mean-ablation value is within [0.5, 2] x native. Prior: unsure
PRICE (registered maximum): native 6; kappa_r 2; program 6; three singles 18; all three 6; manipulability 18; total 56 forwards; 0 backwards; 0 fits. Bar <= 65.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_readings_in_program_v734_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
READINGS = {(1, 4): ("same", 0.05, 0.0037), (5, 5): ("induction", 0.10, 0.0071), (5, 7): ("sink", 0.55, 0.0058)}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.readings_in_program_v734"
FORWARDS_MAX = 65
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, ALL_TOL, CTX_TOL, ADD_TOL, RATIO = 0.002, 0.015, 0.003, 0.003, (0.5, 2.0)
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_readings_in_program_cheap": "all three <= program + 0.015", "pred_c_each_reading_holds_in_context": "single excess <= standalone + 0.003", "pred_d_readings_additive": "all-three excess <= sum + 0.003", "pred_e_reading_heads_values_preserved": "ratios in [0.5, 2]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "all_tol": ALL_TOL, "ctx_tol": CTX_TOL, "add_tol": ADD_TOL, "ratio": RATIO}, "readings": {f"{k[0]}.{k[1]}": list(v) for k, v in READINGS.items()}}
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

    report = {"native": nat, "program_cost": c_prog - nat, "singles": {}, "manipulability": {}}
    for k, (kind, beta, standalone) in READINGS.items():
        c_, fw = ce(ev7, with_readings([k])); forwards += fw
        report["singles"][f"{k[0]}.{k[1]}"] = {"kind": kind, "beta": beta, "cost": c_ - nat, "excess": c_ - c_prog, "standalone": standalone}
        print(f"reading {k[0]}.{k[1]} ({kind} {beta}) inside the program: cost {c_ - nat:+.4f}, excess over the program {c_ - c_prog:+.4f} (standalone single-head cost {standalone:+.4f})")
    all3 = with_readings(list(READINGS)); c_all, fw = ce(ev7, all3); forwards += fw
    report["all_three"] = {"cost": c_all - nat, "excess": c_all - c_prog}
    v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    for k in READINGS:
        state["ablate"] = {k}; c_, fw = ce(ev7, all3); forwards += fw; state["ablate"] = set()
        key = f"{k[0]}.{k[1]}"; report["manipulability"][key] = {"native_value": v701[key], "in_program_value": c_ - c_all, "ratio": (c_ - c_all) / v701[key]}
    sum_single = sum(v["excess"] for v in report["singles"].values())
    print(f"all three readings: cost {c_all - nat:+.4f}, excess {c_all - c_prog:+.4f} (sum of singles {sum_single:+.4f}) | values inside: " + " ".join(f"{k}:{v['native_value']:.4f}->{v['in_program_value']:.4f} ({v['ratio']:.2f}x)" for k, v in report["manipulability"].items()))
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_readings_in_program_cheap": (c_all - nat) <= 0.0723 + ALL_TOL,
                   "pred_c_each_reading_holds_in_context": all(v["excess"] <= v["standalone"] + CTX_TOL for v in report["singles"].values()),
                   "pred_d_readings_additive": (c_all - c_prog) <= sum_single + ADD_TOL, "pred_e_reading_heads_values_preserved": all(RATIO[0] <= v["ratio"] <= RATIO[1] for v in report["manipulability"].values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_readings_in_program_result_v734", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
