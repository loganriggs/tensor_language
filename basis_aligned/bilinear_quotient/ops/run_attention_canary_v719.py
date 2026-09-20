#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_joint_value_replays pred_c_v713_replays pred_d_whole_program_exact_rank_replays pred_e_fresh_replays
"""Attention lane, v719: CANARY — replay the chapter's registered numbers from the saved programs (no fit).

Registered (skip7000, 192 x 512): native 3.13241 (v615); joint value 3.9961 (v701); v713 pattern program as saved 0.0726 (v716/v717 replays);
the whole-attention program of v716 (pattern maps re-truncated to nominal ranks + rank-64 write maps A_h B_h, validation-minimum snapshot)
0.2412 (its exact-rank number). Registered (fresh 512-row window): native 3.5061, v713 0.0782 (v717). Anyone can re-run this to check the
lane's numbers; a drift beyond the bars means an instrument or a file changed. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays                  skip7000 native within 0.002 of 3.13241
    pred_b_joint_value_replays             all-162 mean-ablation joint value within 0.02 of 3.9961
    pred_c_v713_replays                    v713 program cost within 0.003 of 0.0726
    pred_d_whole_program_exact_rank_replays v716 snapshot (pattern + write) cost within 0.005 of 0.2412
    pred_e_fresh_replays                   fresh native within 0.002 of 3.5061 and fresh v713 within 0.003 of 0.0782
PRICE (registered maximum): skip7000: native 6 + all-ablated 6 + v713 6 + v716 6 = 24; kappa_r 2 + 2; fresh: native 16 + v713 16 = 32; total 60 forwards; 0 backwards; 0 fits. Bar <= 70.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_canary_v719_result.json"
SNAP716 = ROOT / "circuits/followups/attention_whole_program_v716_snapshot.pt"
PROGS = {"v713": (ROOT / "circuits/followups/attention_band68_r64_fit_v713_programs.pt", "band68_r64")}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt", "fresh": ROOT / "bilin18_eval_tokens_large.pt"}
REG = {"native": 3.13241, "joint": 3.9961, "v713": 0.0726, "v716": 0.2412, "fresh_native": 3.5061, "fresh_v713": 0.0782}
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.canary_v719"
FORWARDS_MAX = 70
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, JOINT_TOL, COST_TOL, WHOLE_TOL = 0.002, 0.02, 0.003, 0.005
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_joint_value_replays": "+-0.02", "pred_c_v713_replays": "+-0.003", "pred_d_whole_program_exact_rank_replays": "+-0.005", "pred_e_fresh_replays": "+-0.002 / +-0.003"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS), "programs": list(PROGS),
            "bars": {"replay_tol": REPLAY_TOL, "joint_tol": JOINT_TOL, "cost_tol": COST_TOL, "whole_tol": WHOLE_TOL}, "registered": REG}
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

    programs = {}
    for name, (path, arm) in PROGS.items():
        saved = torch.load(path, map_location=dev)[arm]
        maps = {k: {n_: saved[f"{k[0]}.{k[1]}"][n_].to(dev).float() for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in ALL}
        acc = {k: 0 for k in ALL}; nb = 0
        with torch.no_grad():
            for s in range(0, 64, EBATCH):
                idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
                Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                for k in ALL:
                    acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps[k], hd, causal), dmat, off, Tn)
                nb += 1
        programs[name] = {k: {"kind": "lowrank", "kappa": saved[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float(), "kappa_r": acc[k] / nb, "maps": maps[k]} for k in ALL}
    snap = torch.load(SNAP716, map_location=dev)
    maps716 = {k: {n_: snap[f"{k[0]}.{k[1]}"][n_].to(dev).float() for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in ALL}
    acc = {k: 0 for k in ALL}; nb = 0
    with torch.no_grad():
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
            Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            for k in ALL:
                acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps716[k], hd, causal), dmat, off, Tn)
            nb += 1
    prog716 = {k: {"kind": "lowrank", "kappa": snap[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float(), "kappa_r": acc[k] / nb, "maps": maps716[k]} for k in ALL}
    writes716 = {k: (snap[f"{k[0]}.{k[1]}"]["A"].to(dev).float(), snap[f"{k[0]}.{k[1]}"]["B"].to(dev).float()) for k in ALL}
    report = {}
    ev7 = rows["skip7000"]; nat, fw = ce(ev7); forwards += fw
    c713, fw = ce(ev7, programs["v713"]); forwards += fw
    state["ablate"] = set(ALL); c_all, fw = ce(ev7); forwards += fw; state["ablate"] = set()
    c716, fw = ce(ev7, prog716, writes716); forwards += fw
    fr = rows["fresh"]; nat_f, fw = ce(fr); forwards += fw; c713_f, fw = ce(fr, programs["v713"]); forwards += fw
    report = {"skip7000": {"native": nat, "joint_value": c_all - nat, "v713": c713 - nat, "v716_exact_rank": c716 - nat}, "fresh": {"native": nat_f, "v713": c713_f - nat_f}, "registered": REG}
    print(f"skip7000: native {nat:.5f} (reg {REG['native']}) | joint value {c_all - nat:.4f} (reg {REG['joint']}) | v713 {c713 - nat:+.4f} (reg {REG['v713']}) | v716 exact-rank whole program {c716 - nat:+.4f} (reg {REG['v716']})")
    print(f"fresh: native {nat_f:.5f} (reg {REG['fresh_native']}) | v713 {c713_f - nat_f:+.4f} (reg {REG['fresh_v713']})")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    predictions = {"pred_a_native_replays": abs(nat - REG["native"]) <= REPLAY_TOL, "pred_b_joint_value_replays": abs(c_all - nat - REG["joint"]) <= JOINT_TOL,
                   "pred_c_v713_replays": abs(c713 - nat - REG["v713"]) <= COST_TOL, "pred_d_whole_program_exact_rank_replays": abs(c716 - nat - REG["v716"]) <= WHOLE_TOL,
                   "pred_e_fresh_replays": abs(nat_f - REG["fresh_native"]) <= REPLAY_TOL and abs(c713_f - nat_f - REG["fresh_v713"]) <= COST_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_canary_result_v719", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
