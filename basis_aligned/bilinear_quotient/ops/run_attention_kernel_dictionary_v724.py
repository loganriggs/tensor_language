#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_kernel_energy_concentrates pred_c_dictionary8_cheap pred_d_dictionary4_cost pred_e_dictionary16_free
"""Attention lane, v724: a KERNEL DICTIONARY — how many basis kernels do the 162 heads share?

The exact-rank program (v718) carries one kernel per head (162 x 512 numbers, d = 1..512). SVD of that matrix gives basis kernels; each
head's kernel is replaced by its projection onto the top-k basis (k = 4, 8, 16) inside the otherwise unchanged program (content maps and
kappa_r as saved), and priced on skip7000. Also the energy the top-k basis captures. Response only (closed-form edits). CE ADDED; the
program's own cost (0.0723) is the reference.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native within 0.002 of 3.13241 (instrument)
    pred_b_kernel_energy_concentrates the top-8 basis kernels carry >= 95% of the 162 kernels' energy. Prior: likely
    pred_c_dictionary8_cheap       k = 8 costs <= 0.010 above the program (0.0723). Prior: unsure
    pred_d_dictionary4_cost        k = 4 costs > 0.010 above the program (four shapes are not enough). Prior: unsure
    pred_e_dictionary16_free       k = 16 costs <= 0.003 above the program. Prior: likely
PRICE (registered maximum): native 6; kappa_r 2; program 6; three dictionaries 18; total 32 forwards; 0 backwards; 0 fits. Bar <= 40.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_kernel_dictionary_v724_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
KS = (4, 8, 16)
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.kernel_dictionary_v724"
FORWARDS_MAX = 40
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, ENERGY8, COST8, COST4, COST16 = 0.002, 0.95, 0.010, 0.010, 0.003
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_kernel_energy_concentrates": "top-8 energy >= 0.95", "pred_c_dictionary8_cheap": "k=8 <= program + 0.010", "pred_d_dictionary4_cost": "k=4 > program + 0.010", "pred_e_dictionary16_free": "k=16 <= program + 0.003"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "energy8": ENERGY8, "cost8": COST8, "cost4": COST4, "cost16": COST16}, "ks": list(KS)}
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
    c_prog, fw = ce(ev7, program(kappa)); forwards += fw
    Kmat = torch.stack([kappa[k][1:513] for k in ALL]).double()                 # [162, 512]
    U, S, Vh = torch.linalg.svd(Kmat, full_matrices=False); energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    report = {"native": nat, "program_cost": c_prog - nat, "energy_cum": [float(energy[i]) for i in range(32)], "dictionary": {}}
    for kk in KS:
        Kk = (U[:, :kk] * S[:kk]) @ Vh[:kk]
        kap_k = {}
        for i, k in enumerate(ALL):
            v = kappa[k].clone(); v[1:513] = Kk[i].float(); kap_k[k] = v
        c_k, fw = ce(ev7, program(kap_k)); forwards += fw
        report["dictionary"][str(kk)] = {"cost": c_k - nat, "over_program": c_k - c_prog, "energy": float(energy[kk - 1])}
        print(f"dictionary k={kk}: energy {float(energy[kk - 1]):.4f} | cost {c_k - nat:+.4f} (program {c_prog - nat:+.4f}, over {c_k - c_prog:+.4f})")
    print(f"native {nat:.5f} | energy cum (1..16): {[round(float(energy[i]), 3) for i in range(16)]}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    dd = report["dictionary"]
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_kernel_energy_concentrates": dd["8"]["energy"] >= ENERGY8,
                   "pred_c_dictionary8_cheap": dd["8"]["over_program"] <= COST8, "pred_d_dictionary4_cost": dd["4"]["over_program"] > COST4, "pred_e_dictionary16_free": dd["16"]["over_program"] <= COST16}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_kernel_dictionary_result_v724", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
