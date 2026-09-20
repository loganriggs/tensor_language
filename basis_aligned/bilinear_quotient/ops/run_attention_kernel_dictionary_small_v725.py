#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_dictionary2_cost pred_c_dictionary3_cheap pred_d_dictionary1_cost pred_e_basis_shapes
"""Attention lane, v725: the kernel dictionary at k = 1, 2, 3 — the smallest number of shared positional shapes — and what the shapes are.

v724: projecting all 162 kernels onto the top-4 SVD basis costs +0.001 over the exact-rank program (k = 8: +0.0001). This rung prices k = 1,
2, 3 the same way and records the basis kernels (values at d = 1..8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) and every
head's coefficients on the top-4 basis, as data for the reading. Response only. CE ADDED; program reference 0.0723.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays     native within 0.002 of 3.13241 (instrument)
    pred_b_dictionary2_cost   k = 2 costs > 0.010 above the program. Prior: unsure
    pred_c_dictionary3_cheap  k = 3 costs <= 0.005 above the program. Prior: unsure
    pred_d_dictionary1_cost   k = 1 costs > 0.05 above the program (one shape cannot serve both taps and long tails). Prior: likely
    pred_e_basis_shapes       the first basis kernel is a previous-token tap (|value at d = 1| >= 3x |value at d = 2|) and the second is long (|value at d = 64| >= 0.2 x |value at d = 1|). Prior: unsure
PRICE (registered maximum): native 6; kappa_r 2; program 6; three dictionaries 18; total 32 forwards; 0 backwards; 0 fits. Bar <= 40.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_kernel_dictionary_small_v725_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
KS = (1, 2, 3)
DS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.kernel_dictionary_small_v725"
FORWARDS_MAX = 40
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, COST2, COST3, COST1 = 0.002, 0.010, 0.005, 0.05
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_dictionary2_cost": "k=2 > program + 0.010", "pred_c_dictionary3_cheap": "k=3 <= program + 0.005", "pred_d_dictionary1_cost": "k=1 > program + 0.05", "pred_e_basis_shapes": "basis 1 a tap, basis 2 long"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "cost2": COST2, "cost3": COST3, "cost1": COST1}, "ks": list(KS)}
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
    basis = {str(i + 1): {str(d): float(Vh[i, d - 1]) for d in DS} for i in range(4)}
    coef = (U[:, :4] * S[:4]); report["basis"] = basis; report["coefficients"] = {f"{k[0]}.{k[1]}": [float(coef[i, j]) for j in range(4)] for i, k in enumerate(ALL)}
    for i in range(4):
        print(f"basis {i + 1} (sigma {float(S[i]):.3f}): " + " ".join(f"d{d}:{basis[str(i + 1)][str(d)]:+.3f}" for d in DS))
    b1, b2 = basis["1"], basis["2"]
    shapes_ok = abs(b1["1"]) >= 3 * abs(b1["2"]) and abs(b2["64"]) >= 0.2 * abs(b2["1"])
    print(f"native {nat:.5f} | energy cum (1..8): {[round(float(energy[i]), 3) for i in range(8)]} | shapes: basis-1 tap {abs(b1['1']) >= 3 * abs(b1['2'])}, basis-2 long {abs(b2['64']) >= 0.2 * abs(b2['1'])}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    dd = report["dictionary"]
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_dictionary2_cost": dd["2"]["over_program"] > COST2,
                   "pred_c_dictionary3_cheap": dd["3"]["over_program"] <= COST3, "pred_d_dictionary1_cost": dd["1"]["over_program"] > COST1, "pred_e_basis_shapes": bool(shapes_ok)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_kernel_dictionary_small_result_v725", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
