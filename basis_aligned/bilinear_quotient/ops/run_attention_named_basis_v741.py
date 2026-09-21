#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_named4_cheap pred_c_named6_closer pred_d_named4_kernels_only pred_e_rotation_invariant
"""Attention lane, v741: a NAMED basis for the kernel subspace (Logan, 21 Sep: 'is this not a trivial decomposition? why would this set of
four shapes be better than a rotated set?').

The SVD shapes are a convention; only the 4-dimensional span is invariant. On the CPU: no rotation makes heads single-shape (varimax 0.73 vs
random rotations 0.62-0.76 dominant-shape energy), and a greedy pick from simple named functions spans 92% of the kernels' energy with
{exp(-d/6), exp(-d/3), d^-0.5, d^-1} (96% for the SVD's four; 93% adding exp(-d/1) and a tap at d = 2). This rung prices, inside the exact-rank
program (0.0723) and inside the kernels-only program (v646, 0.956): every kernel projected onto (i) the named 4 (two exponential windows,
two power-law tails), (ii) the named 6, (iii) a RANDOM rotation of the SVD 4-space (must equal the SVD 4-shape cost 0.0734 — the invariance
check). Response only. CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays     native within 0.002 of 3.13241 (instrument)
    pred_b_named4_cheap       named-4 kernels inside the program cost <= 0.0723 + 0.010. Prior: unsure
    pred_c_named6_closer      named-6 costs <= named-4 - 0.001 (the tap and the d = 1 window matter). Prior: likely
    pred_d_named4_kernels_only named-4 inside the kernels-only program costs <= 0.956 + 0.05. Prior: unsure
    pred_e_rotation_invariant a random rotation of the SVD 4-space costs within 0.0005 of the SVD 4-shape program (0.0734). Prior: certain (registered as the check)
PRICE (registered maximum): native 6; kappa_r 2; program 6; named-4, named-6, rotated-4 inside the program 18; kernels-only 6 + named-4 6; total 44 forwards; 0 backwards; 0 fits. Bar <= 50.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_named_basis_v741_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
KALL = ROOT / "circuits/followups/embedding_forward_model_kernel_refit_v646_kernels.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.named_basis_v741"
FORWARDS_MAX = 50
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, N4_TOL, N6_GAIN, KO_TOL, ROT_TOL = 0.002, 0.010, 0.001, 0.05, 0.0005
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_named4_cheap": "<= program + 0.010", "pred_c_named6_closer": "<= named4 - 0.001", "pred_d_named4_kernels_only": "<= 0.956 + 0.05", "pred_e_rotation_invariant": "within 0.0005 of 0.0734"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "n4_tol": N4_TOL, "n6_gain": N6_GAIN, "ko_tol": KO_TOL, "rot_tol": ROT_TOL}}
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
    dd = torch.arange(1, 512, dtype=torch.float64, device=dev)
    named = {"exp6": torch.exp(-dd / 6), "exp3": torch.exp(-dd / 3), "pow0.5": dd ** -0.5, "pow1": dd ** -1.0, "exp1": torch.exp(-dd / 1), "tap2": (dd == 2).double()}
    def basis_of(names):
        B = torch.stack([named[n] / named[n].norm() for n in names]); Q, _ = torch.linalg.qr(B.T); return Q.T          # orthonormal rows
    Kmat = torch.stack([kappa[k][1:512] for k in ALL]).double(); U, S, Vh = torch.linalg.svd(Kmat, full_matrices=False)
    gen = torch.Generator(device="cpu").manual_seed(741); R = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))[0].to(dev)
    bases = {"svd4": Vh[:4], "rotated4": R @ Vh[:4], "named4": basis_of(["exp6", "exp3", "pow0.5", "pow1"]), "named6": basis_of(["exp6", "exp3", "pow0.5", "pow1", "exp1", "tap2"])}

    def project(kap, B):
        M = torch.stack([kap[k][1:512] for k in ALL]).double(); P = (M @ B.T) @ B; out = {}
        for i_, k in enumerate(ALL):
            v = kap[k].clone(); v[1:512] = P[i_].float(); out[k] = v
        return out

    kall = torch.load(KALL, map_location=dev)["all162"]; kko = {k: kall[f"{k[0]}.{k[1]}"].to(dev).float() for k in ALL}
    def kernels_only(kap):
        return {k: {"kind": "kernel", "kappa": kap[k]} for k in ALL}
    report = {"native": nat, "program_cost": c_prog - nat, "inside_program": {}, "kernels_only": {}}
    for name, B in bases.items():
        energy = float(((Kmat @ B.T) ** 2).sum() / (Kmat ** 2).sum())
        c_, fw = ce(ev7, program(project(kappa, B))); forwards += fw
        report["inside_program"][name] = {"energy": energy, "cost": c_ - nat, "over_program": c_ - c_prog}
        print(f"{name} inside the program: energy {energy:.4f} | cost {c_ - nat:+.4f} (over the program {c_ - c_prog:+.4f})")
    c_ko, fw = ce(ev7, kernels_only(kko)); forwards += fw; c_kn4, fw = ce(ev7, kernels_only(project(kko, bases["named4"]))); forwards += fw
    report["kernels_only"] = {"as_fitted": c_ko - nat, "named4": c_kn4 - nat}
    print(f"kernels-only program: as fitted {c_ko - nat:+.4f} | named-4 {c_kn4 - nat:+.4f}")
    ip = report["inside_program"]
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_named4_cheap": ip["named4"]["over_program"] <= N4_TOL,
                   "pred_c_named6_closer": ip["named6"]["cost"] <= ip["named4"]["cost"] - N6_GAIN, "pred_d_named4_kernels_only": (c_kn4 - nat) <= 0.956 + KO_TOL,
                   "pred_e_rotation_invariant": abs(ip["rotated4"]["cost"] - 0.0734) <= ROT_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_named_basis_result_v741", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
