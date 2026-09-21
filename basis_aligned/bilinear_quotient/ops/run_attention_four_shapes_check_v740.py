#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_dictionary_transfers pred_c_kernels_only_four_shapes pred_d_kernels_only_own_basis pred_e_kernels_only_two_shapes
"""Attention lane, v740: how real are the FOUR SHAPES? (Logan, 21 Sep) — the dictionary priced on the other two windows, and inside the
kernels-ONLY program where the kernels carry everything.

v724/v725 priced the 4-shape dictionary only on skip7000 and only inside the kernel + content program (0.0723 -> 0.0734), where the content
terms could be covering for the kernels. This rung: (i) the 4-shape program on the fresh 512-row window and skip1200 against the exact-rank
program; (ii) the all-162 FITTED KERNELS-ONLY program (v646: +0.956 on skip7000, no content anywhere) with every kernel projected onto the
v718-derived 4-shape basis, onto its OWN top-4 SVD basis, and onto its own top-2 / top-8. If four shapes suffice there too, the claim is
about the kernels, not about the content covering for them. Response only. CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          skip7000 native within 0.002 of 3.13241; fresh within 0.002 of 3.5061
    pred_b_dictionary_transfers    on fresh and skip1200 the 4-shape program costs <= the exact-rank program + 0.005. Prior: likely
    pred_c_kernels_only_four_shapes kernels-only program with the v718 4-shape basis costs <= 0.956 + 0.05. Prior: unsure
    pred_d_kernels_only_own_basis  kernels-only program with its own top-4 basis costs <= 0.956 + 0.02. Prior: unsure
    pred_e_kernels_only_two_shapes kernels-only program with its own top-2 basis costs > 0.956 + 0.05 (two shapes are not enough there either). Prior: likely
PRICE (registered maximum): skip7000: native 6, kappa_r 2, program 6, dict4 6, kernels-only 6, three projections 18; fresh: native 16, program 16, dict4 16; skip1200: 3 x 3;
total 119 forwards (native 6 + kappa_r 2 + skip7000 program, dict4, kernels-only + 4 projections = 42 + fresh 48 + skip1200 9; the first run tripped a 110 bar at 119 — re-registered); 0 backwards; 0 fits. Bar <= 125.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_four_shapes_check_v740_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
READINGS = {(1, 4): ("same", 0.05, 0.0037), (5, 5): ("induction", 0.10, 0.0071), (5, 7): ("sink", 0.55, 0.0058)}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt", "fresh": ROOT / "bilin18_eval_tokens_large.pt", "skip1200": ROOT / ".rowcache/fineweb_n96_skip1200.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.four_shapes_check_v740"
FORWARDS_MAX = 125
KALL = ROOT / "circuits/followups/embedding_forward_model_kernel_refit_v646_kernels.pt"
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
REPLAY_TOL, DICT_TOL, KO_TOL4, KO_TOL_OWN, KO_TOL2 = 0.002, 0.005, 0.05, 0.02, 0.05
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_dictionary_transfers": "<= program + 0.005 on fresh and skip1200", "pred_c_kernels_only_four_shapes": "<= 0.956 + 0.05", "pred_d_kernels_only_own_basis": "<= 0.956 + 0.02", "pred_e_kernels_only_two_shapes": "> 0.956 + 0.05"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "dict_tol": DICT_TOL, "ko_tol4": KO_TOL4, "ko_tol_own": KO_TOL_OWN, "ko_tol2": KO_TOL2}, "layer2": {str(k): v for k, v in L2.items()}, "steps": STEPS, "readings": {f"{k[0]}.{k[1]}": list(v) for k, v in READINGS.items()}, "gates": {"0": list(L0_SET), "1": list(L1_SET)}, "mean_head": MEAN_HEAD}
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

    Kmat = torch.stack([kappa[k][1:513] for k in ALL]).double(); U, S, Vh = torch.linalg.svd(Kmat, full_matrices=False); K4 = (U[:, :4] * S[:4]) @ Vh[:4]
    kap4 = {}
    for i_, k in enumerate(ALL):
        v = kappa[k].clone(); v[1:513] = K4[i_].float(); kap4[k] = v
    kall = torch.load(KALL, map_location=dev)["all162"]; kko = {k: kall[f"{k[0]}.{k[1]}"].to(dev).float() for k in ALL}

    def kernels_only(kap):
        return {k: {"kind": "kernel", "kappa": kap[k]} for k in ALL}

    def project(kap, basis):
        M = torch.stack([kap[k][1:513] for k in ALL]).double(); P = (M @ basis.T) @ basis; out = {}
        for i_, k in enumerate(ALL):
            v = kap[k].clone(); v[1:513] = P[i_].float(); out[k] = v
        return out

    Mko = torch.stack([kko[k][1:513] for k in ALL]).double(); Uo, So, Vho = torch.linalg.svd(Mko, full_matrices=False); eo = (So ** 2).cumsum(0) / (So ** 2).sum()
    sets = {}
    for sname, rs in rows.items():
        n_, fw = ce(rs); forwards += fw; p_, fw = ce(rs, base_prog); forwards += fw; d_, fw = ce(rs, program(kap4)); forwards += fw
        sets[sname] = {"rows": int(rs.shape[0]), "native": n_, "program": p_ - n_, "dict4": d_ - n_}
        print(f"{sname} ({rs.shape[0]} rows): native {n_:.5f} | exact-rank program {p_ - n_:+.4f} | 4-shape program {d_ - n_:+.4f}")
    c_ko, fw = ce(ev7, kernels_only(kko)); forwards += fw
    c_ko_v718, fw = ce(ev7, kernels_only(project(kko, Vh[:4]))); forwards += fw
    c_ko_own4, fw = ce(ev7, kernels_only(project(kko, Vho[:4]))); forwards += fw
    c_ko_own2, fw = ce(ev7, kernels_only(project(kko, Vho[:2]))); forwards += fw
    c_ko_own8, fw = ce(ev7, kernels_only(project(kko, Vho[:8]))); forwards += fw
    ko = {"as_fitted": c_ko - nat, "v718_basis4": c_ko_v718 - nat, "own_basis4": c_ko_own4 - nat, "own_basis2": c_ko_own2 - nat, "own_basis8": c_ko_own8 - nat, "own_energy_cum": [float(eo[i]) for i in range(8)],
          "basis_overlap_v718_vs_own4": float(((Vh[:4] @ Vho[:4].T) ** 2).sum() / 4)}
    print(f"kernels-only program (v646 fitted kernels, no content): as fitted {c_ko - nat:+.4f} | on the v718 4-shape basis {c_ko_v718 - nat:+.4f} | own top-4 {c_ko_own4 - nat:+.4f} | own top-2 {c_ko_own2 - nat:+.4f} | own top-8 {c_ko_own8 - nat:+.4f} | own energy cum {[round(float(eo[i]), 3) for i in range(8)]} | subspace overlap(v718 4, own 4) {ko['basis_overlap_v718_vs_own4']:.2f}")
    report = {"sets": sets, "kernels_only": ko}
    s7, frs, s12 = sets["skip7000"], sets["fresh"], sets["skip1200"]
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    predictions = {"pred_a_native_replays": abs(s7["native"] - 3.13241) <= REPLAY_TOL and abs(frs["native"] - 3.5061) <= REPLAY_TOL,
                   "pred_b_dictionary_transfers": frs["dict4"] <= frs["program"] + DICT_TOL and s12["dict4"] <= s12["program"] + DICT_TOL,
                   "pred_c_kernels_only_four_shapes": ko["v718_basis4"] <= 0.956 + KO_TOL4, "pred_d_kernels_only_own_basis": ko["own_basis4"] <= 0.956 + KO_TOL_OWN, "pred_e_kernels_only_two_shapes": ko["own_basis2"] > 0.956 + KO_TOL2}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_four_shapes_check_result_v740", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
