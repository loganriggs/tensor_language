#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_read_energy_concentrates pred_c_read256_cheap pred_d_read128_cost pred_e_per_layer_narrower
"""Attention lane, v726: the READ SUBSPACE — how many residual directions do all 162 heads' content maps read from?

The exact-rank program gives each head four maps U_r V_r with V_r (r x 1152) the input-side directions the head reads (r = 16 or 64;
4 x sum r = 11.6k rows in all). SVD of the stacked, per-row-normalised V rows gives the residual directions attention reads through;
we (i) report the energy captured by the top 64 / 128 / 256 / 512 directions, globally and per layer (each layer's 36-144 rows on its
own), and (ii) price projecting EVERY head's V onto the top-k global directions (k = 128, 256, 512) inside the otherwise unchanged
program (kappa_r recomputed). Native rms-normed inputs are 1152-d; a narrow read subspace would mean attention's content side lives in
a small coordinate system of the residual stream. Response only. CE ADDED; program reference 0.0723.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native within 0.002 of 3.13241 (instrument)
    pred_b_read_energy_concentrates the top-256 global directions carry >= 0.8 of the V rows' energy. Prior: unsure
    pred_c_read256_cheap         projecting onto the top-256 costs <= 0.02 over the program. Prior: unsure
    pred_d_read128_cost          projecting onto the top-128 costs > 0.05 over the program. Prior: unsure
    pred_e_per_layer_narrower    for >= 12 of 18 layers, the layer's own top-64 directions carry more of its rows' energy than the global top-64 do of all rows. Prior: likely
PRICE (registered maximum): native 6; kappa_r 2 + 3 x 2; program 6; three projections 18; total 38 forwards; 0 backwards; 0 fits. Bar <= 45.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_read_subspace_v726_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
KS = (128, 256, 512)
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt"}
PROGRAM_COST = 0.0723
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.read_subspace_v726"
FORWARDS_MAX = 40
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, ENERGY256, COST256, COST128, N_LAYERS = 0.002, 0.8, 0.02, 0.05, 12
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_read_energy_concentrates": "top-256 energy >= 0.8", "pred_c_read256_cheap": "k=256 <= program + 0.02", "pred_d_read128_cost": "k=128 > program + 0.05", "pred_e_per_layer_narrower": ">= 12 layers narrower than global at 64"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS),
            "bars": {"replay_tol": REPLAY_TOL, "energy256": ENERGY256, "cost256": COST256, "cost128": COST128, "n_layers": N_LAYERS}, "ks": list(KS)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
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
    facs = {k: {n_: (fac[f"{k[0]}.{k[1]}"][n_][0].to(dev).float(), fac[f"{k[0]}.{k[1]}"][n_][1].to(dev).float()) for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in ALL}
    kappa = {k: fac[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float() for k in ALL}

    def maps_from(proj):
        out = {}
        for k in ALL:
            out[k] = {n_: facs[k][n_][0] @ (facs[k][n_][1] if proj is None else facs[k][n_][1] @ proj) for n_ in facs[k]}
        return out

    def kappa_r_of(maps):
        acc = {k: 0 for k in ALL}; nb = 0; fw = 0
        with torch.no_grad():
            for s in range(0, 64, EBATCH):
                idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); fw += 1
                Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                for k in ALL:
                    acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps[k], hd, causal), dmat, off, Tn)
                nb += 1
        return {k: acc[k] / nb for k in ALL}, fw

    def program(maps, kr):
        return {k: {"kind": "lowrank", "kappa": kappa[k], "kappa_r": kr[k], "maps": maps[k]} for k in ALL}

    ev7 = rows["skip7000"]; nat, fw = ce(ev7); forwards += fw
    maps0 = maps_from(None); kr0, fw = kappa_r_of(maps0); forwards += fw; c_prog, fw = ce(ev7, program(maps0, kr0)); forwards += fw
    rows_all = torch.cat([F.normalize(facs[k][n_][1], dim=1) for k in ALL for n_ in facs[k]]).double()        # [~11.6k, 1152]
    U, S, Vh = torch.linalg.svd(rows_all, full_matrices=False); energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    per_layer = {}
    for l in LAYERS:
        rl = torch.cat([F.normalize(facs[(l, h)][n_][1], dim=1) for h in range(H) for n_ in facs[(l, h)]]).double()
        Sl = torch.linalg.svdvals(rl); el = (Sl ** 2).cumsum(0) / (Sl ** 2).sum()
        gl = (rl @ Vh[:64].T).pow(2).sum() / (rl ** 2).sum()
        per_layer[str(l)] = {"rows": int(rl.shape[0]), "own_top64": float(el[63]), "global_top64_on_layer": float(gl)}
    report = {"native": nat, "program_cost": c_prog - nat, "rows": int(rows_all.shape[0]), "energy": {str(k): float(energy[k - 1]) for k in (32, 64, 128, 256, 512, 768)}, "per_layer": per_layer, "projection": {}}
    for kk in KS:
        P = (Vh[:kk].T @ Vh[:kk]).float()
        mk = maps_from(P); krk, fw = kappa_r_of(mk); forwards += fw; c_k, fw = ce(ev7, program(mk, krk)); forwards += fw
        report["projection"][str(kk)] = {"cost": c_k - nat, "over_program": c_k - c_prog, "energy": float(energy[kk - 1])}
        print(f"read subspace k={kk}: energy {float(energy[kk - 1]):.3f} | cost {c_k - nat:+.4f} (program {c_prog - nat:+.4f}, over {c_k - c_prog:+.4f})")
    n_narrow = sum(1 for v in per_layer.values() if v["own_top64"] > float(energy[63]))
    print(f"native {nat:.5f} | global energy at 32/64/128/256/512: {[round(report['energy'][str(k)], 3) for k in (32, 64, 128, 256, 512)]} | per-layer own top-64: " + " ".join(f"{l}:{per_layer[str(l)]['own_top64']:.2f}" for l in LAYERS) + f" | layers narrower than global top-64 ({float(energy[63]):.2f}): {n_narrow}/18")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    pj = report["projection"]
    predictions = {"pred_a_native_replays": abs(nat - NATIVE_V615) <= REPLAY_TOL, "pred_b_read_energy_concentrates": report["energy"]["256"] >= ENERGY256,
                   "pred_c_read256_cheap": pj["256"]["over_program"] <= COST256, "pred_d_read128_cost": pj["128"]["over_program"] > COST128, "pred_e_per_layer_narrower": n_narrow >= N_LAYERS}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_read_subspace_result_v726", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
