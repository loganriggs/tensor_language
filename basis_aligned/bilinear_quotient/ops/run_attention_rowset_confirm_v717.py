#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_program_cost_transfers pred_c_joint_value_transfers pred_d_head_values_transfer pred_e_in_program_values_transfer
"""Attention lane, v717: ROW-SET confirmation — the chapter's numbers on a window never used for anything.

Every CE number of v701-v716 was priced on the 192 x 512 skip7000 rows (fits on skip80 + skip11000). This rung replays, with no fit, on the
FRESH window bilin18_eval_tokens_large.pt (512 rows x 513 tokens; measured 2026-08-30 as having zero 24-token-prefix overlap with every fit and
eval set) and on skip1200 (96 rows): (i) native CE; (ii) the v713 and v715 pattern programs (saved endpoint maps; kappa_r recomputed on 64 fit
rows as always); (iii) the joint value (all 162 heads mean-ablated); (iv) the native mean-ablation values of the 24 most valuable heads (fresh
window only); (v) their in-program values under v715 (fresh window only). Transfer = agreement with the skip7000 numbers. Response only.
CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          skip7000 native CE within 0.002 of 3.13241 (instrument; the fresh window's native CE is reported, not predicted)
    pred_b_program_cost_transfers  v713 and v715 costs on the fresh window within 0.015 of their skip7000 endpoint costs (replayed here). Prior: likely
    pred_c_joint_value_transfers   the fresh-window joint value within 10% of 3.996. Prior: likely
    pred_d_head_values_transfer    Spearman(fresh native values, skip7000 native values) over the 24 heads >= 0.8. Prior: likely
    pred_e_in_program_values_transfer under v715 on the fresh window: Spearman(fresh native, fresh in-program) >= 0.75 and median ratio in [0.67, 1.5]. Prior: unsure
PRICE (registered maximum): skip7000 (6 forwards per eval): native, v713, v715 = 18; kappa_r 2 x 2 = 4; fresh (16 per eval): native, v713, v715, all-ablated = 64;
skip1200 (3 per eval): the same four = 12; skip7000 all-ablated 6; fresh top-24 native values 24 x 16 = 384; fresh v715 in-program values 24 x 16 = 384;
total ~872 forwards; 0 backwards; 0 fits. Bar <= 900.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_rowset_confirm_v717_result.json"
PROGS = {"v713": (ROOT / "circuits/followups/attention_band68_r64_fit_v713_programs.pt", "band68_r64"), "v715": (ROOT / "circuits/followups/attention_bands08_r64_fit_v715_programs.pt", "bands08_r64")}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
SETS = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt", "fresh": ROOT / "bilin18_eval_tokens_large.pt", "skip1200": ROOT / ".rowcache/fineweb_n96_skip1200.pt"}
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615, JOINT_VALUE = 3.13241, 3.9961
CANDIDATE_ID = "attention.rowset_confirm_v717"
FORWARDS_MAX = 900
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
N_MANIP = 24
REPLAY_TOL, COST_TOL, JOINT_REL, RHO_D, RHO_E, RATIO = 0.002, 0.015, 0.10, 0.8, 0.75, (0.67, 1.5)
PREDICTIONS = {"pred_a_native_replays": "+-0.002 (skip7000)", "pred_b_program_cost_transfers": "fresh within 0.015 of skip7000", "pred_c_joint_value_transfers": "within 10% of 3.996",
               "pred_d_head_values_transfer": "Spearman >= 0.8", "pred_e_in_program_values_transfer": "Spearman >= 0.75 and median ratio in [0.67, 1.5]"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "sets": list(SETS), "programs": list(PROGS),
            "bars": {"replay_tol": REPLAY_TOL, "cost_tol": COST_TOL, "joint_rel": JOINT_REL, "rho_d": RHO_D, "rho_e": RHO_E, "ratio": RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    rows = {name: torch.load(p, map_location="cpu").long() for name, p in SETS.items()}
    fit64 = torch.load(FIT_ROWS, map_location="cpu").long()[:64]
    means = torch.load(MEANS, map_location=dev); v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    state = {"idx": None, "programs": {}, "n": {}, "ablate": set()}
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
            if state["ablate"]:
                z = z.clone()
                for (al, h) in state["ablate"]:
                    if al == l:
                        z[:, h] = means[f"mean_{l}_{h}"]
            return z
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(rs, progs=None):
        state["programs"] = progs or {}; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rs.shape[0], EBATCH):
                idx = rs[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rs[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["programs"] = {}
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
    report = {}
    for sname, rs in rows.items():
        nat, fw = ce(rs); forwards += fw
        costs = {}
        for pname, prog in programs.items():
            c_, fw = ce(rs, prog); forwards += fw; costs[pname] = c_ - nat
        state["ablate"] = set(ALL); c_all, fw = ce(rs); forwards += fw; state["ablate"] = set()
        report[sname] = {"rows": int(rs.shape[0]), "native": nat, "program_cost": costs, "joint_value": c_all - nat}
        print(f"{sname} ({rs.shape[0]} rows): native {nat:.4f} | v713 {costs['v713']:+.4f} | v715 {costs['v715']:+.4f} | joint value {c_all - nat:.3f}")
    top = sorted(v701, key=v701.get, reverse=True)[:N_MANIP]; fresh = rows["fresh"]; nat_f = report["fresh"]["native"]
    values_f, inprog_f = {}, {}
    for key in top:
        l, h = map(int, key.split(".")); state["ablate"] = {(l, h)}
        c_, fw = ce(fresh); forwards += fw; values_f[key] = c_ - nat_f
        c2, fw = ce(fresh, programs["v715"]); forwards += fw; inprog_f[key] = c2 - (nat_f + report["fresh"]["program_cost"]["v715"])
        state["ablate"] = set()

    def spearman(x, y):
        rx = torch.tensor(x).argsort().argsort().double(); ry = torch.tensor(y).argsort().argsort().double(); rx -= rx.mean(); ry -= ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))

    rho_d = spearman([values_f[k] for k in top], [v701[k] for k in top]); rho_e = spearman([values_f[k] for k in top], [inprog_f[k] for k in top])
    ratios = sorted(inprog_f[k] / values_f[k] for k in top if values_f[k] > 0); med_ratio = ratios[len(ratios) // 2]
    report["fresh"]["head_values"] = {k: {"skip7000": v701[k], "fresh": values_f[k], "fresh_in_v715": inprog_f[k]} for k in top}
    report["spearman_fresh_vs_skip7000"] = rho_d; report["spearman_fresh_native_vs_in_program"] = rho_e; report["median_ratio_fresh"] = med_ratio
    print("fresh head values (skip7000 -> fresh -> fresh in v715): " + " ".join(f"{k}:{v701[k]:.3f}->{values_f[k]:.3f}->{inprog_f[k]:.3f}" for k in top))
    print(f"Spearman fresh vs skip7000 values {rho_d:.3f}; fresh native vs in-program {rho_e:.3f}; median ratio {med_ratio:.2f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    s7, fr = report["skip7000"], report["fresh"]
    predictions = {"pred_a_native_replays": abs(s7["native"] - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_program_cost_transfers": all(abs(fr["program_cost"][p] - s7["program_cost"][p]) <= COST_TOL for p in PROGS),
                   "pred_c_joint_value_transfers": abs(fr["joint_value"] - JOINT_VALUE) <= JOINT_REL * JOINT_VALUE, "pred_d_head_values_transfer": rho_d >= RHO_D,
                   "pred_e_in_program_values_transfer": rho_e >= RHO_E and RATIO[0] <= med_ratio <= RATIO[1]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_rowset_confirm_result_v717", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
