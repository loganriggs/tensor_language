#!/usr/bin/env python
"""site_write_certificate_map_probe -- the §2699 second-order Fisher certificate applied to ALL 36 write sites at k = 32 (scored
against §2696's measured price map, docs 96-159) and the JOINT {mlp16, mlp17} certificate at k = 8 / 32 (scored against §2694's
measured `both` prices, docs 96-191). One score pass per batch prices every site at once.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_late_sites_certified pred_c_early_breakdown pred_d_joint_mlp16_17_k8 pred_e_ordering

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs -- LOWER IS BETTER.
Preregistration: polynomial_causal/SITE_WRITE_CERTIFICATE_MAP_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/site_write_certificate_map_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import site_write_pca_truncation_ce_map_probe as M

ROOT = R.ROOT
PREREG = R.POLY / "SITE_WRITE_CERTIFICATE_MAP_PROBE_PREREGISTRATION.md"
PRIOR_MAP = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"
PRIOR_LR = ROOT / "mlp_final_blocks_low_rank_surrogate_probe_results.json"
PRIOR_FC = ROOT / "mlp_final_blocks_fisher_certificate_probe_results.json"
OUT = ROOT / "site_write_certificate_map_probe_results.json"
HASHES = {PREREG: "730eb9bdaa4f453081f1c22e3a6a338c68e093bd0dfa64b159a3acc58eb06ceb",
          PRIOR_MAP: "48bd52ec9201ac97cddcd102cef61885baaaa8a8362b232850159f6f646d0e00",
          PRIOR_LR: "8a88b71455c8087ca5de84cfa83e71df35080236932b4ccbd466ce0bc089326e",
          PRIOR_FC: "1ef013515e16ba491534f7c3b4496bfc6546529d1e816f18840f092cc4307022",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "site_write_certificate_map_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI, CH = M.TI, M.CH
FIT = (0, 96); EVAL_A = (96, 160); EVAL_B = (96, 192)
K_MAP = 32; S_EVAL = 4
LATE_BLOCKS = range(7, NL); LATE_MIN_PRICE = 0.02
EARLY = [("mlp", 0), ("mlp", 1), ("mlp", 2), ("mlp", 3)]
JOINT = [("mlp", 16), ("mlp", 17)]
BARS = {"ce_tol": 1e-4, "cert17_k32_tol": 0.01, "ratio_lo": 0.5, "ratio_hi": 2.0, "cross_min": 0.02, "spearman_min": 0.8}
NULLS = {"late_n_outside": 5, "ratio_lo": 0.25, "ratio_hi": 4.0, "cross_max": 0.0, "spearman_max": 0.4}
ALL_SITES = [(kind, l) for l in range(NL) for kind in ("attn", "mlp")]


def check_hashes():
    for p, h in HASHES.items():
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def sname(s):
    return f"{s[0]}{s[1]}"


def forward_leaves(m, idx, leaf_sites):
    """M.forward semantics with the writes of leaf_sites detached into autograd leaves. Returns logits, {site: leaf}."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = R.rope(Tn)
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
    leaves = {}
    for l, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; h = F.rms_norm(x, (D,))
        def pr(lin):
            return R.rot(F.rms_norm(lin(h).view(B, Tn, NH, HD), (HD,)), cos, sin)
        v = a.c_v(h).view(B, Tn, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        pat = (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q), pr(a.c_k)) / HD) * (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q2), pr(a.c_k2)) / HD)
        pat = pat.masked_fill(~mask, 0.0)
        aw = a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
        if ("attn", l) in leaf_sites:
            aw = aw.detach().requires_grad_(True); leaves[("attn", l)] = aw
        x = x + aw
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        if ("mlp", l) in leaf_sites:
            mw = mw.detach().requires_grad_(True); leaves[("mlp", l)] = mw
        x = x + mw + mlp.Down_bias
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30), leaves


def certificate_pass(m, rows, bases, sites, ks, joint_sets, gen, log):
    """Accumulates single-site cert1/cert2 per (site, k) and joint cert per (set, k) over rows. Returns per-position means."""
    c1 = {(s, k): 0.0 for s in sites for k in ks}; c2 = {(s, k): 0.0 for s in sites for k in ks}
    cj = {(tuple(A), k): 0.0 for A in joint_sets for k in ks}
    n_pos = 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        with torch.enable_grad():
            lg, leaves = forward_leaves(m, idx, set(sites))
            lp = F.log_softmax(lg.float(), -1).reshape(-1, V)
            order = list(sites); leaf_list = [leaves[s] for s in order]
            loss_true = -lp.gather(1, tgt.reshape(-1, 1)).sum()
            g = torch.autograd.grad(loss_true, leaf_list, retain_graph=True)
            p = lp.exp().detach()
            samp = []
            for _ in range(S_EVAL):
                y = torch.multinomial(p, 1, generator=gen).squeeze(1)
                ll = lp.gather(1, y[:, None]).sum()
                samp.append(torch.autograd.grad(ll, leaf_list, retain_graph=True))
        with torch.no_grad():
            deltas = {}
            for si, s in enumerate(order):
                dev = leaves[s].detach().reshape(-1, D) - bases[s]["mu"]; U = bases[s]["U"]
                for k in ks:
                    delta = dev if k == 0 else dev - (dev @ U[:, :k]) @ U[:, :k].T
                    deltas[(s, k)] = delta
                    gd = (g[si].reshape(-1, D) * delta).sum(1)                        # [N]
                    sd = torch.stack([(sm[si].reshape(-1, D) * delta).sum(1) for sm in samp])  # [S, N]
                    c1[(s, k)] += float(gd.sum()); c2[(s, k)] += float(0.5 * (sd ** 2).mean(0).sum())
            for A in joint_sets:
                for k in ks:
                    gd = sum((g[order.index(s)].reshape(-1, D) * deltas[(s, k)]).sum(1) for s in A)
                    sd = sum(torch.stack([(sm[order.index(s)].reshape(-1, D) * deltas[(s, k)]).sum(1) for sm in samp]) for s in A)
                    cj[(tuple(A), k)] += float(gd.sum()) + float(0.5 * (sd ** 2).mean(0).sum())
            n_pos += lp.shape[0]
        del g, samp, leaves, lg, lp, deltas
        log(i + CH)
    single = {(s, k): (c1[(s, k)] + c2[(s, k)]) / n_pos for s in sites for k in ks}
    first = {(s, k): (c1[(s, k)] / n_pos) for s in sites for k in ks}
    joint = {(A, k): v / n_pos for (A, k), v in cj.items()}
    return single, first, joint, n_pos


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
    if smoke:
        gg = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (6, 257), generator=gg); fit, evA, evB = nat[:2], nat[2:4], nat[2:6]
        sites_A = [("attn", 1), ("mlp", 1), ("mlp", 16), ("mlp", 17)]
        measured = {sname(s): float("nan") for s in ALL_SITES}; base_prior = float("nan"); both = {"8": float("nan"), "32": float("nan")}
        fc_prior = {"17_32": float("nan"), "16_32": float("nan"), "17_8": float("nan"), "16_8": float("nan")}
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit, evA, evB = nat[FIT[0]:FIT[1]], nat[EVAL_A[0]:EVAL_A[1]], nat[EVAL_B[0]:EVAL_B[1]]
        sites_A = ALL_SITES
        pm = json.load(open(PRIOR_MAP)); measured = {r["site"]: r["ce_added_k32"] for r in pm["sites"]}; base_prior = pm["baseline_ce_eval"]
        pl = json.load(open(PRIOR_LR)); both = pl["ce_added_ladder"]["both"]; lad = pl["ce_added_ladder"]
        pf = json.load(open(PRIOR_FC)); cp = pf["certificate_pred"]
        fc_prior = {"17_32": cp["17"]["32"], "16_32": cp["16"]["32"], "17_8": cp["17"]["8"], "16_8": cp["16"]["8"]}
    m = R.load_model()
    bases = M.fit_bases(m, fit)
    print(json.dumps({"stage": "bases", "t": round(time.time() - started, 1)}), flush=True)
    ce0_A = M.ce_of(m, evA)
    # ---- Stage A: all sites, k = 32, docs 96-159
    def logA(n):
        print(json.dumps({"stage": "cert_A", "docs": n, "t": round(time.time() - started, 1)}), flush=True)
    single_A, first_A, _, n_A = certificate_pass(m, evA, bases, sites_A, [K_MAP], [], gen, logA)
    # ---- Stage B: mlp16/17 single + joint, k = 8 and 32, docs 96-191
    def logB(n):
        print(json.dumps({"stage": "cert_B", "docs": n, "t": round(time.time() - started, 1)}), flush=True)
    single_B, first_B, joint_B, n_B = certificate_pass(m, evB, bases, JOINT, [8, K_MAP], [JOINT], gen, logB)
    # ---- tables
    rows = []
    for s in sites_A:
        c = single_A[(s, K_MAP)]; meas = measured[sname(s)]
        rows.append({"site": sname(s), "kind": s[0], "block": s[1], "measured_k32": meas, "cert_k32": c,
                     "first_order_share": (first_A[(s, K_MAP)] / c) if c else None, "ratio": (meas / c) if c else None})
    late = [r for r in rows if r["block"] in LATE_BLOCKS and r["measured_k32"] == r["measured_k32"] and r["measured_k32"] >= LATE_MIN_PRICE]
    early = {r["site"]: r["ratio"] for r in rows if (r["kind"], r["block"]) in EARLY}
    inside = lambda x, lo, hi: x is not None and x == x and lo <= x <= hi
    rho = R.spearman(torch.tensor([r["cert_k32"] for r in rows]), torch.tensor([r["measured_k32"] for r in rows])) if not smoke else float("nan")
    jt = {str(k): joint_B[(tuple(JOINT), k)] for k in (8, K_MAP)}
    sg = {f"{s[1]}_{k}": single_B[(s, k)] for s in JOINT for k in (8, K_MAP)}
    cross = {str(k): jt[str(k)] - sg[f"16_{k}"] - sg[f"17_{k}"] for k in (8, K_MAP)}
    ratio_joint = {str(k): (both[str(k)] / jt[str(k)]) if jt[str(k)] else None for k in (8, K_MAP)}
    meas_cross = {str(k): (both[str(k)] - lad["16"][str(k)] - lad["17"][str(k)]) for k in (8, K_MAP)} if not smoke else {}
    inst = {"baseline_ce_A": ce0_A, "prior_baseline_A": base_prior, "abs_diff": abs(ce0_A - base_prior) if not smoke else 0.0,
            "cert17_k32_B": sg["17_32"], "prior_cert17_k32": fc_prior["17_32"], "cert_abs_diff": abs(sg["17_32"] - fc_prior["17_32"]) if not smoke else 0.0,
            "disclosed_vs_2699": {k: {"here": sg[k.replace("_k", "_")], "prior": v} for k, v in {"16_k32": fc_prior["16_32"], "17_k8": fc_prior["17_8"], "16_k8": fc_prior["16_8"]}.items()}}
    preds = {"pred_a_instrument": bool(inst["abs_diff"] <= BARS["ce_tol"] and inst["cert_abs_diff"] <= BARS["cert17_k32_tol"]),
             "pred_b_late_sites_certified": bool(not smoke and len(late) > 0 and all(inside(r["ratio"], BARS["ratio_lo"], BARS["ratio_hi"]) for r in late)),
             "pred_c_early_breakdown": bool(not smoke and "mlp1" in early and not inside(early["mlp1"], BARS["ratio_lo"], BARS["ratio_hi"])),
             "pred_d_joint_mlp16_17_k8": bool(not smoke and inside(ratio_joint["8"], BARS["ratio_lo"], BARS["ratio_hi"]) and cross["8"] >= BARS["cross_min"]),
             "pred_e_ordering": bool(not smoke and rho >= BARS["spearman_min"])}
    nulls = {"b_null_ge_5_late_outside_[.25,4]": bool(not smoke and sum(not inside(r["ratio"], NULLS["ratio_lo"], NULLS["ratio_hi"]) for r in late) >= NULLS["late_n_outside"]),
             "c_null_all_early_inside_[.5,2]": bool(not smoke and len(early) == 4 and all(inside(v, BARS["ratio_lo"], BARS["ratio_hi"]) for v in early.values())),
             "d_null_cross_le_0": bool(cross["8"] <= NULLS["cross_max"]),
             "e_null_spearman_le_.4": bool(not smoke and rho <= NULLS["spearman_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model; Stage A docs 96-159 (baseline 3.11250), Stage B docs 96-191 (baseline 3.08238); LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "k_map": K_MAP, "n_samples": S_EVAL,
           "instrument": inst, "n_positions": {"A": n_A, "B": n_B},
           "sites": rows, "late13": [r["site"] for r in late], "early_ratios": early, "spearman_cert_vs_measured_k32": rho,
           "sum_single_cert_k32_all_sites": sum(r["cert_k32"] for r in rows),
           "stage_B": {"single_cert": sg, "single_first_order": {f"{s[1]}_{k}": first_B[(s, k)] for s in JOINT for k in (8, K_MAP)},
                       "joint_cert": jt, "cross_term": cross, "measured_both": both, "measured_cross": meas_cross, "ratio_joint": ratio_joint},
           "price": {"gpu_forwards": 0, "cpu_doc_forward_equivalents": int(fit.shape[0]) + int(evA.shape[0]) * (2 + 2 * (1 + S_EVAL)) + int(evB.shape[0]) * (1 + 2 * (1 + S_EVAL)),
                     "cpu_seconds": time.time() - started},
           "hashes": {str(p): h for p, h in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "sites", "stage_B", "price")}, indent=1, default=str)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "rho": rho, "late": [(r["site"], round(r["ratio"], 3)) for r in late], "early": early, "stage_B": out["stage_B"], "price": out["price"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
