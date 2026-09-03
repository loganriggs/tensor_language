#!/usr/bin/env python
# BQLANE: cpu
"""late_joint_installation_certificate_probe -- successor to §2701 on a FRESH split (bases fitted on docs 96-191, scored on docs
0-63): (1) single-site k=32 Fisher certificates vs measured prices for the 14 write sites of blocks 11-17; (2) NESTED JOINT
installations A1={mlp16,17} ⊂ A2={mlp14..17} ⊂ A3={mlp11..17} ⊂ A4=all 14 late sites, each certified analytically from one score
pass and measured by one patched forward per document; cross terms X = joint - sum(singles), certified vs measured.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_late14_single_certified_fresh pred_c_nested_joint_certified
#                     pred_d_cross_terms_certified pred_e_superadditive_installation

SIGN CONVENTION (§2135): every "measured" number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER; a
certificate is the analytic second-order prediction of the same quantity (ratio measured/cert = 1 is a perfect price). Nothing
installs into the §312 frontier; bases are activation covariances of the writes.
Preregistration: polynomial_causal/LATE_JOINT_INSTALLATION_CERTIFICATE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_joint_installation_certificate_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import site_write_pca_truncation_ce_map_probe as M
import site_write_certificate_map_probe as CM

ROOT = R.ROOT
PREREG = R.POLY / "LATE_JOINT_INSTALLATION_CERTIFICATE_PROBE_PREREGISTRATION.md"
PRIOR_MAP = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"      # §2696 (old-split singles, eff_rank_fit)
PRIOR_CERT = ROOT / "site_write_certificate_map_probe_results.json"           # §2701 (old-split certificates)
OUT = ROOT / "late_joint_installation_certificate_probe_results.json"
HASHES = {PREREG: "85563b7aa7dcd8326eab33d57e9edb2f95dabf33162707b0ed059444dac1d521",
          PRIOR_MAP: "48bd52ec9201ac97cddcd102cef61885baaaa8a8362b232850159f6f646d0e00",
          PRIOR_CERT: "358eaea00764f04dfc6ec9c9eff22119dc02c047c2f34b8dab389f581c56173f",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_joint_installation_certificate_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = M.TI; CH = M.CH
FIT = (96, 192); EVAL = (0, 64)                       # the FRESH split
K = 32
LATE14 = [(kind, l) for l in range(11, NL) for kind in ("attn", "mlp")]
SETS = {"A1": [("mlp", 16), ("mlp", 17)], "A2": [("mlp", l) for l in range(14, 18)], "A3": [("mlp", l) for l in range(11, 18)], "A4": LATE14}
BARS = {"ce_tol": 1e-4, "eff_rank_tol": 3.0, "rank90_tol": 2, "ratio_lo": 0.5, "ratio_hi": 2.0, "b_min_sites": 13, "e_factor_min": 1.5}
NULLS = {"b_max_sites": 9, "ratio_lo": 0.25, "ratio_hi": 4.0, "e_factor_max": 1.1}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def make_multi_patch(bases, sites, k):
    proj = {s: (bases[s]["mu"], bases[s]["U"][:, :k]) for s in sites}
    def patch(s, w):
        if s not in proj:
            return w
        mu, Uk = proj[s]
        return mu + ((w - mu) @ Uk) @ Uk.T
    return patch


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
    if smoke:
        gg = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (5, 257), generator=gg); fit, ev = nat[:3], nat[3:]
        sites = [("attn", 16), ("mlp", 16), ("mlp", 17)]
        sets = {"A1": [("mlp", 16), ("mlp", 17)], "A4": sites}
        prior_eff, prior_r90 = float("nan"), 4; old_single = {}; old_cert = {}
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit, ev = nat[FIT[0]:FIT[1]], nat[EVAL[0]:EVAL[1]]
        sites = LATE14; sets = SETS
        pm = json.load(open(PRIOR_MAP)); pm17 = [r for r in pm["sites"] if r["site"] == "mlp17"][0]
        prior_eff, prior_r90 = pm17["eff_rank_fit"], pm17["rank_90_fit"]
        old_single = {r["site"]: r["ce_added_k32"] for r in pm["sites"]}
        pc = json.load(open(PRIOR_CERT)); old_cert = {r["site"]: r["cert_k32"] for r in pc["sites"]}
    log = lambda stage: (lambda n: print(json.dumps({"stage": stage, "docs": n, "t": round(time.time() - started, 1)}), flush=True))
    # instrument
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce_manual = float(F.cross_entropy(M.forward(m := R.load_model(), idx).reshape(-1, V), tgt.reshape(-1)))
    ce_module = float(m(idx.contiguous(), tgt.contiguous()))
    bases = M.fit_bases(m, fit); log("fit")(int(fit.shape[0]))
    ident = make_multi_patch(bases, [("mlp", 17)], D)
    ce_ident = float(F.cross_entropy(M.forward(m, idx, ident).reshape(-1, V), tgt.reshape(-1)))
    eff17 = bases[("mlp", 17)]["spec"]["eff_rank"]; r90_17 = bases[("mlp", 17)]["spec"]["rank_90"]
    # certificates: one score pass, all late sites as leaves, joint sets summed inside the square
    single_c, first_c, joint_c, n_pos = CM.certificate_pass(m, ev, bases, sites, [K], [tuple(A) for A in sets.values()], gen, log("cert"))
    # measured
    ce0 = M.ce_of(m, ev); log("base")(int(ev.shape[0]))
    single_m = {}
    for s in sites:
        single_m[s] = M.ce_of(m, ev, make_multi_patch(bases, [s], K)) - ce0; log(f"single_{CM.sname(s)}")(int(ev.shape[0]))
    joint_m = {}
    for name, A in sets.items():
        joint_m[name] = M.ce_of(m, ev, make_multi_patch(bases, A, K)) - ce0; log(f"joint_{name}")(int(ev.shape[0]))
    rows = []
    for s in sites:
        c = single_c[(s, K)]; mm = single_m[s]
        rows.append({"site": CM.sname(s), "kind": s[0], "block": s[1], "measured_k32": mm, "cert_k32": c,
                     "first_order_share": (first_c[(s, K)] / c if c != 0 else float("nan")), "ratio": (mm / c if c != 0 else float("inf")),
                     "old_split_measured": old_single.get(CM.sname(s)), "old_split_cert": old_cert.get(CM.sname(s))})
        print(json.dumps(rows[-1]), flush=True)
    sets_out = {}
    for name, A in sets.items():
        jc = joint_c[(tuple(A), K)]; jm = joint_m[name]
        sc = sum(single_c[(s, K)] for s in A); sm = sum(single_m[s] for s in A)
        sets_out[name] = {"sites": [CM.sname(s) for s in A], "n": len(A), "joint_measured": jm, "joint_cert": jc,
                          "ratio": (jm / jc if jc != 0 else float("inf")), "sum_single_measured": sm, "sum_single_cert": sc,
                          "X_measured": jm - sm, "X_cert": jc - sc,
                          "X_ratio": ((jm - sm) / (jc - sc) if (jc - sc) != 0 else float("inf")),
                          "superadditivity_factor": (jm / sm if sm != 0 else float("inf"))}
        print(json.dumps({name: sets_out[name]}), flush=True)
    inside = lambda r, lo, hi: (lo <= r <= hi)
    n_in = sum(inside(r["ratio"], BARS["ratio_lo"], BARS["ratio_hi"]) for r in rows)
    have = [n for n in ("A1", "A2", "A3", "A4") if n in sets_out]
    a4 = sets_out.get("A4"); a3 = sets_out.get("A3", a4)
    def d_ok(S):
        return bool(S["X_measured"] > 0 and S["X_cert"] > 0 and inside(S["X_ratio"], BARS["ratio_lo"], BARS["ratio_hi"]))
    preds = {
        'pred_a_instrument': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"] and abs(ce_ident - ce_manual) <= BARS["ce_tol"]
                                  and (smoke or (abs(eff17 - prior_eff) <= BARS["eff_rank_tol"] and abs(r90_17 - prior_r90) <= BARS["rank90_tol"]))),
        'pred_b_late14_single_certified_fresh': bool(n_in >= BARS["b_min_sites"]),
        'pred_c_nested_joint_certified': bool(all(inside(sets_out[n]["ratio"], BARS["ratio_lo"], BARS["ratio_hi"]) for n in have)),
        'pred_d_cross_terms_certified': bool(d_ok(a3) and d_ok(a4)),
        'pred_e_superadditive_installation': bool(a4["superadditivity_factor"] >= BARS["e_factor_min"]),
    }
    nulls = {"b_null_le_9_of_14_inside": bool(n_in <= NULLS["b_max_sites"]),
             "c_null_A4_ratio_outside_[.25,4]": bool(not inside(a4["ratio"], NULLS["ratio_lo"], NULLS["ratio_hi"])),
             "d_null_A4_Xcert_le_0_or_Xratio_outside_[.25,4]": bool(a4["X_cert"] <= 0 or not inside(a4["X_ratio"], NULLS["ratio_lo"], NULLS["ratio_hi"])),
             "e_null_factor_le_1.1": bool(a4["superadditivity_factor"] <= NULLS["e_factor_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (bases fit on docs 96-191); LOWER IS BETTER; ratio = measured/certificate",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "k": K, "n_samples": CM.S_EVAL, "fit_docs": list(FIT), "eval_docs": list(EVAL),
           "instrument": {"ce_manual": ce_manual, "ce_module": ce_module, "ce_identity_patch_mlp17": ce_ident, "eff_rank_mlp17_fresh": eff17, "eff_rank_mlp17_prior": prior_eff,
                          "rank90_mlp17_fresh": r90_17, "rank90_mlp17_prior": prior_r90},
           "baseline_ce_eval": ce0, "n_positions": n_pos, "n_inside_[.5,2]": n_in, "sites": rows, "sets": sets_out,
           "price": {"gpu_forwards": 0, "cpu_doc_forward_equivalents": int(fit.shape[0]) + int(ev.shape[0]) * (6 + 1 + len(sites) + len(sets)) + 8, "cpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "sets": {n: (sets_out[n]["ratio"], sets_out[n]["X_ratio"]) for n in sets_out}}, indent=1))


if __name__ == "__main__":
    main()
