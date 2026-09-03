#!/usr/bin/env python
# BQLANE: cpu
"""pairwise_fisher_subset_price_probe -- the second-order joint Fisher certificate is exactly pairwise, so 14 single certificates
+ 91 pairwise cross terms X_st from ONE score pass define a subset-price model J(A) = sum c_s + sum_{s<t} X_st over all 2^14
installations of the late write sites (blocks 11-17, k=32). Tested against measured prices of 12 random subsets, the 4 nested
sets of §2703, and a certificate-chosen BEST7 / WORST7 design pair, on a fresh split (bases docs 96-191, scored docs 64-95).

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_pairwise_model_prices_random_subsets pred_c_cross_terms_positive
#                     pred_d_ordering pred_e_design_gain

SIGN CONVENTION (§2135): every "measured" number is CE ADDED ABOVE THE REAL MODEL on held-out docs 64-95 -- LOWER IS BETTER; J(A)
is the analytic second-order prediction of the same quantity (ratio = measured/J). Nothing installs into the §312 frontier.
Preregistration: polynomial_causal/PAIRWISE_FISHER_SUBSET_PRICE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time, itertools
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/pairwise_fisher_subset_price_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import site_write_pca_truncation_ce_map_probe as M
import site_write_certificate_map_probe as CM
import late_joint_installation_certificate_probe as LJ

ROOT = R.ROOT
PREREG = R.POLY / "PAIRWISE_FISHER_SUBSET_PRICE_PROBE_PREREGISTRATION.md"
PRIOR_LJ = ROOT / "late_joint_installation_certificate_probe_results.json"   # §2703
OUT = ROOT / "pairwise_fisher_subset_price_probe_results.json"
HASHES = {PREREG: "36a777cbd7863a45f232bab297a97bc6f02542693577d8370f315e4059243a71",
          PRIOR_LJ: "2bdb4ea7af834cb698c16c81df11a8bdae0782f00ed6112eb6e0a1341c10f11a",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "pairwise_fisher_subset_price_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = M.TI
FIT = (96, 192); EVAL = (64, 96)
K = 32
LATE14 = LJ.LATE14; NESTED = LJ.SETS
SIZES = [3, 3, 3, 3, 5, 5, 5, 5, 8, 8, 8, 8]
BARS = {"ce_tol": 1e-4, "identity_rel": 1e-6, "a_stab_min": 12, "b_lo": 0.7, "b_hi": 1.4, "b_min": 10, "c_min_pos": 80, "d_spearman_min": 0.9, "e_max": 0.5}
NULLS = {"b_max": 6, "c_min_neg": 30, "d_spearman_max": 0.5, "e_min": 0.8}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def draw_random_subsets(sites, sizes, seed=1):
    g = torch.Generator().manual_seed(seed); out = []
    for n in sizes:
        perm = torch.randperm(len(sites), generator=g)[:n].sort().values.tolist()
        out.append(tuple(sites[i] for i in perm))
    return out


def J_of(A, c, X):
    A = list(A)
    return sum(c[s] for s in A) + sum(X[frozenset((a, b))] for a, b in itertools.combinations(A, 2))


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
        sites = [("attn", 16), ("mlp", 16), ("attn", 17), ("mlp", 17)]
        nested = {"A1": [("mlp", 16), ("mlp", 17)], "A4": sites}; sizes = [2, 3]; prior_c = {}
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit, ev = nat[FIT[0]:FIT[1]], nat[EVAL[0]:EVAL[1]]
        sites = LATE14; nested = NESTED; sizes = SIZES
        prior_c = {r["site"]: r["cert_k32"] for r in json.load(open(PRIOR_LJ))["sites"]}
    log = lambda stage: (lambda n: print(json.dumps({"stage": stage, "docs": n, "t": round(time.time() - started, 1)}), flush=True))
    m = R.load_model()
    idx = ev[:4, :TI]; tgt = ev[:4, 1:TI + 1]
    ce_manual = float(F.cross_entropy(M.forward(m, idx).reshape(-1, V), tgt.reshape(-1)))
    ce_module = float(m(idx.contiguous(), tgt.contiguous()))
    bases = M.fit_bases(m, fit); log("fit")(int(fit.shape[0]))
    pairs = [tuple(p) for p in itertools.combinations(sites, 2)]
    random_sets = draw_random_subsets(sites, sizes)
    named = {f"R{i + 1}_n{len(A)}": A for i, A in enumerate(random_sets)}
    named.update({n: tuple(A) for n, A in nested.items()})
    joint_sets = pairs + list(named.values())
    single_c, first_c, joint_c, n_pos = CM.certificate_pass(m, ev, bases, sites, [K], joint_sets, gen, log("cert"))
    c = {s: single_c[(s, K)] for s in sites}
    X = {frozenset(p): joint_c[(p, K)] - c[p[0]] - c[p[1]] for p in pairs}
    # designs chosen from the certificate BEFORE measurement
    designs = {}
    if not smoke:
        cands = [A for A in itertools.combinations(sites, 7) if sum(s[0] == "mlp" for s in A) >= 4]
        scored = sorted(((J_of(A, c, X), A) for A in cands), key=lambda t: t[0])
        designs = {"BEST7": scored[0][1], "WORST7": scored[-1][1]}
        named.update(designs)
    # measured
    ce0 = M.ce_of(m, ev); log("base")(int(ev.shape[0]))
    single_m = {}
    for s in sites:
        single_m[s] = M.ce_of(m, ev, LJ.make_multi_patch(bases, [s], K)) - ce0; log(f"single_{CM.sname(s)}")(int(ev.shape[0]))
    sets_out = {}
    for name, A in named.items():
        jm = M.ce_of(m, ev, LJ.make_multi_patch(bases, list(A), K)) - ce0; log(f"set_{name}")(int(ev.shape[0]))
        J = J_of(A, c, X); direct = joint_c[(tuple(A), K)] if (tuple(A), K) in joint_c else float("nan")
        sets_out[name] = {"sites": [CM.sname(s) for s in A], "n": len(A), "measured": jm, "J": J, "direct_joint_cert": direct,
                          "identity_rel_err": (abs(direct - J) / max(abs(J), 1e-12) if direct == direct else float("nan")),
                          "ratio": (jm / J if J != 0 else float("inf")), "sum_single_measured": sum(single_m[s] for s in A),
                          "X_measured": jm - sum(single_m[s] for s in A), "X_model": J - sum(c[s] for s in A)}
        print(json.dumps({name: sets_out[name]}), flush=True)
    rows = [{"site": CM.sname(s), "measured_k32": single_m[s], "cert_k32": c[s], "ratio": (single_m[s] / c[s] if c[s] else float("inf")),
             "first_order_share": (first_c[(s, K)] / c[s] if c[s] else float("nan")), "prior_cert_docs0_63": prior_c.get(CM.sname(s))} for s in sites]
    Xmat = [[(X[frozenset((a, b))] if a != b else c[a]) for b in sites] for a in sites]
    kinds = lambda p: "".join(sorted(s[0][0] for s in p))   # 'aa', 'am', 'mm'
    Xby = {kk: [X[frozenset(p)] for p in pairs if kinds(p) == kk] for kk in ("aa", "am", "mm")}
    rnames = [n for n in named if n.startswith("R")]; sixteen = rnames + list(nested.keys())
    n_b = sum(BARS["b_lo"] <= sets_out[n]["ratio"] <= BARS["b_hi"] for n in rnames)
    n_pos_pairs = sum(v > 0 for v in X.values()); n_neg_pairs = sum(v < 0 for v in X.values())
    rho = R.spearman(torch.tensor([sets_out[n]["J"] for n in sixteen]), torch.tensor([sets_out[n]["measured"] for n in sixteen]))
    stab = sum(0.5 <= (c[s] / prior_c[CM.sname(s)]) <= 2.0 for s in sites if CM.sname(s) in prior_c and prior_c[CM.sname(s)])
    ident_ok = all(sets_out[n]["identity_rel_err"] <= BARS["identity_rel"] for n in sets_out if sets_out[n]["identity_rel_err"] == sets_out[n]["identity_rel_err"])
    e_ratio = (sets_out["BEST7"]["measured"] / sets_out["WORST7"]["measured"]) if designs else float("nan")
    preds = {
        'pred_a_instrument': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"] and ident_ok and (smoke or stab >= BARS["a_stab_min"])),
        'pred_b_pairwise_model_prices_random_subsets': bool(n_b >= (BARS["b_min"] if not smoke else 1)),
        'pred_c_cross_terms_positive': bool(n_pos_pairs >= (BARS["c_min_pos"] if not smoke else 1)),
        'pred_d_ordering': bool(rho >= BARS["d_spearman_min"]),
        'pred_e_design_gain': bool(designs and e_ratio <= BARS["e_max"]),
    }
    nulls = {"b_null_le_6_of_12": bool(n_b <= NULLS["b_max"]), "c_null_ge_30_negative_pairs": bool(n_neg_pairs >= NULLS["c_min_neg"]),
             "d_null_spearman_le_.5": bool(rho <= NULLS["d_spearman_max"]), "e_null_best_ge_.8_worst": bool(designs and e_ratio >= NULLS["e_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 64-95 (bases fit on docs 96-191); LOWER IS BETTER; ratio = measured/J",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "k": K, "n_samples": CM.S_EVAL, "fit_docs": list(FIT), "eval_docs": list(EVAL),
           "instrument": {"ce_manual": ce_manual, "ce_module": ce_module, "n_sites_cert_stable_vs_2703": stab, "pairwise_identity_ok": ident_ok},
           "baseline_ce_eval": ce0, "n_positions": n_pos, "sites": rows, "site_order": [CM.sname(s) for s in sites], "X_matrix_with_c_on_diagonal": Xmat,
           "X_by_kind_mean": {kk: (sum(v) / len(v) if v else float("nan")) for kk, v in Xby.items()}, "n_pairs_positive": n_pos_pairs, "n_pairs_negative": n_neg_pairs,
           "sets": sets_out, "n_random_in_[.7,1.4]": n_b, "spearman_J_vs_measured_16": rho, "designs": {k: [CM.sname(s) for s in v] for k, v in designs.items()},
           "best7_over_worst7_measured": e_ratio,
           "price": {"gpu_forwards": 0, "cpu_doc_forward_equivalents": int(fit.shape[0]) + int(ev.shape[0]) * (6 + 1 + len(sites) + len(named)) + 8, "cpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "X_by_kind_mean", "spearman_J_vs_measured_16", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "n_b": n_b, "rho": rho, "e_ratio": e_ratio, "designs": out["designs"]}, indent=1))


if __name__ == "__main__":
    main()
