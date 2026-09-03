#!/usr/bin/env python
"""early_mlp_isolated_token_program_probe -- why is mlp1's write causally dense (§2696: .883 CE added at k=32)? Two axes on the same
held-out documents: (i) the write-PCA rank ladder k in {32,64,128,256,512} (H-dense: slow decay), (ii) the ISOLATED-TOKEN PROGRAM
F_l[t] = the mlp-l write the native model produces on the length-1 sequence [t], installed at every position (H-token: a per-token
lookup table is a better simple program than any 32-d subspace). mlp1 registered; mlp0/2/3 and a joint early-table arm disclosed.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_mlp1_dense pred_c_isolated_token_program_mlp1 pred_d_token_R2_mlp1

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-159 -- LOWER IS BETTER.
Preregistration: polynomial_causal/EARLY_MLP_ISOLATED_TOKEN_PROGRAM_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/early_mlp_isolated_token_program_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import site_write_pca_truncation_ce_map_probe as M

ROOT = R.ROOT
PREREG = R.POLY / "EARLY_MLP_ISOLATED_TOKEN_PROGRAM_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "site_write_pca_truncation_ce_map_probe_results.json"
OUT = ROOT / "early_mlp_isolated_token_program_probe_results.json"
HASHES = {PREREG: "b17191d40391892a0141db2a37167abe00461e1b2af5e527e09c4d37f526eabb", PRIOR: "48bd52ec9201ac97cddcd102cef61885baaaa8a8362b232850159f6f646d0e00",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "early_mlp_isolated_token_program_probe"
D, NL, V = R.D, R.NL, R.V
TI, FIT, EVAL, CH = M.TI, M.FIT, M.EVAL, M.CH
SITES = [0, 1, 2, 3]
KS = [32, 64, 128, 256, 512]
TB = 2048                        # tokens per length-1 batch when building the tables
BARS = {"ce_tol": 1e-4, "k32_repro_tol": 0.015, "k32_prior": None, "dense_k256_min": 0.20, "table_max": 0.40, "r2_min": 0.5}
NULLS = {"dense_k256_max": 0.05, "table_min": None, "r2_max": 0.2}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


@torch.no_grad()
def build_tables(m, sites, vocab):
    """F_l[t] = mlp-l write of the native model on the length-1 sequence [t]; rows for every t in vocab (int tensor)."""
    tabs = {l: torch.zeros(int(vocab.numel()), D) for l in sites}
    for i in range(0, vocab.numel(), TB):
        ids = vocab[i:i + TB].view(-1, 1)
        def collect(s, w):
            if s[0] == "mlp" and s[1] in tabs:
                tabs[s[1]][i:i + ids.shape[0]] = w[:, 0, :]
        M.forward(m, ids, collect=collect)
    return tabs


def make_table_patch(tabs, sites):
    """mw_l(pos) := F_l[token at pos]; sites = list of blocks to replace jointly. Needs the input ids: set via closure holder."""
    holder = {}
    def patch(s, w):
        if s[0] == "mlp" and s[1] in sites:
            return tabs[s[1]][holder["idx"]]
        return w
    return patch, holder


def ce_tables(m, rows, tabs, sites):
    patch, holder = make_table_patch(tabs, sites)
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]; holder["idx"] = idx
        lg = M.forward(m, idx, patch)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


@torch.no_grad()
def table_fit(m, rows, tabs, bases, sites):
    """R2 and median cosine of the native write vs the table entry, over EVAL positions, per site."""
    num = {l: 0.0 for l in sites}; den = {l: 0.0 for l in sites}; cosines = {l: [] for l in sites}
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]
        def collect(s, w):
            if s[0] == "mlp" and s[1] in sites:
                l = s[1]; f = tabs[l][idx]; mu = bases[("mlp", l)]["mu"]
                num[l] += float(((w - f) ** 2).sum()); den[l] += float(((w - mu) ** 2).sum())
                cosines[l].append(F.cosine_similarity(w.reshape(-1, D), f.reshape(-1, D), dim=1))
        M.forward(m, idx, collect=collect)
    return {str(l): {"r2": 1.0 - num[l] / den[l], "median_cos": float(torch.cat(cosines[l]).median())} for l in sites}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    check_hashes()
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    pj = json.load(open(PRIOR)); prior_base = pj["baseline_ce_eval"]
    prior_k32 = {r["block"]: r["ce_added_k32"] for r in pj["sites"] if r["kind"] == "mlp"}
    m = R.load_model()
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (6, 257), generator=g); fit, ev = nat[:3], nat[3:]
        vocab = torch.unique(ev.reshape(-1)); ks = [32, 256]; sites = [1, 3]
    else:
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit, ev = nat[FIT[0]:FIT[1]], nat[EVAL[0]:EVAL[1]]
        vocab = torch.arange(V); ks = KS; sites = SITES
    bases = M.fit_bases(m, fit)
    ce0 = M.ce_of(m, ev)
    ladder = {str(l): {} for l in sites}
    for l in sites:
        for k in ks:
            ladder[str(l)][str(k)] = M.ce_of(m, ev, M.make_patch(bases, ("mlp", l), k)) - ce0
            print(json.dumps({"stage": "ladder", "block": l, "k": k, "ce_added": ladder[str(l)][str(k)]}), flush=True)
    t0 = time.time()
    tabs_full = build_tables(m, sites, vocab)
    if smoke:   # smoke vocab is the unique eval ids: remap through a lookup
        full = {l: torch.zeros(V, D) for l in sites}
        for l in sites:
            full[l][vocab] = tabs_full[l]
        tabs = full
    else:
        tabs = tabs_full
    table_seconds = time.time() - t0
    table_ce = {str(l): ce_tables(m, ev, tabs, [l]) - ce0 for l in sites}
    for l in sites:
        print(json.dumps({"stage": "table", "block": l, "ce_added": table_ce[str(l)]}), flush=True)
    joint_ce = ce_tables(m, ev, tabs, sites) - ce0
    fitq = table_fit(m, ev, tabs, bases, sites)
    tab_norm = {str(l): float(tabs[l].norm(dim=1).mean()) for l in sites}
    l1 = "1" if 1 in sites else str(sites[0]); k256 = "256"
    preds = {
        'pred_a_instrument': bool(abs(ce0 - prior_base) <= BARS["ce_tol"] and abs(ladder[l1]["32"] - prior_k32.get(int(l1), float("nan"))) <= BARS["k32_repro_tol"]),
        'pred_b_mlp1_dense': bool(ladder[l1][k256] >= BARS["dense_k256_min"]),
        'pred_c_isolated_token_program_mlp1': bool(table_ce[l1] <= BARS["table_max"]),
        'pred_d_token_R2_mlp1': bool(fitq[l1]["r2"] >= BARS["r2_min"]),
    }
    nulls = {"b_null_k256_le_.05": bool(ladder[l1][k256] <= NULLS["dense_k256_max"]),
             "c_null_table_ge_k32_price": bool(table_ce[l1] >= prior_k32.get(int(l1), float("inf"))),
             "d_null_r2_le_.2": bool(fitq[l1]["r2"] <= NULLS["r2_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 96-159; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "sites": sites, "ks": ks,
           "baseline_ce_eval": ce0, "prior_baseline_ce_eval": prior_base, "prior_k32_mlp": {str(k): v for k, v in prior_k32.items() if k in sites},
           "ladder_ce_added": ladder, "isolated_token_table_ce_added": table_ce, "joint_early_tables_ce_added": joint_ce,
           "table_fit_eval": fitq, "table_mean_norm": tab_norm, "n_fit_docs": int(fit.shape[0]), "n_eval_docs": int(ev.shape[0]),
           "price": {"gpu_forwards": 0, "cpu_doc_forwards": int(fit.shape[0]) + int(ev.shape[0]) * (2 + len(sites) * len(ks) + len(sites) + 1),
                     "table_length1_forwards": int(vocab.numel()), "table_seconds": table_seconds, "cpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "ladder_ce_added", "isolated_token_table_ce_added", "joint_early_tables_ce_added", "table_fit_eval", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ladder": ladder, "tables": table_ce, "joint": joint_ce}, indent=1))


if __name__ == "__main__":
    main()
