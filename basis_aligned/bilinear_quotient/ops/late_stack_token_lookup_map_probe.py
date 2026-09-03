#!/usr/bin/env python
"""late_stack_token_lookup_map_probe -- extends §2717 (mlp16+17 message = half a current-token lookup inside the 16-dim late core) to the
whole late MLP stack mlp11-17 (MEAN value 1.885 nat, §2716) and to each of mlp11..15 singly. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_STACK_TOKEN_LOOKUP_MAP_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_stack_token_lookup_map_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_stack_token_lookup_map_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_STACK_TOKEN_LOOKUP_MAP_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_message_token_lookup_probe_results.json"   # §2717
OUT = ROOT / "late_stack_token_lookup_map_probe_results.json"
HASHES = {PREREG: "c443b0e69631864f3f770c4a06a03b847c7e13e63d0bae3ad281840f43a3af51", PRIOR: "84a3d8a9a335c8ca790ff9c03205821f34d9d631ed23bc06ade1400d4a0633bd",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_stack_token_lookup_map_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_MEAN7 = 1.885; LAM = 1e-2
MID5 = [("mlp", l) for l in range(11, 16)]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.03, "b_min": 0.35, "c_min": 0.60, "d_med_min": 0.35, "e_mult": 0.80}
NULLS = {"b_max": 0.15, "c_max": 0.35, "d_med_max": 0.15, "e_mult": 0.50}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


class GAcc:
    """float64 second-moment accumulator on DEV."""
    def __init__(self):
        self.S = torch.zeros(D, D, dtype=torch.float64, device=DEV); self.mu = torch.zeros(D, dtype=torch.float64, device=DEV); self.cnt = 0
    def add(self, w):
        X = w.reshape(-1, D).double(); self.S += X.T @ X; self.mu += X.sum(0); self.cnt += X.shape[0]
    def cov(self):
        mu = self.mu / self.cnt; return mu, (self.S / self.cnt - torch.outer(mu, mu)).cpu()
    def finish(self):
        mu, C = self.cov()
        ev, U = torch.linalg.eigh(C); U = U.flip(1)
        return {"mu": mu.float(), "U": U.float().to(DEV), "spec": R.spectrum(C), "C": C}


def pooled_basis(bs):
    """One basis from the equal-weight mean of the per-site centred covariances (= PCA of the pooled per-site-centred writes)."""
    C = sum(b["C"] for b in bs) / len(bs)
    ev, U = torch.linalg.eigh(C); U = U.flip(1)
    return {"U": U.float().to(DEV), "spec": R.spectrum(C), "C": C}


def captured(b, U, k):
    """Fraction of site b's write variance inside span(U[:, :k])."""
    Uk = U[:, :k].cpu().double(); return float(torch.trace(Uk.T @ b["C"] @ Uk) / torch.trace(b["C"]))


@torch.no_grad()
def forward(m, idx, patch=None, collect=None):
    """tt_model semantics; patch: dict site -> fn(w) applied to that site's write; collect(site, w)."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = (t.to(DEV) for t in R.rope(Tn))
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
    def apply(s, w, x):
        if collect is not None:
            collect(s, w)
        return patch[s](w, x) if patch and s in patch else w
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
        x = x + apply(("attn", l), aw, x)
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        x = x + apply(("mlp", l), mw, x) + mlp.Down_bias
    if collect is not None:
        collect(("final", -1), x)
    if collect is not None:
        collect(("final", -1), x)
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows, patch=None, hook=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        if hook is not None:
            hook(idx)
        lg = forward(m, idx, patch)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


def fit(m, rows, sites):
    acc = {s: GAcc() for s in sites}
    for i in range(0, rows.shape[0], CH):
        forward(m, rows[i:i + CH, :TI], collect=lambda s, w: acc[s].add(w) if s in acc else None)
    return {s: acc[s].finish() for s in sites}


def pooled(bs, normalise):
    C = sum((b["C"] / torch.trace(b["C"]) if normalise else b["C"]) for b in bs) / len(bs)
    ev, U = torch.linalg.eigh(C); U = U.flip(1)
    return {"U": U.double(), "spec": R.spectrum(C)}


def emb(m, idx):
    return F.rms_norm(m.transformer.wte(idx), (D,))


def prev_ids(idx):
    return torch.cat([idx[:, :1], idx[:, :-1]], 1)


class Lookup:
    """w -> mu + P (A phi(t)); phi set per chunk by hook(idx). P = None (full), or a [D,k] orthonormal basis (core-restricted)."""
    def __init__(self, m, mu, A, P=None, use_prev=False):
        self.m, self.mu, self.A, self.P, self.use_prev = m, mu.to(DEV), A.to(DEV).float(), (None if P is None else P.to(DEV).float()), use_prev
        self.phi = None
    def hook(self, idx):
        self.phi = emb(self.m, prev_ids(idx) if self.use_prev else idx)
    def __call__(self, w, x):
        y = self.phi @ self.A
        if self.P is not None:
            y = (y @ self.P) @ self.P.T
        return self.mu + y


def ridge_fit(m, rows, sites, use_prev=False):
    """Ridge of the centred write on phi = rms_norm(wte(t)) over all fit tokens; returns per-site (mu, A, r2_fit)."""
    G = torch.zeros(D, D, dtype=torch.float64, device=DEV); B = {s: torch.zeros(D, D, dtype=torch.float64, device=DEV) for s in sites}
    sp = torch.zeros(D, dtype=torch.float64, device=DEV); sw = {s: torch.zeros(D, dtype=torch.float64, device=DEV) for s in sites}
    ss = {s: 0.0 for s in sites}; n = 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; box = {}
        forward(m, idx, collect=lambda s_, w: box.__setitem__(s_, w) if s_ in sites else None)
        phi = emb(m, prev_ids(idx) if use_prev else idx).reshape(-1, D).double(); n += phi.shape[0]
        G += phi.T @ phi; sp += phi.sum(0)
        for s_ in sites:
            W_ = box[s_].reshape(-1, D).double(); B[s_] += phi.T @ W_; sw[s_] += W_.sum(0); ss[s_] += float((W_ ** 2).sum())
    mp = sp / n; Gc = G / n - torch.outer(mp, mp); out = {}
    for s_ in sites:
        mw = sw[s_] / n; Bc = B[s_] / n - torch.outer(mp, mw)
        A = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / D * torch.eye(D, dtype=torch.float64, device=DEV), Bc)
        var_w = ss[s_] / n - float((mw ** 2).sum()); expl = float(torch.trace(Bc.T @ A))
        out[s_] = {"mu": mw.float(), "A": A.float(), "mphi": mp.float(), "r2_fit": expl / var_w}
    return out


def r2_heldout(m, rows, fits, use_prev=False):
    num = {s: 0.0 for s in fits}; den = {s: 0.0 for s in fits}
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; box = {}
        forward(m, idx, collect=lambda s_, w: box.__setitem__(s_, w) if s_ in fits else None)
        phi = emb(m, prev_ids(idx) if use_prev else idx).reshape(-1, D)
        for s_ in fits:
            W_ = box[s_].reshape(-1, D); pred = fits[s_]["mu"] + (phi - fits[s_]["mphi"]) @ fits[s_]["A"]
            num[s_] += float(((W_ - pred) ** 2).sum()); den[s_] += float(((W_ - fits[s_]["mu"]) ** 2).sum())
    return {s: 1 - num[s] / den[s] for s in fits}


def meanpatch(mu):
    mu = mu.to(DEV); return lambda w, x: mu.expand_as(w)


def oracle_core(mu, P):
    mu = mu.to(DEV); P = P.to(DEV).float(); return lambda w, x: mu + ((w - mu) @ P) @ P.T


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    m = R.load_model().to(DEV)
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (8, 257), generator=g); fit_rows, ev = nat[:4].to(DEV), nat[4:].to(DEV)
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV)
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    bases = fit(m, fit_rows, LATE7); log(stage="fit_cov")
    core = pooled([bases[s] for s in LATE7], False); P = core["U"][:, :KM]
    cur = ridge_fit(m, fit_rows, LATE7, use_prev=False)
    for s_ in LATE7:
        cur[s_]["mu"] = bases[s_]["mu"].to(DEV)
        cur[s_]["mu_eff"] = cur[s_]["mu"] - (cur[s_]["mphi"].to(DEV) @ cur[s_]["A"].to(DEV))
    r2 = {"cur_fit": {f"{s[0]}{s[1]}": cur[s]["r2_fit"] for s in LATE7}, "cur_heldout": {f"{s[0]}{s[1]}": v for s, v in r2_heldout(m, ev, cur).items()}}
    log(stage="ridge", r2=r2)
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    def lk2(fits, sites, Pm=None):
        objs = {s: Lookup(m, fits[s]["mu_eff"] if Pm is None else fits[s]["mu"], fits[s]["A"], Pm, False) for s in sites}
        if Pm is not None:   # restricted: mu + P A (phi - mphi)  ==  (mu - P A mphi) + P A phi
            for s in sites:
                objs[s].mu = (fits[s]["mu"].to(DEV) - ((fits[s]["mphi"].to(DEV) @ fits[s]["A"].to(DEV)) @ Pm.to(DEV).float()) @ Pm.to(DEV).float().T)
        def hook(idx):
            for o in objs.values():
                o.hook(idx)
        return objs, hook
    arms = {}
    arms["MEAN7"] = ce_of(m, ev, {s: meanpatch(bases[s]["mu"]) for s in LATE7}) - ce0
    o, h = lk2(cur, LATE7); arms["CUR7"] = ce_of(m, ev, o, h) - ce0
    o, h = lk2(cur, LATE7, P); arms["CUR7_M"] = ce_of(m, ev, o, h) - ce0
    arms["ORACLE7_M"] = ce_of(m, ev, {s: oracle_core(bases[s]["mu"], P) for s in LATE7}) - ce0
    log(stage="joint", **{k: round(v, 4) for k, v in arms.items()})
    single = {}
    for s_ in MID5:
        nm = f"{s_[0]}{s_[1]}"
        mean_s = ce_of(m, ev, {s_: meanpatch(bases[s_]["mu"])}) - ce0
        o, h = lk2(cur, [s_]); cur_s = ce_of(m, ev, o, h) - ce0
        single[nm] = {"MEAN": mean_s, "CUR": cur_s, "rec_CUR": (1 - cur_s / mean_s) if mean_s > 0 else float("nan")}
        log(stage="single", site=nm, **{k: round(v, 4) for k, v in single[nm].items()})
    rec = {k: 1 - arms[k] / arms["MEAN7"] for k in ("CUR7", "CUR7_M", "ORACLE7_M")} if arms["MEAN7"] > 0 else {k: float("nan") for k in ("CUR7", "CUR7_M", "ORACLE7_M")}
    recs5 = sorted(v["rec_CUR"] for v in single.values()); med5 = recs5[len(recs5) // 2]
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["MEAN7"] - PRIOR_MEAN7) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok and all(v > 0 for v in r2["cur_heldout"].values())),
        'pred_b_late_stack_half_lookup': bool(rec["CUR7"] >= BARS["b_min"]),
        'pred_c_core_carries_the_stack': bool(rec["ORACLE7_M"] >= BARS["c_min"]),
        'pred_d_lookup_is_generic_across_late_sites': bool(med5 >= BARS["d_med_min"]),
        'pred_e_core_restriction_is_cheap_for_the_stack': bool(rec["CUR7_M"] >= BARS["e_mult"] * rec["CUR7"]),
    }
    nulls = {"b_null_rec_cur7_le_.15": bool(rec["CUR7"] <= NULLS["b_max"]), "c_null_rec_oracle7_le_.35": bool(rec["ORACLE7_M"] <= NULLS["c_max"]),
             "d_null_median_single_rec_le_.15": bool(med5 <= NULLS["d_med_max"]), "e_null_rec_cur7M_le_.5_rec_cur7": bool(rec["CUR7_M"] <= NULLS["e_mult"] * rec["CUR7"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); rec = 1 - CE(arm)/CE(MEAN of the same site set) (HIGHER = better)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "k_core": KM, "ridge_lambda_rel": LAM,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "mean_late7": arms["MEAN7"], "prior_mean_late7": PRIOR_MEAN7},
           "ce_added_joint_late7": arms, "recovery_late7": rec, "single_site_mlp11_15": single, "median_single_rec_cur": med5, "ridge_r2": r2, "core_spectrum": core["spec"],
           "price": {"gpu_doc_forwards": 2 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + 1 + 4 + 10), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added_joint_late7", "recovery_late7", "single_site_mlp11_15", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added_joint_late7": arms, "recovery_late7": rec, "single_site_mlp11_15": single}, indent=1))


if __name__ == "__main__":
    main()
