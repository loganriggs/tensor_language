#!/usr/bin/env python
"""late_pool_map_rank_curve_probe -- input-covariance-weighted rank curve of the one-shot linear context map (x_hat_11 -> sum of mlp11-15 writes,
§2724) and a quadratic-in-top-32-input-PCs upgrade. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_rank_128_keeps_most pred_c_rank_16_is_not_the_map pred_d_quadratic_helps_modestly
#                     pred_e_map_is_broad

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_POOL_MAP_RANK_CURVE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_pool_map_rank_curve_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_pool_map_rank_curve_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_POOL_MAP_RANK_CURVE_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_pool_surrogate_probe_results.json"   # §2724
OUT = ROOT / "late_pool_map_rank_curve_probe_results.json"
HASHES = {PREREG: "3a1a746ccc3aaddcd13f3231c68ee667cfc5e5e7e494ad0ae528b49317eeb661", PRIOR: "c3e1b9f387927cba3b360872b8d2014b1b5d6a33486e981c6e77ef5656de29f8",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_pool_map_rank_curve_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_POOLMEAN = 0.724; PRIOR_ONESHOT = 0.345; LAM = 1e-2
POOL = [("mlp", l) for l in range(11, 16)]
KS = [16, 32, 64, 128, 256, 512, 1152]; KQ = 32
BARS = {"ce_tol": 1e-4, "repro_tol": 0.02, "repro_tol_pool": 0.03, "b_frac": 0.80, "c_frac": 0.30, "d_gain": 0.03, "e_eff": 100.0}
NULLS = {"b_frac": 0.50, "c_frac": 0.60, "d_gain": 0.005, "e_eff": 40.0}


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
        if collect is not None:
            collect(("mlpin", l), xhat)
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


IU = torch.triu_indices(KM, KM)


def feats(mode, e, xh, Um):
    Um = Um.to(xh.device, xh.dtype); IU_ = IU.to(xh.device)
    """Feature rows [N, F] for one chunk. e = token embedding rows, xh = rms-normed MLP input rows, Um = [D, KM] core basis."""
    if mode == "CUR":
        return e
    if mode == "FULLIN":
        return torch.cat([e, xh], 1)
    if mode == "IN":
        return xh
    if mode == "INTOK":
        return torch.cat([xh, e], 1)
    c = xh @ Um; q = (c[:, IU_[0]] * c[:, IU_[1]])
    return torch.cat([e, c, q], 1)          # COREIN: 1152 + 16 + 136


class Surrogate:
    """w -> mu + P (A phi) with phi built from the token (hook) and the pre-write residual x."""
    def __init__(self, m, mode, fit, Pm, Um):
        self.m, self.mode, self.Pm, self.Um = m, mode, (None if Pm is None else Pm.to(DEV).float()), Um.to(DEV).float()
        self.A = fit["A"].to(DEV).float(); self.mphi = fit["mphi"].to(DEV).float(); self.mu = fit["mu"].to(DEV).float(); self.e = None
    def hook(self, idx):
        self.e = emb(self.m, idx)
    def __call__(self, w, x):
        B, T, _ = w.shape
        phi = feats(self.mode, self.e.reshape(-1, D), F.rms_norm(x, (D,)).reshape(-1, D), self.Um)
        y = ((phi - self.mphi) @ self.A).view(B, T, D)
        if self.Pm is None:
            return self.mu + y
        return self.mu + (y @ self.Pm) @ self.Pm.T


def collect_pass(m, rows, sites):
    """Yields per chunk: idx, {site: write rows}, {site: rms-normed MLP input rows}."""
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; box = {}
        def col(s_, w):
            if s_ in sites or (s_[0] == "mlpin" and ("mlp", s_[1]) in sites):
                box[s_] = w
        forward(m, idx, collect=col)
        yield idx, {s_: box[s_].reshape(-1, D) for s_ in sites}, {s_: F.rms_norm(box[("mlpin", s_[1])], (D,)).reshape(-1, D) for s_ in sites}


def ridge_fit(m, rows, sites, mode, Um):
    G = None; B = {}; sp = None; sw = {}; ss = {}; n = 0
    for idx, W_, X_ in collect_pass(m, rows, sites):
        e = emb(m, idx).reshape(-1, D)
        for s_ in sites:
            phi = feats(mode, e, X_[s_], Um).double(); Y = W_[s_].double()
            if G is None:
                nf = phi.shape[1]; G = torch.zeros(nf, nf, dtype=torch.float64, device=DEV); sp = torch.zeros(nf, dtype=torch.float64, device=DEV)
                for t in sites:
                    B[t] = torch.zeros(nf, D, dtype=torch.float64, device=DEV); sw[t] = torch.zeros(D, dtype=torch.float64, device=DEV); ss[t] = 0.0
            if s_ == sites[0]:
                G += phi.T @ phi; sp += phi.sum(0); n += phi.shape[0]
            B[s_] += phi.T @ Y; sw[s_] += Y.sum(0); ss[s_] += float((Y ** 2).sum())
    out = {}
    for s_ in sites:
        # features differ per site only through the input rows; G above uses the first site's features -> refit G per site for COREIN/FULLIN
        pass
    # exact per-site Gram (features depend on the site's own input rows)
    for s_ in sites:
        G = None; sp = None; n = 0
        for idx, W_, X_ in collect_pass(m, rows, [s_]):
            phi = feats(mode, emb(m, idx).reshape(-1, D), X_[s_], Um).double()
            if G is None:
                nf = phi.shape[1]; G = torch.zeros(nf, nf, dtype=torch.float64, device=DEV); sp = torch.zeros(nf, dtype=torch.float64, device=DEV)
            G += phi.T @ phi; sp += phi.sum(0); n += phi.shape[0]
        mp = sp / n; Gc = G / n - torch.outer(mp, mp); mw = sw[s_] / n; Bc = B[s_] / n - torch.outer(mp, mw)
        A = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / Gc.shape[0] * torch.eye(Gc.shape[0], dtype=torch.float64, device=DEV), Bc)
        out[s_] = {"mu": mw.float(), "A": A.float(), "mphi": mp.float(), "r2_fit": float(torch.trace(Bc.T @ A)) / (ss[s_] / n - float((mw ** 2).sum()))}
    return out


def r2_core_heldout(m, rows, fits, mode, Um):
    """Held-out R^2 of the CORE-PROJECTED write."""
    num = {s: 0.0 for s in fits}; den = {s: 0.0 for s in fits}; Umf = Um.to(DEV).float()
    for idx, W_, X_ in collect_pass(m, rows, list(fits)):
        e = emb(m, idx).reshape(-1, D)
        for s_ in fits:
            phi = feats(mode, e, X_[s_], Um); pred = fits[s_]["mu"] + (phi - fits[s_]["mphi"]) @ fits[s_]["A"]
            num[s_] += float((((W_[s_] - pred) @ Umf) ** 2).sum()); den[s_] += float((((W_[s_] - fits[s_]["mu"]) @ Umf) ** 2).sum())
    return {s: 1 - num[s] / den[s] for s in fits}


def meanpatch(mu):
    mu = mu.to(DEV); return lambda w, x: mu.expand_as(w)


def oracle_core(mu, P):
    mu = mu.to(DEV); P = P.to(DEV).float(); return lambda w, x: mu + ((w - mu) @ P) @ P.T


def fit_filler(m, rows, sites, Pm):
    """Per site: mean of the non-core input x_perp and a ridge map e(t) -> x_perp (both on the fit set)."""
    Pm = Pm.to(DEV).float(); out = {}
    for s_ in sites:
        G = None
        for idx, W_, X_ in collect_pass(m, rows, [s_]):
            e = emb(m, idx).reshape(-1, D).double(); xh = X_[s_].double(); xp = xh - (xh @ Pm.double()) @ Pm.double().T
            if G is None:
                G = torch.zeros(D, D, dtype=torch.float64, device=DEV); Bm = torch.zeros(D, D, dtype=torch.float64, device=DEV)
                se = torch.zeros(D, dtype=torch.float64, device=DEV); sx = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
            G += e.T @ e; Bm += e.T @ xp; se += e.sum(0); sx += xp.sum(0); n += e.shape[0]
        me = se / n; mx = sx / n; Gc = G / n - torch.outer(me, me); Bc = Bm / n - torch.outer(me, mx)
        A = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / D * torch.eye(D, dtype=torch.float64, device=DEV), Bc)
        out[s_] = {"xbar_perp": mx.float(), "A_fill": A.float(), "me": me.float()}
    return out


class WeightsArm:
    """The block's OWN mlp applied to x' = P x_hat + filler; filler in {MEAN, TOK, RAND}; optional core output restriction."""
    def __init__(self, m, l, fill, mode, mu, Pm, restrict):
        self.mlp = m.transformer.h[l].mlp; self.fill = fill; self.mode = mode; self.mu = mu.to(DEV); self.Pm = Pm.to(DEV).float(); self.restrict = restrict; self.e = None
    def hook(self, idx):
        self.idx = idx
    def __call__(self, w, x):
        B, T, _ = w.shape
        xh = F.rms_norm(x, (D,)); core = (xh @ self.Pm) @ self.Pm.T
        if self.mode == "MEAN":
            fl = self.fill["xbar_perp"]
        elif self.mode == "TOK":
            e = F.rms_norm(self.mlp_wte(self.idx), (D,))
            fl = self.fill["xbar_perp"] + (e - self.fill["me"]) @ self.fill["A_fill"]
        else:  # RAND: x_perp of a random other position in the same chunk (fixed permutation per call, seeded)
            xp = xh - core; flat = xp.reshape(-1, D)
            g = torch.Generator(device="cpu").manual_seed(int(self.idx[0, 0]) + 7 * T)
            perm = torch.randperm(flat.shape[0], generator=g).to(DEV)
            fl = flat[perm].view(B, T, D)
        xp_ = core + fl
        wr = self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
        if self.restrict:
            wr = self.mu + ((wr - self.mu) @ self.Pm) @ self.Pm.T
        return wr


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
    bases = fit(m, fit_rows, POOL); log(stage="fit_cov")
    core = pooled([bases[s] for s in POOL], False); P = core["U"][:, :KM]   # only used to satisfy Surrogate's Um slot (mode IN ignores it)
    def accumulate(featfn, dim):
        G = torch.zeros(dim, dim, dtype=torch.float64, device=DEV); Bm = torch.zeros(dim, D, dtype=torch.float64, device=DEV)
        sp = torch.zeros(dim, dtype=torch.float64, device=DEV); sw = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
        for idx, W_, X_ in collect_pass(m, fit_rows, POOL):
            phi = featfn(X_[("mlp", 11)]).double(); Y = sum(W_[s_] for s_ in POOL).double()
            G += phi.T @ phi; Bm += phi.T @ Y; sp += phi.sum(0); sw += Y.sum(0); n += phi.shape[0]
        mp = sp / n; mw = sw / n; Gc = G / n - torch.outer(mp, mp); Bc = Bm / n - torch.outer(mp, mw)
        A = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / dim * torch.eye(dim, dtype=torch.float64, device=DEV), Bc)
        return {"mu": mw.float(), "A": A.float(), "mphi": mp.float(), "Gc": Gc, "A64": A}
    lin = accumulate(lambda xh: xh, D); log(stage="lin_fit")
    Gc = lin["Gc"]; evals, Ug = torch.linalg.eigh(Gc); evals = evals.clamp_min(1e-12)
    Gh = (Ug * evals.sqrt()) @ Ug.T; Gih = (Ug * evals.rsqrt()) @ Ug.T
    M = Gh @ lin["A64"]; U, S, Vh = torch.linalg.svd(M)
    eff_rank = float((S.sum() ** 2) / (S ** 2).sum()); cum = (S ** 2).cumsum(0) / (S ** 2).sum()
    rank90 = int((cum < 0.90).sum().item()) + 1
    def trunc(k):
        Ak = Gih @ ((U[:, :k] * S[:k]) @ Vh[:k])
        return {"mu": lin["mu"], "A": Ak.float(), "mphi": lin["mphi"]}
    top = Ug.flip(1)[:, :KQ].float(); mp32 = lin["mphi"]
    IUq = torch.triu_indices(KQ, KQ).to(DEV)
    def qfeat(xh):
        c = (xh - mp32) @ top
        return torch.cat([xh, c[:, IUq[0]] * c[:, IUq[1]]], 1)
    quad = accumulate(qfeat, D + IUq.shape[1]); log(stage="quad_fit")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    zero = lambda w, x: torch.zeros_like(w)
    class Lin:
        def __init__(self, f, featfn):
            self.A = f["A"].to(DEV); self.mphi = f["mphi"].to(DEV); self.mu = f["mu"].to(DEV); self.featfn = featfn
        def hook(self, idx):
            pass
        def __call__(self, w, x):
            B, T, _ = w.shape
            phi = self.featfn(F.rms_norm(x, (D,)).reshape(-1, D))
            return self.mu + ((phi - self.mphi) @ self.A).view(B, T, D)
    def run_pool(obj):
        patch = {("mlp", 11): obj, **{s: zero for s in POOL[1:]}}
        return ce_of(m, ev, patch, obj.hook) - ce0
    arms = {"POOL_MEAN": ce_of(m, ev, {s: meanpatch(bases[s]["mu"]) for s in POOL}) - ce0}
    for k in KS:
        arms[f"LIN_k{k}"] = run_pool(Lin(trunc(k), lambda xh: xh)); log(stage="arm", k=k, ce=round(arms[f"LIN_k{k}"], 4))
    arms["QUAD32"] = run_pool(Lin(quad, qfeat)); log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    full = arms["LIN_k1152"]; rec = {k: 1 - v / arms["POOL_MEAN"] for k, v in arms.items()}
    rf = rec["LIN_k1152"]
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["POOL_MEAN"] - PRIOR_POOLMEAN) <= BARS["repro_tol_pool"] and abs(full - PRIOR_ONESHOT) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_rank_128_keeps_most': bool(rec["LIN_k128"] >= BARS["b_frac"] * rf),
        'pred_c_rank_16_is_not_the_map': bool(rec["LIN_k16"] <= BARS["c_frac"] * rf),
        'pred_d_quadratic_helps_modestly': bool(arms["QUAD32"] <= full - BARS["d_gain"]),
        'pred_e_map_is_broad': bool(eff_rank >= BARS["e_eff"]),
    }
    nulls = {"b_null_rank128_le_half": bool(rec["LIN_k128"] <= NULLS["b_frac"] * rf), "c_null_rank16_ge_.60": bool(rec["LIN_k16"] >= NULLS["c_frac"] * rf),
             "d_null_quad_no_gain": bool(arms["QUAD32"] >= full - NULLS["d_gain"]), "e_null_eff_rank_le_40": bool(eff_rank <= NULLS["e_eff"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); rec = 1 - CE/CE(POOL_MEAN)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "ridge_lambda_rel": LAM, "ks": KS, "k_quad_pcs": KQ,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "pool_mean": arms["POOL_MEAN"], "prior_pool_mean": PRIOR_POOLMEAN, "full": full, "prior_oneshot": PRIOR_ONESHOT},
           "ce_added": arms, "recovery_of_pool_mean": rec, "recovery_fraction_of_full": {k: (v / rf if rf else None) for k, v in rec.items()},
           "weighted_map_spectrum": {"eff_rank": eff_rank, "rank_90": rank90, "top_singular": [float(v) for v in S[:16]], "sum_sq": float((S ** 2).sum())},
           "input_cov_spectrum": {"eff_rank": float((evals.sum() ** 2) / (evals ** 2).sum()), "top16": [float(v) for v in evals.flip(0)[:16]]},
           "price": {"gpu_doc_forwards": 3 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (2 + len(KS) + 1), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "weighted_map_spectrum", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "eff_rank": eff_rank}, indent=1))


if __name__ == "__main__":
    main()
