#!/usr/bin/env python
"""late_tail_cross_unit_probe -- BELOW THE MLP BLOCK, second cut: is the core-gated linear read of the tail (the cross term Lc*Rt + Lt*Rc,
83% of the late MLPs' 768-cost, §2780) carried by FEW hidden units? Rank each late block's hidden units by the fit-set output energy of their
cross term (mean cross_j^2 x ||Down[:, j]||^2); keep the cross term on the top-h units only (h = 256/512/1024/2048 of 4608), against a seeded
random-h control; tail*tail dropped throughout (§2780: .0087). CUDA lane-1.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_top_1024_units_recover_most_of_the_cross_gain pred_c_top_512_units_recover_much
#                     pred_d_random_1024_units_recover_little pred_e_cross_energy_concentrated_on_units

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier (§2125).
Preregistration: polynomial_causal/LATE_TAIL_CROSS_UNIT_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_tail_cross_unit_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_tail_cross_unit_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_TAIL_CROSS_UNIT_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_tail_product_term_probe_results.json"   # §2780
OUT = ROOT / "late_tail_cross_unit_probe_results.json"
HASHES = {PREREG: "cce1c499305f25e8dfc86861bfddfc32eadc18ef3d11bad387b3a81dce1a0812", PRIOR: "044ce5ad6bb1e0b4d54b902aa3ca1d1e9d321cebc1c33c0f64246f557a8842ea",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_tail_cross_unit_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_SPLIT8_1024 = 0.0374; PRIOR_LATE_MLP_768 = 0.1249; PRIOR_LATE_MLP_896 = 0.0662; PRIOR_DROP_TT_768 = 0.0087; LAM = 1e-2
K = 1024; SPLIT = 8; EARLY = list(range(0, 11)); LATE = list(range(11, 18))
SITES = [("mlp", l) for l in range(NL)]; ASITES = [("attn", l) for l in range(NL)]; FINAL = ("final", -1)
ESITES = [(kd, l) for l in EARLY for kd in ("attn", "mlp")]; LSITES = [("mlp", l) for l in LATE] + [("attn", l) for l in LATE]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "b_min": 0.60, "c_min": 0.40, "d_max": 0.35, "e_max": 1500.0}
NULLS = {"b_max": 0.35, "c_max": 0.20, "d_min": 0.55, "e_min": 2500.0}


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
    if patch and ("final", -1) in patch:
        x = patch[("final", -1)](x, x)
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
    st = {s_: {"Gx": torch.zeros(D, D, dtype=torch.float64, device=DEV), "sx": torch.zeros(D, dtype=torch.float64, device=DEV)} for s_ in SITES + ASITES}
    wst = {s_: {"Gw": torch.zeros(D, D, dtype=torch.float64, device=DEV), "sw": torch.zeros(D, dtype=torch.float64, device=DEV)} for s_ in SITES + ASITES}
    class Rec:
        def __init__(self, s_): self.s = s_
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)).reshape(-1, D).double(); st[self.s]["Gx"] += xh.T @ xh; st[self.s]["sx"] += xh.sum(0); return w
    n = 0
    for i in range(0, fit_rows.shape[0], CH):
        idx = fit_rows[i:i + CH, :TI]; box = {}
        forward(m, idx, patch={s_: Rec(s_) for s_ in ASITES}, collect=lambda s_, w: box.__setitem__(s_, w) if (s_[0] == "mlpin" and ("mlp", s_[1]) in SITES) or s_ in SITES + ASITES else None)
        for s_ in SITES:
            xh = F.rms_norm(box[("mlpin", s_[1])], (D,)).reshape(-1, D).double(); st[s_]["Gx"] += xh.T @ xh; st[s_]["sx"] += xh.sum(0)
        for s_ in SITES + ASITES:
            ww = box[s_].reshape(-1, D).double(); wst[s_]["Gw"] += ww.T @ ww; wst[s_]["sw"] += ww.sum(0)
        n += idx.numel()
    me = None
    heads = {}
    for s_ in SITES + ASITES:
        mx = st[s_]["sx"] / n; Cx = st[s_]["Gx"] / n - torch.outer(mx, mx); evx, Ux = torch.linalg.eigh(Cx); Ux = Ux.flip(1)
        heads[s_] = {"U": Ux.float(), "mx": mx.float(), "Afull": None, "Cx": Cx, "eff_rank": R.spectrum(Cx.cpu())["eff_rank"]}
    wheads = {}
    for s_ in SITES + ASITES:
        mw = wst[s_]["sw"] / n; Cw = wst[s_]["Gw"] / n - torch.outer(mw, mw); _, Uw = torch.linalg.eigh(Cw); Uw = Uw.flip(1)
        wheads[s_] = {"U": Uw.float(), "mu": mw.float(), "Cw": Cw, "eff_rank": R.spectrum(Cw.cpu())["eff_rank"]}
    del st, wst
    log(stage="heads_done")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    class OwnHead:
        """Block l with its own weights on a k-dim input subspace (its own top-k input PCs, or a supplied basis U_); the rest of the input
        is the fit-set constant ('const') or the constant plus the ridge token read ('tok', §2730 recipe); output unrestricted."""
        def __init__(self, l, k, fill, U_=None):
            h = heads[("mlp", l)]; self.mlp = m.transformer.h[l].mlp; self.U = h["U"][:, :k] if U_ is None else U_; self.mx = h["mx"]; self.fill = fill
            self.A = h["Afull"]
        def hook(self, idx):
            self.idx = idx
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); core_ = (xh @ self.U) @ self.U.T
            if self.fill == "tok":
                e = F.rms_norm(m.transformer.wte(self.idx), (D,)); fb = self.mx + (e - me.float()) @ self.A
            else:
                fb = self.mx
            xp_ = core_ + fb - (fb @ self.U) @ self.U.T
            return self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
    class AttnHead:
        """Attention block l recomputed from a projected input h = mx + U U^T (xhat - mx) (or the exact xhat when full); own weights;
        the block-0 value residual v1 is recomputed exactly from the token embedding."""
        def __init__(self, l, k, U_=None, full=False):
            self.a = m.transformer.h[l].attn; h = heads[("attn", l)]; self.U = (h["U"] if U_ is None else U_)[:, :k]; self.mx = h["mx"]; self.full = full; self.l = l
        def hook(self, idx):
            B, Tn = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); b0 = m.transformer.h[0]
            self.v1 = b0.attn.c_v(F.rms_norm(b0.lambdas[0] * x0 + b0.lambdas[1] * x0, (D,))).view(B, Tn, NH, HD)
            self.cos, self.sin = (t.to(DEV) for t in R.rope(Tn)); self.mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
        def __call__(self, w, x):
            B, Tn = x.shape[:2]; xh = F.rms_norm(x, (D,)); a = self.a
            h = xh if self.full else self.mx + ((xh - self.mx) @ self.U) @ self.U.T
            def pr(lin):
                return R.rot(F.rms_norm(lin(h).view(B, Tn, NH, HD), (HD,)), self.cos, self.sin)
            v = a.c_v(h).view(B, Tn, NH, HD)
            if self.l != 0:
                v = (1 - a.lamb) * v + a.lamb * self.v1.view_as(v)
            pat = (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q), pr(a.c_k)) / HD) * (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q2), pr(a.c_k2)) / HD)
            pat = pat.masked_fill(~self.mask, 0.0)
            return a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
    class Buf:
        def __init__(self): self.acc = None
        def hook(self, idx): self.acc = None
        def add(self, t): self.acc = t.clone() if self.acc is None else self.acc + t
    class Route:
        """Split site s's write into its part in the read core and its remainder; mode 'delete' drops the remainder, 'readout' moves
        it to the buffer (added before the final norm), 'hidden' keeps it in the stream and books it for subtraction at the end."""
        def __init__(self, s_, k, U_, mode, buf):
            self.U = U_[:, :k]; self.mu = wheads[s_]["mu"]; self.mode = mode; self.buf = buf
        def hook(self, idx):
            pass
        def __call__(self, w, x):
            c = w - self.mu; inside = (c @ self.U) @ self.U.T; out = c - inside
            if self.mode == "delete":
                return self.mu + inside
            self.buf.add(out)
            return self.mu + inside if self.mode == "readout" else w
    class FinalAdd:
        def __init__(self, buf, sign): self.buf = buf; self.sign = sign
        def hook(self, idx):
            pass
        def __call__(self, x, _):
            return x if self.buf.acc is None else x + self.sign * self.buf.acc
    class Both:
        def __init__(self, reader, writer): self.r = reader; self.w = writer
        def hook(self, idx):
            self.r.hook(idx); self.w.hook(idx)
        def __call__(self, w, x):
            return self.w(self.r(w, x), x)
    def core_of(sites):
        C = sum(heads[s_]["Cx"] for s_ in sites) / len(sites); _, U_ = torch.linalg.eigh(C); return U_.flip(1).float()
    U_late = core_of(LSITES)
    def run(patch, extra=()):
        objs = list(patch.values()) + list(extra)
        def hook(idx):
            for o in objs:
                o.hook(idx)
        return ce_of(m, ev, patch, hook) - ce0
    def head(s_, k, U_=None):
        return OwnHead(s_[1], k, "const", None if U_ is None else U_[:, :k]) if s_[0] == "mlp" else AttnHead(s_[1], k, U_)
    def blocks(ls):
        return [(kd, l) for l in ls for kd in ("attn", "mlp") if 0 <= l < NL]
    OWN8 = blocks(range(0, SPLIT)); SET8 = blocks(range(SPLIT, NL)); U_8 = core_of(SET8)
    def split8(k):
        patch = {s_: head(s_, k) for s_ in OWN8}
        for s_ in SET8:
            patch[s_] = head(s_, k, U_8)
        return run(patch)
    HID = m.transformer.h[SPLIT].mlp.Left.weight.shape[0]
    class TermHead(OwnHead):
        """MLP l reads the bus core c (top-k of U_8; rest constant) with t the dropped tail; product = Lc*Rc + mask*(Lc*Rt + Lt*Rc) (+ Lt*Rt if tt).
        mask: None = all hidden units keep their cross term; a bool[HID] keeps it on those units only; 'none' = core*core only."""
        def __init__(self, l, k, mask=None, tt=False, rec=None):
            super().__init__(l, k, "const", U_8[:, :k]); self.mask = mask; self.tt = tt; self.rec = rec
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); c = (xh @ self.U) @ self.U.T + self.mx - (self.mx @ self.U) @ self.U.T; t = xh - c
            Lc = self.mlp.Left(c); Rc = self.mlp.Right(c); Lt = self.mlp.Left(t); Rt = self.mlp.Right(t)
            cross = Lc * Rt + Lt * Rc
            if self.rec is not None:
                self.rec["E"] += (cross.reshape(-1, HID).double() ** 2).sum(0); self.rec["n"] += cross.reshape(-1, HID).shape[0]
                return w
            if isinstance(self.mask, str):
                prod = Lc * Rc
            elif self.mask is None:
                prod = Lc * Rc + cross
            else:
                prod = Lc * Rc + cross * self.mask
            if self.tt:
                prod = prod + Lt * Rt
            return self.mlp.Down(prod)
    recs = {l: {"E": torch.zeros(HID, dtype=torch.float64, device=DEV), "n": 0} for l in range(SPLIT, NL)}
    for i in range(0, fit_rows.shape[0], CH):
        idx = fit_rows[i:i + CH, :TI]
        pt = {("mlp", l): TermHead(l, 768, rec=recs[l]) for l in range(SPLIT, NL)}
        for o in pt.values():
            o.hook(idx)
        forward(m, idx, patch=pt)
    energy = {}; order = {}; pr_list = []
    for l in range(SPLIT, NL):
        col = (m.transformer.h[l].mlp.Down.weight.double() ** 2).sum(0)
        e_ = recs[l]["E"] / recs[l]["n"] * col; energy[l] = e_; order[l] = torch.argsort(e_, descending=True)
        pr_list.append(float(e_.sum() ** 2 / (e_ ** 2).sum()))
    pr_mean = sum(pr_list) / len(pr_list)
    def cum_share(l, h):
        e_ = energy[l]; return float(e_[order[l][:h]].sum() / e_.sum())
    log(stage="unit_energy", participation_ratio_mean=round(pr_mean, 1), pr_by_block=[round(v, 1) for v in pr_list],
        share_top1024=[round(cum_share(l, 1024), 3) for l in range(SPLIT, NL)])
    def mask_top(l, h):
        mk = torch.zeros(HID, dtype=torch.bool, device=DEV); mk[order[l][:h]] = True; return mk
    def mask_rand(l, h, seed):
        g = torch.Generator().manual_seed(seed * 100 + l); mk = torch.zeros(HID, dtype=torch.bool); mk[torch.randperm(HID, generator=g)[:h]] = True; return mk.to(DEV)
    def cross_top(h):
        return run({("mlp", l): TermHead(l, 768, mask=mask_top(l, h)) for l in range(SPLIT, NL)})
    def cross_rand(h, seed=0):
        return run({("mlp", l): TermHead(l, 768, mask=mask_rand(l, h, seed)) for l in range(SPLIT, NL)})
    arms = {"SPLIT8_1024": split8(1024), "LATE_MLP_768": run({("mlp", l): TermHead(l, 768, mask="none") for l in range(SPLIT, NL)}),
            "DROP_TT_768": run({("mlp", l): TermHead(l, 768) for l in range(SPLIT, NL)}), "CROSS_TOP_1024": cross_top(1024)}
    if not smoke:
        arms["CROSS_TOP_256"] = cross_top(256); arms["CROSS_TOP_512"] = cross_top(512); arms["CROSS_TOP_2048"] = cross_top(2048); arms["CROSS_RAND_1024"] = cross_rand(1024)
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    C0 = arms["LATE_MLP_768"]; CT = arms["DROP_TT_768"]; gain = max(C0 - CT, 1e-9)
    def rec_of(name):
        return (C0 - arms[name]) / gain if name in arms else None
    r1024 = rec_of("CROSS_TOP_1024"); r512 = rec_of("CROSS_TOP_512"); rr = rec_of("CROSS_RAND_1024")
    summ = {"cross_gain_768": C0 - CT, "recovery": {k: rec_of(k) for k in arms if k.startswith("CROSS_")}, "participation_ratio_mean": pr_mean, "participation_ratio_by_block": pr_list,
            "energy_share_top": {str(h): [cum_share(l, h) for l in range(SPLIT, NL)] for h in (256, 512, 1024, 2048)}, "hidden_dim": HID}
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["SPLIT8_1024"] - PRIOR_SPLIT8_1024) <= BARS["repro_tol"] and abs(C0 - PRIOR_LATE_MLP_768) <= BARS["repro_tol"] and abs(CT - PRIOR_DROP_TT_768) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_top_1024_units_recover_most_of_the_cross_gain': bool(r1024 is not None and r1024 >= BARS["b_min"]),
        'pred_c_top_512_units_recover_much': bool(r512 is not None and r512 >= BARS["c_min"]),
        'pred_d_random_1024_units_recover_little': bool(rr is not None and rr <= BARS["d_max"]),
        'pred_e_cross_energy_concentrated_on_units': bool(pr_mean <= BARS["e_max"]),
    }
    nulls = {"b_null_rec1024_le_.35": bool(r1024 is not None and r1024 <= NULLS["b_max"]), "c_null_rec512_le_.20": bool(r512 is not None and r512 <= NULLS["c_max"]),
             "d_null_rand1024_ge_.55": bool(rr is not None and rr >= NULLS["d_min"]), "e_null_pr_ge_2500": bool(pr_mean >= NULLS["e_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "split": SPLIT,
           "program": "blocks 8-17 MLP reads through the bus (top-768 of U_8; rest constant); the cross term Lc*Rt + Lt*Rc is kept on the top-h hidden units by fit-set output energy (CROSS_TOP_h) or a seeded random h units (CROSS_RAND_h); tail*tail dropped throughout",
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "split8_1024": arms["SPLIT8_1024"], "prior_split8_1024": PRIOR_SPLIT8_1024, "late_mlp_768": C0, "prior_late_mlp_768": PRIOR_LATE_MLP_768, "drop_tt_768": CT, "prior_drop_tt_768": PRIOR_DROP_TT_768},
           "ce_added": arms, "summary": summ,
           "price": {"gpu_doc_forwards": 2 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
