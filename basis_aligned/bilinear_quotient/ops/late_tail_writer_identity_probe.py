#!/usr/bin/env python
"""late_tail_writer_identity_probe -- WHICH late block's tail write matters, and whether §2790's per-distance profile was a fade in time
or writer identity: the late-origin residual is split exactly (scalar lambda mixing) into one component per late writer block j; each
arm removes ONE writer's lambda-propagated tail write from every downstream late MLP reader (W8..W17), replaced by its fit-set mean.
The fit pass records the full 10 x 10 reader x writer tail-energy matrix; the distance slope is fitted WITH writer fixed effects.
Control for §2792's caveat on §2790. CUDA lane-1.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_no_true_fade_with_writer_fixed_effects pred_c_later_writers_write_more_tail
#                     pred_d_single_writer_cost_tracks_energy pred_e_single_writers_sum_near_the_whole

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_TAIL_WRITER_IDENTITY_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_tail_writer_identity_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_tail_writer_identity_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_TAIL_WRITER_IDENTITY_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_tail_read_operator_rank_probe_results.json"   # §2791
OUT = ROOT / "late_tail_writer_identity_probe_results.json"
HASHES = {PREREG: "12b625191dc2c7a2c103a1ab2a1902dd53f20803f1baf713c227a2b52861acee", PRIOR: "fcfd6bc76b308a21e9468e80f844b5b41dab571fea21b8d9792fe34db296aa11",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_tail_writer_identity_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_SPLIT8_1024 = 0.0374; PRIOR_LATE_MLP_768 = 0.1249; PRIOR_EARLY_TAIL_ONLY = 0.0711; PRIOR_D_LE2 = 0.0302; LAM = 1e-2
K = 1024; SPLIT = 8; EARLY = list(range(0, 11)); LATE = list(range(11, 18))
SITES = [("mlp", l) for l in range(NL)]; ASITES = [("attn", l) for l in range(NL)]; FINAL = ("final", -1)
ESITES = [(kd, l) for l in EARLY for kd in ("attn", "mlp")]; LSITES = [("mlp", l) for l in LATE] + [("attn", l) for l in LATE]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "b_beta_min": -0.11, "c_rho_min": 0.6, "d_rho_min": 0.6, "e_lo": 0.5, "e_hi": 1.2}
NULLS = {"b_beta_max": -0.22, "c_rho_max": 0.2, "d_rho_max": 0.2, "e_hi": 1.5}


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
    LB = list(range(SPLIT, NL))
    Uk = U_8[:, :768]
    def perp(z):
        return z - (z @ Uk) @ Uk.T
    def writers(l, win):
        """late writer blocks j <= l whose distance d = l - j falls in the window: ('le', n) -> d <= n; ('gt', n) -> d > n; 'all' -> every
        late writer j <= l (== the parent's 'both'); None -> nothing (exact). Block l's own attention write is the d = 0 writer."""
        if win is None:
            return []
        if win == "all":
            return [j for j in LB if j <= l]
        op, n = win
        if op == "one":
            return [n] if n <= l else []
        return [j for j in LB if j <= l and ((l - j <= n) if op == "le" else (l - j > n))]
    class Tracker:
        """Attention-site recorder for late block l: one lambda-propagated component per late writer block j (c_j <- lam0 c_j at every later
        block entry; block j's attention write is added at ('attn', j), its MLP write + Down bias at ('mlp', j)). x = y + sum_j c_j exactly
        at every late site (y = early origin). Returns the write unchanged."""
        def __init__(self, l, box):
            self.l = l; self.box = box
        def hook(self, idx):
            pass
        def __call__(self, w, x):
            b = self.box
            if self.l == SPLIT:
                b["c"] = {}
            else:
                lam0 = m.transformer.h[self.l].lambdas[0]
                for j in list(b["c"]):
                    b["c"][j] = lam0 * b["c"][j]
            b["c"][self.l] = w
            return w
    class RecHead(OwnHead):
        """MLP l: core exact; from the tail, the lambda-propagated writes of the blocks in the window are removed (each replaced by its
        fit-set mean, so the tail keeps its mean). Then records its own write (+ bias) into c_l."""
        def __init__(self, l, win, box, means):
            super().__init__(l, 768, "const", Uk); self.win = win; self.box = box; self.means = means; self.l = l
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); scale = xh.norm(dim=-1, keepdim=True) / x.norm(dim=-1, keepdim=True).clamp_min(1e-12); b = self.box
            js = writers(self.l, self.win); xp_ = xh
            if js:
                xp_ = xh - perp(sum(b["c"][j] * scale - self.means[(self.l, j)] for j in js))
            out = self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
            b["c"][self.l] = b["c"][self.l] + out + self.mlp.Down_bias
            return out
    WIN = {"DROP_ALL": "all", "D_LE2": ("le", 2)}; WIN.update({f"W{j}": ("one", j) for j in LB})
    # fit pass: per-(reader, writer) means of the scaled component; pooled tail energy of each window's composite and of each distance
    acc = {(l, j): torch.zeros(D, dtype=torch.float64, device=DEV) for l in LB for j in LB if j <= l}
    en = {nm: 0.0 for nm in WIN}; E = {(l, j): 0.0 for l in LB for j in LB if j <= l}; xn = {l: 0.0 for l in LB}; efull = 0.0; ntok = 0
    class MeanRec(RecHead):
        def __init__(self, l, box): super().__init__(l, None, box, None)
        def __call__(self, w, x):
            nonlocal efull
            xh = F.rms_norm(x, (D,)); scale = xh.norm(dim=-1, keepdim=True) / x.norm(dim=-1, keepdim=True).clamp_min(1e-12); b = self.box
            for j in writers(self.l, "all"):
                cj = b["c"][j] * scale
                acc[(self.l, j)] += cj.reshape(-1, D).double().sum(0); E[(self.l, j)] += float(perp(cj).pow(2).sum())
            for nm, win in WIN.items():
                js = writers(self.l, win)
                if js:
                    en[nm] += float(perp(sum(b["c"][j] * scale for j in js)).pow(2).sum())
            efull += float(perp(xh).pow(2).sum()); xn[self.l] += float(x.pow(2).sum())
            return super().__call__(w, x)
    for i in range(0, fit_rows.shape[0], CH):
        idx = fit_rows[i:i + CH, :TI]; bx = {}
        pt = {("attn", l): Tracker(l, bx) for l in LB}; pt.update({("mlp", l): MeanRec(l, bx) for l in LB})
        for o in pt.values():
            o.hook(idx)
        forward(m, idx, patch=pt); ntok += idx.numel()
    means = {k_: (v / ntok).float() for k_, v in acc.items()}
    eshare = {nm: (en[nm] / en["DROP_ALL"] if en["DROP_ALL"] > 0 else float("nan")) for nm in WIN}
    Emat = {f"{l}<-{j}": E[(l, j)] / ntok for (l, j) in E}                     # per-token tail energy of writer j's component as read by l
    # distance slope WITH writer fixed effects: log E_lj = a_j + beta * (l - j), pairs with d >= 1 (block j's MLP write included)
    pairs = [(l, j) for (l, j) in E if l > j and E[(l, j)] > 0]
    import math
    byw = {}
    for (l, j) in pairs:
        byw.setdefault(j, []).append((l - j, math.log(E[(l, j)] / ntok)))
    num = den = 0.0
    for j, pts in byw.items():
        if len(pts) < 2:
            continue
        md = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
        num += sum((p[0] - md) * (p[1] - my) for p in pts); den += sum((p[0] - md) ** 2 for p in pts)
    beta_fe = num / den if den > 0 else float("nan")
    # the same slope WITHOUT fixed effects (the §2790 pooled-by-distance reading), for the record
    md = sum(l - j for (l, j) in pairs) / len(pairs); my = sum(math.log(E[(l, j)] / ntok) for (l, j) in pairs) / len(pairs)
    beta_pooled = sum(((l - j) - md) * (math.log(E[(l, j)] / ntok) - my) for (l, j) in pairs) / max(sum(((l - j) - md) ** 2 for (l, j) in pairs), 1e-12)
    # what the reader sees: a write's per-block retention in the reader's NORMALISED input is lam0_l^2 * |x_{l-1}|^2 / |x_l|^2 (raw writes only
    # grow by lam0 >= .88; the fade, if any, is the residual norm growing under the reader's rms_norm)
    pred_ret = {l: float(m.transformer.h[l].lambdas[0]) ** 2 * xn[l - 1] / max(xn[l], 1e-12) for l in LB if l > SPLIT}
    log_ret_mean = sum(math.log(v) for v in pred_ret.values()) / len(pred_ret)
    wenergy = {j: sum(E[(l, j)] for l in LB if l >= j) / ntok for j in LB}                     # pooled over readers
    wpair = {j: sum(E[(l, j)] for l in LB if l > j) / max(len([l for l in LB if l > j]), 1) / ntok for j in LB if j < NL - 1}   # mean per downstream reader
    log(stage="tail_energy", late_origin_share_of_tail=round(en["DROP_ALL"] / max(efull, 1e-12), 3), beta_fe=round(beta_fe, 4), beta_pooled=round(beta_pooled, 4), writer_pair_energy={str(j): round(v, 1) for j, v in wpair.items()})
    def rec(win):
        bx = {}
        patch = {("attn", l): Tracker(l, bx) for l in LB}
        patch.update({("mlp", l): RecHead(l, win, bx, means) for l in LB})
        return run(patch)
    arms = {"SPLIT8_1024": split8(1024), "LATE_MLP_768": run({("mlp", l): head(("mlp", l), 768, U_8) for l in LB})}
    arms.update({nm: rec(win) for nm, win in WIN.items()})
    if smoke:
        arms["EXACT_CHECK"] = rec(None)
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    C0 = arms["LATE_MLP_768"]; A_ = arms["DROP_ALL"]
    def frac(nm):
        return arms[nm] / A_ if abs(A_) > 1e-9 else float("inf")
    wcost = {j: arms[f"W{j}"] for j in LB}
    js_pair = sorted(wpair); js_all = sorted(wcost)
    rho_depth = R.spearman([float(j) for j in js_pair], [wpair[j] for j in js_pair])
    rho_cost_energy = R.spearman([wcost[j] for j in js_all], [wenergy[j] for j in js_all])
    summ = {"writer_cost": {str(j): wcost[j] for j in LB}, "writer_cost_share": {str(j): frac(f"W{j}") for j in LB}, "writer_energy_pooled": {str(j): wenergy[j] for j in LB},
            "writer_energy_per_reader": {str(j): wpair[j] for j in wpair}, "energy_matrix_reader_from_writer": Emat,
            "beta_distance_writer_fixed_effects": beta_fe, "retention_per_block_fe": math.exp(beta_fe) if beta_fe == beta_fe else float("nan"),
            "beta_distance_pooled": beta_pooled, "retention_per_block_pooled": math.exp(beta_pooled),
            "reader_rms_x": {str(l): math.sqrt(xn[l] / ntok / D) for l in LB}, "predicted_retention_lambda_norm": {str(l): v for l, v in pred_ret.items()},
            "predicted_retention_geomean": math.exp(log_ret_mean), "beta_fe_minus_log_predicted_retention": beta_fe - log_ret_mean,
            "spearman_writer_depth_vs_per_reader_energy": rho_depth, "spearman_writer_cost_vs_pooled_energy": rho_cost_energy,
            "sum_single_writers_over_all": sum(wcost.values()) / A_ if abs(A_) > 1e-9 else float("inf"), "max_single_writer_share": max(frac(f"W{j}") for j in LB),
            "d_le2_share": frac("D_LE2"), "late_origin_share_of_tail_energy": en["DROP_ALL"] / max(efull, 1e-12), "lambda0_late": {str(l): float(m.transformer.h[l].lambdas[0]) for l in LB}}
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["SPLIT8_1024"] - PRIOR_SPLIT8_1024) <= BARS["repro_tol"] and abs(C0 - PRIOR_LATE_MLP_768) <= BARS["repro_tol"]
                   and abs(A_ - PRIOR_EARLY_TAIL_ONLY) <= BARS["repro_tol"] and abs(arms["D_LE2"] - PRIOR_D_LE2) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_no_true_fade_with_writer_fixed_effects': bool(beta_fe >= BARS["b_beta_min"]),
        'pred_c_later_writers_write_more_tail': bool(rho_depth >= BARS["c_rho_min"]),
        'pred_d_single_writer_cost_tracks_energy': bool(rho_cost_energy >= BARS["d_rho_min"]),
        'pred_e_single_writers_sum_near_the_whole': bool(BARS["e_lo"] <= summ["sum_single_writers_over_all"] <= BARS["e_hi"]),
    }
    nulls = {"b_null_beta_le_-.22": bool(beta_fe <= NULLS["b_beta_max"]), "c_null_rho_le_.2": bool(rho_depth <= NULLS["c_rho_max"]),
             "d_null_rho_le_.2": bool(rho_cost_energy <= NULLS["d_rho_max"]), "e_null_sum_ge_1.5": bool(summ["sum_single_writers_over_all"] >= NULLS["e_hi"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "split": SPLIT, "windows": {k: str(v) for k, v in WIN.items()},
           "program": "blocks 8-17 MLP reads: core (top-768 of U_8) exact; tail drops ONE late writer block's lambda-propagated write (attention + MLP + bias) from every downstream reader (W8..W17), replaced by its fit-set mean; plus DROP_ALL and D_LE2 as §2790 instruments",
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "split8_1024": arms["SPLIT8_1024"], "prior_split8_1024": PRIOR_SPLIT8_1024, "late_mlp_768": C0, "prior_late_mlp_768": PRIOR_LATE_MLP_768,
                          "drop_all": A_, "prior_early_tail_only": PRIOR_EARLY_TAIL_ONLY, "d_le2": arms["D_LE2"], "prior_d_le2": PRIOR_D_LE2},
           "ce_added": arms, "summary": summ,
           "price": {"gpu_doc_forwards": 2 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
