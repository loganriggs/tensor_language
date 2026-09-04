#!/usr/bin/env python
"""late_tail_rewrite_chain_probe -- IS §2794's chain a literal re-write? Each late MLP's TAIL WRITE (its output in the 384 tail coordinates)
is ridge-regressed, out of sample, on each earlier late writer's lambda-propagated tail component as the MLP reads it (45 reader-writer
pairs), on the MLP's full tail input, and on the previous block's component. If writer j+1's tail content is a transform of writer j's,
transfer R^2 falls with distance, adjacent transfer dominates far, and transfer rank-tracks §2794's loss-metric pair cosine kappa.
No drops, no CE arms beyond the two instruments. CUDA lane-1.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_transfer_falls_with_distance pred_c_adjacent_transfer_dominates_far
#                     pred_d_transfer_tracks_loss_coherence pred_e_tail_write_is_mostly_linear_in_tail_input

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. R^2 values are
out-of-sample explained variance, not CE. Descriptive; nothing installs into the §312 frontier (§2118 stays closed).
Preregistration: polynomial_causal/LATE_TAIL_REWRITE_CHAIN_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_tail_rewrite_chain_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_tail_rewrite_chain_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_TAIL_REWRITE_CHAIN_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_tail_writer_pair_coherence_probe_results.json"   # §2794
OUT = ROOT / "late_tail_rewrite_chain_probe_results.json"
HASHES = {PREREG: "276fe663cfbd047ddbeaca846a3ce44fb3ae5314fb32ca0b36cf6b8a69a85dfe", PRIOR: "ccc0b6323fc20d83d6df4cd9aa6d541a2cdc41c84cef86decf8b4ee260b6bd26",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_tail_rewrite_chain_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_SPLIT8_1024 = 0.0374; PRIOR_LATE_MLP_768 = 0.1249; PRIOR_EARLY_TAIL_ONLY = 0.0711; PRIOR_KAPPA = {"8_9": 0.1413, "8_10": 0.1321, "8_11": 0.0886, "8_12": 0.0742, "8_13": 0.1033, "8_14": 0.116, "8_15": 0.0612, "8_16": 0.0215, "9_10": 0.1906, "9_11": 0.1913, "9_12": 0.1083, "9_13": 0.0883, "9_14": 0.105, "9_15": 0.0732, "9_16": 0.0391, "10_11": 0.1787, "10_12": 0.169, "10_13": 0.1655, "10_14": 0.1609, "10_15": 0.0902, "10_16": 0.0528, "11_12": 0.2033, "11_13": 0.1688, "11_14": 0.1675, "11_15": 0.1335, "11_16": 0.048, "12_13": 0.2133, "12_14": 0.2045, "12_15": 0.1557, "12_16": 0.0685, "13_14": 0.3061, "13_15": 0.203, "13_16": 0.1355, "14_15": 0.2973, "14_16": 0.1666, "15_16": 0.1451}; LAM = 1e-2
K = 1024; SPLIT = 8; EARLY = list(range(0, 11)); LATE = list(range(11, 18))
SITES = [("mlp", l) for l in range(NL)]; ASITES = [("attn", l) for l in range(NL)]; FINAL = ("final", -1)
ESITES = [(kd, l) for l in EARLY for kd in ("attn", "mlp")]; LSITES = [("mlp", l) for l in LATE] + [("attn", l) for l in LATE]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "b_rho": -0.4, "c_ratio": 2.0, "d_rho": 0.5, "e_r2": 0.5}
NULLS = {"b_rho": 0.0, "c_ratio": 1.2, "d_rho": 0.1, "e_r2": 0.2}


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
    Uk = U_8[:, :768]; Ut = U_8[:, 768:]                  # tail coordinates: 384 dims orthogonal to the bus-768 core
    NT = Ut.shape[1]
    def perp(z):
        return z - (z @ Uk) @ Uk.T
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
    class Mom:
        """Centred second moments of (X [N,p], Y [N,q]) in float64: enough for a closed-form ridge fit and an out-of-sample R^2."""
        def __init__(self, p, q):
            self.A = torch.zeros(p, p, dtype=torch.float64, device=DEV); self.B = torch.zeros(p, q, dtype=torch.float64, device=DEV)
            self.Cyy = torch.zeros(q, q, dtype=torch.float64, device=DEV); self.sx = torch.zeros(p, dtype=torch.float64, device=DEV)
            self.sy = torch.zeros(q, dtype=torch.float64, device=DEV); self.n = 0
        def add(self, X, Y):
            X = X.double(); Y = Y.double(); self.A += X.T @ X; self.B += X.T @ Y; self.Cyy += Y.T @ Y; self.sx += X.sum(0); self.sy += Y.sum(0); self.n += X.shape[0]
        def centred(self):
            n = self.n; mx = self.sx / n; my = self.sy / n
            return self.A - n * torch.outer(mx, mx), self.B - n * torch.outer(mx, my), self.Cyy - n * torch.outer(my, my), mx, my
    def ridge(fit, lam_frac):
        A, B, _, mx, my = fit.centred()
        lam = lam_frac * torch.trace(A) / A.shape[0]
        beta = torch.linalg.solve(A + lam * torch.eye(A.shape[0], dtype=A.dtype, device=DEV), B)
        return beta, mx, my
    def r2_oos(ev_mom, beta, mx, my):
        """out-of-sample R^2 of Y ~ my + (X - mx) beta on the eval moments (mean model = fit-set my)."""
        A, B, Cyy, ex, ey = ev_mom.centred(); n = ev_mom.n
        dx = ex - mx; dy = ey - my
        # residual = (Yc + dy) - (Xc + dx) beta ; sum of squares via moments
        rss = torch.trace(Cyy) + n * float(dy @ dy) - 2 * torch.trace(B.T @ beta) - 2 * n * float(dy @ (dx @ beta)) \
              + torch.trace(beta.T @ A @ beta) + n * float((dx @ beta) @ (dx @ beta))
        tss = torch.trace(Cyy) + n * float(dy @ dy)
        return 1.0 - float(rss) / max(float(tss), 1e-12)
    PAIRS = [(l, j) for l in LB for j in LB if j < l]                  # reader l's tail write regressed on writer j's tail component, j < l
    moms = {sp: {("pair", l, j): Mom(NT, NT) for (l, j) in PAIRS} for sp in ("fit", "ev")}
    for sp in moms:
        moms[sp].update({("full", l): Mom(NT, NT) for l in LB}); moms[sp].update({("prev", l): Mom(NT, NT) for l in LB if l > SPLIT})
    class WriteRec(OwnHead):
        """MLP l exact (no drop); records its tail write against each earlier writer's tail component, the full tail input, and the previous
        block's component, in tail coordinates (@ Ut). Then adds its write (+ bias) to c_l."""
        def __init__(self, l, box, split):
            super().__init__(l, 768, "const", Uk); self.box = box; self.l = l; self.split = split
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); scale = xh.norm(dim=-1, keepdim=True) / x.norm(dim=-1, keepdim=True).clamp_min(1e-12); b = self.box
            out = self.mlp.Down(self.mlp.Left(xh) * self.mlp.Right(xh))
            Y = (out @ Ut).reshape(-1, NT); M = moms[self.split]
            for j in LB:
                if j < self.l:
                    M[("pair", self.l, j)].add(((b["c"][j] * scale) @ Ut).reshape(-1, NT), Y)
            M[("full", self.l)].add((xh @ Ut).reshape(-1, NT), Y)
            if self.l > SPLIT:
                M[("prev", self.l)].add(((b["c"][self.l - 1] * scale) @ Ut).reshape(-1, NT), Y)
            b["c"][self.l] = b["c"][self.l] + out + self.mlp.Down_bias
            return out
    for split, rows_ in (("fit", fit_rows), ("ev", ev)):
        for i in range(0, rows_.shape[0], CH):
            idx = rows_[i:i + CH, :TI]; bx = {}
            pt = {("attn", l): Tracker(l, bx) for l in LB}; pt.update({("mlp", l): WriteRec(l, bx, split) for l in LB})
            for o in pt.values():
                o.hook(idx)
            forward(m, idx, patch=pt)
    LAMF = 1e-2
    r2 = {}
    for key in moms["fit"]:
        beta, mx, my = ridge(moms["fit"][key], LAMF); r2[key] = r2_oos(moms["ev"][key], beta, mx, my)
    import math
    r2_pair = {(l, j): r2[("pair", l, j)] for (l, j) in PAIRS}; dist = {(l, j): l - j for (l, j) in PAIRS}
    def med(v):
        v = sorted(v); return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    r2_by_d = {d: med([r2_pair[p] for p in PAIRS if dist[p] == d]) for d in sorted(set(dist.values()))}
    rho_dist = R.spearman([float(dist[p]) for p in PAIRS], [r2_pair[p] for p in PAIRS])
    far = [r2_pair[p] for p in PAIRS if dist[p] >= 5]
    mf = med(far); ratio_adj_far = (r2_by_d[1] / mf) if mf > 1e-4 else (float('inf') if r2_by_d[1] > 0 else float('nan'))
    r2_full = {l: r2[("full", l)] for l in LB}; r2_prev = {l: r2[("prev", l)] for l in LB if l > SPLIT}
    # §2794's loss-metric pair cosine kappa_{j,k} (writers 8..16, j<k) against the transfer R^2 of writer k's write from writer j's component
    kap = PRIOR_KAPPA
    common = [(l, j) for (l, j) in PAIRS if f"{j}_{l}" in kap]
    rho_kappa = R.spearman([r2_pair[p] for p in common], [kap[f"{p[1]}_{p[0]}"] for p in common])
    arms = {"SPLIT8_1024": split8(1024), "LATE_MLP_768": run({("mlp", l): head(("mlp", l), 768, U_8) for l in LB})}
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    summ = {"r2_pair_reader_from_writer": {f"{l}<-{j}": v for (l, j), v in r2_pair.items()}, "median_r2_by_distance": {str(d): v for d, v in r2_by_d.items()},
            "spearman_distance_vs_r2": rho_dist, "median_r2_adjacent": r2_by_d[1], "median_r2_far_ge5": med(far), "adjacent_over_far": ratio_adj_far,
            "r2_full_tail_input": {str(l): v for l, v in r2_full.items()}, "median_r2_full_tail_input": med(list(r2_full.values())),
            "r2_prev_block_component": {str(l): v for l, v in r2_prev.items()}, "spearman_r2_vs_kappa_2794": rho_kappa, "n_common_pairs": len(common),
            "ridge_lambda_frac_of_mean_eigen": LAMF, "n_fit_tokens": moms["fit"][("full", SPLIT)].n, "n_eval_tokens": moms["ev"][("full", SPLIT)].n}
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["SPLIT8_1024"] - PRIOR_SPLIT8_1024) <= BARS["repro_tol"] and abs(arms["LATE_MLP_768"] - PRIOR_LATE_MLP_768) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_transfer_falls_with_distance': bool(rho_dist <= BARS["b_rho"]),
        'pred_c_adjacent_transfer_dominates_far': bool(ratio_adj_far >= BARS["c_ratio"]),
        'pred_d_transfer_tracks_loss_coherence': bool(rho_kappa >= BARS["d_rho"]),
        'pred_e_tail_write_is_mostly_linear_in_tail_input': bool(summ["median_r2_full_tail_input"] >= BARS["e_r2"]),
    }
    nulls = {"b_null_rho_ge_0": bool(rho_dist >= NULLS["b_rho"]), "c_null_ratio_le_1.2": bool(ratio_adj_far <= NULLS["c_ratio"]),
             "d_null_rho_le_.1": bool(rho_kappa <= NULLS["d_rho"]), "e_null_r2_le_.2": bool(summ["median_r2_full_tail_input"] <= NULLS["e_r2"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); R^2 values are out-of-sample on docs 0-63 (fits on docs 96-191) and are NOT CE numbers",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "split": SPLIT,
           "program": "no drops: each late MLP l's tail write (out @ Ut, 384 dims) is ridge-regressed (lambda = 1e-2 x mean eigenvalue) on (i) each earlier late writer j's lambda-propagated tail component as l reads it, (ii) l's full tail input xh @ Ut, (iii) block l-1's component; out-of-sample R^2 on docs 0-63",
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "split8_1024": arms["SPLIT8_1024"], "prior_split8_1024": PRIOR_SPLIT8_1024, "late_mlp_768": arms["LATE_MLP_768"], "prior_late_mlp_768": PRIOR_LATE_MLP_768},
           "ce_added": arms, "summary": summ,
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + 1 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
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
