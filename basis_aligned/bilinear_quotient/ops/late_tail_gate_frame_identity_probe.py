#!/usr/bin/env python
"""late_tail_gate_frame_identity_probe -- WHAT is the shared late tail-read gate frame? §2806: one pooled 256-dim core frame gates all ten
late tail reads at .0442 (own per-block frames + .011 at every rank). This rung prices three NAMED 256-dim core subspaces with the same
instrument -- the core's variance frame (top PCs of the pooled core moment), the token-identity frame (unembedding top right-singular
directions projected into the core), and the data-free weight frame (top eigenvectors of the unwhitened pooled gate Gram) -- against the
shared frame and a random one; extends the shared ladder to 384 and 512 modes; and tests, at those two NEW ranks, the regularity seen post
hoc in §2806 that SHARED_k = 1 - (pooled captured energy at k) to within .003 at k = 32/64/128/256. CUDA lane-1.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_gate_frame_is_not_the_variance_frame pred_c_shared_ladder_reaches_025_at_512
#                     pred_d_cost_equals_lost_pooled_energy_at_new_ranks pred_e_gate_frame_is_not_the_token_identity_frame

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER.
Descriptive; nothing installs into the §312 frontier; bases are data second moments scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_TAIL_GATE_FRAME_IDENTITY_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_tail_gate_frame_identity_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_tail_gate_frame_identity_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_TAIL_GATE_FRAME_IDENTITY_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_tail_gate_shared_frame_probe_results.json"   # §2806
OUT = ROOT / "late_tail_gate_frame_identity_probe_results.json"
HASHES = {PREREG: "6f6f6dc89d3170450dacf140b66a6a43fbded7408a2f01d5ca8fac128bd68817", PRIOR: "0e4fd8e263e2018b9b0cac12b8a8ff2bc965066895891f4fdf40883a05be66ef",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_tail_gate_frame_identity_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_SPLIT8_1024 = 0.0374; PRIOR_LATE_MLP_768 = 0.1249; PRIOR_EARLY_TAIL_ONLY = 0.0711; PRIOR_JOINT_GATE0 = 0.3668; PRIOR_GATE128 = 0.0627; PRIOR_SUM_GATE0 = 0.1895; PRIOR_B8_GATE0 = 0.0304; PRIOR_B9_GATE0 = 0.0209; PRIOR_B16_GATE0 = 0.0181; PRIOR_B17_GATE0 = 0.0458; PRIOR_B17_GATE16 = 0.0083; PRIOR_B16_GATE64 = 0.0051; PRIOR_TWO_BLOCK = 0.01491; PRIOR_FULL = 0.0722; PRIOR_SHARED256 = 0.0442; ALT = (64, 96); LAM = 1e-2
K = 1024; SPLIT = 8; EARLY = list(range(0, 11)); LATE = list(range(11, 18))
SITES = [("mlp", l) for l in range(NL)]; ASITES = [("attn", l) for l in range(NL)]; FINAL = ("final", -1)
ESITES = [(kd, l) for l in EARLY for kd in ("attn", "mlp")]; LSITES = [("mlp", l) for l in LATE] + [("attn", l) for l in LATE]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "exact_tol": 1e-3, "gram_tol": 1e-6, "frame_tol": 1e-6, "floor": 0.002, "b_margin": 0.02, "c_cost": 0.025, "d_dev": 0.01, "e_margin": 0.02}
NULLS = {"b_margin": 0.005, "c_cost": 0.04, "d_dev": 0.02, "e_margin": 0.005}


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
    Uk = U_8[:, :768]; Ut = U_8[:, 768:]
    def perp(z):
        return z - (z @ Uk) @ Uk.T
    def core(z):
        return (z @ Uk) @ Uk.T
    # ---- exact weight-side gate Gram: G_ij = E_t <J_i t, J_j t> for the core coordinates i, j, where J(c) t = W_D[(L c) o (R t) + (R c) o (L t)]
    #      = sum_i c_i J_i t and E over the tail second moment M_t; then weighted by the core second moment M_c (independence factorisation).
    def gate_gram(l):
        mlp = m.transformer.h[l].mlp; h = heads[("mlp", l)]
        M = h["Cx"].double().to(DEV) + torch.outer(h["mx"].double(), h["mx"].double())
        Ukd, Utd = Uk.double(), Ut.double(); Mc = Ukd.T @ M @ Ukd; Mt = Utd.T @ M @ Utd
        Wl, Wr, Wd = mlp.Left.weight.double(), mlp.Right.weight.double(), mlp.Down.weight.double()
        Lk, Rk, Lt, Rt = Wl @ Ukd, Wr @ Ukd, Wl @ Utd, Wr @ Utd
        Gd = Wd.T @ Wd
        Mrr = Gd * (Rt @ Mt @ Rt.T); Mll = Gd * (Lt @ Mt @ Lt.T); Mrl = Gd * (Rt @ Mt @ Lt.T).T
        G = Lk.T @ Mrr @ Lk + Rk.T @ Mll @ Rk + Rk.T @ Mrl @ Lk + Lk.T @ Mrl.T @ Rk
        G = 0.5 * (G + G.T)
        ev_c, V_c = torch.linalg.eigh(Mc); ev_c = ev_c.clamp_min(1e-8 * float(ev_c.max()))
        Mc_h = (V_c * ev_c.sqrt()) @ V_c.T; Mc_ih = (V_c / ev_c.sqrt()) @ V_c.T
        Gw = Mc_h @ G @ Mc_h; Gw = 0.5 * (Gw + Gw.T); evw, Vw = torch.linalg.eigh(Gw); evw = evw.flip(0).clamp_min(0); Vw = Vw.flip(1)
        return {"G": G, "Gw": Gw, "evw": evw, "Vw": Vw, "Mc_h": Mc_h, "Mc_ih": Mc_ih, "Mt": Mt, "Lk": Lk, "Rk": Rk, "Lt": Lt, "Rt": Rt, "Wd": Wd, "eff_rank_core_input": R.spectrum(Mc.cpu())["eff_rank"]}
    def direct_check(g, n_pairs=6, seed=0):
        """materialise J_i = W_D[diag(L_i) R_t + diag(R_i) L_t] for random i, j and compare tr(J_i^T J_j M_t) with the Gram formula."""
        gen = torch.Generator().manual_seed(seed); worst = 0.0
        for _ in range(n_pairs):
            i, j = (int(v) for v in torch.randint(0, 768, (2,), generator=gen))
            Ji = g["Wd"] @ (g["Lk"][:, i:i + 1] * g["Rt"] + g["Rk"][:, i:i + 1] * g["Lt"]); Jj = g["Wd"] @ (g["Lk"][:, j:j + 1] * g["Rt"] + g["Rk"][:, j:j + 1] * g["Lt"])
            d = float(torch.trace(Ji.T @ Jj @ g["Mt"])); f = float(g["G"][i, j]); worst = max(worst, abs(d - f) / max(abs(d), 1e-12))
        return worst
    gates = {}
    for l in LB:
        gates[l] = gate_gram(l)
        gates[l]["check"] = direct_check(gates[l]) if l in (SPLIT, NL - 1) else None
        gates[l]["spec"] = R.spectrum(gates[l]["Gw"].cpu()); gates[l]["spec_unweighted"] = R.spectrum(gates[l]["G"].cpu())
        pr_ = gates[l]["evw"] / gates[l]["evw"].sum(); gates[l]["captured"] = {k: float(pr_[:k].sum()) for k in (16, 64, 128, 256)}
        gates[l]["trace_ratio_gw_over_g"] = float(gates[l]["evw"].sum())
        for nm in ("Mt", "Lt", "Rt", "Wd"):
            gates[l][nm] = None
        torch.cuda.empty_cache()
    log(stage="gate_grams", check_8=gates[SPLIT]["check"], check_17=gates[NL - 1]["check"], eff=[round(gates[l]["spec"]["eff_rank"], 1) for l in LB], r90=[gates[l]["spec"]["rank_90"] for l in LB])
    def med(v):
        v = sorted(v); return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    class Gate(OwnHead):
        """MLP l = MLP(c) + J(Pi_k c) t + MLP(t): the cross term's gate input c is replaced by its projection onto the top-k gate modes (in the
        core-second-moment-whitened metric; k = 0 removes the cross term). rec: fit-pass cross-term output energy (exact, no factorisation)."""
        def __init__(self, l, k, rec=None):
            super().__init__(l, 768, "const", Uk); self.l = l; self.k = k; self.rec = rec
            g = gates[l]
            self.Pi = None if k is None else (g["Mc_h"] @ g["Vw"][:, :k] @ g["Vw"][:, :k].T @ g["Mc_ih"]).float() if k > 0 else torch.zeros(768, 768, device=DEV)
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); c = core(xh); t = xh - c; mlp = self.mlp
            Mc_ = mlp.Down(mlp.Left(c) * mlp.Right(c)); Mt_ = mlp.Down(mlp.Left(t) * mlp.Right(t))
            if self.Pi is None:
                cross = mlp.Down(mlp.Left(xh) * mlp.Right(xh)) - Mc_ - Mt_
            else:
                cg = ((xh @ Uk) @ self.Pi.T) @ Uk.T
                cross = mlp.Down(mlp.Left(cg) * mlp.Right(t) + mlp.Right(cg) * mlp.Left(t))
            if self.rec is not None:
                self.rec[("e_cross", self.l)] += float(cross.pow(2).sum()); self.rec[("n", self.l)] += xh.shape[0] * xh.shape[1]
            return Mc_ + cross + Mt_
    rec = {}
    for l in LB:
        rec[("e_cross", l)] = 0.0; rec[("n", l)] = 0
    for i in range(0, fit_rows.shape[0], CH):
        forward(m, fit_rows[i:i + CH, :TI], patch={("mlp", l): Gate(l, None, rec=rec) for l in LB})
    indep = {l: gates[l]["trace_ratio_gw_over_g"] / max(rec[("e_cross", l)] / rec[("n", l)], 1e-12) for l in LB}
    log(stage="fit", indep_factor=[round(indep[l], 3) for l in LB])
    # --- read spaces, pooled frame, transferred and random frames (weight-side, exact; G_l kept per block) ---
    def orth(A):
        Q, _ = torch.linalg.qr(A); return Q
    def read_space(l, k):
        g = gates[l]; return orth(g["Mc_ih"] @ g["Vw"][:, :k])
    def overlap(Q1, Q2):
        return float((Q1.T @ Q2).pow(2).sum() / Q1.shape[1])
    def randQ(k, seed):
        return orth(torch.randn(768, k, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).to(DEV))
    Mc_of = {l: gates[l]["Mc_h"] @ gates[l]["Mc_h"] for l in LB}
    def retained(l, Pi):
        g = gates[l]; Pd = Pi.double(); return float(torch.trace(g["G"] @ Pd @ Mc_of[l] @ Pd.T) / torch.trace(g["G"] @ Mc_of[l]))
    KS = (16, 64, 128)
    adj = {k: {f"{l}_{l + 1}": overlap(read_space(l, k), read_space(l + 1, k)) for l in LB[:-1]} for k in KS}
    adj_rand = {k: overlap(randQ(k, 1), randQ(k, 2)) for k in KS}
    far = {k: overlap(read_space(SPLIT, k), read_space(NL - 1, k)) for k in KS}                        # blocks 8 vs 17
    Mcb = sum(Mc_of.values()) / len(LB); Gp = sum(gates[l]["G"] / torch.trace(gates[l]["G"]) for l in LB)
    evb, Vb = torch.linalg.eigh(Mcb); evb = evb.clamp_min(1e-8 * float(evb.max())); Mcb_h = (Vb * evb.sqrt()) @ Vb.T; Mcb_ih = (Vb / evb.sqrt()) @ Vb.T
    Gwp = Mcb_h @ Gp @ Mcb_h; Gwp = 0.5 * (Gwp + Gwp.T); evp, Vp = torch.linalg.eigh(Gwp); evp = evp.flip(0).clamp_min(0); Vp = Vp.flip(1)
    pooled_spec = R.spectrum(Gwp.cpu()); pooled_captured = {k: float(evp[:k].sum() / evp.sum()) for k in (16, 32, 64, 128, 256, 384, 512)}
    def pi_shared(k):
        return (Mcb_h @ Vp[:, :k] @ Vp[:, :k].T @ Mcb_ih).float()
    def pi_own(l, k):
        g = gates[l]; return (g["Mc_h"] @ g["Vw"][:, :k] @ g["Vw"][:, :k].T @ g["Mc_ih"]).float()
    Qr = randQ(128, 0); pi_rand = (Mcb_h @ Qr @ Qr.T @ Mcb_ih).float()
    def pi_orth(Q):
        return (Q @ Q.T).float()
    def ret_pool(Pi):
        Pd = Pi.double(); return float(torch.trace(Gp @ Pd @ Mcb @ Pd.T) / torch.trace(Gp @ Mcb))
    evm, Um = torch.linalg.eigh(Mcb); Um = Um.flip(1)                                                  # core-variance (PCA) frame of the pooled core moment
    evg, Ug = torch.linalg.eigh(Gp); Ug = Ug.flip(1)                                                   # data-free weight frame (unwhitened pooled Gram)
    Wu = m.lm_head.weight.detach().double(); _, su, Vhu = torch.linalg.svd(Wu, full_matrices=False); Qu = orth(Uk.double().T @ Vhu[:256].T); del Wu
    Qr = randQ(256, 0)
    frames = {"SHARED_256": pi_shared(256), "SHARED_384": pi_shared(384), "SHARED_512": pi_shared(512),
              "PCA_128": pi_orth(Um[:, :128]), "PCA_256": pi_orth(Um[:, :256]), "PCA_512": pi_orth(Um[:, :512]),
              "WEIGHT_256": pi_orth(Ug[:, :256]), "UNEMB_256": pi_orth(Qu), "RANDOM_256": (Mcb_h @ Qr @ Qr.T @ Mcb_ih).float()}
    reads = {"SHARED_256": orth(Mcb_ih @ Vp[:, :256]), "PCA_256": Um[:, :256], "WEIGHT_256": Ug[:, :256], "UNEMB_256": Qu, "RANDOM_256": orth(Mcb_ih @ Qr)}
    overlap_with_shared = {nm: overlap(Q, reads["SHARED_256"]) for nm, Q in reads.items()}
    ret = {nm: {str(l): retained(l, Pi) for l in LB} for nm, Pi in frames.items()}
    ret["OWN_128"] = {str(l): retained(l, pi_own(l, 128)) for l in LB}
    ret_pooled = {nm: ret_pool(Pi) for nm, Pi in frames.items()}
    frame_check = max(max(abs(ret["OWN_128"][str(l)] - gates[l]["captured"][128]) for l in LB), abs(ret_pooled["SHARED_256"] - pooled_captured[256]))
    unemb_in_core = float((Uk.double().T @ Vhu[:256].T).pow(2).sum() / 256)                            # how much of the unembedding top-256 lies in the core
    log(stage="frames", pooled_captured=pooled_captured, overlap_with_shared=overlap_with_shared, ret_pooled=ret_pooled, frame_check=frame_check, unemb_in_core=unemb_in_core,
        spec_G_eff=R.spectrum(Gp.cpu())["eff_rank"], spec_M_eff=R.spectrum(Mcb.cpu())["eff_rank"])
    class GateP(Gate):
        """MLP l with the cross term gated by a SUPPLIED core projector Pi (shared / named / random frame) instead of its own modes."""
        def __init__(self, l, Pi):
            super().__init__(l, None); self.Pi = Pi
    def many(kmap):
        patch = {("mlp", j): Gate(j, None) for j in LB}
        for j, kj in kmap.items():
            patch[("mlp", j)] = Gate(j, kj)
        return run(patch)
    def many_pi(Pi):
        return run({("mlp", j): GateP(j, Pi) for j in LB})
    arms = {"SPLIT8_1024": split8(1024), "GATE_EXACT": many({}), "UNIFORM_128": many({l: 128 for l in LB})}
    for nm, Pi in frames.items():
        arms[nm] = many_pi(Pi)
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    law = {str(k): {"cost": arms[f"SHARED_{k}"], "lost_pooled_energy": 1.0 - pooled_captured[k], "deviation": arms[f"SHARED_{k}"] - (1.0 - pooled_captured[k])} for k in (256, 384, 512)}
    law_off = {nm: {"cost": arms[nm], "lost_pooled_energy": 1.0 - ret_pooled[nm], "deviation": arms[nm] - (1.0 - ret_pooled[nm])} for nm in ("PCA_128", "PCA_256", "PCA_512", "WEIGHT_256", "UNEMB_256", "RANDOM_256")}
    dev_new = max(abs(law["384"]["deviation"]), abs(law["512"]["deviation"])); dev_new_min = min(abs(law["384"]["deviation"]), abs(law["512"]["deviation"]))
    summ = {"pooled_captured": {str(k): v for k, v in pooled_captured.items()}, "pooled_gram": {"eff_rank": pooled_spec["eff_rank"], "rank_90": pooled_spec["rank_90"]},
            "overlap_with_shared_256_read_space": overlap_with_shared, "random_pair_expected": 256 / 768, "retained_pooled_energy": ret_pooled, "retained_energy_per_block": ret,
            "law_shared_ladder": law, "law_prior_ranks_2806": {"32": {"cost": 0.1209, "lost": 0.118}, "64": {"cost": 0.0981, "lost": 0.097}, "128": {"cost": 0.0740, "lost": 0.074}, "256": {"cost": 0.0442, "lost": 0.046}},
            "law_off_family": law_off, "max_dev_new_ranks": dev_new, "unemb_top256_energy_in_core": unemb_in_core,
            "pca_256_minus_shared_256": arms["PCA_256"] - arms["SHARED_256"], "unemb_256_minus_shared_256": arms["UNEMB_256"] - arms["SHARED_256"], "weight_256_minus_shared_256": arms["WEIGHT_256"] - arms["SHARED_256"],
            "prior_uniform_gate128_2799": PRIOR_GATE128, "prior_shared_256_2806": PRIOR_SHARED256, "direct_check_rel_err": {"8": gates[SPLIT]["check"], "17": gates[NL - 1]["check"]}, "frame_check": frame_check, "n_fit_tokens": rec[("n", SPLIT)]}
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["SPLIT8_1024"] - PRIOR_SPLIT8_1024) <= BARS["repro_tol"] and abs(arms["GATE_EXACT"]) <= BARS["exact_tol"]
                   and abs(arms["UNIFORM_128"] - PRIOR_GATE128) <= BARS["repro_tol"] and abs(arms["SHARED_256"] - PRIOR_SHARED256) <= BARS["repro_tol"]
                   and max(gates[SPLIT]["check"], gates[NL - 1]["check"]) <= BARS["gram_tol"] and frame_check <= BARS["frame_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_gate_frame_is_not_the_variance_frame': bool(arms["PCA_256"] - arms["SHARED_256"] >= BARS["b_margin"]),
        'pred_c_shared_ladder_reaches_025_at_512': bool(arms["SHARED_512"] <= BARS["c_cost"]),
        'pred_d_cost_equals_lost_pooled_energy_at_new_ranks': bool(dev_new <= BARS["d_dev"]),
        'pred_e_gate_frame_is_not_the_token_identity_frame': bool(arms["UNEMB_256"] - arms["SHARED_256"] >= BARS["e_margin"]),
    }
    nulls = {"b_null_pca_within_.005": bool(arms["PCA_256"] - arms["SHARED_256"] <= NULLS["b_margin"]),
             "c_null_shared512_ge_.04": bool(arms["SHARED_512"] >= NULLS["c_cost"]),
             "d_null_both_deviations_ge_.02": bool(dev_new_min >= NULLS["d_dev"]),
             "e_null_unemb_within_.005": bool(arms["UNEMB_256"] - arms["SHARED_256"] <= NULLS["e_margin"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "split": SPLIT,
           "program": "named 256-dim core frames (pooled core-variance PCs; unembedding top-256 projected into the core; data-free weight frame; random) applied identically at all ten late MLPs' tail-read gates, against the §2806 shared frame; the shared ladder at 384/512; the cost = lost-pooled-energy regularity at the two new ranks",
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "split8_1024": arms["SPLIT8_1024"], "prior_split8_1024": PRIOR_SPLIT8_1024, "gate_exact": arms["GATE_EXACT"],
                          "uniform_128": arms["UNIFORM_128"], "prior_uniform_128": PRIOR_GATE128, "shared_256": arms["SHARED_256"], "prior_shared_256": PRIOR_SHARED256, "direct_check_rel_err": summ["direct_check_rel_err"], "frame_check": frame_check},
           "ce_added": arms, "summary": summ,
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
