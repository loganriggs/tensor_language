#!/usr/bin/env python
"""late_tail_readout_identity_probe -- is the late tail's readout value a FREQUENCY/CALIBRATION signal or TOKEN IDENTITY? §2800: withholding
the late MLPs' tail writes (blocks 8-17, the 384 dims outside the bus-768 core, centred, lambda-propagated in one accumulator A) from the
FINAL READOUT alone costs .1130, 98% of it on targets not in the context and monotone in target frequency (unseen targets +.285/token,
count >= 256 targets -.063/token). The simplest account is a single "rare-token" logit shift; this rung prices it: keep only the tail
direction d_f that best drives the unembedding along the centred log-frequency vector u (KEEP_FREQ1), drop only that direction
(DROP_FREQ1), compare with the top unembed / activation-PCA / random directions and with d_f added to the 32-dim unembed frame of §2798,
split the recovery by target class and frequency, and record the change in mean predictive entropy. CUDA lane-1.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_frequency_direction_alone_recovers_a_real_share pred_c_frequency_direction_beats_the_top_unembed_direction
#                     pred_d_channel_is_mostly_identity_not_frequency pred_e_withholding_the_channel_raises_entropy
#                     pred_f_frequency_direction_value_sits_on_novel_targets

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER; recovered(P) =
1 - KEEP_P / FINAL_ONLY. Descriptive; nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_TAIL_READOUT_IDENTITY_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_tail_readout_identity_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
from target_token_classes import target_token_classes

if not torch.cuda.is_available():
    raise RuntimeError("late_tail_readout_identity_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_TAIL_READOUT_IDENTITY_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_tail_readout_content_probe_results.json"   # §2800
OUT = ROOT / "late_tail_readout_identity_probe_results.json"
HASHES = {PREREG: "45a38dca3429ae5d116a6b52993fec9417813994750af31648e020d2357ac8a9", PRIOR: "3061e47d3477b28264b6a43d74ff291cd2b154102eb741ef97d077656c70502b",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_tail_readout_identity_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_SPLIT8_1024 = 0.0374; PRIOR_LATE_MLP_768 = 0.1249; PRIOR_EARLY_TAIL_ONLY = 0.0711; PRIOR_ALL_TAILOUT = 0.1459; PRIOR_READERS_ONLY = 0.0404; PRIOR_FINAL_ONLY = 0.1130; PRIOR_REC_WU32 = 0.193; N_RAND = 3; LAM = 1e-2
K = 1024; SPLIT = 8; EARLY = list(range(0, 11)); LATE = list(range(11, 18))
SITES = [("mlp", l) for l in range(NL)]; ASITES = [("attn", l) for l in range(NL)]; FINAL = ("final", -1)
ESITES = [(kd, l) for l in EARLY for kd in ("attn", "mlp")]; LSITES = [("mlp", l) for l in LATE] + [("attn", l) for l in LATE]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "none_tol": 1e-3, "b_rec": 0.15, "c_mult": 2.0, "d_drop": 0.85, "e_ent": 0.05, "f_share": 0.7}
NULLS = {"b_rec": 0.05, "c_mult": 1.0, "d_drop": 0.6, "e_ent": 0.0, "f_share": 0.4}


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
    class Box:
        """One lambda-propagated accumulator A of the late MLPs' centred TAIL writes (perp(write) - fit mean, blocks in `writers`), in stream
        units. Each consumer in `mlps` / `attns` / `final` computes from x - A instead of x; the stream itself keeps every real write, so
        the consumers NOT named see the tail exactly as the real model does. With all consumers named, x - A is the write-site drop of §2796."""
        def __init__(self, writers, mlps=(), attns=(), final=False):
            self.writers = set(writers); self.mlps = set(mlps); self.attns = set(attns); self.final = final; self.A = None
        def hook(self, idx):
            self.A = None
        def see(self, x, on):
            return x - self.A if (on and self.A is not None) else x
    class CAttn(AttnHead):
        """Late attention l: propagates A by lambda0 at block entry (Tracker semantics), then recomputes attention exactly from x - A if it is a consumer."""
        def __init__(self, l, box):
            super().__init__(l, D, full=True); self.box = box; self.l = l
        def __call__(self, w, x):
            b = self.box
            if b.A is not None and self.l != SPLIT:
                b.A = m.transformer.h[self.l].lambdas[0] * b.A
            return super().__call__(w, b.see(x, self.l in b.attns))
    class CMLP(OwnHead):
        """Late MLP l: computes from x - A if it is a consumer; if it is a writer, adds its centred tail write to A. rec: fit-pass sums."""
        def __init__(self, l, box, mu=None, rec=None):
            super().__init__(l, 768, "const", Uk); self.l = l; self.box = box; self.mu = mu; self.rec = rec
        def __call__(self, w, x):
            b = self.box; xh = F.rms_norm(b.see(x, self.l in b.mlps), (D,)); out = self.mlp.Down(self.mlp.Left(xh) * self.mlp.Right(xh))
            if self.rec is not None:
                r = self.rec; pm = perp(out).reshape(-1, D); r[("sum", self.l)] += pm.double().sum(0); r[("n", self.l)] += pm.shape[0]
            if self.l in b.writers:
                d = perp(out) - self.mu[self.l]; b.A = d if b.A is None else b.A + d
            return out
    class CFinal:
        def __init__(self, box): self.box = box
        def hook(self, idx):
            pass
        def __call__(self, x, _):
            return self.box.see(x, self.box.final)
    rec = {}
    for l in LB:
        rec[("sum", l)] = torch.zeros(D, dtype=torch.float64, device=DEV); rec[("n", l)] = 0
    for i in range(0, fit_rows.shape[0], CH):
        forward(m, fit_rows[i:i + CH, :TI], patch={("mlp", l): CMLP(l, Box(()), rec=rec) for l in LB})
    mu = {l: (rec[("sum", l)] / rec[("n", l)]).float() for l in LB}
    log(stage="fit", n_fit_tokens=rec[("n", SPLIT)], mu_norm_17=round(float(mu[17].norm()), 3))
    def consumer(writers, mlps=(), attns=(), final=False, keep=None):
        box = Box(writers, mlps, attns, final)
        patch = {("mlp", l): CMLP(l, box, mu=mu) for l in LB}; patch.update({("attn", l): CAttn(l, box) for l in LB})
        patch[("final", -1)] = CFinal(box) if keep is None else CFinalKeep(box, keep)
        return run(patch, extra=(box,))
    Ut = U_8[:, 768:]
    class CFinalKeep(CFinal):
        """Final readout sees x - A + (the part of A inside the k-dim tail frame P): the readout is denied everything the late MLPs wrote to the
        tail EXCEPT its component in span(P). P is 384 x k in tail coordinates (columns orthonormal)."""
        def __init__(self, box, P): super().__init__(box); self.P = P
        def __call__(self, x, _):
            if self.box.A is None:
                return x
            return x - self.box.A + (((self.box.A @ Ut) @ self.P) @ self.P.T) @ Ut.T
    class FinalCov(CFinal):
        """Fit pass: second moments of A's tail coordinates at the final site."""
        def __init__(self, box, acc): super().__init__(box); self.acc = acc
        def __call__(self, x, _):
            if self.box.A is not None:
                Y = (self.box.A @ Ut).reshape(-1, 384).double(); self.acc["S"] += Y.T @ Y; self.acc["s"] += Y.sum(0); self.acc["n"] += Y.shape[0]
            return x
    acc = {"S": torch.zeros(384, 384, dtype=torch.float64, device=DEV), "s": torch.zeros(384, dtype=torch.float64, device=DEV), "n": 0}
    for i in range(0, fit_rows.shape[0], CH):
        box = Box(LB); patch = {("mlp", l): CMLP(l, box, mu=mu) for l in LB}; patch.update({("attn", l): CAttn(l, box) for l in LB}); patch[("final", -1)] = FinalCov(box, acc)
        idx = fit_rows[i:i + CH, :TI]
        for o in patch.values():
            o.hook(idx)
        box.hook(idx); forward(m, idx, patch)
    mA = acc["s"] / acc["n"]; C_A = acc["S"] / acc["n"] - torch.outer(mA, mA)
    evA, UA = torch.linalg.eigh(C_A); UA = UA.flip(1).float(); specA = R.spectrum(C_A.cpu())
    # the unembedding's own tail frame: right singular vectors of W_U restricted to the tail (V x 384), ordered by logit energy
    WU = (m.lm_head.weight.detach().to(DEV).float() @ Ut).double(); _, sWU, VhWU = torch.linalg.svd(WU, full_matrices=False); VWU = VhWU.T.float()
    sWU = sWU.float(); pr_ = (sWU ** 2) / (sWU ** 2).sum(); eff_WU = float(torch.exp(-(pr_ * pr_.log()).sum()))
    # tail-covariance energy of A that the unembedding's top-k frame captures, and the frame overlap at k = 32
    def cap(P):
        return float(torch.trace(P.double().T @ C_A @ P.double()) / torch.trace(C_A))
    # --- the frequency direction: unit vector u over the vocabulary = centred log(1 + fit-corpus count); its best tail direction d_f ∝ WUᵀu ---
    cnt = torch.bincount(fit_rows.reshape(-1), minlength=V).double()
    u = torch.log1p(cnt); u = u - u.mean(); u = u / u.norm()
    d_f = WU.T @ u; d_f = (d_f / d_f.norm()).float().unsqueeze(1)                       # 384 x 1
    def orth(*cols):
        Mx = torch.cat(cols, 1); Q, _ = torch.linalg.qr(torch.cat([Mx, torch.eye(384, device=DEV)], 1)); return Q
    Q_f = orth(d_f); P_dropf = Q_f[:, 1:]                                              # orthonormal complement of d_f (384 x 383)
    P_f_wu32 = orth(d_f, VWU[:, :32])[:, :33]
    g = torch.Generator(device="cpu").manual_seed(0)
    rand_dirs = [F.normalize(torch.randn(384, 1, generator=g), dim=0).to(DEV) for _ in range(N_RAND)]
    wu_share_f = float((WU @ d_f.double()).pow(2).sum() / WU.pow(2).sum())            # share of W_U's tail logit energy along d_f
    frame_facts = {"cos_df_WU1": float((d_f[:, 0] @ VWU[:, 0]).abs()), "cos_df_PCA1": float((d_f[:, 0] @ UA[:, 0]).abs()), "tail_cov_energy_along_df": cap(d_f),
                   "tail_cov_energy_WU1": cap(VWU[:, :1]), "tail_cov_energy_PCA1": cap(UA[:, :1]), "WU_logit_energy_share_along_df": wu_share_f,
                   "cos_u_vs_WU_meanA": float((u @ (WU @ (mA / mA.norm()))) / (WU @ (mA / mA.norm())).norm())}
    log(stage="frames", **{k: round(v, 4) for k, v in frame_facts.items()})
    def tok_loss(patch=None, extra=()):
        """Per-token held-out CE [n_eval_docs, TI] (double) and the mean predictive entropy (nat) under a patch; same forward as ce_of."""
        objs = (list(patch.values()) if patch else []) + list(extra); outs = []; ent = 0.0; n = 0
        for i in range(0, ev.shape[0], CH):
            idx = ev[i:i + CH, :TI]; tgt = ev[i:i + CH, 1:TI + 1]
            for o in objs:
                o.hook(idx)
            lp = F.log_softmax(forward(m, idx, patch).reshape(-1, V).float(), -1)
            outs.append(F.nll_loss(lp, tgt.reshape(-1), reduction="none").view(idx.shape[0], TI)); ent += float(-(lp.exp() * lp).sum()); n += lp.shape[0]
        return torch.cat(outs, 0).double(), ent / n
    def final_tok(keep=None):
        box = Box(LB, final=True)
        patch = {("mlp", l): CMLP(l, box, mu=mu) for l in LB}; patch.update({("attn", l): CAttn(l, box) for l in LB})
        patch[("final", -1)] = CFinal(box) if keep is None else CFinalKeep(box, keep)
        return tok_loss(patch, extra=(box,))
    arms = {"SPLIT8_1024": split8(1024)}
    base, ent0 = tok_loss()
    tok = {"FINAL_ONLY": final_tok(), "KEEP_ALL": final_tok(torch.eye(384, device=DEV)), "KEEP_FREQ1": final_tok(d_f), "DROP_FREQ1": final_tok(P_dropf),
           "KEEP_WU1": final_tok(VWU[:, :1]), "KEEP_PCA1": final_tok(UA[:, :1]), "KEEP_FREQ1_WU32": final_tok(P_f_wu32)}
    tok.update({f"KEEP_RAND1_{i}": final_tok(rd) for i, rd in enumerate(rand_dirs)})
    ents = {k: v[1] for k, v in tok.items()}; delta = {k: v[0] - base for k, v in tok.items()}
    arms.update({k: float(v.mean()) for k, v in delta.items()})
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    def sdiv(a, b):
        return (a / b) if (a is not None and b is not None and abs(b) > 1e-12) else None
    F_ = arms["FINAL_ONLY"]
    rec_ = {k[5:]: 1.0 - sdiv(arms[k], F_) for k in arms if k.startswith("KEEP_") and k != "KEEP_ALL"}
    rec_rand = sum(rec_[f"RAND1_{i}"] for i in range(N_RAND)) / N_RAND
    drop_frac = sdiv(arms["DROP_FREQ1"], F_)
    cls = target_token_classes(ev[:, :TI], ev[:, 1:TI + 1])
    fc = cnt[ev[:, 1:TI + 1]]
    def by_class(dv):
        return {c: float(dv[cls[c]].sum()) / int(cls[c].sum()) if int(cls[c].sum()) else None for c in ("induction", "repeat", "novel")}
    def by_freq(dv):
        return {"unseen": float(dv[fc == 0].mean()) if int((fc == 0).sum()) else None, "common_ge64": float(dv[fc >= 64].mean()) if int((fc >= 64).sum()) else None}
    recov_f = delta["FINAL_ONLY"] - delta["KEEP_FREQ1"]                                 # per-token CE recovered by keeping d_f (LOWER KEEP = more recovered)
    tot_recov = float(recov_f.sum())
    novel_share_of_recovery = sdiv(float(recov_f[cls["novel"]].sum()), tot_recov)
    summ = {"recovered_fraction_of_readout_consumption": rec_, "recovered_rand1_mean": rec_rand, "drop_freq1_over_final_only": drop_frac,
            "freq1_over_wu1": sdiv(rec_["FREQ1"], rec_["WU1"]), "freq1_over_pca1": sdiv(rec_["FREQ1"], rec_["PCA1"]),
            "freq1_wu32_minus_wu32_prior_2798": rec_["FREQ1_WU32"] - PRIOR_REC_WU32, "prior_rec_wu32_2798": PRIOR_REC_WU32,
            "entropy": {"baseline": ent0, **ents, "final_only_minus_baseline": ents["FINAL_ONLY"] - ent0, "keep_freq1_minus_baseline": ents["KEEP_FREQ1"] - ent0},
            "per_token_by_class": {k: by_class(delta[k]) for k in ("FINAL_ONLY", "KEEP_FREQ1", "DROP_FREQ1")},
            "per_token_by_freq": {k: by_freq(delta[k]) for k in ("FINAL_ONLY", "KEEP_FREQ1", "DROP_FREQ1")},
            "novel_share_of_freq1_recovery": novel_share_of_recovery, "frame_facts": frame_facts, "prior_final_only_2797": PRIOR_FINAL_ONLY, "n_fit_tokens": rec[("n", SPLIT)]}
    def ge(v, b):
        return bool(v is not None and v >= b)
    def le(v, b):
        return bool(v is not None and v <= b)
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["SPLIT8_1024"] - PRIOR_SPLIT8_1024) <= BARS["repro_tol"] and abs(F_ - PRIOR_FINAL_ONLY) <= BARS["repro_tol"]
                   and abs(arms["KEEP_ALL"]) <= BARS["none_tol"]) if not smoke else bool(abs(arms["KEEP_ALL"]) <= BARS["none_tol"])
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_frequency_direction_alone_recovers_a_real_share': ge(rec_["FREQ1"], BARS["b_rec"]),
        'pred_c_frequency_direction_beats_the_top_unembed_direction': bool(rec_["WU1"] is not None and rec_["FREQ1"] >= BARS["c_mult"] * rec_["WU1"]),
        'pred_d_channel_is_mostly_identity_not_frequency': ge(drop_frac, BARS["d_drop"]),
        'pred_e_withholding_the_channel_raises_entropy': ge(ents["FINAL_ONLY"] - ent0, BARS["e_ent"]),
        'pred_f_frequency_direction_value_sits_on_novel_targets': ge(novel_share_of_recovery, BARS["f_share"]),
    }
    nulls = {"b_null_rec_freq1_le_.05": le(rec_["FREQ1"], NULLS["b_rec"]), "c_null_freq1_le_wu1": bool(rec_["WU1"] is not None and rec_["FREQ1"] <= NULLS["c_mult"] * rec_["WU1"]),
             "d_null_drop_frac_le_.6": le(drop_frac, NULLS["d_drop"]), "e_null_entropy_change_le_0": le(ents["FINAL_ONLY"] - ent0, NULLS["e_ent"]),
             "f_null_novel_share_le_.4": le(novel_share_of_recovery, NULLS["f_share"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); recovered = 1 - KEEP/FINAL_ONLY; entropy in nats",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "split": SPLIT, "n_rand": N_RAND,
           "program": "the §2797/§2798 keep instrument (late-MLP centred tail writes A withheld from the FINAL READOUT except their component in a tail frame P) with P = the frequency direction d_f ∝ (W_U Ut)ᵀ u (u = centred log(1+count) over the vocabulary, fit corpus), its 383-dim complement, the top unembed / PCA directions, random directions, and d_f ⊕ WU32; per-token deltas by class (ops/target_token_classes.py) and frequency; mean predictive entropy",
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "split8_1024": arms["SPLIT8_1024"], "prior_split8_1024": PRIOR_SPLIT8_1024, "final_only": F_, "prior_final_only": PRIOR_FINAL_ONLY,
                          "keep_all": arms["KEEP_ALL"], "tok_mean_baseline": float(base.mean())},
           "ce_added": arms, "summary": summ,
           "price": {"gpu_doc_forwards": 2 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (2 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
